"""Main module for the Lore service"""

import threading
import json
import time
from transformers import AutoModelForCausalLM, AutoTokenizer
import structlog
from prometheus_client import start_http_server
from valkey import Valkey

from config import Settings
from models import ChunkAck, Question, Response, Error
from metrics import Metrics
from buffer import ChunkBuffer


class LoreService:
    """Service for processing student questions with AI-generated explanations"""

    def __init__(self):
        # Load configuration
        self.settings = Settings()

        # Initialize structured logging
        self.log = structlog.get_logger()
        self.log.info("initializing_service", log_level=self.settings.LOG_LEVEL)

        # Initialize metrics
        self.metrics = Metrics()

        # Set up buffer for managing unacknowledged text chunks
        self.active_buffers: dict[int, ChunkBuffer] = {}
        self.buffer_lock = threading.Lock()

        # Initialize Valkey client
        self.valkey = Valkey(
            host=self.settings.VALKEY_HOST,
            port=self.settings.VALKEY_PORT,
            db=self.settings.VALKEY_DB,
        )

        # Initialize model
        self.log.info("initializing_model", model=self.settings.MODEL_NAME)
        self.model = AutoModelForCausalLM.from_pretrained(
            self.settings.MODEL_NAME, device_map=self.settings.DEVICE
        )
        self.tokenizer = AutoTokenizer.from_pretrained(self.settings.MODEL_NAME)

        # System prompt
        self.SYSTEM_PROMPT = """You are a programming concept explainer.
        You must ONLY provide natural language explanations.
        Never generate or include code examples.
        Never use code formatting, backticks, or code blocks.
        Explain concepts using analogies and plain language instead.
        The explanations must be generated to be understandable by a beginner.
        The explanations must assume the user has no prior knowledge of the concept."""

    def generate_response(self, question: Question) -> None:
        """Generate and buffer response chunks with metrics and logging"""
        start_time = time.time()
        total_tokens = 0
        buffer = ChunkBuffer(question.request_id)

        with self.buffer_lock:
            self.active_buffers[question.request_id] = buffer

        try:
            # Time the model inference
            with self.metrics.model_inference_time.time():
                full_prompt = f"{self.SYSTEM_PROMPT}\n\nQuestion: {question.question}"
                inputs = self.tokenizer(
                    full_prompt,
                    return_tensors="pt",
                    max_length=self.settings.MODEL_MAX_LENGTH,
                    truncation=True
                )

                # Stream tokens
                for token in self.model.generate(
                    inputs["input_ids"],
                    max_length=self.settings.MODEL_MAX_LENGTH,
                    temperature=self.settings.MODEL_TEMPERATURE,
                    num_return_sequences=1,
                    streaming=True
                ):
                    chunk = self.tokenizer.decode(token, skip_special_tokens=True)
                    total_tokens += 1

                    chunk_obj = buffer.add_chunk(chunk)

                    if chunk_obj:
                        self.valkey.publish(
                            f'{self.settings.RESPONSE_CHANNEL_PREFIX}{question.request_id}',
                            chunk_obj.model_dump_json()
                        )

                    # If buffer is full, wait for acks
                    while len(buffer.buffer) >= buffer.window_size:
                        time.sleep(0.2)

            # Mark final chunk and send completion metrics
            final_chunk = buffer.add_chunk("", is_final=True)
            if final_chunk:
                self.valkey.publish(
                    f'{self.settings.RESPONSE_CHANNEL_PREFIX}{question.request_id}',
                    final_chunk.model_dump_json()
                )

            completion_time = time.time() - start_time

            # Update all metrics
            self.metrics.questions_processed.labels(
                cohort_id=str(question.cohort_id or 'none')
            ).inc()
            self.metrics.response_time.observe(completion_time)
            self.metrics.token_usage.labels(
                cohort_id=str(question.cohort_id or 'none')
            ).inc(total_tokens)

            self.log.info("response_generated",
                    request_id=question.request_id,
                    completion_time=completion_time,
                    token_count=total_tokens)

        except Exception as e:
            self.metrics.errors_total.labels(
                error_type=type(e).__name__
            ).inc()

            error = Error(
                request_id=question.request_id,
                error_type=type(e).__name__,
                error_message=str(e),
            )
            self.log.error("error_generating_response",
                        error=error.model_dump())

            # Try to send error message to client
            error_chunk = buffer.add_chunk(
                f"Error generating response: {str(e)}",
                is_final=True
            )
            if error_chunk:
                self.valkey.publish(
                    f'{self.settings.RESPONSE_CHANNEL_PREFIX}{question.request_id}',
                    error_chunk.model_dump_json()
                )

        finally:
            # Clean up buffer if it's empty
            with self.buffer_lock:
                if question.request_id in self.active_buffers:
                    if not self.active_buffers[question.request_id].buffer:
                        del self.active_buffers[question.request_id]
    def handle_message(self, message: dict) -> None:
        """Process incoming message from Valkey"""
        start_time = time.time()
        data = None

        try:
            # Parse and validate payload
            data = json.loads(message["data"])
            question = Question(**data)

            self.log.info(
                "received_question",
                request_id=question.request_id,
                user_id=question.user_id,
                cohort_id=question.cohort_id,
            )

            # Update queue metrics
            self.metrics.queue_length.inc()

            # Generate response
            response = self.generate_response(question)

            # Publish response
            self.valkey.publish(
                f"{self.settings.RESPONSE_CHANNEL_PREFIX}{question.request_id}",
                response.model_dump_json(),
            )

            # Update metrics
            self.metrics.questions_processed.labels(
                cohort_id=str(question.cohort_id or "none")
            ).inc()
            self.metrics.response_time.observe(response.completion_time)

            self.log.info(
                "response_generated",
                request_id=question.request_id,
                completion_time=response.completion_time,
                token_count=response.token_count,
            )

        except Exception as e:
            self.metrics.errors_total.labels(error_type=type(e).__name__).inc()

            error = Error(
                request_id=data.get("request_id") if data else None,
                error_type=type(e).__name__,
                error_message=str(e),
            )
            self.log.error("error_processing_question", error=error.model_dump())
        finally:
            self.metrics.queue_length.dec()

    def handle_ack(self, message: dict) -> None:
        """Handle acknowledgment messages from clients"""
        try:
            ack = ChunkAck(**json.loads(message["data"]))

            with self.buffer_lock:
                if ack.request_id in self.active_buffers:
                    buffer = self.active_buffers[ack.request_id]
                    next_chunk = buffer.ack_received(ack.sequence_number)

                    if next_chunk:
                        # Send next chunk
                        self.valkey.publish(
                            f"{self.settings.RESPONSE_CHANNEL_PREFIX}{ack.request_id}",
                            next_chunk.model_dump_json(),
                        )

                    # Clean up completed buffers
                    if not buffer.buffer and buffer.is_complete:
                        del self.active_buffers[ack.request_id]
        except Exception as e:
            self.log.error("error_processing_ack", error=str(e))

    def start(self) -> None:
        """Start the service"""
        # Start Prometheus metrics server
        start_http_server(self.settings.PROMETHEUS_PORT)
        self.log.info("metrics_server_started", port=self.settings.PROMETHEUS_PORT)

        # Subscribe to questions channel
        pubsub = self.valkey.pubsub()
        pubsub.subscribe(self.settings.QUESTION_CHANNEL)

        self.log.info(
            "service_started",
            channel=self.settings.QUESTION_CHANNEL,
            valkey_host=self.settings.VALKEY_HOST,
        )

        # Listen for messages
        try:
            for message in pubsub.listen():
                if message["type"] == "message":
                    self.handle_message(message)
        except Exception as e:
            error = Error(
                error_type=type(e).__name__,
                error_message=str(e),
            )
            self.log.error("service_error", error=error.model_dump())
            raise


def main() -> None:
    """Main entry point for the service"""
    try:
        service = LoreService()
        service.start()
    except Exception as e:
        # Create a temporary logger for startup errors
        startup_log = structlog.get_logger()
        error = Error(
            error_type=type(e).__name__,
            error_message=str(e),
        )
        startup_log.error("startup_failed", error=error.model_dump())
        raise


if __name__ == "__main__":
    main()
