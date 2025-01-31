"""Main module for the Lore service"""

import threading
import json
import time
from typing import Optional
from transformers import AutoModelForCausalLM, AutoTokenizer
import structlog
from prometheus_client import start_http_server
from valkey import Valkey

from config import Settings
from models import ChunkAck, Question, Error, MessageType
from metrics import Metrics
from buffer import ChunkBuffer


class LoreService:
    """Service for processing student questions with AI-generated explanations"""

    def __init__(self) -> None:
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
        self.buffer_condition = threading.Condition(self.buffer_lock)

        # Initialize Valkey client
        self._init_valkey()

        # Initialize model
        self._init_model()

    def _init_valkey(self) -> None:
        """Initialize Valkey client with error handling"""
        try:
            self.log.info("connecting_to_valkey", host=self.settings.VALKEY_HOST)
            self.valkey = Valkey(
                host=self.settings.VALKEY_HOST,
                port=self.settings.VALKEY_PORT,
                db=self.settings.VALKEY_DB,
            )
        except Exception as e:
            self.log.error("valkey_initialization_failed", error=str(e))
            raise

    def _init_model(self) -> None:
        """Initialize AI model with error handling"""
        try:
            self.log.info("initializing_model", model=self.settings.MODEL_NAME)

            self.model = AutoModelForCausalLM.from_pretrained(
                self.settings.MODEL_NAME,
                device_map=self.settings.DEVICE
            )
            self.tokenizer = AutoTokenizer.from_pretrained(self.settings.MODEL_NAME)
            self.tokenizer.pad_token = self.tokenizer.eos_token
        except Exception as e:
            self.log.error("model_initialization_failed", error=str(e))
            raise

    def _clean_stale_buffers(self, max_age: float = 300.0) -> None:
        """Remove buffers that haven't been updated in the specified time"""
        current_time = time.time()
        with self.buffer_condition:
            stale_buffers = [
                request_id for request_id, buffer in self.active_buffers.items()
                if (current_time - buffer.last_update) > max_age
            ]
            for request_id in stale_buffers:
                self.log.warning("removing_stale_buffer", request_id=request_id)
                del self.active_buffers[request_id]

            # Notify waiting threads that buffer state has changed
            if stale_buffers:
                self.buffer_condition.notify_all()

    def generate_response(self, question: Question) -> None:
        """Generate and buffer response chunks with metrics and logging"""
        start_time = time.time()
        total_tokens = 0
        buffer = ChunkBuffer(question.request_id, self.buffer_condition)  # Pass condition here

        with self.buffer_condition:
            self.active_buffers[question.request_id] = buffer

        try:
            # Time the model inference
            with self.metrics.model_inference_time.time():
                full_prompt = f"{self.settings.SYSTEM_PROMPT}\n\nQuestion: {question.question}"

                self.log.info("configuring_model", request_id=question.request_id)
                inputs = self.tokenizer(
                    full_prompt,
                    return_tensors="pt",
                    max_length=self.settings.MODEL_MAX_LENGTH,
                    truncation=True
                )

                attention_mask = inputs["input_ids"].ne(self.tokenizer.pad_token_id)

                # Stream tokens
                self.log.info("generating_tokens", request_id=question.request_id)

                # Inside generate_response method
                tokens = self.model.generate(
                    inputs["input_ids"],
                    do_sample=True,
                    temperature=0.7,
                    max_new_tokens=100,
                    attention_mask=attention_mask,
                    pad_token_id=self.tokenizer.eos_token_id,
                    min_length=30,  # Force minimum generation length
                    repetition_penalty=1.2,  # Reduce repetitive text
                    no_repeat_ngram_size=2
                )

                # Get full response minus the prompt
                prompt_length = len(inputs["input_ids"][0])
                generated_tokens = tokens[0][prompt_length:]
                self.log.info("decoding_token", request_id=question.request_id)
                response = self.tokenizer.decode(generated_tokens, skip_special_tokens=True)


                self.log.info("generating_chunks", request_id=question.request_id)
                chunks = [response[i:i+50] for i in range(0, len(response), 50)]

                for chunk in chunks:
                    total_tokens += 1
                    self.log.info("adding_chunk", request_id=question.request_id)
                    chunk_obj = buffer.add_chunk(chunk)

                    # Move the buffer check outside of condition to prevent deadlock
                    if chunk_obj:
                        self.log.info("publishing_chunk", request_id=question.request_id)
                        self.valkey.publish(
                            f'{self.settings.RESPONSE_CHANNEL_PREFIX}{question.request_id}',
                            chunk_obj.model_dump_json()
                        )

                        # Wait for acknowledgment after publishing
                        with self.buffer_condition:
                            while len(buffer.buffer) >= buffer.window_size:
                                self.log.debug("waiting_for_ack",
                                    request_id=question.request_id,
                                    buffer_size=len(buffer.buffer))
                                self.buffer_condition.wait(timeout=1.0)  # Add timeout to prevent indefinite wait

            # Mark final chunk and send completion metrics
            final_chunk = buffer.add_chunk("", is_final=True)
            if final_chunk:
                self.valkey.publish(
                    f'{self.settings.RESPONSE_CHANNEL_PREFIX}{question.request_id}',
                    final_chunk.model_dump_json()
                )

            completion_time = time.time() - start_time

            # Update all metrics
            self.metrics.questions_processed.inc()
            self.metrics.response_time.observe(completion_time)
            self.metrics.token_usage.inc(total_tokens)

            self.log.info("response_generated",
                    request_id=question.request_id,
                    completion_time=completion_time,
                    token_count=total_tokens)

        except Exception as e:
            error = self._handle_error(e, question.request_id)
            self._send_error_chunk(error, buffer)

        finally:
            self._cleanup_buffer(question.request_id)

    def _handle_error(self, e: Exception, request_id: Optional[int]) -> Error:
        """Standardized error handling"""
        error = Error(
            request_id=request_id,
            error_type=type(e).__name__,
            error_message=str(e),
        )
        self.metrics.errors_total.labels(error_type=error.error_type).inc()
        self.log.error("error_occurred", error=error.model_dump())
        return error

    def _send_error_chunk(self, error: Error, buffer: ChunkBuffer) -> None:
        """Send error message to client through buffer"""
        if error.request_id:
            error_chunk = buffer.add_chunk(
                f"Error: {error.error_message}",
                is_final=True
            )
            if error_chunk:
                self.valkey.publish(
                    f'{self.settings.RESPONSE_CHANNEL_PREFIX}{error.request_id}',
                    error_chunk.model_dump_json()
                )

    def _cleanup_buffer(self, request_id: int) -> None:
        """Clean up buffer if empty"""
        with self.buffer_lock:
            if request_id in self.active_buffers:
                if not self.active_buffers[request_id].buffer:
                    del self.active_buffers[request_id]

    def handle_message(self, message: dict) -> None:
        """Process incoming message from Valkey"""
        try:
            data = json.loads(message["data"])
            message_type = data.get("type", MessageType.QUESTION)

            if message_type == MessageType.QUESTION:
                self._handle_question_message(data)
            elif message_type == MessageType.ACK:
                self._handle_ack_message(data)
            else:
                self.log.warning("unknown_message_type", type=message_type)

        except json.JSONDecodeError as e:
            self._handle_error(e, None)
        except Exception as e:
            self._handle_error(e, None)

    def _handle_question_message(self, data: dict) -> None:
        """Handle incoming question message"""
        try:
            question = Question(**data)
            self.log.info(
                "received_question",
                request_id=question.request_id,
                user_id=question.user_id,
            )
            self.metrics.queue_length.inc()
            self.generate_response(question)
        finally:
            self.metrics.queue_length.dec()

    def _handle_ack_message(self, data: dict) -> None:
        """Handle acknowledgment message"""
        try:
            ack = ChunkAck(**data)
            self.log.info("received_ack", request_id=ack.request_id, sequence_number=data['sequence_number'])

            with self.buffer_condition:
                if ack.request_id in self.active_buffers:
                    buffer = self.active_buffers[ack.request_id]
                    next_chunk = buffer.ack_received(ack.sequence_number)

                    # Notify waiting threads before publishing next chunk
                    self.buffer_condition.notify_all()

                    if next_chunk:
                        self.valkey.publish(
                            f"{self.settings.RESPONSE_CHANNEL_PREFIX}{ack.request_id}",
                            next_chunk.model_dump_json(),
                        )

        except Exception as e:
            self._handle_error(e, data.get("request_id"))

    def start(self) -> None:
        """Start the service"""
        try:
            # Start Prometheus metrics server
            start_http_server(self.settings.PROMETHEUS_PORT)
            self.log.info("metrics_server_started", port=self.settings.PROMETHEUS_PORT)

            # Subscribe to channels
            pubsub = self.valkey.pubsub()
            # pubsub.subscribe(self.settings.QUESTION_CHANNEL)
            # pubsub.subscribe(self.settings.ACK_CHANNEL)
            pubsub.subscribe(self.settings.QUESTION_CHANNEL, self.settings.ACK_CHANNEL)

            while True:
                message = pubsub.get_message()
                if not message:
                    break
                self.log.info("subscription_confirmation", message=message)
                if message['type'] == 'subscribe':
                    self.log.info("subscribed_to_channel",
                        channel=message['channel'].decode(),
                        total_subscriptions=message['data']
                    )

            self.log.info(
                "service_started",
                channels=[self.settings.QUESTION_CHANNEL, self.settings.ACK_CHANNEL],
                valkey_host=self.settings.VALKEY_HOST,
            )

            try:
                # Main message loop
                message_count = 0
                last_clean = time.time()

                while True:
                    # Get message with short timeout
                    message = pubsub.get_message(timeout=0.01)  # Shorter timeout for more frequent checks

                    if message and message["type"] == "message":
                        self.log.debug("received_message", message=message)
                        self.handle_message(message)
                        message_count += 1

                    # Clean stale buffers every 100 messages or every 5 seconds
                    current_time = time.time()
                    if message_count >= 100 or (current_time - last_clean) >= 5.0:
                        if self.active_buffers:
                            self._clean_stale_buffers()
                        message_count = 0
                        last_clean = current_time

            except KeyboardInterrupt:
                self.log.info("Shutting down gracefully...")
            except Exception as e:
                self.log.error("Fatal error", error=str(e))
                raise

        except Exception as e:
            error = self._handle_error(e, None)
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

# /Users/chortlehoort/Library/Caches/pypoetry/virtualenvs/lore-r_e3AbgO-py3.10/lib/python3.10/site-packages/transformers/generation/configuration_utils.py:628: UserWarning: `do_sample` is set to `False`. However, `temperature` is set to `0.7` -- this flag is only used in sample-based generation modes. You should set `do_sample=True` or unset `temperature`.
#   warnings.warn(
# The attention mask and the pad token id were not set. As a consequence, you may observe unexpected behavior. Please pass your input's `attention_mask` to obtain reliable results.
# Setting `pad_token_id` to `eos_token_id`:2 for open-end generation.
# The attention mask is not set and cannot be inferred from input because pad token is same as eos token. As a consequence, you may observe unexpected behavior. Please pass your input's `attention_mask` to obtain reliable results.
# Both `max_new_tokens` (=20) and `max_length`(=500) seem to have been set. `max_new_tokens` will take precedence. Please refer to the documentation for more information. (https://huggingface.co/docs/transformers/main/en/main_classes/text_generation)