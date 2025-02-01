"""Main module for the Lore service"""

import threading
import json
import time
import queue  # Add this import
from typing import Optional
from transformers import AutoModelForCausalLM, AutoTokenizer
import structlog
from prometheus_client import start_http_server
from valkey import Valkey
import concurrent.futures
from concurrent.futures import ThreadPoolExecutor

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

        self.response_queue = queue.Queue()

        # Replace individual threads with thread pool
        self.executor = ThreadPoolExecutor(
            max_workers=2,
            thread_name_prefix="lore"
        )
        self.futures = []

        self.retransmit_interval = 1.0
        self.retransmit_thread = threading.Thread(
            target=self._check_retransmissions,
            daemon=True
        )

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
        buffer = ChunkBuffer(question.request_id)  # Pass condition here

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
            self.log.info("received_ack",
                request_id=ack.request_id,
                sequence_number=ack.sequence_number)

            with self.buffer_condition:
                if ack.request_id in self.active_buffers:
                    buffer = self.active_buffers[ack.request_id]
                    next_chunk = buffer.ack_received(ack.sequence_number)

                    # Notify waiting threads before publishing next chunk
                    self.buffer_condition.notify_all()

                    if next_chunk:
                        self.log.debug("sending_next_chunk",
                            request_id=ack.request_id,
                            sequence_number=next_chunk.sequence_number)
                        self.valkey.publish(
                            f"{self.settings.RESPONSE_CHANNEL_PREFIX}{ack.request_id}",
                            next_chunk.model_dump_json()
                        )

        except Exception as e:
            self._handle_error(e, data.get("request_id"))

    def _check_retransmissions(self) -> None:
        """Periodically check for chunks that need retransmission"""
        while True:
            time.sleep(self.retransmit_interval)
            for buffer in self.active_buffers.values():
                chunks = buffer.check_retransmissions()
                for chunk in chunks:
                    self.log.info("retransmitting_chunk",
                        request_id=chunk.request_id,
                        sequence_number=chunk.sequence_number)
                    self.valkey.publish(
                        f'{self.settings.RESPONSE_CHANNEL_PREFIX}{chunk.request_id}',
                        chunk.model_dump_json()
                    )

    def _process_responses(self) -> None:
        """Process responses in worker thread"""
        while True:
            try:
                question = self.response_queue.get()
                self.generate_response(question)
            except Exception as e:
                self.log.error("response_generation_error", error=str(e))
            finally:
                self.response_queue.task_done()

    def _process_questions(self) -> None:
        """Dedicated thread for handling questions and response generation"""
        pubsub = self.valkey.pubsub()
        self.log.info("subscribing_to_channel", channel=self.settings.QUESTION_CHANNEL)
        pubsub.subscribe(self.settings.QUESTION_CHANNEL)

        while True:
            message = pubsub.get_message(timeout=0.1)
            if message and message["type"] == "message":
                self.log.debug("received_message", message=message, channel=self.settings.QUESTION_CHANNEL)
                try:
                    data = json.loads(message["data"])
                    self._handle_question_message(data)
                except Exception as e:
                    self.log.error("question_processing_error", error=str(e))

    def _process_acks(self) -> None:
        """Dedicated thread for handling ACK messages"""
        pubsub = self.valkey.pubsub()
        self.log.info("subscribing_to_channel", channel=self.settings.ACK_CHANNEL)
        pubsub.subscribe(self.settings.ACK_CHANNEL)

        while True:
            message = pubsub.get_message(timeout=0.1)
            if message and message["type"] == "message":
                self.log.debug("received_message", message=message, channel=self.settings.ACK_CHANNEL)
                try:
                    data = json.loads(message["data"])
                    self._handle_ack_message(data)
                except Exception as e:
                    self.log.error("ack_processing_error", error=str(e))

    def start(self) -> None:
        """Start the service"""
        try:
            # Start Prometheus metrics server
            start_http_server(self.settings.PROMETHEUS_PORT)
            self.log.info("metrics_server_started", port=self.settings.PROMETHEUS_PORT)

            # Submit long-running tasks to thread pool
            self.log.info("starting_question_processing_thread")
            self.futures.append(self.executor.submit(self._process_questions))
            self.log.info("starting_ack_processing_thread")
            self.futures.append(self.executor.submit(self._process_acks))

            # Wait for completion (or KeyboardInterrupt)
            concurrent.futures.wait(self.futures, return_when=concurrent.futures.FIRST_EXCEPTION)

        except KeyboardInterrupt:
            self.log.info("Shutting down gracefully...")
            self.executor.shutdown(wait=True)
        except Exception as e:
            self.log.error("Fatal error", error=str(e))
            self.executor.shutdown(wait=False)
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
