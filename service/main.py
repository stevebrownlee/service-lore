"""Main module for the Lore service"""

import threading
import json
import time
import queue
from typing import Optional
from transformers import AutoModelForCausalLM, AutoTokenizer, TextIteratorStreamer
import structlog
import torch
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

        structlog.configure(
            processors=[
                structlog.contextvars.merge_contextvars,
                structlog.processors.add_log_level,
                structlog.processors.TimeStamper(fmt="iso"),
                structlog.processors.format_exc_info,
                structlog.processors.JSONRenderer()
            ],
            wrapper_class=structlog.make_filtering_bound_logger(self.settings.LOG_LEVEL),
            context_class=dict,
            logger_factory=structlog.PrintLoggerFactory(),
            cache_logger_on_first_use=True
        )

        # Initialize structured logging
        self.log = structlog.get_logger()

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

        self.retransmit_interval = 1.0

        # Create dedicated threads for message handling
        self.question_thread = threading.Thread(
            target=self._process_questions,
            name="question_processor",
            daemon=True
        )
        self.ack_thread = threading.Thread(
            target=self._process_acks,
            name="ack_processor",
            daemon=True
        )
        self.retransmit_thread = threading.Thread(
            target=self._check_retransmissions,
            name="retransmission_checker",
            daemon=True
        )

        # Track thread health
        self.running = threading.Event()

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
                torch_dtype=torch.float16,
                device_map="cpu"
            ).to(self.settings.DEVICE)

            self.tokenizer = AutoTokenizer.from_pretrained(self.settings.MODEL_NAME)

        except Exception as e:
            self.log.error("model_initialization_failed", error=str(e))
            raise

    def _clean_stale_buffers(self, max_age: float = 300.0) -> None:
        """Remove buffers that haven't been updated in the specified time"""
        current_time = time.time()
        with self.buffer_condition:
            stale_buffers = []
            for request_id, buffer in self.active_buffers.items():
                # Check both buffer age and if all chunks are expired
                if ((current_time - buffer.last_update) > max_age or
                    (not buffer.buffer and current_time - buffer.created_at > buffer.max_chunk_age)):
                    stale_buffers.append(request_id)

            for request_id in stale_buffers:
                self.log.warning("removing_stale_buffer",
                               request_id=request_id,
                               age=current_time - self.active_buffers[request_id].last_update)
                del self.active_buffers[request_id]

            # Notify waiting threads that buffer state has changed
            if stale_buffers:
                self.buffer_condition.notify_all()

    def generate_response(self, question: Question) -> None:
        """Generate and buffer response chunks with metrics and logging"""
        start_time = time.time()
        total_tokens = 0
        buffer = ChunkBuffer(question.request_id)

        with self.buffer_condition:
            self.active_buffers[question.request_id] = buffer

        try:
            with self.metrics.model_inference_time.time():
                # TinyLlama uses ChatML format
                messages = [
                    {
                        "role": "system",
                        "content": self.settings.SYSTEM_PROMPT
                    },
                    {
                        "role": "user",
                        "content": f"{question.question}"
                    }
                ]

                # Use chat template
                self.log.info("defining_prompt", messages=messages)
                prompt = self.tokenizer.apply_chat_template(
                    messages,
                    tokenize=False,
                    add_generation_prompt=True
                )

                # Prepare inputs
                self.log.info("generating_inputs", request_id=question.request_id)
                inputs = self.tokenizer(
                    prompt,
                    return_tensors="pt",
                    padding=True,
                    truncation=True,
                    max_length=self.settings.MODEL_MAX_LENGTH
                ).to(self.settings.DEVICE)

                # Configure streamer for TinyLlama
                self.log.info("configuring_streamer", request_id=question.request_id)
                streamer = TextIteratorStreamer(
                    self.tokenizer,
                    timeout=None,
                    skip_prompt=True,
                    skip_special_tokens=True
                )

                # TinyLlama specific generation parameters
                self.log.info("starting_generation", request_id=question.request_id)
                thread = threading.Thread(
                    target=self.model.generate,
                    kwargs={
                        "input_ids": inputs["input_ids"],
                        "streamer": streamer,
                        "max_new_tokens": self.settings.MODEL_MAX_LENGTH,
                        "do_sample": True,
                        "temperature": self.settings.MODEL_TEMPERATURE,
                        "top_p": 0.9,
                        "top_k": 50,
                        "repetition_penalty": 1.2,
                        "no_repeat_ngram_size": 3
                    }
                )
                thread.start()

                # Process generated chunks
                current_chunk = ""
                for new_text in streamer:
                    if new_text:
                        current_chunk += new_text
                        total_tokens += len(self.tokenizer(new_text)["input_ids"])

                        # Buffer chunks when they reach defined size
                        if len(current_chunk) >= self.settings.BUFFER_CHUNK_SIZE:
                            self.log.info("adding_chunk",
                                request_id=question.request_id,
                                chunk_size=len(current_chunk)
                            )
                            chunk_obj = buffer.add_chunk(current_chunk)
                            current_chunk = ""

                            if chunk_obj:
                                self.log.info("publishing_chunk",
                                    request_id=question.request_id)
                                self.valkey.publish(
                                    f'{self.settings.RESPONSE_CHANNEL_PREFIX}{question.request_id}',
                                    chunk_obj.model_dump_json()
                                )

                                with self.buffer_condition:
                                    while len(buffer.buffer) >= buffer.window_size:
                                        self.log.debug("waiting_for_ack",
                                            request_id=question.request_id,
                                            buffer_size=len(buffer.buffer))
                                        self.buffer_condition.wait(timeout=1.0)

                # Handle any remaining text
                if current_chunk:
                    chunk_obj = buffer.add_chunk(current_chunk)
                    if chunk_obj:
                        self.valkey.publish(
                            f'{self.settings.RESPONSE_CHANNEL_PREFIX}{question.request_id}',
                            chunk_obj.model_dump_json()
                        )

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
        """Handle incoming questions in dedicated thread"""
        pubsub = self.valkey.pubsub()
        pubsub.subscribe(self.settings.QUESTION_CHANNEL)

        while self.running.is_set():
            try:
                message = pubsub.get_message(timeout=0.1)
                if message and message["type"] == "message":
                    data = json.loads(message["data"])
                    self._handle_question_message(data)
            except Exception as e:
                self.log.error("question_processing_error", error=str(e))

    def _process_acks(self) -> None:
        """Handle ACK messages in dedicated thread"""
        pubsub = self.valkey.pubsub()
        pubsub.subscribe(self.settings.ACK_CHANNEL)

        while self.running.is_set():
            try:
                message = pubsub.get_message(timeout=0.1)
                if message and message["type"] == "message":
                    data = json.loads(message["data"])
                    self._handle_ack_message(data)
            except Exception as e:
                self.log.error("ack_processing_error", error=str(e))

    def start(self) -> None:
        """Start the service"""
        try:
            # Start Prometheus metrics server
            start_http_server(
                port=self.settings.METRICS_PORT,
                addr=self.settings.METRICS_HOST
            )
            self.log.info("metrics_endpoint_started",
                host=self.settings.METRICS_HOST,
                port=self.settings.METRICS_PORT
            )

            # Set running flag and start threads
            self.running.set()
            self.question_thread.start()
            self.ack_thread.start()
            self.retransmit_thread.start()

            self.log.info("service_started",
                question_thread=self.question_thread.name,
                ack_thread=self.ack_thread.name
            )

            # Keep main thread alive
            while True:
                time.sleep(1)
                if not (self.question_thread.is_alive() and
                       self.ack_thread.is_alive() and
                       self.retransmit_thread.is_alive()):
                    raise RuntimeError("A worker thread has died")

        except KeyboardInterrupt:
            self.log.info("Shutting down gracefully...")
            self.running.clear()  # Signal threads to stop
            self.question_thread.join(timeout=5.0)
            self.ack_thread.join(timeout=5.0)
            self.retransmit_thread.join(timeout=5.0)
        except Exception as e:
            self.log.error("Fatal error", error=str(e))
            self.running.clear()
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
