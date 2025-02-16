from prometheus_client import Counter, Histogram, Gauge

class Metrics:
    def __init__(self):
        # Existing metrics
        self.questions_processed = Counter(
            'questions_processed_total',
            'Total questions processed',
        )

        self.errors_total = Counter(
            'errors_total',
            'Total errors encountered',
            ['error_type']  # Track different types of errors
        )

        self.response_time = Histogram(
            'response_generation_seconds',
            'Time spent generating responses'
        )

        self.queue_length = Gauge(
            'question_queue_length',
            'Number of questions in queue'
        )

        self.model_inference_time = Histogram(
            'model_inference_seconds',
            'Time spent on model inference'
        )

        self.token_usage = Counter(
            'token_usage_total',
            'Total tokens used in responses',
        )

        # New metrics for buffer monitoring
        self.active_buffers = Gauge(
            'active_buffers_total',
            'Number of active chunk buffers'
        )

        self.stale_buffers_cleaned = Counter(
            'stale_buffers_cleaned_total',
            'Number of stale buffers cleaned up'
        )

        self.chunks_sent = Counter(
            'chunks_sent_total',
            'Number of text chunks sent to clients',
        )

        self.chunks_acknowledged = Counter(
            'chunks_acknowledged_total',
            'Number of chunks acknowledged by clients',
        )

        self.chunk_retry_count = Counter(
            'chunk_retry_total',
            'Number of chunk retransmissions',
        )

        self.buffer_wait_time = Histogram(
            'buffer_wait_seconds',
            'Time spent waiting for buffer space'
        )
