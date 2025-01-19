# metrics.py
from prometheus_client import Counter, Histogram, Gauge

class Metrics:
    def __init__(self):
        self.questions_processed = Counter(
            'questions_processed_total',
            'Total questions processed',
            ['cohort_id']  # Allow tracking by cohort
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
            ['cohort_id']
        )