"""Buffer for managing unacknowledged text chunks sent to client for a specific request"""
import threading, time
from typing import Optional
import structlog
from models import TextChunk
from config import Settings

class ChunkBuffer:
    """Manages unacknowledged chunks for a specific request"""
    def __init__(self, request_id: int, condition: threading.Condition, window_size: int = 5):
        self.request_id: int = request_id
        self.window_size: int = window_size
        self.buffer: dict[int, TextChunk] = {}
        self.next_sequence: int = 0
        self.last_acked: int = -1
        self.last_update: float = time.time()
        self.buffer_condition = condition  # Use the service's condition
        self.log = structlog.get_logger()
        self.settings = Settings()

    def _update_timestamp(self) -> None:
        """Update the last_update timestamp"""
        self.last_update = time.time()

    def add_chunk(self, chunk: str, is_final: bool = False) -> Optional[TextChunk]:
        """Add a chunk to the buffer. Returns the chunk if it's ready to send."""
        with self.buffer_condition:
            if len(self.buffer) < self.window_size:
                chunk_obj = TextChunk(
                    request_id=self.request_id,
                    sequence_number=self.next_sequence,
                    chunk=chunk,
                    is_final=is_final
                )
                self.buffer[self.next_sequence] = chunk_obj
                self.next_sequence += 1
                self._update_timestamp()
                return chunk_obj
            return None

    def ack_received(self, sequence_number: int) -> Optional[TextChunk]:
        """Process acknowledgment and return next chunk to send if available"""
        with self.buffer_condition:
            if sequence_number in self.buffer:
                del self.buffer[sequence_number]
                self.last_acked = sequence_number
                self._update_timestamp()

                # Notify waiting threads that space is available
                self.buffer_condition.notify_all()

                if len(self.buffer) < self.window_size:
                    return self.get_next_chunk()
        return None

    def get_next_chunk(self) -> Optional[TextChunk]:
        """Get the next chunk that should be sent"""
        with self.buffer_condition:
            # Find the lowest sequence number in the buffer
            if self.buffer:
                self._update_timestamp()
                return min(self.buffer.values(), key=lambda x: x.sequence_number)
        return None