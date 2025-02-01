"""Buffer for managing unacknowledged text chunks sent to client for a specific request"""
import threading, time
from typing import Optional
import structlog
from models import TextChunk
from config import Settings

class ChunkBuffer:
    """Manages unacknowledged chunks for a specific request"""
    def __init__(self, request_id: int, window_size: int = 5):
        self.request_id: int = request_id
        self.window_size: int = window_size
        self.buffer: dict[int, TextChunk] = {}
        self.next_sequence: int = 0
        self.window_start: int = 0  # Track start of sliding window
        self.last_acked: int = -1
        self.unacked_chunks: dict[int, TextChunk] = {}  # Store unacknowledged chunks
        self.retransmit_timeout: float = 1.0  # Seconds
        self.last_transmit: dict[int, float] = {}  # Track transmission times
        self.condition = threading.Condition()
        self.log = structlog.get_logger()
        self.settings = Settings()

    def _update_timestamp(self) -> None:
        """Update the last_update timestamp"""
        self.last_update = time.time()

    def add_chunk(self, chunk: str, is_final: bool = False) -> Optional[TextChunk]:
        """Add chunk to buffer using sliding window protocol"""
        with self.condition:
            if len(self.buffer) < self.window_size:
                chunk_obj = TextChunk(
                    request_id=self.request_id,
                    sequence_number=self.next_sequence,
                    chunk=chunk,
                    is_final=is_final
                )
                self.buffer[self.next_sequence] = chunk_obj
                self.unacked_chunks[self.next_sequence] = chunk_obj
                self.last_transmit[self.next_sequence] = time.time()
                self.next_sequence += 1
                self._update_timestamp()
                return chunk_obj
            return None

    def ack_received(self, sequence_number: int) -> Optional[TextChunk]:
        """Handle cumulative acknowledgments like TCP"""
        with self.condition:
            # Remove all chunks up to and including this sequence number
            for seq in range(self.window_start, sequence_number + 1):
                if seq in self.unacked_chunks:
                    del self.unacked_chunks[seq]
                    del self.last_transmit[seq]
                    if seq in self.buffer:
                        del self.buffer[seq]

            # Update window start
            self.window_start = sequence_number + 1
            self.last_acked = sequence_number
            self._update_timestamp()

            # Check for next chunk to send
            if len(self.buffer) < self.window_size:
                return self.get_next_chunk()
        return None

    def get_next_chunk(self) -> Optional[TextChunk]:
        """Get the next chunk that should be sent"""
        with self.condition:
            # Find the lowest sequence number in the buffer
            if self.buffer:
                self._update_timestamp()
                return min(self.buffer.values(), key=lambda x: x.sequence_number)
        return None

    def check_retransmissions(self) -> list[TextChunk]:
        """Check for chunks that need retransmission"""
        current_time = time.time()
        retransmit = []

        with self.condition:
            for seq, transmit_time in self.last_transmit.items():
                if current_time - transmit_time > self.retransmit_timeout:
                    if seq in self.unacked_chunks:
                        chunk = self.unacked_chunks[seq]
                        self.last_transmit[seq] = current_time
                        retransmit.append(chunk)

        return retransmit