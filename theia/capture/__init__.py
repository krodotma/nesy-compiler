"""
Theia Capture Module — Vision capture subsystem (L0).

Components:
    ring_buffer: Thread-safe ring buffer for frame storage
    screen: Cross-platform screen capture
    ingest: HTTP endpoint for frame ingestion
"""

from dataclasses import dataclass, field
from typing import List, Optional, Any
import time


@dataclass
class Frame:
    """Single captured frame with metadata."""
    timestamp: float
    data: str  # Base64 JPEG
    width: int = 0
    height: int = 0
    source: str = "screen"
    meta: dict = field(default_factory=dict)


class RingBuffer:
    """
    Thread-safe ring buffer for frame storage.
    
    Ported from VisionEye.tsx ring buffer pattern.
    Default capacity: 60 frames (1 minute at 1 FPS).
    """
    
    def __init__(self, capacity: int = 60):
        self.capacity = capacity
        self._buffer: List[Frame] = []
        self._lock = None  # Lazy init for threading
    
    def push(self, frame: Frame) -> None:
        """Add frame to buffer, evicting oldest if full."""
        if len(self._buffer) >= self.capacity:
            self._buffer.pop(0)
        self._buffer.append(frame)
    
    def get_all(self) -> List[Frame]:
        """Return all frames in buffer."""
        return list(self._buffer)
    
    def clear(self) -> None:
        """Clear all frames."""
        self._buffer.clear()
    
    def __len__(self) -> int:
        return len(self._buffer)


class CaptureModule:
    """
    Main capture module interface.
    
    Usage:
        capture = CaptureModule()
        capture.start()
        frames = capture.get_frames()
        capture.stop()
    """
    
    def __init__(self, buffer_size: int = 60, fps: float = 1.0, monitor_index: int = 1):
        self.buffer = RingBuffer(capacity=buffer_size)
        self.fps = fps
        self.monitor_index = monitor_index
        self._running = False
        self._capture_task = None
    
    def start(self) -> None:
        """Start screen capture loop."""
        self._running = True
        # TODO: Implement async capture loop
    
    def stop(self) -> None:
        """Stop screen capture loop."""
        self._running = False
    
    def get_frames(self) -> List[Frame]:
        """Get all captured frames."""
        return self.buffer.get_all()
    
    def ingest(self, frames: List[str], timestamp: Optional[float] = None) -> int:
        """
        Ingest frames from external source (e.g., dashboard VisionEye).
        
        Args:
            frames: List of base64 JPEG strings
            timestamp: Optional timestamp for batch
            
        Returns:
            Number of frames ingested
        """
        ts = timestamp or time.time()
        for i, data in enumerate(frames):
            frame = Frame(
                timestamp=ts + (i * 0.001),  # Offset for ordering
                data=data,
                source="external",
            )
            self.buffer.push(frame)
        return len(frames)
    
    def status(self) -> dict:
        """Return capture status."""
        return {
            "running": self._running,
            "buffer_size": len(self.buffer),
            "buffer_capacity": self.buffer.capacity,
            "fps": self.fps,
        }


# Compatibility alias
WindowCapture = CaptureModule

__all__ = ["CaptureModule", "RingBuffer", "Frame", "WindowCapture"]

