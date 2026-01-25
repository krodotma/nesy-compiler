#!/usr/bin/env python3
"""
dialogos_streaming.py - Multi-Agent Streaming & Priority Queue

Extends Dialogos with:
- D5: Parallel streaming cells for concurrent multi-agent responses
- D6: Priority queue with backpressure management

Part of Dialogos Phase 2
"""

import asyncio
import heapq
import time
import uuid
from dataclasses import dataclass, field
from enum import IntEnum
from typing import Any, AsyncIterator, Callable, Dict, List, Optional

# ============================================================================
# Types
# ============================================================================

class Priority(IntEnum):
    """Request priority levels (lower = higher priority)."""
    CRITICAL = 0   # Ring 0 constitutional
    HIGH = 1       # SAGENT requests
    NORMAL = 2     # Standard operations
    LOW = 3        # Background/batch
    BULK = 4       # Large batch jobs


@dataclass(order=True)
class PriorityRequest:
    """Request wrapper for priority queue."""
    priority: int
    timestamp: float = field(compare=True)
    req_id: str = field(compare=False)
    provider: str = field(compare=False)
    prompt: str = field(compare=False)
    context: Dict[str, Any] = field(compare=False, default_factory=dict)
    actor: str = field(compare=False, default="unknown")
    callback: Optional[Callable] = field(compare=False, default=None)


@dataclass
class StreamChunk:
    """Single chunk from a streaming response."""
    req_id: str
    provider: str
    index: int
    content: str
    is_final: bool = False
    latency_ms: int = 0


# ============================================================================
# Priority Queue with Backpressure (D6)
# ============================================================================

class DialogosPriorityQueue:
    """
    Priority queue for Dialogos requests with backpressure management.
    
    Features:
    - Priority-based scheduling (CRITICAL > HIGH > NORMAL > LOW > BULK)
    - Configurable queue size limits per priority
    - Backpressure signals when queue is saturated
    - Fair scheduling within same priority (FIFO)
    """
    
    DEFAULT_LIMITS = {
        Priority.CRITICAL: 100,
        Priority.HIGH: 50,
        Priority.NORMAL: 200,
        Priority.LOW: 100,
        Priority.BULK: 500,
    }
    
    def __init__(self, limits: Optional[Dict[Priority, int]] = None):
        self._heap: List[PriorityRequest] = []
        self._limits = limits or self.DEFAULT_LIMITS
        self._counts: Dict[Priority, int] = {p: 0 for p in Priority}
        self._lock = asyncio.Lock()
        self._not_empty = asyncio.Event()
        
    async def enqueue(self, request: PriorityRequest) -> bool:
        """
        Add request to queue. Returns False if backpressure applies.
        """
        async with self._lock:
            priority = Priority(request.priority)
            
            # Check backpressure
            if self._counts[priority] >= self._limits[priority]:
                return False  # Backpressure: reject request
            
            heapq.heappush(self._heap, request)
            self._counts[priority] += 1
            self._not_empty.set()
            return True
    
    async def dequeue(self) -> PriorityRequest:
        """Get highest priority request (blocks if empty)."""
        while True:
            async with self._lock:
                if self._heap:
                    request = heapq.heappop(self._heap)
                    priority = Priority(request.priority)
                    self._counts[priority] -= 1
                    
                    if not self._heap:
                        self._not_empty.clear()
                    return request
            
            await self._not_empty.wait()
    
    def size(self) -> int:
        return len(self._heap)
    
    def is_saturated(self, priority: Priority) -> bool:
        return self._counts[priority] >= self._limits[priority]
    
    def get_stats(self) -> Dict[str, Any]:
        return {
            "total_queued": len(self._heap),
            "by_priority": {p.name: self._counts[p] for p in Priority},
            "saturation": {p.name: self.is_saturated(p) for p in Priority},
        }


# ============================================================================
# Streaming Cell Manager (D5)
# ============================================================================

class StreamingCellManager:
    """
    Manages parallel streaming responses from multiple providers.
    
    Features:
    - Concurrent streaming from multiple LLM cells
    - Unified stream multiplexing
    - Per-cell backpressure handling
    - Graceful cancellation
    """
    
    def __init__(self, max_concurrent: int = 5):
        self.max_concurrent = max_concurrent
        self._active_streams: Dict[str, asyncio.Task] = {}
        self._semaphore = asyncio.Semaphore(max_concurrent)
        self._output_queue: asyncio.Queue[StreamChunk] = asyncio.Queue()
        
    async def start_stream(
        self,
        req_id: str,
        provider: str,
        stream_generator: AsyncIterator[str],
    ) -> str:
        """
        Start a new streaming cell.
        Returns the stream ID for tracking.
        """
        stream_id = f"{req_id}:{provider}"
        
        async def _stream_worker():
            async with self._semaphore:
                index = 0
                start_time = time.time()
                
                try:
                    async for chunk in stream_generator:
                        latency_ms = int((time.time() - start_time) * 1000)
                        await self._output_queue.put(StreamChunk(
                            req_id=req_id,
                            provider=provider,
                            index=index,
                            content=chunk,
                            latency_ms=latency_ms,
                        ))
                        index += 1
                    
                    # Final chunk
                    latency_ms = int((time.time() - start_time) * 1000)
                    await self._output_queue.put(StreamChunk(
                        req_id=req_id,
                        provider=provider,
                        index=index,
                        content="",
                        is_final=True,
                        latency_ms=latency_ms,
                    ))
                    
                except asyncio.CancelledError:
                    # Send cancellation marker
                    await self._output_queue.put(StreamChunk(
                        req_id=req_id,
                        provider=provider,
                        index=index,
                        content="[CANCELLED]",
                        is_final=True,
                    ))
                finally:
                    if stream_id in self._active_streams:
                        del self._active_streams[stream_id]
        
        task = asyncio.create_task(_stream_worker())
        self._active_streams[stream_id] = task
        return stream_id
    
    async def cancel_stream(self, stream_id: str) -> bool:
        """Cancel an active stream."""
        if stream_id in self._active_streams:
            self._active_streams[stream_id].cancel()
            return True
        return False
    
    async def get_next_chunk(self, timeout: float = 30.0) -> Optional[StreamChunk]:
        """Get next available chunk from any active stream."""
        try:
            return await asyncio.wait_for(
                self._output_queue.get(),
                timeout=timeout
            )
        except asyncio.TimeoutError:
            return None
    
    async def iter_chunks(self) -> AsyncIterator[StreamChunk]:
        """Iterate over all chunks from all active streams."""
        while self._active_streams or not self._output_queue.empty():
            chunk = await self.get_next_chunk(timeout=1.0)
            if chunk:
                yield chunk
    
    def active_count(self) -> int:
        return len(self._active_streams)
    
    def get_stats(self) -> Dict[str, Any]:
        return {
            "active_streams": self.active_count(),
            "max_concurrent": self.max_concurrent,
            "queue_size": self._output_queue.qsize(),
        }


# ============================================================================
# Enhanced Dialogos Dispatcher
# ============================================================================

class EnhancedDialogosDispatcher:
    """
    Enhanced Dialogos dispatcher with streaming and priority queue.
    """
    
    def __init__(self, max_concurrent: int = 5):
        self.queue = DialogosPriorityQueue()
        self.streaming = StreamingCellManager(max_concurrent)
        self._running = False
        self._worker_task: Optional[asyncio.Task] = None
        
    async def submit(
        self,
        provider: str,
        prompt: str,
        priority: Priority = Priority.NORMAL,
        context: Optional[Dict[str, Any]] = None,
        actor: str = "unknown",
    ) -> Optional[str]:
        """
        Submit a request to the priority queue.
        Returns req_id if accepted, None if backpressure applies.
        """
        req_id = str(uuid.uuid4())
        request = PriorityRequest(
            priority=priority.value,
            timestamp=time.time(),
            req_id=req_id,
            provider=provider,
            prompt=prompt,
            context=context or {},
            actor=actor,
        )
        
        if await self.queue.enqueue(request):
            return req_id
        return None  # Backpressure
    
    async def start(self):
        """Start the dispatcher worker."""
        self._running = True
        self._worker_task = asyncio.create_task(self._process_loop())
        
    async def stop(self):
        """Stop the dispatcher."""
        self._running = False
        if self._worker_task:
            self._worker_task.cancel()
            try:
                await self._worker_task
            except asyncio.CancelledError:
                pass
    
    async def _process_loop(self):
        """Main processing loop."""
        while self._running:
            try:
                request = await asyncio.wait_for(
                    self.queue.dequeue(),
                    timeout=1.0
                )
                # Process request (integrate with actual provider here)
                print(f"[Dialogos] Processing {request.req_id[:8]} priority={Priority(request.priority).name}")
                
            except asyncio.TimeoutError:
                continue
            except asyncio.CancelledError:
                break
    
    def get_stats(self) -> Dict[str, Any]:
        return {
            "queue": self.queue.get_stats(),
            "streaming": self.streaming.get_stats(),
            "running": self._running,
        }


# ============================================================================
# Singleton
# ============================================================================

enhanced_dispatcher = EnhancedDialogosDispatcher()

# ============================================================================
# CLI
# ============================================================================

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Dialogos Streaming CLI")
    parser.add_argument("--stats", action="store_true", help="Show dispatcher stats")
    parser.add_argument("--test", action="store_true", help="Run test submission")
    
    args = parser.parse_args()
    
    async def run_test():
        await enhanced_dispatcher.start()
        
        # Test submissions at different priorities
        for priority in [Priority.HIGH, Priority.NORMAL, Priority.LOW]:
            req_id = await enhanced_dispatcher.submit(
                provider="claude-opus",
                prompt="Test prompt",
                priority=priority,
            )
            print(f"Submitted {req_id} at priority {priority.name}")
        
        await asyncio.sleep(2)
        print(f"Stats: {enhanced_dispatcher.get_stats()}")
        await enhanced_dispatcher.stop()
    
    if args.test:
        asyncio.run(run_test())
    
    if args.stats:
        print(enhanced_dispatcher.get_stats())
