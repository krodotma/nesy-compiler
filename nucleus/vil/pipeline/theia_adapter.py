"""
VIL-Theia Integration Adapter
Connects Theia VLM Specialist to VIL Coordinator pipeline.

This adapter provides:
1. VLM inference methods (infer, describe_screen, find_element, suggest_action)
2. VIL event emission for all Theia operations
3. Vision-to-ICL pipeline integration
4. CMP tracking for vision operations
5. Trace ID propagation

Version: 1.0
Date: 2026-01-25
"""

import base64
import time
from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass
import numpy as np

try:
    from theia.vlm.specialist import VLMSpecialist, TheiaConfig
    from theia.vlm.icl import ScreenshotICL, ScreenshotExample
    THEIA_AVAILABLE = True
except ImportError:
    THEIA_AVAILABLE = False
    VLMSpecialist = None
    TheiaConfig = None

from nucleus.vil.events import (
    VILEventType,
    VisionEvent,
    LearningEvent,
    create_vision_event,
    create_learning_event,
    create_trace_id,
    CmpMetrics,
)
from nucleus.vil.interfaces import VILVisionInterface, VILLearningInterface


# Constants
PHI = 1.618033988749895
CMP_DISCOUNT = 1 / PHI


@dataclass
class VLMInference:
    """VLM inference result."""
    response: str
    confidence: float
    trace_id: str
    latency_ms: float
    model_used: str = "theia-specialist"
    metadata: Dict[str, Any] = None


class TheiaVILAdapter(VILVisionInterface, VILLearningInterface):
    """
    Adapter connecting Theia VLM Specialist to VIL pipeline.

    Extends Theia with:
    - VIL event emission
    - Trace ID propagation
    - CMP tracking
    - Vision-to-ICL pipeline
    """

    def __init__(
        self,
        theia_config: Optional[TheiaConfig] = None,
        bus_emitter: Optional[callable] = None,
    ):
        self.bus_emitter = bus_emitter
        self._trace_counter = 0

        if THEIA_AVAILABLE and theia_config is not None:
            self.theia: Optional[VLMSpecialist] = VLMSpecialist(config=theia_config)
            self.theia.boot()
        else:
            self.theia = None
            print("[TheiaVILAdapter] Theia not available, using mock mode")

        # Vision CMP tracking
        self.vision_cmp = CmpMetrics(
            capture=0.0,
            analysis=0.0,
            lineage="theia-vil-adapter",
            generation=0,
            fitness=0.0,
        )

        # Statistics
        self.stats = {
            "inferences": 0,
            "frames_processed": 0,
            "icl_examples": 0,
            "avg_latency_ms": 0.0,
        }

    # === VLM Inference Methods (Theia Extension) ===

    async def infer(
        self,
        image: str,
        prompt: str,
        use_icl: bool = True,
        trace_id: Optional[str] = None,
    ) -> VLMInference:
        """
        Run VLM inference on image with prompt.

        Args:
            image: Base64-encoded image
            prompt: Text prompt for VLM
            use_icl: Use in-context learning from past examples
            trace_id: Optional trace ID for correlation

        Returns:
            VLMInference with response, confidence, trace_id
        """
        start_time = time.time()
        trace_id = trace_id or self._create_trace_id("infer")

        # Emit vision capture event
        await self._emit_vision_event(
            frame_id=trace_id,
            image_data=image,
            metadata={"prompt": prompt, "use_icl": use_icl}
        )

        # Run inference
        if self.theia and self.theia.icl:
            # Use ICL for retrieval
            if use_icl:
                similar = self.theia.icl.retrieve_similar(
                    query_embedding=self._embed_image(image),
                    k=3
                )
                # Build ICL context
                icl_context = self._build_icl_context(similar)
                full_prompt = f"{icl_context}\n\n{prompt}"
            else:
                full_prompt = prompt

            # Mock inference (would call actual VLM here)
            response = self._mock_inference(image, full_prompt)
            confidence = 0.85  # Placeholder

            # Add to ICL if successful
            if confidence > 0.7:
                self._add_icl_example(image, prompt, response)
        else:
            # Fallback mock
            response = self._mock_inference(image, prompt)
            confidence = 0.5

        latency_ms = (time.time() - start_time) * 1000

        # Update CMP
        self.vision_cmp.analysis = confidence
        self.vision_cmp.generation += 1
        self.vision_cmp.fitness = (
            self.vision_cmp.fitness * CMP_DISCOUNT +
            confidence * PHI * (1 - CMP_DISCOUNT)
        )

        # Emit processed event
        await self._emit_vision_event(
            frame_id=trace_id,
            image_data=None,  # Already emitted
            metadata={
                "response": response[:200],
                "confidence": confidence,
                "latency_ms": latency_ms,
                "cmp_fitness": self.vision_cmp.fitness,
            }
        )

        # Update stats
        self.stats["inferences"] += 1
        self.stats["frames_processed"] += 1
        self.stats["avg_latency_ms"] = (
            (self.stats["avg_latency_ms"] * (self.stats["inferences"] - 1) + latency_ms) /
            self.stats["inferences"]
        )

        return VLMInference(
            response=response,
            confidence=confidence,
            trace_id=trace_id,
            latency_ms=latency_ms,
            metadata={"cmp": self.vision_cmp.to_dict()},
        )

    async def describe_screen(
        self,
        image: str,
        detail_level: str = "medium",
    ) -> str:
        """
        Describe what's on screen.

        Args:
            image: Base64-encoded screenshot
            detail_level: "brief", "medium", or "detailed"

        Returns:
            Text description of screen content
        """
        trace_id = self._create_trace_id("describe")
        prompt = self._describe_prompt(detail_level)

        result = await self.infer(image, prompt, trace_id=trace_id)
        return result.response

    async def find_element(
        self,
        image: str,
        description: str,
    ) -> Dict[str, Any]:
        """
        Find UI element on screen for browser automation.

        Args:
            image: Base64-encoded screenshot
            description: Element description (e.g., "submit button")

        Returns:
            Dict with element location and confidence
        """
        trace_id = self._create_trace_id("find_element")
        prompt = f"Find the element: {description}\n\nRespond with JSON: {{'x': <pixel>, 'y': <pixel>, 'confidence': <0-1>}}"

        result = await self.infer(image, prompt, trace_id=trace_id)

        # Parse JSON response
        import json
        try:
            element_data = json.loads(result.response)
        except:
            element_data = {"x": -1, "y": -1, "confidence": 0.0}

        return {
            **element_data,
            "trace_id": trace_id,
            "description": description,
        }

    async def suggest_action(
        self,
        image: str,
        goal: str,
    ) -> Dict[str, Any]:
        """
        Suggest next action for autonomous navigation.

        Args:
            image: Base64-encoded screenshot
            goal: Current goal (e.g., "login to account")

        Returns:
            Dict with suggested action and reasoning
        """
        trace_id = self._create_trace_id("suggest_action")
        prompt = f"Goal: {goal}\n\nSuggest the next action. Respond with action and reasoning."

        result = await self.infer(image, prompt, trace_id=trace_id, use_icl=True)

        return {
            "action": result.response,
            "confidence": result.confidence,
            "trace_id": trace_id,
            "goal": goal,
        }

    # === VILVisionInterface Methods ===

    async def capture_frame(self) -> Optional[Dict[str, Any]]:
        """Capture frame from vision source."""
        if self.theia:
            perception = self.theia.perceive()
            return {
                "frame_id": self._create_trace_id("capture"),
                "image_data": None,  # Theia handles capture internally
                "width": 1920,
                "height": 1080,
                "timestamp": time.time(),
                "entropy": perception.get("embedding_norm", 0.0),
                "has_input": perception.get("has_input", False),
            }
        return None

    async def get_ring_buffer(self) -> List[Dict[str, Any]]:
        """Get ring buffer contents."""
        # Theia doesn't expose ring buffer, return empty
        return []

    async def analyze_entropy(self, frame: Dict[str, Any]) -> float:
        """Calculate visual entropy."""
        return frame.get("entropy", 0.0)

    async def emit_capture_event(self, frame: Dict[str, Any]) -> None:
        """Emit vision capture event."""
        await self._emit_vision_event(**frame)

    # === VILLearningInterface Methods ===

    async def add_task(
        self,
        task_id: str,
        prompt: str,
        response: str,
        success: bool = True,
    ) -> Dict[str, Any]:
        """Add learning task for meta-learning."""
        if self.theia and self.theia.icl:
            example = ScreenshotExample(
                id=task_id,
                image_embedding=np.random.randn(32),  # Placeholder
                action=prompt,
                outcome="success" if success else "failure",
                timestamp=time.time(),
            )
            self.theia.icl.add_example(example)
            self.stats["icl_examples"] += 1

        return {"task_id": task_id, "success": success}

    async def inner_loop(
        self,
        task_id: str,
        num_steps: int = 5,
        lr: float = 0.1,
    ) -> Dict[str, float]:
        """Run inner loop adaptation."""
        # Mock inner loop
        return {
            "loss": 0.1,
            "accuracy": 0.9,
            "gradient_norm": 0.5,
        }

    async def outer_loop(
        self,
        num_tasks: int = 10,
        meta_lr: float = 0.01,
    ) -> Dict[str, float]:
        """Run meta-learning outer loop."""
        # Mock outer loop
        return {
            "meta_loss": 0.15,
            "meta_accuracy": 0.85,
            "task_diversity": 0.7,
        }

    async def record_episode(
        self,
        episode_id: str,
        states: List[Any],
        actions: List[Any],
        rewards: List[float],
        next_states: List[Any],
    ) -> None:
        """Record RL episode."""
        pass  # Theia handles this internally

    async def compute_meta_gradient(
        self,
        tasks: List[Dict[str, Any]],
    ) -> np.ndarray:
        """Compute meta-gradient across tasks."""
        return np.random.randn(32)  # Placeholder

    # === Private Methods ===

    def _create_trace_id(self, prefix: str) -> str:
        """Create unique trace ID."""
        self._trace_counter += 1
        return f"theia_{prefix}_{self._trace_counter}_{int(time.time() * 1000)}"

    def _embed_image(self, image_b64: str) -> np.ndarray:
        """Create embedding from base64 image."""
        # Simple hash-based embedding (would use real vision encoder)
        import hashlib
        h = hashlib.sha256(image_b64.encode()).digest()
        return np.frombuffer(h[:32], dtype=np.uint8).astype(np.float32) / 255.0

    def _mock_inference(self, image: str, prompt: str) -> str:
        """Mock VLM inference (placeholder for actual model)."""
        return f"[Mock VLM Response] Analyzed image with prompt: '{prompt[:50]}...'"

    def _describe_prompt(self, detail_level: str) -> str:
        """Generate prompt for screen description."""
        prompts = {
            "brief": "Briefly describe what you see on this screen in one sentence.",
            "medium": "Describe the main elements on this screen in 2-3 sentences.",
            "detailed": "Provide a detailed description of all visible UI elements, their positions, and relationships.",
        }
        return prompts.get(detail_level, prompts["medium"])

    def _build_icl_context(self, similar: List[Tuple]) -> str:
        """Build ICL context from similar examples."""
        if not similar:
            return "No similar past examples found."

        context = "Similar past examples:\n"
        for example, similarity in similar[:3]:
            context += f"- {example.action} (similarity: {similarity:.2f})\n"
        return context

    def _add_icl_example(self, image: str, prompt: str, response: str):
        """Add successful inference to ICL buffer."""
        if self.theia and self.theia.icl:
            example = ScreenshotExample.from_screenshot(
                image_b64=image,
                action=response[:100],  # Truncate for action
                outcome="success",
            )
            self.theia.icl.add_example(example)
            self.stats["icl_examples"] += 1

    async def _emit_vision_event(
        self,
        frame_id: str,
        image_data: Optional[str],
        metadata: Optional[Dict[str, Any]] = None,
        **kwargs
    ):
        """Emit vision event to bus."""
        if not self.bus_emitter:
            return

        event = VisionEvent(
            frame_id=frame_id,
            image_data=image_data,
            source="theia-vil-adapter",
            metadata=metadata or {},
            **kwargs
        )
        event.cmp = self.vision_cmp

        try:
            self.bus_emitter(event.to_bus_event())
        except Exception as e:
            print(f"[TheiaVILAdapter] Bus emission error: {e}")

    # === Statistics ===

    def get_stats(self) -> Dict[str, Any]:
        """Get adapter statistics."""
        return {
            **self.stats,
            "cmp_fitness": self.vision_cmp.fitness,
            "cmp_generation": self.vision_cmp.generation,
            "theia_available": THEIA_AVAILABLE and self.theia is not None,
            "icl_buffer_size": (
                len(self.theia.icl.examples)
                if self.theia and self.theia.icl else 0
            ),
        }


def create_theia_adapter(
    bus_emitter: Optional[callable] = None,
) -> TheiaVILAdapter:
    """Create Theia-VIL adapter with default config."""
    config = TheiaConfig() if THEIA_AVAILABLE else None
    return TheiaVILAdapter(theia_config=config, bus_emitter=bus_emitter)


__all__ = [
    "TheiaVILAdapter",
    "create_theia_adapter",
    "VLMInference",
    "THEIA_AVAILABLE",
]
