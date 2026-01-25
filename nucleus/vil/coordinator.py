"""
VIL (Vision-Integration-Learning) Coordinator
Central orchestration layer for vision-metalearning integration.

The VILCoordinator manages:
- Vision input from VisionEye/Theia
- Learning updates from Omega Metalearning
- CMP tracking from Clade Manager
- Synthesis from CGP/EGGP
- Geometric embedding updates

Version: 1.0
Date: 2026-01-25
"""

import asyncio
from typing import Optional, Dict, Any, List, Callable
from dataclasses import dataclass, field
from datetime import datetime
import logging

from .events import (
    VILEvent,
    VisionEvent,
    LearningEvent,
    SynthesisEvent,
    CMPEvent,
    IntegrationEvent,
    VILEventType,
    create_vil_event,
    create_integration_event,
    create_trace_id,
)


logger = logging.getLogger(__name__)


# Constants from Learning Tower L8
PHI = 1.618033988749895  # Golden ratio
CMP_DISCOUNT = 1 / PHI  # ~0.618 per generation
GLOBAL_CMP_FLOOR = 0.236  # 1/PHI^3


@dataclass
class VILState:
    """Global VIL state."""
    vision_buffer: List[Dict[str, Any]] = field(default_factory=list)
    icl_examples: List[Dict[str, Any]] = field(default_factory=list)
    active_clades: Dict[str, float] = field(default_factory=dict)
    synthesis_cache: Dict[str, str] = field(default_factory=dict)
    geometric_embeddings: Dict[str, List[float]] = field(default_factory=dict)
    cmp_score: float = 0.0
    generation: int = 0
    total_episodes: int = 0
    total_rewards: float = 0.0


@dataclass
class VILConfig:
    """VIL Coordinator configuration."""
    buffer_size: int = 60
    icl_buffer_size: int = 5
    meta_lr: float = 0.01
    inner_lr: float = 0.1
    phi_weighted: bool = True
    geometric_enabled: bool = True
    cgp_enabled: bool = True
    cmp_tracking: bool = True
    event_emission: bool = True


class VILCoordinator:
    """
    Central coordinator for Vision-Integration-Learning pipeline.

    Architecture:
    ┌─────────────────────────────────────────────────────────────┐
    │                     VILCoordinator                           │
    ├─────────────────────────────────────────────────────────────┤
    │                                                             │
    │  ┌──────────┐  ┌──────────┐  ┌──────────┐  ┌──────────┐  │
    │  │  Vision  │  │ Learning │  │Synthesis │  │   CMP    │  │
    │  │  Input   │→ │  Update  │→ │  Engine  │→ │ Tracking │  │
    │  └──────────┘  └──────────┘  └──────────┘  └──────────┘  │
    │         ↓             ↓              ↓            ↓         │
    │  ┌──────────────────────────────────────────────────────┐  │
    │  │           Geometric Continuous Layer                  │  │
    │  │    (S^n + H^n embeddings, attractor dynamics)         │  │
    │  └──────────────────────────────────────────────────────┘  │
    └─────────────────────────────────────────────────────────────┘
    """

    def __init__(
        self,
        config: Optional[VILConfig] = None,
        bus_emitter: Optional[Callable] = None,
    ):
        self.config = config or VILConfig()
        self.state = VILState()
        self._bus_emitter = bus_emitter
        self._subscribers: Dict[VILEventType, List[Callable]] = {}
        self._running = False
        self._trace_id = create_trace_id("vil_coord")

        logger.info(f"VILCoordinator initialized with trace_id: {self._trace_id}")

    # === Vision Input ===

    async def process_vision_frame(
        self,
        frame_id: str,
        image_data: Optional[str],
        width: int = 1920,
        height: int = 1080,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> VisionEvent:
        """
        Process incoming vision frame from VisionEye/Theia.

        Flow:
        1. Add to ring buffer (max size: buffer_size)
        2. Calculate visual entropy
        3. Emit vision.capture event
        4. Trigger VLM inference if needed
        """
        # Create vision event
        event = VisionEvent(
            frame_id=frame_id,
            image_data=image_data,
            width=width,
            height=height,
            source="vil.coordinator",
            metadata=metadata or {},
        )

        # Update ring buffer
        self.state.vision_buffer.append({
            "frame_id": frame_id,
            "timestamp": datetime.utcnow().isoformat(),
            "width": width,
            "height": height,
        })
        if len(self.state.vision_buffer) > self.config.buffer_size:
            self.state.vision_buffer.pop(0)

        # Emit event
        await self._emit(event)

        logger.debug(f"Processed vision frame: {frame_id}, buffer size: {len(self.state.vision_buffer)}")
        return event

    # === Learning Integration ===

    async def process_meta_update(
        self,
        task_id: str,
        loss: float,
        reward: float,
        inner_loss: float = 0.0,
        outer_loss: float = 0.0,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> LearningEvent:
        """
        Process metalearning update from Omega/Learning Tower L8.

        Updates:
        1. ICL buffer with successful examples
        2. CMP score based on reward
        3. Geometric embeddings
        """
        # Update state
        self.state.total_episodes += 1
        self.state.total_rewards += reward

        # Update CMP (phi-weighted)
        if self.config.phi_weighted:
            reward_weight = reward * PHI
            self.state.cmp_score = (
                self.state.cmp_score * CMP_DISCOUNT +
                reward_weight * (1 - CMP_DISCOUNT)
            )

        # Create learning event
        event = LearningEvent(
            event_type=VILEventType.LEARN_META_UPDATE,
            task_id=task_id,
            loss=loss,
            reward=reward,
            inner_loss=inner_loss,
            outer_loss=outer_loss,
            episode_id=str(self.state.total_episodes),
            source="vil.coordinator",
            metadata=metadata or {},
        )

        # Update CMP in event
        event.cmp.fitness = self.state.cmp_score
        event.cmp.generation = self.state.generation

        await self._emit(event)

        logger.info(
            f"Meta update: task={task_id}, loss={loss:.4f}, "
            f"reward={reward:.4f}, cmp={self.state.cmp_score:.4f}"
        )
        return event

    async def add_icl_example(
        self,
        image: str,
        prompt: str,
        response: str,
        success: bool = True,
    ) -> LearningEvent:
        """
        Add in-context learning example to buffer.

        Used by VLM Specialist for few-shot learning.
        """
        example = {
            "image": image,
            "prompt": prompt,
            "response": response,
            "success": success,
            "timestamp": datetime.utcnow().isoformat(),
        }

        self.state.icl_examples.append(example)
        if len(self.state.icl_examples) > self.config.icl_buffer_size:
            self.state.icl_examples.pop(0)

        event = LearningEvent(
            event_type=VILEventType.LEARN_ICL_EXAMPLE,
            icl_examples=len(self.state.icl_examples),
            icl_buffer_size=self.config.icl_buffer_size,
            source="vil.coordinator",
            data={"example": example},
        )

        await self._emit(event)

        logger.debug(f"Added ICL example, buffer size: {len(self.state.icl_examples)}")
        return event

    # === Synthesis Integration ===

    async def process_synthesis(
        self,
        program_id: str,
        source_image: str,
        target_goal: str,
        generated_code: str,
        confidence: float,
        genome_id: str = "",
    ) -> SynthesisEvent:
        """
        Process program synthesis from CGP/EGGP.

        Caches synthesized programs and tracks evolution.
        """
        self.state.synthesis_cache[program_id] = generated_code

        event = SynthesisEvent(
            event_type=VILEventType.SYNTH_PROGRAM,
            program_id=program_id,
            source_image=source_image,
            target_goal=target_goal,
            generated_code=generated_code,
            confidence=confidence,
            genome_id=genome_id,
            generation=self.state.generation,
            source="vil.coordinator",
        )

        await self._emit(event)

        logger.info(f"Synthesis: program={program_id}, confidence={confidence:.4f}")
        return event

    # === CMP Integration ===

    async def update_clade_cmp(
        self,
        clade_id: str,
        fitness: float,
        pressure: float = 1.0,
        mutation_rate: float = 0.1,
        state: str = "active",
    ) -> CMPEvent:
        """
        Update CMP metrics for a clade.

        Integrates with Clade Manager for evolutionary tracking.
        """
        phi_weighted = fitness * PHI if self.config.phi_weighted else fitness
        self.state.active_clades[clade_id] = phi_weighted

        event = CMPEvent(
            event_type=VILEventType.CMP_FITNESS,
            clade_id=clade_id,
            fitness=fitness,
            phi_weighted=phi_weighted,
            pressure=pressure,
            mutation_rate=mutation_rate,
            state=state,
            source="vil.coordinator",
        )

        await self._emit(event)

        logger.debug(f"CMP update: clade={clade_id}, fitness={fitness:.4f}")
        return event

    async def merge_clades(
        self,
        parent_clades: List[str],
        new_clade: str,
        merged_fitness: float,
    ) -> CMPEvent:
        """
        Merge clades when fitness threshold reached.
        """
        # Remove merged clades
        for clade in parent_clades:
            self.state.active_clades.pop(clade, None)

        # Add new merged clade
        self.state.active_clades[new_clade] = merged_fitness
        self.state.generation += 1

        event = CMPEvent(
            event_type=VILEventType.CMP_MERGE,
            clade_id=new_clade,
            fitness=merged_fitness,
            phi_weighted=merged_fitness * PHI,
            state="merged",
            source="vil.coordinator",
            data={"parent_clades": parent_clades},
        )

        await self._emit(event)

        logger.info(
            f"Clade merge: {parent_clades} → {new_clade} "
            f"(fitness={merged_fitness:.4f}, gen={self.state.generation})"
        )
        return event

    # === Geometric Integration ===

    async def update_geometric_embedding(
        self,
        entity_id: str,
        embedding: List[float],
        manifold: str = "spherical",
    ) -> VILEvent:
        """
        Update geometric embedding for an entity.

        Supports:
        - Spherical (S^n): Local geometry
        - Hyperbolic (H^n): Hierarchical structure
        - Fiber bundles: Curvature and transport
        """
        if self.config.geometric_enabled:
            self.state.geometric_embeddings[entity_id] = embedding

        event = VILEvent(
            event_type=VILEventType.GEOM_EMBEDDING,
            source="vil.coordinator",
            data={
                "entity_id": entity_id,
                "manifold": manifold,
                "embedding_dim": len(embedding),
            },
        )

        await self._emit(event)

        logger.debug(f"Geometric update: entity={entity_id}, manifold={manifold}")
        return event

    # === Integration Orchestration ===

    async def start_phase(
        self,
        zone: int,
        phase: str,
        step: int,
        agent: str,
    ) -> IntegrationEvent:
        """Mark start of integration phase/step."""
        event = create_integration_event(
            zone=zone,
            phase=phase,
            step=step,
            agent=agent,
            status="in_progress",
        )
        event.source = "vil.coordinator"

        await self._emit(event)

        logger.info(f"Phase start: Zone {zone}, {phase}, Step {step} by {agent}")
        return event

    async def complete_step(
        self,
        zone: int,
        phase: str,
        step: int,
        agent: str,
        wip_percent: float = 100.0,
    ) -> IntegrationEvent:
        """Mark completion of integration step."""
        event = create_integration_event(
            zone=zone,
            phase=phase,
            step=step,
            agent=agent,
            status="completed",
            wip_percent=wip_percent,
        )
        event.source = "vil.coordinator"

        await self._emit(event)

        logger.info(
            f"Step complete: Zone {zone}, {phase}, Step {step} "
            f"by {agent} ({wip_percent:.0f}%)"
        )
        return event

    async def emit_error(
        self,
        zone: int,
        step: int,
        agent: str,
        error: str,
    ) -> IntegrationEvent:
        """Emit error event."""
        event = create_integration_event(
            zone=zone,
            phase="error",
            step=step,
            agent=agent,
            status="error",
        )
        event.source = "vil.coordinator"
        event.data["error"] = error

        await self._emit(event)

        logger.error(f"Error: Zone {zone}, Step {step} by {agent}: {error}")
        return event

    # === Event Management ===

    def subscribe(
        self,
        event_type: VILEventType,
        handler: Callable,
    ) -> None:
        """Subscribe to specific event type."""
        if event_type not in self._subscribers:
            self._subscribers[event_type] = []
        self._subscribers[event_type].append(handler)

    async def _emit(self, event: VILEvent) -> None:
        """Emit event to subscribers and bus."""
        # Notify subscribers
        if event.event_type in self._subscribers:
            for handler in self._subscribers[event.event_type]:
                try:
                    if asyncio.iscoroutinefunction(handler):
                        await handler(event)
                    else:
                        handler(event)
                except Exception as e:
                    logger.error(f"Subscriber error: {e}")

        # Emit to bus if configured
        if self._bus_emitter and self.config.event_emission:
            try:
                self._bus_emitter(event.to_bus_event())
            except Exception as e:
                logger.error(f"Bus emission error: {e}")

    # === State Queries ===

    def get_state(self) -> VILState:
        """Get current VIL state."""
        return self.state

    def get_stats(self) -> Dict[str, Any]:
        """Get VIL statistics."""
        return {
            "vision_buffer_size": len(self.state.vision_buffer),
            "icl_examples": len(self.state.icl_examples),
            "active_clades": len(self.state.active_clades),
            "synthesis_cache": len(self.state.synthesis_cache),
            "geometric_embeddings": len(self.state.geometric_embeddings),
            "cmp_score": self.state.cmp_score,
            "generation": self.state.generation,
            "total_episodes": self.state.total_episodes,
            "total_rewards": self.state.total_rewards,
            "avg_reward": (
                self.state.total_rewards / self.state.total_episodes
                if self.state.total_episodes > 0
                else 0.0
            ),
        }


# Factory function

def create_vil_coordinator(
    config: Optional[VILConfig] = None,
    bus_emitter: Optional[Callable] = None,
) -> VILCoordinator:
    """Create VIL Coordinator with default config."""
    return VILCoordinator(config=config, bus_emitter=bus_emitter)


__all__ = [
    "VILCoordinator",
    "create_vil_coordinator",
    "VILState",
    "VILConfig",
    "PHI",
    "CMP_DISCOUNT",
    "GLOBAL_CMP_FLOOR",
]
