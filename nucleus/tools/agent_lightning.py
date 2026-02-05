#!/usr/bin/env python3
"""
agent_lightning.py - Agent-Lightning Trainer Spine

DUALITY-BIND E10: OpenTelemetry spans → shaped rewards → training signals.

Ring: 1 (Operator)
Protocol: DKIN v29
"""

import argparse
import json
import sys
import time
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Dict, List, Any, Optional
import os

SPAN_DIR = Path(os.environ.get("PLURIBUS_SPAN_DIR", ".pluribus/spans"))


@dataclass
class TrainingSpan:
    """A span for training (OpenTelemetry-inspired)."""
    span_id: str
    trace_id: str
    dag_id: str
    node_id: str
    lineage_id: str
    parent_span_id: Optional[str] = None
    
    # Metrics
    start_time: float = 0.0
    end_time: float = 0.0
    duration_ms: float = 0.0
    
    # Guard results
    guard_results: Dict[str, str] = field(default_factory=dict)
    guard_score: float = 1.0
    
    # Shaped rewards
    task_reward: float = 0.0
    coordination_reward: float = 0.0
    efficiency_reward: float = 0.0
    shaped_reward: float = 0.0
    
    # Attributes
    actor: str = ""
    action: str = ""
    attributes: Dict[str, Any] = field(default_factory=dict)


class AgentLightning:
    """Trainer spine for shaped rewards and credit assignment."""
    
    def __init__(self, span_dir: Path = None):
        self.span_dir = span_dir or SPAN_DIR
        self.span_dir.mkdir(parents=True, exist_ok=True)
        self.spans: Dict[str, TrainingSpan] = {}
    
    def start_span(
        self,
        span_id: str,
        trace_id: str,
        dag_id: str,
        node_id: str,
        lineage_id: str,
        actor: str = "",
        action: str = "",
    ) -> TrainingSpan:
        """Start a new training span."""
        span = TrainingSpan(
            span_id=span_id,
            trace_id=trace_id,
            dag_id=dag_id,
            node_id=node_id,
            lineage_id=lineage_id,
            start_time=time.time(),
            actor=actor,
            action=action,
        )
        self.spans[span_id] = span
        return span
    
    def end_span(self, span_id: str, success: bool = True, reward: float = 0.0):
        """End a span and compute rewards."""
        if span_id not in self.spans:
            return
        
        span = self.spans[span_id]
        span.end_time = time.time()
        span.duration_ms = (span.end_time - span.start_time) * 1000
        span.task_reward = reward if success else -0.5
        
        # Compute shaped reward
        span.shaped_reward = self._compute_shaped_reward(span)
        
        # Persist
        self._save_span(span)
    
    def _compute_shaped_reward(self, span: TrainingSpan) -> float:
        """
        Compute shaped reward from components.
        
        shaped_reward = w_task * task + w_guard * guard + w_coord * coord + w_eff * eff
        """
        weights = {
            "task": 0.5,
            "guard": 0.3,
            "coordination": 0.1,
            "efficiency": 0.1,
        }
        
        # Efficiency: bonus for fast completion
        target_ms = 1000  # 1 second target
        efficiency = max(0, 1 - (span.duration_ms / target_ms))
        span.efficiency_reward = efficiency
        
        shaped = (
            weights["task"] * span.task_reward +
            weights["guard"] * span.guard_score +
            weights["coordination"] * span.coordination_reward +
            weights["efficiency"] * span.efficiency_reward
        )
        
        return shaped
    
    def _save_span(self, span: TrainingSpan):
        """Save span to disk."""
        path = self.span_dir / f"{span.span_id}.json"
        with open(path, "w") as f:
            json.dump(asdict(span), f)
    
    def load_spans(self, trace_id: str = None) -> List[TrainingSpan]:
        """Load spans from disk."""
        spans = []
        for path in self.span_dir.glob("*.json"):
            try:
                data = json.loads(path.read_text())
                span = TrainingSpan(**data)
                if trace_id is None or span.trace_id == trace_id:
                    spans.append(span)
            except (json.JSONDecodeError, OSError, TypeError, KeyError):
                continue  # Skip malformed or unreadable span files
        return spans
    
    def compute_credit_assignment(self, trace_id: str) -> Dict[str, float]:
        """
        Compute credit assignment for spans in a trace.
        Uses Laplacian-based diffusion for causal credit.
        """
        spans = self.load_spans(trace_id)
        if not spans:
            return {}
        
        # Simple credit: proportional to shaped reward
        total_reward = sum(s.shaped_reward for s in spans)
        if total_reward == 0:
            return {s.span_id: 1.0 / len(spans) for s in spans}
        
        credit = {}
        for s in spans:
            credit[s.span_id] = s.shaped_reward / total_reward
        
        return credit
    
    def get_training_batch(self, batch_size: int = 32) -> List[Dict[str, Any]]:
        """Get a batch of spans for training."""
        spans = self.load_spans()[:batch_size]
        return [asdict(s) for s in spans]


def main():
    parser = argparse.ArgumentParser(description="Agent-Lightning Trainer")
    subparsers = parser.add_subparsers(dest="command", required=True)
    
    p_start = subparsers.add_parser("start", help="Start a span")
    p_start.add_argument("--span-id", required=True)
    p_start.add_argument("--trace-id", required=True)
    p_start.add_argument("--dag-id", default="dag-0")
    p_start.add_argument("--node-id", default="node-0")
    p_start.add_argument("--lineage-id", default="lineage-0")
    
    p_end = subparsers.add_parser("end", help="End a span")
    p_end.add_argument("--span-id", required=True)
    p_end.add_argument("--success", type=bool, default=True)
    p_end.add_argument("--reward", type=float, default=1.0)
    
    p_credit = subparsers.add_parser("credit", help="Compute credit")
    p_credit.add_argument("trace_id")
    
    subparsers.add_parser("batch", help="Get training batch")
    
    args = parser.parse_args()
    lightning = AgentLightning()
    
    if args.command == "start":
        span = lightning.start_span(
            args.span_id, args.trace_id, args.dag_id,
            args.node_id, args.lineage_id
        )
        print(f"✅ Started span: {span.span_id}")
        return 0
    elif args.command == "end":
        lightning.spans[args.span_id] = lightning.load_spans()[0] if lightning.load_spans() else None
        if lightning.spans.get(args.span_id):
            lightning.end_span(args.span_id, args.success, args.reward)
            print(f"✅ Ended span: {args.span_id}")
        return 0
    elif args.command == "credit":
        credit = lightning.compute_credit_assignment(args.trace_id)
        print(f"Credit assignment for {args.trace_id}:")
        for k, v in credit.items():
            print(f"  {k}: {v:.3f}")
        return 0
    elif args.command == "batch":
        batch = lightning.get_training_batch()
        print(f"Training batch: {len(batch)} spans")
        return 0


if __name__ == "__main__":
    sys.exit(main())
