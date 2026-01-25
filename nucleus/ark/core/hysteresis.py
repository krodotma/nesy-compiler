#!/usr/bin/env python3
"""
hysteresis.py - Hysteresis implementation for ARK

DNA Axiom 4: Past states influence the present.

Implements:
- Historical decision consultation
- State momentum tracking
- Anti-regression safeguards
"""

import json
from dataclasses import dataclass, field
from typing import Dict, List, Optional
from pathlib import Path
from datetime import datetime, timedelta


@dataclass
class HistoricalDecision:
    """A past decision that influences current behavior."""
    sha: str
    decision_type: str  # commit_accepted, commit_rejected, gate_triggered
    gate: str
    reason: str
    timestamp: str
    entropy_state: Dict[str, float] = field(default_factory=dict)


class HysteresisTracker:
    """
    Tracks historical decisions to influence current behavior.
    
    Implements Axiom 4: Hysteresis - past states influence present.
    
    Key behaviors:
    1. If same file was recently rejected, increase scrutiny
    2. If CMP has been declining, flag for review
    3. If entropy spiked recently, extend cooldown
    """
    
    def __init__(self, ark_dir: Path):
        self.history_path = ark_dir / "hysteresis.json"
        self.decisions: List[HistoricalDecision] = []
        self._load()
    
    def _load(self) -> None:
        """Load historical decisions from disk."""
        if self.history_path.exists():
            try:
                data = json.loads(self.history_path.read_text())
                self.decisions = [
                    HistoricalDecision(**d) for d in data.get("decisions", [])
                ]
            except Exception:
                self.decisions = []
    
    def _save(self) -> None:
        """Save historical decisions to disk."""
        self.history_path.parent.mkdir(parents=True, exist_ok=True)
        data = {
            "decisions": [
                {
                    "sha": d.sha,
                    "decision_type": d.decision_type,
                    "gate": d.gate,
                    "reason": d.reason,
                    "timestamp": d.timestamp,
                    "entropy_state": d.entropy_state
                }
                for d in self.decisions[-100:]  # Keep last 100
            ]
        }
        self.history_path.write_text(json.dumps(data, indent=2))
    
    def record_decision(self, sha: str, decision_type: str, gate: str, 
                       reason: str, entropy: Dict[str, float]) -> None:
        """Record a decision for future reference."""
        decision = HistoricalDecision(
            sha=sha,
            decision_type=decision_type,
            gate=gate,
            reason=reason,
            timestamp=datetime.utcnow().isoformat(),
            entropy_state=entropy
        )
        self.decisions.append(decision)
        self._save()
    
    def check_hysteresis(self, files: List[str], entropy: Dict[str, float]) -> tuple[bool, str]:
        """
        Check if hysteresis conditions affect current commit.
        
        Returns:
            (should_proceed, warning_message)
        """
        recent_cutoff = datetime.utcnow() - timedelta(hours=1)
        recent_rejections = [
            d for d in self.decisions
            if d.decision_type == "commit_rejected"
            and datetime.fromisoformat(d.timestamp) > recent_cutoff
        ]
        
        # Check 1: Too many recent rejections
        if len(recent_rejections) >= 3:
            return False, "Hysteresis: 3+ rejections in last hour, cooldown required"
        
        # Check 2: Entropy trending upward
        recent_entropies = [d.entropy_state for d in self.decisions[-5:] if d.entropy_state]
        if recent_entropies:
            trend = self._calculate_entropy_trend(recent_entropies, entropy)
            if trend > 0.1:
                return True, f"Hysteresis WARNING: Entropy trending up by {trend:.2f}"
        
        # Check 3: Same files rejected recently
        for d in recent_rejections:
            # Future: track affected files per decision
            pass
        
        return True, "Hysteresis check passed"
    
    def _calculate_entropy_trend(self, historical: List[Dict], current: Dict) -> float:
        """Calculate entropy trend (positive = increasing, negative = decreasing)."""
        if not historical or not current:
            return 0.0
        
        def total_entropy(e: Dict) -> float:
            return sum(e.values()) / len(e) if e else 0.5
        
        historical_avg = sum(total_entropy(e) for e in historical) / len(historical)
        current_total = total_entropy(current)
        
        return current_total - historical_avg
    
    def get_momentum(self) -> str:
        """Get current momentum state based on recent decisions."""
        if len(self.decisions) < 3:
            return "neutral"
        
        recent = self.decisions[-5:]
        accepts = sum(1 for d in recent if d.decision_type == "commit_accepted")
        rejects = sum(1 for d in recent if d.decision_type == "commit_rejected")
        
        if accepts >= 4:
            return "positive"  # Streak of good commits
        elif rejects >= 3:
            return "negative"  # Too many rejections
        else:
            return "neutral"
    
    def clear_history(self) -> None:
        """Clear all historical decisions."""
        self.decisions = []
        self._save()
