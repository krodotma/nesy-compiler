#!/usr/bin/env python3
"""
checkpoint.py - Model Checkpointing for ARK CL

P2-034: Implement checkpoint saving
P2-035: Add model versioning in .ark/
P2-037: Implement warm-start from checkpoints

Features:
- Checkpoint saving with metadata
- Version tagging
- Best model tracking
- Warm-start restoration
"""

import json
import time
import shutil
from dataclasses import dataclass, field, asdict
from typing import Dict, List, Optional, Any
from pathlib import Path
import logging

logger = logging.getLogger("ARK.CL.Checkpoint")


@dataclass
class ModelCheckpoint:
    """A saved model checkpoint."""
    version: str
    timestamp: float
    cmp_mean: float  # Mean CMP during training
    success_rate: float
    curriculum_level: int
    parameters: Dict[str, float]
    fisher: Dict[str, float]  # EWC Fisher values
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict:
        return asdict(self)
    
    @classmethod
    def from_dict(cls, data: Dict) -> "ModelCheckpoint":
        return cls(**data)
    
    @property
    def score(self) -> float:
        """Compute quality score for ranking checkpoints."""
        return self.cmp_mean * 0.4 + self.success_rate * 0.4 + (self.curriculum_level / 5) * 0.2


class CheckpointManager:
    """
    Manages model checkpoints for ARK CL.
    
    P2-034, P2-035, P2-037: Checkpoint management
    """
    
    def __init__(
        self,
        checkpoint_dir: Optional[str] = None,
        max_checkpoints: int = 10,
        keep_best: int = 3
    ):
        self.checkpoint_dir = Path(checkpoint_dir) if checkpoint_dir else Path("~/.ark/checkpoints").expanduser()
        self.max_checkpoints = max_checkpoints
        self.keep_best = keep_best
        self.checkpoints: List[ModelCheckpoint] = []
        self.best_version: Optional[str] = None
        
        self._ensure_dir()
        self._load_index()
    
    def _ensure_dir(self) -> None:
        """Create checkpoint directory."""
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
    
    def _index_path(self) -> Path:
        """Get path to checkpoint index."""
        return self.checkpoint_dir / "index.json"
    
    def _load_index(self) -> None:
        """Load checkpoint index."""
        index_path = self._index_path()
        if index_path.exists():
            try:
                data = json.loads(index_path.read_text())
                self.checkpoints = [
                    ModelCheckpoint.from_dict(c) for c in data.get("checkpoints", [])
                ]
                self.best_version = data.get("best_version")
                logger.debug("Loaded %d checkpoints", len(self.checkpoints))
            except Exception as e:
                logger.warning("Failed to load checkpoint index: %s", e)
    
    def _save_index(self) -> None:
        """Save checkpoint index."""
        try:
            data = {
                "checkpoints": [c.to_dict() for c in self.checkpoints],
                "best_version": self.best_version,
                "updated_at": time.time()
            }
            self._index_path().write_text(json.dumps(data, indent=2))
        except Exception as e:
            logger.warning("Failed to save checkpoint index: %s", e)
    
    def _checkpoint_path(self, version: str) -> Path:
        """Get path for a checkpoint file."""
        return self.checkpoint_dir / f"checkpoint_{version}.json"
    
    def save(
        self,
        parameters: Dict[str, float],
        fisher: Dict[str, float],
        cmp_mean: float,
        success_rate: float,
        curriculum_level: int,
        version: Optional[str] = None,
        metadata: Optional[Dict] = None
    ) -> str:
        """
        Save a new checkpoint.
        
        P2-034: Implement checkpoint saving
        """
        # Generate version if not provided
        if version is None:
            version = f"v{len(self.checkpoints) + 1}_{int(time.time())}"
        
        checkpoint = ModelCheckpoint(
            version=version,
            timestamp=time.time(),
            cmp_mean=cmp_mean,
            success_rate=success_rate,
            curriculum_level=curriculum_level,
            parameters=parameters,
            fisher=fisher,
            metadata=metadata or {}
        )
        
        # Save checkpoint file
        checkpoint_path = self._checkpoint_path(version)
        checkpoint_path.write_text(json.dumps(checkpoint.to_dict(), indent=2))
        
        # Add to index
        self.checkpoints.append(checkpoint)
        
        # Update best if this is better
        if self.best_version is None or checkpoint.score > self._get_checkpoint(self.best_version).score:
            self.best_version = version
            logger.info("New best checkpoint: %s (score=%.3f)", version, checkpoint.score)
        
        # Cleanup old checkpoints
        self._cleanup()
        
        self._save_index()
        logger.info("Saved checkpoint %s", version)
        return version
    
    def _get_checkpoint(self, version: str) -> Optional[ModelCheckpoint]:
        """Get checkpoint by version."""
        for c in self.checkpoints:
            if c.version == version:
                return c
        return None
    
    def load(self, version: str) -> Optional[ModelCheckpoint]:
        """
        Load a checkpoint by version.
        
        P2-037: Warm-start from checkpoints
        """
        checkpoint_path = self._checkpoint_path(version)
        if not checkpoint_path.exists():
            logger.warning("Checkpoint %s not found", version)
            return None
        
        try:
            data = json.loads(checkpoint_path.read_text())
            checkpoint = ModelCheckpoint.from_dict(data)
            logger.info("Loaded checkpoint %s", version)
            return checkpoint
        except Exception as e:
            logger.error("Failed to load checkpoint %s: %s", version, e)
            return None
    
    def load_best(self) -> Optional[ModelCheckpoint]:
        """Load the best checkpoint."""
        if self.best_version:
            return self.load(self.best_version)
        return None
    
    def load_latest(self) -> Optional[ModelCheckpoint]:
        """Load the most recent checkpoint."""
        if not self.checkpoints:
            return None
        latest = max(self.checkpoints, key=lambda c: c.timestamp)
        return self.load(latest.version)
    
    def _cleanup(self) -> None:
        """Remove old checkpoints, keeping best and recent ones."""
        if len(self.checkpoints) <= self.max_checkpoints:
            return
        
        # Sort by score (descending)
        by_score = sorted(self.checkpoints, key=lambda c: c.score, reverse=True)
        best_versions = {c.version for c in by_score[:self.keep_best]}
        
        # Sort by time (descending)
        by_time = sorted(self.checkpoints, key=lambda c: c.timestamp, reverse=True)
        recent_versions = {c.version for c in by_time[:self.max_checkpoints - self.keep_best]}
        
        # Keep union
        keep_versions = best_versions | recent_versions
        
        # Remove others
        to_remove = [c for c in self.checkpoints if c.version not in keep_versions]
        for checkpoint in to_remove:
            checkpoint_path = self._checkpoint_path(checkpoint.version)
            if checkpoint_path.exists():
                checkpoint_path.unlink()
            self.checkpoints.remove(checkpoint)
            logger.debug("Removed old checkpoint %s", checkpoint.version)
    
    def list_versions(self) -> List[Dict[str, Any]]:
        """List all available checkpoints."""
        return [
            {
                "version": c.version,
                "timestamp": c.timestamp,
                "score": c.score,
                "cmp_mean": c.cmp_mean,
                "success_rate": c.success_rate,
                "is_best": c.version == self.best_version
            }
            for c in sorted(self.checkpoints, key=lambda c: c.timestamp, reverse=True)
        ]
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get checkpoint statistics."""
        return {
            "total_checkpoints": len(self.checkpoints),
            "best_version": self.best_version,
            "latest_version": max(self.checkpoints, key=lambda c: c.timestamp).version if self.checkpoints else None,
            "checkpoint_dir": str(self.checkpoint_dir),
            "versions": [c.version for c in self.checkpoints]
        }
