#!/usr/bin/env python3
"""
training.py - Neural Gate Training Pipeline

Implements online learning from commit history for the Neural Gate model.
Part of the ARK Continual Learning system.
"""

import logging
import time
from pathlib import Path
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, field

logger = logging.getLogger("ARK.Training")

# Try importing torch
try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from torch.optim import Adam
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False
    logger.warning("PyTorch not available - training disabled")


@dataclass
class TrainingConfig:
    """Configuration for neural gate training."""
    batch_size: int = 32
    learning_rate: float = 0.001
    ewc_lambda: float = 0.4  # Elastic Weight Consolidation strength
    checkpoint_interval: int = 100  # Save every N experiences
    min_experiences: int = 50  # Minimum before training starts


@dataclass
class TrainingMetrics:
    """Metrics from a training epoch."""
    loss: float
    accuracy: float
    experiences_used: int
    epoch: int
    timestamp: float = field(default_factory=time.time)


class NeuralTrainer:
    """
    Trains the Neural Gate model from commit experience.
    
    Implements:
    1. Experience sampling from replay buffer
    2. EWC (Elastic Weight Consolidation) for continual learning
    3. Periodic checkpointing
    """
    
    def __init__(
        self,
        config: Optional[TrainingConfig] = None,
        checkpoint_dir: Optional[Path] = None,
    ):
        self.config = config or TrainingConfig()
        self.checkpoint_dir = checkpoint_dir or Path.cwd() / ".ark" / "checkpoints"
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self._model = None
        self._optimizer = None
        self._fisher_diag: Optional[Dict[str, Any]] = None
        self._epoch = 0
    
    @property
    def model(self):
        """Lazy-load the neural gate model."""
        if self._model is None and HAS_TORCH:
            try:
                from nucleus.ark.neural.model import NeuralGateModel
                self._model = NeuralGateModel()
                self._optimizer = Adam(
                    self._model.parameters(),
                    lr=self.config.learning_rate,
                )
                self._load_latest_checkpoint()
            except ImportError:
                logger.warning("NeuralGateModel not available")
        return self._model
    
    def train_epoch(
        self,
        experiences: List[Dict[str, Any]],
    ) -> Optional[TrainingMetrics]:
        """
        Train one epoch on sampled experiences.
        
        Args:
            experiences: List of experience dicts with:
                - entropy: Dict[str, float] (H* vector)
                - success: bool (whether commit was good)
                - reward: float (CMP delta)
        
        Returns:
            TrainingMetrics if training succeeded, None otherwise
        """
        if not HAS_TORCH or not self.model:
            logger.warning("Training unavailable (no PyTorch or model)")
            return None
        
        if len(experiences) < self.config.min_experiences:
            logger.info(f"Insufficient experiences: {len(experiences)} < {self.config.min_experiences}")
            return None
        
        self.model.train()
        total_loss = 0.0
        correct = 0
        
        # Process in batches
        for i in range(0, len(experiences), self.config.batch_size):
            batch = experiences[i:i + self.config.batch_size]
            
            # Prepare tensors
            entropy_batch = torch.tensor([
                [exp["entropy"].get(k, 0.5) for k in sorted(exp["entropy"].keys())]
                for exp in batch
            ], dtype=torch.float32)
            
            target_batch = torch.tensor([
                1.0 if exp.get("success", True) else 0.0
                for exp in batch
            ], dtype=torch.float32).unsqueeze(1)
            
            # Forward pass
            self._optimizer.zero_grad()
            predictions = self.model(entropy_batch)
            
            # Compute loss (BCE + EWC regularization)
            bce_loss = F.binary_cross_entropy(predictions, target_batch)
            ewc_loss = self._compute_ewc_loss() if self._fisher_diag else 0.0
            loss = bce_loss + self.config.ewc_lambda * ewc_loss
            
            # Backward pass
            loss.backward()
            self._optimizer.step()
            
            total_loss += loss.item() * len(batch)
            correct += ((predictions > 0.5) == target_batch).sum().item()
        
        self._epoch += 1
        accuracy = correct / len(experiences)
        avg_loss = total_loss / len(experiences)
        
        metrics = TrainingMetrics(
            loss=avg_loss,
            accuracy=accuracy,
            experiences_used=len(experiences),
            epoch=self._epoch,
        )
        
        # Checkpoint periodically
        if self._epoch % self.config.checkpoint_interval == 0:
            self._save_checkpoint()
            self._update_fisher(experiences)
        
        logger.info(f"Epoch {self._epoch}: loss={avg_loss:.4f}, accuracy={accuracy:.2%}")
        return metrics
    
    def _compute_ewc_loss(self) -> float:
        """Compute Elastic Weight Consolidation loss."""
        if not self._fisher_diag or not HAS_TORCH:
            return 0.0
        
        ewc_loss = 0.0
        for name, param in self.model.named_parameters():
            if name in self._fisher_diag:
                fisher = self._fisher_diag[name]["fisher"]
                old_param = self._fisher_diag[name]["param"]
                ewc_loss += (fisher * (param - old_param).pow(2)).sum()
        
        return ewc_loss
    
    def _update_fisher(self, experiences: List[Dict[str, Any]]) -> None:
        """Update Fisher information diagonal for EWC."""
        if not HAS_TORCH or not self.model:
            return
        
        self._fisher_diag = {}
        self.model.eval()
        
        # Sample subset for Fisher computation
        sample = experiences[:min(100, len(experiences))]
        
        for exp in sample:
            entropy = torch.tensor([
                [exp["entropy"].get(k, 0.5) for k in sorted(exp["entropy"].keys())]
            ], dtype=torch.float32)
            
            self.model.zero_grad()
            output = self.model(entropy)
            output.backward()
            
            for name, param in self.model.named_parameters():
                if param.grad is not None:
                    if name not in self._fisher_diag:
                        self._fisher_diag[name] = {
                            "fisher": param.grad.data.clone().pow(2),
                            "param": param.data.clone(),
                        }
                    else:
                        self._fisher_diag[name]["fisher"] += param.grad.data.clone().pow(2)
        
        # Normalize
        for name in self._fisher_diag:
            self._fisher_diag[name]["fisher"] /= len(sample)
        
        logger.info(f"Updated Fisher diagonal with {len(sample)} experiences")
    
    def _save_checkpoint(self) -> None:
        """Save model checkpoint."""
        if not HAS_TORCH or not self.model:
            return
        
        path = self.checkpoint_dir / f"neural_gate_epoch_{self._epoch}.pt"
        torch.save({
            "epoch": self._epoch,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self._optimizer.state_dict(),
            "fisher_diag": self._fisher_diag,
        }, path)
        
        # Also save as "latest"
        latest_path = self.checkpoint_dir / "neural_gate_latest.pt"
        torch.save({
            "epoch": self._epoch,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self._optimizer.state_dict(),
            "fisher_diag": self._fisher_diag,
        }, latest_path)
        
        logger.info(f"Saved checkpoint: {path}")
    
    def _load_latest_checkpoint(self) -> bool:
        """Load latest checkpoint if available."""
        if not HAS_TORCH:
            return False
        
        latest_path = self.checkpoint_dir / "neural_gate_latest.pt"
        if not latest_path.exists():
            return False
        
        try:
            checkpoint = torch.load(latest_path, weights_only=True)
            self.model.load_state_dict(checkpoint["model_state_dict"])
            self._optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
            self._epoch = checkpoint["epoch"]
            self._fisher_diag = checkpoint.get("fisher_diag")
            logger.info(f"Loaded checkpoint from epoch {self._epoch}")
            return True
        except Exception as e:
            logger.warning(f"Failed to load checkpoint: {e}")
            return False
