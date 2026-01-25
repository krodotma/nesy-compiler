#!/usr/bin/env python3
"""
ewc.py - Elastic Weight Consolidation for ARK CL

P2-025: Implement forgetting mechanism (EWC-style)
P2-026: Add catastrophic forgetting prevention

EWC prevents catastrophic forgetting by:
1. Computing Fisher Information Matrix (FIM) for important parameters
2. Adding regularization penalty when weights drift from optimal values
3. Allowing continual learning without forgetting previous tasks

Reference: Kirkpatrick et al., "Overcoming catastrophic forgetting in NNs"
"""

import json
import time
import numpy as np
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any, Tuple
from pathlib import Path
import logging

logger = logging.getLogger("ARK.CL.EWC")


@dataclass
class EWCConfig:
    """Configuration for Elastic Weight Consolidation."""
    lambda_ewc: float = 5000.0  # EWC regularization strength
    fisher_samples: int = 200   # Samples for Fisher computation
    online_ewc: bool = True     # Use online EWC (single Fisher, updated)
    decay: float = 0.9          # Decay factor for online EWC
    importance_threshold: float = 0.1  # Min importance to track
    
    def to_dict(self) -> Dict:
        return {
            "lambda_ewc": self.lambda_ewc,
            "fisher_samples": self.fisher_samples,
            "online_ewc": self.online_ewc,
            "decay": self.decay,
            "importance_threshold": self.importance_threshold
        }
    
    @classmethod
    def from_dict(cls, data: Dict) -> "EWCConfig":
        return cls(**{k: v for k, v in data.items() if k in cls.__dataclass_fields__})


@dataclass
class ParameterImportance:
    """Importance of a parameter for a specific task."""
    param_name: str
    optimal_value: float
    fisher_diagonal: float  # Importance (Fisher Information)
    task_id: str
    timestamp: float = field(default_factory=time.time)
    
    def to_dict(self) -> Dict:
        return {
            "param_name": self.param_name,
            "optimal_value": self.optimal_value,
            "fisher_diagonal": self.fisher_diagonal,
            "task_id": self.task_id,
            "timestamp": self.timestamp
        }
    
    @classmethod
    def from_dict(cls, data: Dict) -> "ParameterImportance":
        return cls(**data)


class ElasticWeightConsolidation:
    """
    EWC implementation for preventing catastrophic forgetting.
    
    P2-025: Forgetting mechanism
    P2-026: Catastrophic forgetting prevention
    
    For ARK's gate learning:
    - Parameters = gate threshold configs
    - Tasks = different types of commits (by file type, etymology, etc.)
    - Fisher = importance of each threshold for each commit type
    """
    
    def __init__(
        self,
        config: Optional[EWCConfig] = None,
        storage_path: Optional[str] = None
    ):
        self.config = config or EWCConfig()
        self.storage_path = Path(storage_path) if storage_path else Path("~/.ark/cl/ewc_state.json").expanduser()
        
        # Fisher Information for each parameter
        self.fisher: Dict[str, float] = {}
        # Optimal parameter values (star values)
        self.optimal_params: Dict[str, float] = {}
        # Per-task importance
        self.task_importance: Dict[str, Dict[str, ParameterImportance]] = {}
        # Current task
        self.current_task: str = "default"
        
        self._load()
    
    def _load(self) -> None:
        """Load EWC state from storage."""
        if self.storage_path.exists():
            try:
                data = json.loads(self.storage_path.read_text())
                self.fisher = data.get("fisher", {})
                self.optimal_params = data.get("optimal_params", {})
                self.current_task = data.get("current_task", "default")
                
                for task_id, params in data.get("task_importance", {}).items():
                    self.task_importance[task_id] = {
                        name: ParameterImportance.from_dict(imp)
                        for name, imp in params.items()
                    }
                
                logger.debug("Loaded EWC state: %d params, %d tasks", 
                            len(self.fisher), len(self.task_importance))
            except Exception as e:
                logger.warning("Failed to load EWC state: %s", e)
    
    def _save(self) -> None:
        """Save EWC state to storage."""
        try:
            self.storage_path.parent.mkdir(parents=True, exist_ok=True)
            
            data = {
                "fisher": self.fisher,
                "optimal_params": self.optimal_params,
                "current_task": self.current_task,
                "task_importance": {
                    task_id: {name: imp.to_dict() for name, imp in params.items()}
                    for task_id, params in self.task_importance.items()
                },
                "config": self.config.to_dict(),
                "saved_at": time.time()
            }
            
            self.storage_path.write_text(json.dumps(data, indent=2))
        except Exception as e:
            logger.warning("Failed to save EWC state: %s", e)
    
    def register_task(self, task_id: str, params: Dict[str, float]) -> None:
        """
        Register a new task with its optimal parameters.
        
        Call after training on a new task (commit type).
        """
        self.current_task = task_id
        
        # Store optimal parameters
        for name, value in params.items():
            self.optimal_params[name] = value
        
        # Initialize importance for this task
        if task_id not in self.task_importance:
            self.task_importance[task_id] = {}
        
        self._save()
        logger.info("Registered task %s with %d params", task_id, len(params))
    
    def compute_fisher(
        self,
        params: Dict[str, float],
        gradients_fn,  # Callable that returns {param: gradient}
        samples: Optional[List[Any]] = None
    ) -> Dict[str, float]:
        """
        Compute Fisher Information Matrix diagonal.
        
        Fisher diagonal = E[grad^2] for each parameter.
        High Fisher = parameter is important for current task.
        
        Args:
            params: Current parameter values
            gradients_fn: Function returning gradients for a sample
            samples: Samples to compute Fisher over
        """
        fisher = {name: 0.0 for name in params}
        n_samples = len(samples) if samples else self.config.fisher_samples
        
        # For ARK, we approximate gradients from CMP deltas
        # In a real implementation, this would use actual gradients
        for sample in (samples or []):
            try:
                grads = gradients_fn(sample)
                for name, grad in grads.items():
                    fisher[name] += grad ** 2
            except Exception:
                pass
        
        # Normalize
        if n_samples > 0:
            fisher = {name: f / n_samples for name, f in fisher.items()}
        
        # Apply threshold
        fisher = {
            name: max(f, 0.0) if f > self.config.importance_threshold else 0.0
            for name, f in fisher.items()
        }
        
        # Online EWC: combine with previous Fisher
        if self.config.online_ewc and self.fisher:
            for name in fisher:
                if name in self.fisher:
                    fisher[name] = (
                        self.config.decay * self.fisher[name] +
                        (1 - self.config.decay) * fisher[name]
                    )
        
        self.fisher = fisher
        
        # Store per-task importance
        for name, f in fisher.items():
            if f > 0:
                self.task_importance.setdefault(self.current_task, {})[name] = ParameterImportance(
                    param_name=name,
                    optimal_value=params.get(name, 0.0),
                    fisher_diagonal=f,
                    task_id=self.current_task
                )
        
        self._save()
        return fisher
    
    def ewc_loss(self, params: Dict[str, float]) -> float:
        """
        Compute EWC regularization loss.
        
        L_EWC = (lambda/2) * sum_i F_i * (theta_i - theta*_i)^2
        
        This penalizes deviation from optimal parameters,
        weighted by their importance (Fisher).
        """
        if not self.fisher or not self.optimal_params:
            return 0.0
        
        loss = 0.0
        for name, value in params.items():
            if name in self.fisher and name in self.optimal_params:
                fisher_i = self.fisher[name]
                optimal_i = self.optimal_params[name]
                loss += fisher_i * (value - optimal_i) ** 2
        
        return (self.config.lambda_ewc / 2) * loss
    
    def ewc_gradient(self, params: Dict[str, float]) -> Dict[str, float]:
        """
        Compute EWC gradient for regularization.
        
        d(L_EWC)/d(theta_i) = lambda * F_i * (theta_i - theta*_i)
        """
        grads = {}
        
        for name, value in params.items():
            if name in self.fisher and name in self.optimal_params:
                fisher_i = self.fisher[name]
                optimal_i = self.optimal_params[name]
                grads[name] = self.config.lambda_ewc * fisher_i * (value - optimal_i)
            else:
                grads[name] = 0.0
        
        return grads
    
    def consolidate(self, params: Dict[str, float]) -> None:
        """
        Consolidate current task knowledge.
        
        Call this after training on a task to lock in the learned parameters.
        """
        # Update optimal params
        self.optimal_params.update(params)
        
        # Increase importance of frequently used params
        for name in params:
            if name in self.fisher:
                # Slight boost to reinforce
                self.fisher[name] *= 1.05
        
        self._save()
        logger.info("Consolidated knowledge for task %s", self.current_task)
    
    def get_frozen_params(self, threshold: float = 0.8) -> List[str]:
        """
        Get parameters that should be frozen (very high importance).
        
        These params are critical for previous tasks and should not change.
        """
        if not self.fisher:
            return []
        
        max_fisher = max(self.fisher.values()) if self.fisher else 1.0
        normalized = {name: f / max_fisher for name, f in self.fisher.items()}
        
        return [name for name, f in normalized.items() if f >= threshold]
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get EWC statistics for monitoring."""
        return {
            "current_task": self.current_task,
            "tracked_params": len(self.fisher),
            "tracked_tasks": len(self.task_importance),
            "top_important_params": [
                {"param": name, "fisher": f}
                for name, f in sorted(self.fisher.items(), 
                                     key=lambda x: x[1], reverse=True)[:10]
            ],
            "frozen_params": self.get_frozen_params(),
            "config": self.config.to_dict()
        }
