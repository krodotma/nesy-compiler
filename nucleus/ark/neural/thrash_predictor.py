#!/usr/bin/env python3
"""
thrash_predictor.py - Thrash Probability Predictor

P2-046: Implement thrash probability predictor
P2-047: Add quality score predictor
P2-048: Create calibrated confidence intervals

Predicts:
- Thrash probability: Will this change cause agentic thrash?
- Quality score: Expected CMP after commit
- Confidence intervals: Uncertainty quantification
"""

import time
import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any, Tuple

logger = logging.getLogger("ARK.Neural.ThrashPredictor")

try:
    import torch
    import torch.nn as nn
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False


@dataclass
class ThrashPrediction:
    """Result of thrash prediction."""
    thrash_probability: float  # 0-1 probability of thrash
    quality_score: float       # Expected CMP (0-1)
    confidence_interval: Tuple[float, float]  # (low, high) CI
    risk_level: str            # low, medium, high, critical
    recommendation: str        # PROCEED, REVIEW, ABORT
    latency_ms: float = 0.0
    factors: Dict[str, float] = field(default_factory=dict)
    
    def to_dict(self) -> Dict:
        return {
            "thrash_probability": self.thrash_probability,
            "quality_score": self.quality_score,
            "confidence_interval": list(self.confidence_interval),
            "risk_level": self.risk_level,
            "recommendation": self.recommendation,
            "latency_ms": self.latency_ms,
            "factors": self.factors
        }


if HAS_TORCH:
    class ThrashPredictorModel(nn.Module):
        """
        Neural model for thrash prediction.
        
        P2-046: Thrash probability predictor
        """
        
        def __init__(self, input_dim: int = 23, hidden_dim: int = 64):
            super().__init__()
            
            self.encoder = nn.Sequential(
                nn.Linear(input_dim, hidden_dim),
                nn.BatchNorm1d(hidden_dim),
                nn.ReLU(),
                nn.Dropout(0.2),
                nn.Linear(hidden_dim, hidden_dim),
                nn.ReLU()
            )
            
            # Thrash probability head
            self.thrash_head = nn.Sequential(
                nn.Linear(hidden_dim, 32),
                nn.ReLU(),
                nn.Linear(32, 1),
                nn.Sigmoid()
            )
            
            # Quality score head
            self.quality_head = nn.Sequential(
                nn.Linear(hidden_dim, 32),
                nn.ReLU(),
                nn.Linear(32, 1),
                nn.Sigmoid()
            )
            
            # Calibration temperature
            self.temperature = nn.Parameter(torch.ones(1))
        
        def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
            features = self.encoder(x)
            thrash_prob = self.thrash_head(features)
            quality = self.quality_head(features)
            
            # Apply temperature scaling
            thrash_calibrated = thrash_prob / self.temperature
            
            return thrash_calibrated, quality


class ThrashPredictor:
    """
    High-level thrash prediction interface.
    
    Combines:
    - Thrash probability estimation
    - Quality score prediction
    - Calibrated confidence intervals
    """
    
    def __init__(
        self,
        model_path: Optional[str] = None,
        calibration_samples: int = 100
    ):
        self.model = None
        self.calibration_samples = calibration_samples
        
        if HAS_TORCH:
            self.model = ThrashPredictorModel()
            if model_path:
                self._load_model(model_path)
    
    def _load_model(self, path: str) -> None:
        """Load pretrained model."""
        try:
            state_dict = torch.load(path, map_location="cpu")
            self.model.load_state_dict(state_dict)
            self.model.eval()
            logger.info("Loaded thrash predictor from %s", path)
        except Exception as e:
            logger.warning("Failed to load model: %s", e)
    
    def predict(self, features: List[float]) -> ThrashPrediction:
        """
        Predict thrash probability and quality.
        
        Args:
            features: Feature vector from FeatureExtractor
            
        Returns:
            ThrashPrediction with probability, quality, and confidence
        """
        start_time = time.time()
        
        if HAS_TORCH and self.model:
            return self._neural_predict(features, start_time)
        else:
            return self._heuristic_predict(features, start_time)
    
    def _neural_predict(
        self, 
        features: List[float],
        start_time: float
    ) -> ThrashPrediction:
        """Neural prediction with calibration."""
        x = torch.tensor([features], dtype=torch.float32)
        
        self.model.eval()
        with torch.no_grad():
            thrash_prob, quality = self.model(x)
            thrash_prob = thrash_prob.item()
            quality_score = quality.item()
        
        # Compute confidence interval via MC dropout
        confidence_interval = self._compute_confidence_interval(x)
        
        return self._build_prediction(
            thrash_prob, quality_score, confidence_interval, start_time
        )
    
    def _compute_confidence_interval(
        self, 
        x: 'torch.Tensor',
        alpha: float = 0.05
    ) -> Tuple[float, float]:
        """
        Compute confidence interval via MC dropout.
        
        P2-048: Create calibrated confidence intervals
        """
        if not HAS_TORCH or not self.model:
            return (0.3, 0.7)
        
        # Enable dropout for MC sampling
        self.model.train()
        samples = []
        
        for _ in range(self.calibration_samples):
            with torch.no_grad():
                thrash, _ = self.model(x)
                samples.append(thrash.item())
        
        self.model.eval()
        
        # Compute percentiles
        samples.sort()
        lower_idx = int(alpha / 2 * len(samples))
        upper_idx = int((1 - alpha / 2) * len(samples))
        
        return (samples[lower_idx], samples[upper_idx])
    
    def _heuristic_predict(
        self, 
        features: List[float],
        start_time: float
    ) -> ThrashPrediction:
        """Fallback heuristic prediction."""
        # Extract key features (entropy values are first 8)
        entropy_values = features[:8] if len(features) >= 8 else [0.5] * 8
        
        # Thrash probability based on structural entropy and complexity
        h_struct = entropy_values[5]  # h_struct
        c_load = entropy_values[6]    # c_load
        
        thrash_prob = h_struct * 0.5 + c_load * 0.3 + 0.1
        thrash_prob = min(1.0, max(0.0, thrash_prob))
        
        # Quality score inverse of thrash and entropy
        h_total = sum(entropy_values) / len(entropy_values)
        quality_score = 1.0 - (thrash_prob * 0.4 + h_total * 0.6)
        
        # Simple confidence interval based on variance proxy
        variance = 0.1 + h_total * 0.2
        confidence_interval = (
            max(0.0, thrash_prob - variance),
            min(1.0, thrash_prob + variance)
        )
        
        return self._build_prediction(
            thrash_prob, quality_score, confidence_interval, start_time
        )
    
    def _build_prediction(
        self,
        thrash_prob: float,
        quality_score: float,
        confidence_interval: Tuple[float, float],
        start_time: float
    ) -> ThrashPrediction:
        """Build prediction object from values."""
        # Determine risk level
        if thrash_prob >= 0.8:
            risk_level = "critical"
            recommendation = "ABORT"
        elif thrash_prob >= 0.6:
            risk_level = "high"
            recommendation = "REVIEW"
        elif thrash_prob >= 0.4:
            risk_level = "medium"
            recommendation = "REVIEW"
        else:
            risk_level = "low"
            recommendation = "PROCEED"
        
        latency = (time.time() - start_time) * 1000
        
        return ThrashPrediction(
            thrash_probability=thrash_prob,
            quality_score=quality_score,
            confidence_interval=confidence_interval,
            risk_level=risk_level,
            recommendation=recommendation,
            latency_ms=latency,
            factors={
                "neural_available": HAS_TORCH and self.model is not None,
                "ci_width": confidence_interval[1] - confidence_interval[0]
            }
        )
    
    def predict_from_entropy(self, entropy: Dict[str, float]) -> ThrashPrediction:
        """Predict from entropy dict directly."""
        keys = ["h_info", "h_miss", "h_conj", "h_alea",
                "h_epis", "h_struct", "c_load", "h_goal_drift"]
        features = [entropy.get(k, 0.5) for k in keys]
        # Pad to expected dimension
        features.extend([0.5] * (23 - len(features)))
        return self.predict(features)
