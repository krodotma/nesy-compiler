#!/usr/bin/env python3
"""
bayesian.py - Bayesian Neural Gate with Uncertainty Quantification

P2-049: Implement Bayesian neural gate
P2-050: Add uncertainty quantification

Implements:
- Bayesian neural network for gates
- Epistemic uncertainty (model uncertainty)
- Aleatoric uncertainty (data uncertainty)
- Monte Carlo dropout for inference
"""

import time
import math
import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any, Tuple

logger = logging.getLogger("ARK.Neural.Bayesian")

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False


@dataclass
class UncertaintyEstimate:
    """Uncertainty quantification result."""
    mean: float                    # Mean prediction
    std: float                     # Standard deviation
    epistemic: float               # Model uncertainty
    aleatoric: float               # Data uncertainty
    total_uncertainty: float       # Combined uncertainty
    confidence: float              # 1 - uncertainty
    samples: List[float] = field(default_factory=list)
    
    def to_dict(self) -> Dict:
        return {
            "mean": self.mean,
            "std": self.std,
            "epistemic": self.epistemic,
            "aleatoric": self.aleatoric,
            "total_uncertainty": self.total_uncertainty,
            "confidence": self.confidence,
            "n_samples": len(self.samples)
        }


if HAS_TORCH:
    class BayesianLinear(nn.Module):
        """
        Bayesian linear layer with weight uncertainty.
        
        Uses variational inference with reparameterization trick.
        """
        
        def __init__(self, in_features: int, out_features: int):
            super().__init__()
            self.in_features = in_features
            self.out_features = out_features
            
            # Weight mean and log variance
            self.weight_mu = nn.Parameter(
                torch.Tensor(out_features, in_features).normal_(0, 0.1)
            )
            self.weight_log_var = nn.Parameter(
                torch.Tensor(out_features, in_features).fill_(-5)
            )
            
            # Bias mean and log variance
            self.bias_mu = nn.Parameter(torch.zeros(out_features))
            self.bias_log_var = nn.Parameter(torch.full((out_features,), -5.0))
            
            # Prior
            self.prior_mu = 0.0
            self.prior_log_var = 0.0
        
        def forward(self, x: torch.Tensor) -> torch.Tensor:
            if self.training:
                # Sample weights using reparameterization
                weight_std = torch.exp(0.5 * self.weight_log_var)
                weight = self.weight_mu + weight_std * torch.randn_like(weight_std)
                
                bias_std = torch.exp(0.5 * self.bias_log_var)
                bias = self.bias_mu + bias_std * torch.randn_like(bias_std)
            else:
                # Use mean for inference (MAP estimate)
                weight = self.weight_mu
                bias = self.bias_mu
            
            return F.linear(x, weight, bias)
        
        def kl_divergence(self) -> torch.Tensor:
            """KL divergence from prior for regularization."""
            kl_weight = self._kl_normal(
                self.weight_mu, self.weight_log_var,
                self.prior_mu, self.prior_log_var
            )
            kl_bias = self._kl_normal(
                self.bias_mu, self.bias_log_var,
                self.prior_mu, self.prior_log_var
            )
            return kl_weight + kl_bias
        
        def _kl_normal(
            self, 
            mu_q: torch.Tensor, 
            log_var_q: torch.Tensor,
            mu_p: float,
            log_var_p: float
        ) -> torch.Tensor:
            """KL divergence between two normal distributions."""
            var_q = torch.exp(log_var_q)
            var_p = math.exp(log_var_p)
            
            kl = 0.5 * (
                log_var_p - log_var_q + 
                (var_q + (mu_q - mu_p)**2) / var_p - 1
            )
            return kl.sum()
    
    
    class BayesianGateModel(nn.Module):
        """
        Bayesian neural gate with uncertainty quantification.
        
        P2-049: Implement Bayesian neural gate
        """
        
        def __init__(self, input_dim: int = 8, hidden_dim: int = 32):
            super().__init__()
            
            self.layer1 = BayesianLinear(input_dim, hidden_dim)
            self.layer2 = BayesianLinear(hidden_dim, hidden_dim)
            self.output = BayesianLinear(hidden_dim, 1)
            
            # Aleatoric uncertainty (learned)
            self.log_var_output = BayesianLinear(hidden_dim, 1)
        
        def forward(
            self, 
            x: torch.Tensor
        ) -> Tuple[torch.Tensor, torch.Tensor]:
            """
            Forward pass returning mean and variance.
            
            Returns:
                (mean, log_variance) tuple
            """
            h = F.relu(self.layer1(x))
            h = F.relu(self.layer2(h))
            
            mean = torch.sigmoid(self.output(h))
            log_var = self.log_var_output(h)
            
            return mean, log_var
        
        def kl_divergence(self) -> torch.Tensor:
            """Total KL divergence for ELBO loss."""
            return (
                self.layer1.kl_divergence() +
                self.layer2.kl_divergence() +
                self.output.kl_divergence() +
                self.log_var_output.kl_divergence()
            )


class BayesianNeuralGate:
    """
    High-level Bayesian neural gate interface.
    
    P2-049: Bayesian neural gate
    P2-050: Uncertainty quantification
    """
    
    def __init__(
        self,
        input_dim: int = 8,
        hidden_dim: int = 32,
        n_samples: int = 30,
        model_path: Optional[str] = None
    ):
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.n_samples = n_samples
        self.model = None
        
        if HAS_TORCH:
            self.model = BayesianGateModel(input_dim, hidden_dim)
            if model_path:
                self._load(model_path)
    
    def _load(self, path: str) -> None:
        """Load model weights."""
        try:
            state_dict = torch.load(path, map_location="cpu")
            self.model.load_state_dict(state_dict)
            self.model.eval()
        except Exception as e:
            logger.warning("Failed to load Bayesian model: %s", e)
    
    def predict(self, entropy: Dict[str, float]) -> UncertaintyEstimate:
        """
        Predict with full uncertainty quantification.
        
        Uses MC dropout for epistemic uncertainty.
        """
        if not HAS_TORCH or not self.model:
            return self._fallback_predict(entropy)
        
        # Convert to tensor
        keys = ["h_info", "h_miss", "h_conj", "h_alea",
                "h_epis", "h_struct", "c_load", "h_goal_drift"]
        values = [entropy.get(k, 0.5) for k in keys]
        x = torch.tensor([values], dtype=torch.float32)
        
        return self._mc_predict(x)
    
    def _mc_predict(self, x: 'torch.Tensor') -> UncertaintyEstimate:
        """
        Monte Carlo prediction for uncertainty.
        
        P2-050: Add uncertainty quantification
        """
        self.model.train()  # Enable dropout for MC
        
        means = []
        variances = []
        
        for _ in range(self.n_samples):
            with torch.no_grad():
                mean, log_var = self.model(x)
                means.append(mean.item())
                variances.append(torch.exp(log_var).item())
        
        self.model.eval()
        
        # Compute statistics
        mean_pred = sum(means) / len(means)
        
        # Epistemic uncertainty: variance of means
        epistemic = sum((m - mean_pred)**2 for m in means) / len(means)
        
        # Aleatoric uncertainty: mean of variances
        aleatoric = sum(variances) / len(variances)
        
        # Total uncertainty
        total = math.sqrt(epistemic + aleatoric)
        
        # Standard deviation
        std = math.sqrt(sum((m - mean_pred)**2 for m in means) / len(means))
        
        # Confidence: inverse of uncertainty
        confidence = max(0.0, 1.0 - total)
        
        return UncertaintyEstimate(
            mean=mean_pred,
            std=std,
            epistemic=epistemic,
            aleatoric=aleatoric,
            total_uncertainty=total,
            confidence=confidence,
            samples=means
        )
    
    def _fallback_predict(self, entropy: Dict[str, float]) -> UncertaintyEstimate:
        """Heuristic fallback."""
        h_total = sum(entropy.values()) / max(len(entropy), 1)
        
        # Simple estimate
        mean = 1.0 - h_total
        std = 0.2
        epistemic = 0.1
        aleatoric = 0.1
        
        return UncertaintyEstimate(
            mean=mean,
            std=std,
            epistemic=epistemic,
            aleatoric=aleatoric,
            total_uncertainty=math.sqrt(epistemic + aleatoric),
            confidence=0.7,
            samples=[mean]
        )
    
    def get_gate_decision(
        self, 
        entropy: Dict[str, float],
        threshold: float = 0.5,
        uncertainty_threshold: float = 0.3
    ) -> Tuple[str, UncertaintyEstimate]:
        """
        Get gate decision with uncertainty-aware thresholding.
        
        Returns:
            (decision, uncertainty_estimate)
        """
        estimate = self.predict(entropy)
        
        # If too uncertain, recommend review
        if estimate.total_uncertainty > uncertainty_threshold:
            decision = "REVIEW"
        elif estimate.mean > threshold:
            decision = "ACCEPT"
        else:
            decision = "REJECT"
        
        return decision, estimate
