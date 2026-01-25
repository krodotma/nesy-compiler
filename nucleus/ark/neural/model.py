#!/usr/bin/env python3
"""
model.py - Neural Gate Model Architecture

P2-041: Upgrade NeuralAdapter to use torch
P2-042: Create learned gate model architecture

Implements a hybrid neural gate model:
- Feature encoder (MLP for H* entropy vector)
- Code understanding layer (simplified CodeBERT-style)
- Gate heads (Inertia, Entelecheia, Homeostasis)
"""

import time
import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any, Tuple
from pathlib import Path

logger = logging.getLogger("ARK.Neural.Model")

# Check for torch availability
try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False
    logger.info("PyTorch not available - using fallback mock models")


@dataclass
class NeuralGateConfig:
    """Configuration for neural gate model."""
    hidden_dim: int = 128
    num_layers: int = 2
    dropout: float = 0.1
    entropy_dim: int = 8  # H* vector dimension
    code_dim: int = 256   # Code embedding dimension
    num_gates: int = 3    # Inertia, Entelecheia, Homeostasis
    use_attention: bool = True
    calibrate: bool = True
    device: str = "cpu"
    
    def to_dict(self) -> Dict:
        return {
            "hidden_dim": self.hidden_dim,
            "num_layers": self.num_layers,
            "dropout": self.dropout,
            "entropy_dim": self.entropy_dim,
            "code_dim": self.code_dim,
            "num_gates": self.num_gates
        }


@dataclass
class GatePrediction:
    """Result of neural gate prediction."""
    gate_name: str
    probability: float  # 0-1 probability of passing
    decision: str       # ACCEPT, REJECT, REVIEW
    confidence: float   # Model confidence
    latency_ms: float = 0.0
    features_used: List[str] = field(default_factory=list)
    explanation: str = ""


if HAS_TORCH:
    class EntropyEncoder(nn.Module):
        """Encode H* entropy vector."""
        
        def __init__(self, config: NeuralGateConfig):
            super().__init__()
            self.config = config
            
            self.encoder = nn.Sequential(
                nn.Linear(config.entropy_dim, config.hidden_dim),
                nn.LayerNorm(config.hidden_dim),
                nn.GELU(),
                nn.Dropout(config.dropout),
                nn.Linear(config.hidden_dim, config.hidden_dim),
                nn.GELU()
            )
        
        def forward(self, entropy_vec: torch.Tensor) -> torch.Tensor:
            return self.encoder(entropy_vec)
    
    
    class CodeEncoder(nn.Module):
        """Encode code/diff content (simplified)."""
        
        def __init__(self, config: NeuralGateConfig):
            super().__init__()
            self.config = config
            
            # Simple embedding + attention
            self.embedding = nn.Embedding(30000, 64)  # Vocab ~ 30k tokens
            self.attention = nn.MultiheadAttention(64, num_heads=4, batch_first=True)
            self.projection = nn.Linear(64, config.hidden_dim)
        
        def forward(self, token_ids: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
            embedded = self.embedding(token_ids)
            attended, _ = self.attention(embedded, embedded, embedded, key_padding_mask=mask)
            pooled = attended.mean(dim=1)  # Average pooling
            return self.projection(pooled)
    
    
    class HistoryEncoder(nn.Module):
        """
        Encode patch history for thrash detection.
        
        Claude Opus Critic R&D insight: "A Transformer can detect 
        cyclical churn. Include History Vector: Embedding of last 
        5 patches to this file (To detect loops)."
        """
        
        def __init__(self, config: NeuralGateConfig, history_len: int = 5):
            super().__init__()
            self.config = config
            self.history_len = history_len
            
            # LSTM to capture temporal patterns in patch history
            self.lstm = nn.LSTM(
                input_size=config.entropy_dim,
                hidden_size=config.hidden_dim // 2,
                num_layers=1,
                batch_first=True,
                bidirectional=True
            )
            
            # Attention over history to find recurring patterns
            self.history_attention = nn.MultiheadAttention(
                config.hidden_dim, num_heads=2, batch_first=True
            )
            
            # Cycle detection head
            self.cycle_detector = nn.Sequential(
                nn.Linear(config.hidden_dim, config.hidden_dim // 2),
                nn.ReLU(),
                nn.Linear(config.hidden_dim // 2, 1),
                nn.Sigmoid()
            )
        
        def forward(self, history: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
            """
            Args:
                history: (batch, history_len, entropy_dim) - past H* vectors
            
            Returns:
                features: (batch, hidden_dim) - history embedding
                cycle_prob: (batch, 1) - probability of cyclical pattern
            """
            # LSTM encoding
            lstm_out, _ = self.lstm(history)  # (batch, seq, hidden)
            
            # Self-attention to find recurring patterns
            attended, attn_weights = self.history_attention(
                lstm_out, lstm_out, lstm_out
            )
            
            # Pool to single vector
            features = attended.mean(dim=1)
            
            # Detect cycles
            cycle_prob = self.cycle_detector(features)
            
            return features, cycle_prob
    
    
    class GateHead(nn.Module):
        """Decision head for a specific gate."""
        
        def __init__(self, config: NeuralGateConfig, gate_name: str):
            super().__init__()
            self.gate_name = gate_name
            self.config = config
            
            self.head = nn.Sequential(
                nn.Linear(config.hidden_dim * 2, config.hidden_dim),
                nn.LayerNorm(config.hidden_dim),
                nn.GELU(),
                nn.Dropout(config.dropout),
                nn.Linear(config.hidden_dim, 1),
                nn.Sigmoid()
            )
        
        def forward(self, features: torch.Tensor) -> torch.Tensor:
            return self.head(features)
    
    
    class NeuralGateModel(nn.Module):
        """
        Full neural gate model with multiple gate heads.
        
        P2-042: Create learned gate model architecture
        """
        
        def __init__(self, config: NeuralGateConfig):
            super().__init__()
            self.config = config
            
            self.entropy_encoder = EntropyEncoder(config)
            self.code_encoder = CodeEncoder(config)
            
            # Gate heads
            self.gate_heads = nn.ModuleDict({
                "inertia": GateHead(config, "inertia"),
                "entelecheia": GateHead(config, "entelecheia"),
                "homeostasis": GateHead(config, "homeostasis")
            })
            
            # Temperature for calibration
            self.temperature = nn.Parameter(torch.ones(1))
        
        def forward(
            self,
            entropy_vec: torch.Tensor,
            token_ids: Optional[torch.Tensor] = None,
            mask: Optional[torch.Tensor] = None
        ) -> Dict[str, torch.Tensor]:
            # Encode entropy
            entropy_features = self.entropy_encoder(entropy_vec)
            
            # Encode code if provided
            if token_ids is not None:
                code_features = self.code_encoder(token_ids, mask)
            else:
                code_features = torch.zeros_like(entropy_features)
            
            # Concatenate features
            combined = torch.cat([entropy_features, code_features], dim=-1)
            
            # Get predictions for each gate
            predictions = {}
            for gate_name, head in self.gate_heads.items():
                logits = head(combined)
                # Apply temperature scaling for calibration
                calibrated = logits / self.temperature
                predictions[gate_name] = calibrated
            
            return predictions
        
        def predict(self, entropy_vec: torch.Tensor, **kwargs) -> Dict[str, float]:
            """Get gate probabilities."""
            self.eval()
            with torch.no_grad():
                outputs = self.forward(entropy_vec, **kwargs)
                return {k: v.item() for k, v in outputs.items()}


class NeuralGate:
    """
    High-level neural gate interface.
    
    Wraps the PyTorch model with convenience methods.
    """
    
    def __init__(
        self,
        config: Optional[NeuralGateConfig] = None,
        model_path: Optional[str] = None
    ):
        self.config = config or NeuralGateConfig()
        self.model_path = Path(model_path) if model_path else None
        self.model = None
        self._cache: Dict[str, GatePrediction] = {}
        self._cache_ttl = 60  # seconds
        
        self._init_model()
    
    def _init_model(self) -> None:
        """Initialize or load model."""
        if HAS_TORCH:
            self.model = NeuralGateModel(self.config)
            
            if self.model_path and self.model_path.exists():
                try:
                    state_dict = torch.load(self.model_path, map_location="cpu")
                    self.model.load_state_dict(state_dict)
                    logger.info("Loaded neural gate model from %s", self.model_path)
                except Exception as e:
                    logger.warning("Failed to load model: %s", e)
        else:
            self.model = None
    
    def predict(
        self,
        entropy: Dict[str, float],
        code_content: str = "",
        use_cache: bool = True
    ) -> Dict[str, GatePrediction]:
        """
        Predict gate decisions from entropy and code.
        
        Returns predictions for all gates.
        """
        start_time = time.time()
        
        # Check cache
        cache_key = f"{hash(tuple(sorted(entropy.items())))}"
        if use_cache and cache_key in self._cache:
            cached = self._cache[cache_key]
            if (time.time() - cached.latency_ms / 1000) < self._cache_ttl:
                return {cached.gate_name: cached}
        
        predictions = {}
        
        if HAS_TORCH and self.model:
            # Convert entropy to tensor
            entropy_vec = self._entropy_to_tensor(entropy)
            
            # Get model predictions
            with torch.no_grad():
                outputs = self.model.forward(entropy_vec.unsqueeze(0))
            
            for gate_name, prob_tensor in outputs.items():
                prob = prob_tensor.item()
                predictions[gate_name] = self._make_prediction(
                    gate_name, prob, start_time
                )
        else:
            # Fallback: heuristic predictions
            predictions = self._fallback_predict(entropy, start_time)
        
        # Cache results
        for pred in predictions.values():
            self._cache[cache_key] = pred
        
        return predictions
    
    def _entropy_to_tensor(self, entropy: Dict[str, float]) -> 'torch.Tensor':
        """Convert entropy dict to tensor."""
        keys = ["h_info", "h_miss", "h_conj", "h_alea", 
                "h_epis", "h_struct", "c_load", "h_goal_drift"]
        values = [entropy.get(k, 0.5) for k in keys]
        return torch.tensor(values, dtype=torch.float32)
    
    def _make_prediction(
        self, 
        gate_name: str, 
        probability: float, 
        start_time: float
    ) -> GatePrediction:
        """Create GatePrediction from probability."""
        if probability > 0.8:
            decision = "ACCEPT"
            confidence = probability
        elif probability > 0.5:
            decision = "REVIEW"
            confidence = 0.5
        else:
            decision = "REJECT"
            confidence = 1 - probability
        
        return GatePrediction(
            gate_name=gate_name,
            probability=probability,
            decision=decision,
            confidence=confidence,
            latency_ms=(time.time() - start_time) * 1000,
            features_used=["entropy_vector"],
            explanation=f"Neural gate {gate_name}: {probability:.2%} pass probability"
        )
    
    def _fallback_predict(
        self, 
        entropy: Dict[str, float],
        start_time: float
    ) -> Dict[str, GatePrediction]:
        """Heuristic fallback when no model available."""
        h_total = sum(entropy.values()) / max(len(entropy), 1)
        
        predictions = {}
        
        # Inertia: based on structural entropy
        h_struct = entropy.get("h_struct", 0.5)
        inertia_prob = 1.0 - h_struct
        predictions["inertia"] = self._make_prediction("inertia", inertia_prob, start_time)
        
        # Entelecheia: based on goal drift
        h_goal = entropy.get("h_goal_drift", 0.5)
        ente_prob = 1.0 - h_goal
        predictions["entelecheia"] = self._make_prediction("entelecheia", ente_prob, start_time)
        
        # Homeostasis: based on total entropy
        homeo_prob = 1.0 - h_total
        predictions["homeostasis"] = self._make_prediction("homeostasis", homeo_prob, start_time)
        
        return predictions
    
    def save(self, path: Optional[str] = None) -> bool:
        """Save model to file."""
        if not HAS_TORCH or not self.model:
            return False
        
        save_path = Path(path) if path else self.model_path
        if not save_path:
            return False
        
        try:
            save_path.parent.mkdir(parents=True, exist_ok=True)
            torch.save(self.model.state_dict(), save_path)
            logger.info("Saved neural gate model to %s", save_path)
            return True
        except Exception as e:
            logger.error("Failed to save model: %s", e)
            return False
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get model statistics."""
        return {
            "torch_available": HAS_TORCH,
            "model_loaded": self.model is not None,
            "config": self.config.to_dict(),
            "cache_size": len(self._cache),
            "model_path": str(self.model_path) if self.model_path else None
        }
