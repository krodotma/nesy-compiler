"""
ARK Neural Gate Enhancement

Phase 2.3 Implementation (P2-041 to P2-060)

Provides:
- PyTorch-based neural gate models
- Feature extraction pipeline (AST, embeddings)
- Thrash probability prediction
- Bayesian uncertainty quantification
- Calibrated confidence intervals
"""

from .model import NeuralGate, NeuralGateConfig, GatePrediction
from .features import FeatureExtractor, CodeFeatures, CommitFeatures
from .thrash_predictor import ThrashPredictor, ThrashPrediction
from .bayesian import BayesianNeuralGate, UncertaintyEstimate
