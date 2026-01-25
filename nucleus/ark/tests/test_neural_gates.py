import unittest
import sys
from unittest.mock import MagicMock, patch
from nucleus.ark.neural.model import NeuralGate, HAS_TORCH
from nucleus.ark.neural.thrash_predictor import ThrashPredictor

class TestNeuralGates(unittest.TestCase):
    def test_neural_gate_prediction_fallback(self):
        """Test neural gate prediction logic (fallback or mocked)."""
        entropy = {"h_struct": 0.5, "h_total": 0.5, "h_goal_drift": 0.1}
        
        gate = NeuralGate()
        
        # This should work regardless of torch availability
        # If torch is missing, it uses _fallback_predict logic
        preds = gate.predict(entropy, use_cache=False)
        
        self.assertIsInstance(preds, dict)
        self.assertIn("inertia", preds)
        self.assertIn("entelecheia", preds)
        
        # Verify specific fallback behavior if no torch
        if not HAS_TORCH:
            # Fallback uses 1.0 - h_struct for inertia
            self.assertAlmostEqual(preds["inertia"].probability, 0.5)

    def test_thrash_predictor(self):
        """Test thrash predictor logic."""
        tp = ThrashPredictor()
        
        # Test Low Risk
        safe_entropy = {"h_churn": 0.1, "h_debt": 0.1}
        res_safe = tp.predict_from_entropy(safe_entropy)
        self.assertIn(res_safe.risk_level, ["low", "medium"])
        
        # Test High Risk
        danger_entropy = {"h_churn": 0.9, "h_debt": 0.9, "h_struct": 0.9}
        res_danger = tp.predict_from_entropy(danger_entropy)
        self.assertIn(res_danger.risk_level, ["high", "critical"])

if __name__ == '__main__':
    unittest.main()
