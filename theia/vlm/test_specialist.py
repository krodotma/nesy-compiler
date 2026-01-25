"""
Tests for Theia VLM Specialist.

Verifies:
- Boot sequence
- Perception cycle (mocked capture)
- Action execution (automata + synthesis)
- Status reporting
"""

import unittest
import numpy as np
from unittest.mock import MagicMock, patch
from theia.vlm.specialist import VLMSpecialist, TheiaConfig

class TestVLMSpecialist(unittest.TestCase):
    
    def setUp(self):
        self.config = TheiaConfig(
            capture_index=0,
            memory_capacity=10,
            vps_mode=True
        )
        self.specialist = VLMSpecialist(self.config)
        
        # Mock capture to prevent actual screen access during tests
        self.specialist.capture = MagicMock()
        self.specialist.capture.capture_frame.return_value = np.zeros((100, 100, 3), dtype=np.uint8)

    def test_boot(self):
        """Test boot sequence."""
        status = self.specialist.boot()
        self.assertTrue(self.specialist.active)
        self.assertEqual(status["status"], "active")
        self.assertEqual(status["layers"], 10)
        self.assertIn("boot_time", status)

    def test_perception_cycle(self):
        """Test full perception cycle."""
        self.specialist.boot()
        
        perception = self.specialist.perceive()
        
        self.assertTrue(perception["has_input"])
        self.assertIn("embedding_norm", perception)
        self.assertIn("prescience", perception)
        self.assertIn("metacog", perception)
        
        # Verify prescience structure
        prescience = perception["prescience"]
        self.assertIn("scales", prescience)
        self.assertIn("global_coherence", prescience)

    def test_action_execution(self):
        """Test action generation."""
        self.specialist.boot()
        
        result = self.specialist.act("test_intention")
        
        self.assertEqual(result["action"], "test_intention")
        self.assertIn("program_id", result)
        self.assertIn("automaton_state", result)
        self.assertIn("learning_applied", result)
        
        # Verify automaton advanced
        # Initial state is usually 0, hashing intention should move it
        self.assertIsNotNone(result["automaton_state"])

    def test_status_report(self):
        """Test status reporting."""
        self.specialist.boot()
        report = self.specialist.status_report()
        
        self.assertTrue(report["active"])
        self.assertGreater(report["uptime"], 0)
        self.assertEqual(report["vps_mode"], True)
        self.assertIn("automata_state", report)

    def tearDown(self):
        self.specialist.shutdown()

if __name__ == "__main__":
    unittest.main()
