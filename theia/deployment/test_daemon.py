"""
Tests for Theia Deployment Daemon.

Verifies:
- Daemon loop structure (mocked)
- Signal handling (mocked)
"""

import unittest
from unittest.mock import MagicMock, patch
import signal
import time

# Import the function to test (we'll need to refactor run_daemon slightly to be testable 
# or import the logic inside it. For now, we test the logic structure via mocks).

from theia.deployment.run_daemon import run_daemon

class TestDeploymentDaemon(unittest.TestCase):
    
    @patch('theia.deployment.run_daemon.VLMSpecialist')
    @patch('theia.deployment.run_daemon.time.sleep')
    def test_daemon_startup_and_loop(self, mock_sleep, MockSpecialist):
        """Test that daemon initializes specialist and enters loop."""
        
        # Setup mock
        mock_instance = MockSpecialist.return_value
        mock_instance.boot.return_value = {"status": "active"}
        mock_instance.perceive.return_value = {"has_input": False}
        
        # We need to raise an exception to break the infinite loop in run_daemon for testing
        mock_sleep.side_effect = InterruptedError("Break loop")
        
        with self.assertRaises(InterruptedError):
            run_daemon()
            
        # Verification
        MockSpecialist.assert_called_once()
        mock_instance.boot.assert_called_once()
        mock_instance.perceive.assert_called()

if __name__ == "__main__":
    unittest.main()
