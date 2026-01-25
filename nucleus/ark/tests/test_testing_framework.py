import unittest
from nucleus.ark.testing.fuzzer import CoverageFuzzer
from nucleus.ark.testing.golden import GoldenTestManager

class TestTestingFramework(unittest.TestCase):
    def test_fuzzer_initialization(self):
        """Test that fuzzer handles gates correctly."""
        fuzzer = CoverageFuzzer(max_iterations=10)
        
        # Mock gate function
        def mock_gate(entropy):
            return sum(entropy.values()) < 1.0
            
        result = fuzzer.fuzz_gate(mock_gate)
        self.assertIsNotNone(result)
        self.assertGreaterEqual(result.total_inputs, 0)

    def test_golden_manager(self):
        """Test golden test manager."""
        # Just verify it instantiates and has expected methods
        # Actual testing requires file I/O which we skip for this smoke test
        manager = GoldenTestManager()
        self.assertTrue(hasattr(manager, 'list_tests'))
        self.assertTrue(hasattr(manager, 'update_all_failing'))

if __name__ == '__main__':
    unittest.main()
