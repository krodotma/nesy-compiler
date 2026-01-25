import unittest
from nucleus.ark.gates.entelecheia import EntelecheiaGate, EntelecheiaContext
from nucleus.ark.specs.ltl_validator import LTLValidator, SpecLoader, LTLSpec


class TestEntelecheiaGate(unittest.TestCase):
    def test_basic_purpose_check(self):
        """Test basic purpose validation."""
        gate = EntelecheiaGate()
        
        # With purpose
        ctx = EntelecheiaContext(purpose="Add user authentication")
        self.assertTrue(gate.check(ctx))
        
        # Without purpose
        ctx_empty = EntelecheiaContext(purpose="")
        self.assertFalse(gate.check(ctx_empty))
    
    def test_cosmetic_rejection(self):
        """Test cosmetic changes are rejected without spec."""
        gate = EntelecheiaGate()
        
        ctx = EntelecheiaContext(purpose="formatting fix", is_cosmetic=True)
        self.assertFalse(gate.check(ctx))
    
    def test_liveness_bypass(self):
        """Test liveness gain bypasses other checks."""
        gate = EntelecheiaGate()
        
        ctx = EntelecheiaContext(purpose="", liveness_gain=0.5)
        self.assertTrue(gate.check(ctx))


class TestLTLValidator(unittest.TestCase):
    def test_forbidden_pattern(self):
        """Test forbidden pattern detection."""
        spec = LTLSpec(
            name="test",
            formula="",
            invariants=[],
            allowed_patterns=[],
            forbidden_patterns=["os\\.system\\("],
        )
        
        # Mock loader
        class MockLoader:
            def load(self, ref):
                return spec
        
        validator = LTLValidator(loader=MockLoader())
        
        # Diff with forbidden pattern
        diff = "+    os.system('rm -rf /')"
        passed, reason = validator.validate("test", diff, "clean up")
        self.assertFalse(passed)
        self.assertIn("Forbidden pattern", reason)
    
    def test_invariant_no_delete(self):
        """Test NoDelete invariant."""
        spec = LTLSpec(
            name="test",
            formula="",
            invariants=["NoDelete<test_>"],
            allowed_patterns=[],
            forbidden_patterns=[],
        )
        
        class MockLoader:
            def load(self, ref):
                return spec
        
        validator = LTLValidator(loader=MockLoader())
        
        # Diff that deletes a test
        diff = "-def test_something():\n-    pass"
        passed, reason = validator.validate("test", diff, "refactor")
        self.assertFalse(passed)
        self.assertIn("Invariant violated", reason)


if __name__ == '__main__':
    unittest.main()
