#!/usr/bin/env python3
# test_distillation.py - Comprehensive Test Suite for Distillation System

import unittest
import tempfile
import shutil
from pathlib import Path
import sys

# Add project root to path
sys.path.insert(0, str(Path(__file__).parents[3]))

from nucleus.tools.distillation.neural_adapter import NeuralAdapter
from nucleus.tools.distillation.triplet_dna import InertiaGate, EntelecheiaGate, HomeostasisGate, DNAContext
from nucleus.tools.distillation.distill_engine import DistillationEngine


class TestNeuralAdapter(unittest.TestCase):
    """Test Neural Gate functionality."""
    
    def setUp(self):
        self.adapter = NeuralAdapter()
    
    def test_simple_code_low_complexity(self):
        """Test that simple code has low complexity."""
        code = """
def hello():
    return "world"
"""
        features = self.adapter.extract_features_from_code(code)
        self.assertLess(features[0], 0.3)  # Low complexity
        self.assertLess(features[1], 0.3)  # Low depth
        self.assertEqual(features[4], 0.0)  # No anti-patterns
        
        thrash_prob = self.adapter.predict_thrash(features)
        self.assertLess(thrash_prob, 0.5)
    
    def test_complex_code_high_complexity(self):
        """Test that complex code is detected."""
        code = """
def complex_function(x, y, z):
    if x > 0:
        if y > 0:
            if z > 0:
                for i in range(10):
                    while i < 5:
                        if i % 2 == 0:
                            return True
    return False
"""
        features = self.adapter.extract_features_from_code(code)
        self.assertGreater(features[0], 0.3)  # Higher complexity
        self.assertGreater(features[1], 0.3)  # Deeper nesting
        
        thrash_prob = self.adapter.predict_thrash(features)
        self.assertGreater(thrash_prob, 0.3)
    
    def test_anti_pattern_detection(self):
        """Test that anti-patterns are detected."""
        code = """
class AbstractFactoryFactory:
    pass

class UserManager:
    pass
"""
        features = self.adapter.extract_features_from_code(code)
        self.assertGreater(features[4], 0.5)  # Anti-patterns detected
        
        thrash_prob = self.adapter.predict_thrash(features)
        self.assertGreater(thrash_prob, 0.8)  # High thrash probability
    
    def test_unparseable_code(self):
        """Test that unparseable code returns max thrash."""
        code = "def broken syntax here"
        features = self.adapter.extract_features_from_code(code)
        self.assertEqual(features[0], 1.0)  # Max complexity for broken code
        
        thrash_prob = self.adapter.predict_thrash(features)
        self.assertGreater(thrash_prob, 0.8)


class TestTripletDNA(unittest.TestCase):
    """Test DNA Gate functionality."""
    
    def test_inertia_gate_isolation(self):
        """Test InertiaGate rejects isolated code."""
        gate = InertiaGate()
        
        # Low isolation - should pass
        context = DNAContext(
            source_node="test.py",
            target_graph=None,
            system_entropy={'isolation_score': 0.2}
        )
        self.assertTrue(gate.check(context))
        
        # High isolation - should fail
        context.system_entropy['isolation_score'] = 0.9
        self.assertFalse(gate.check(context))
    
    def test_homeostasis_gate_entropy(self):
        """Test HomeostasisGate rejects high system entropy."""
        gate = HomeostasisGate()
        
        # Low entropy - should pass
        context = DNAContext(
            source_node="test.py",
            target_graph=None,
            system_entropy={'h_total': 0.3}
        )
        self.assertTrue(gate.check(context))
        
        # High entropy - should fail
        context.system_entropy['h_total'] = 0.9
        self.assertFalse(gate.check(context))


class TestDistillationEngine(unittest.TestCase):
    """Test end-to-end distillation workflow."""
    
    def setUp(self):
        """Create temporary directories for testing."""
        self.test_dir = tempfile.mkdtemp()
        self.source_dir = Path(self.test_dir) / "source"
        self.target_dir = Path(self.test_dir) / "target"
        self.source_dir.mkdir()
        self.target_dir.mkdir()
    
    def tearDown(self):
        """Clean up temporary directories."""
        shutil.rmtree(self.test_dir)
    
    def test_accept_good_code(self):
        """Test that good code is accepted and copied."""
        # Create a simple, clean Python file
        good_file = self.source_dir / "good_module.py"
        good_file.write_text("""
def calculate_sum(a, b):
    return a + b

def calculate_product(a, b):
    return a * b
""")
        
        engine = DistillationEngine(str(self.target_dir))
        engine.process_repo(str(self.source_dir))
        
        # Check that file was copied
        target_file = self.target_dir / "good_module.py"
        self.assertTrue(target_file.exists())
    
    def test_reject_bad_code(self):
        """Test that bad code (anti-patterns) is rejected."""
        # Create a file with anti-patterns
        bad_file = self.source_dir / "bad_module.py"
        bad_file.write_text("""
class AbstractFactoryFactory:
    '''Over-engineered bloat'''
    pass

class UserManager:
    '''Generic Manager anti-pattern'''
    def do_everything(self):
        pass
""")
        
        engine = DistillationEngine(str(self.target_dir))
        engine.process_repo(str(self.source_dir))
        
        # Check that file was NOT copied
        target_file = self.target_dir / "bad_module.py"
        self.assertFalse(target_file.exists())
    
    def test_reject_complex_code(self):
        """Test that overly complex code is rejected."""
        complex_file = self.source_dir / "complex_module.py"
        
        # Generate very complex nested code
        code = "def mega_function():\n"
        for i in range(15):
            code += "    " * (i + 1) + f"if x{i} > 0:\n"
        code += "    " * 16 + "return True\n"
        
        complex_file.write_text(code)
        
        engine = DistillationEngine(str(self.target_dir))
        engine.process_repo(str(self.source_dir))
        
        # High complexity should be rejected
        target_file = self.target_dir / "complex_module.py"
        self.assertFalse(target_file.exists())


if __name__ == '__main__':
    unittest.main()
