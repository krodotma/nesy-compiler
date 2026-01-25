#!/usr/bin/env python3
"""
test_ark_operational.py - Operational tests for ARK

These are REAL tests that prove ARK works, not theoretical simulations.
Each test creates actual files, runs actual commands, and verifies actual results.
"""

import os
import sys
import shutil
import tempfile
import unittest
from pathlib import Path

# Add pluribus to path
sys.path.insert(0, '/Users/kroma/pluribus')

from nucleus.ark.core.repository import ArkRepository
from nucleus.ark.core.context import ArkCommitContext, Witness
from nucleus.ark.gates.inertia import InertiaGate, InertiaContext
from nucleus.ark.gates.entelecheia import EntelecheiaGate, EntelecheiaContext
from nucleus.ark.gates.homeostasis import HomeostasisGate, HomeostasisContext
from nucleus.ark.ribosome.gene import Gene
from nucleus.ark.ribosome.clade import Clade
from nucleus.ark.ribosome.genome import OrganismGenome
from nucleus.ark.rhizom.dag import RhizomDAG, RhizomNode
from nucleus.ark.rhizom.etymology import EtymologyExtractor
from nucleus.ark.rhizom.lineage import LineageTracker
from nucleus.ark.portal.ingest import IngestPipeline
from nucleus.ark.portal.distill import DistillationPipeline
from nucleus.ark.synthesis.ltl_spec import PluribusLTLSpec, LTLVerifier
from nucleus.ark.synthesis.grammar import GrammarFilter, SynthesisGrammar


class TestArkRepository(unittest.TestCase):
    """Test ArkRepository with actual file operations."""
    
    def setUp(self):
        """Create a temporary directory for each test."""
        self.test_dir = tempfile.mkdtemp(prefix="ark_test_")
        self.repo = ArkRepository(self.test_dir)
    
    def tearDown(self):
        """Clean up temporary directory."""
        shutil.rmtree(self.test_dir, ignore_errors=True)
    
    def test_init_creates_ark_directory(self):
        """PROOF: ark init creates .ark directory with rhizome.json"""
        result = self.repo.init()
        
        self.assertTrue(result, "init() should return True")
        self.assertTrue((Path(self.test_dir) / ".git").exists(), ".git should exist")
        self.assertTrue((Path(self.test_dir) / ".ark").exists(), ".ark should exist")
        self.assertTrue((Path(self.test_dir) / ".ark" / "rhizom.json").exists(), "rhizom.json should exist")
        print("✅ PROOF: ark init creates proper directory structure")
    
    def test_status_returns_dna_metrics(self):
        """PROOF: ark status shows entropy vector"""
        self.repo.init()
        status = self.repo.status()
        
        self.assertIn("initialized", status, "Status should have initialized flag")
        self.assertTrue(status["initialized"], "Should be initialized")
        self.assertIn("current_entropy", status, "Status should have entropy")
        self.assertIn("h_struct", status["current_entropy"], "Entropy should have h_struct")
        print(f"✅ PROOF: ark status returns entropy: {status['current_entropy']}")
    
    def test_commit_with_cell_cycle(self):
        """PROOF: ark commit runs through G1-S-G2-M phases"""
        self.repo.init()
        
        # Create a test file
        test_file = Path(self.test_dir) / "test_module.py"
        test_file.write_text('"""Test module."""\ndef hello(): pass\n')
        
        context = ArkCommitContext(
            etymology="Test commit for verification",
            purpose="Prove Cell Cycle works",
            cmp=0.7,
            entropy={"h_struct": 0.3, "h_doc": 0.2}  # Low entropy = should pass
        )
        
        sha = self.repo.commit("test: Verify Cell Cycle", context)
        
        self.assertIsNotNone(sha, "Commit should succeed")
        self.assertEqual(len(sha), 40, "SHA should be 40 chars")
        print(f"✅ PROOF: Cell Cycle commit succeeded: {sha[:8]}")
    
    def test_commit_rejected_high_entropy(self):
        """PROOF: High entropy commits are rejected by G1"""
        self.repo.init()
        
        test_file = Path(self.test_dir) / "chaotic.py"
        test_file.write_text("x=1\n")
        
        context = ArkCommitContext(
            purpose="This should fail",
            entropy={"h_struct": 0.9, "h_doc": 0.9, "h_test": 0.9}  # Very high entropy
        )
        
        sha = self.repo.commit("feat: Should be rejected", context)
        
        self.assertIsNone(sha, "High entropy commit should be rejected")
        print("✅ PROOF: High entropy commit correctly rejected by G1")


class TestDNAGates(unittest.TestCase):
    """Test DNA gates with real scenarios."""
    
    def test_inertia_blocks_core_without_witness(self):
        """PROOF: Inertia gate blocks changes to core files without witness"""
        gate = InertiaGate()
        
        # Try to modify world_router.py without witness
        context = InertiaContext(
            files=["nucleus/tools/world_router.py"],
            has_witness=False,
            has_formal_proof=False
        )
        
        result = gate.check(context)
        
        self.assertFalse(result, "Should block without witness")
        print("✅ PROOF: Inertia gate blocks core file modification without witness")
    
    def test_inertia_allows_core_with_witness(self):
        """PROOF: Inertia gate allows changes with witness"""
        gate = InertiaGate()
        
        context = InertiaContext(
            files=["nucleus/tools/world_router.py"],
            has_witness=True,
            has_formal_proof=False
        )
        
        result = gate.check(context)
        
        self.assertTrue(result, "Should allow with witness")
        print("✅ PROOF: Inertia gate allows core file modification with witness")
    
    def test_entelecheia_rejects_cosmetic_changes(self):
        """PROOF: Entelecheia gate rejects purposeless cosmetic changes"""
        gate = EntelecheiaGate()
        
        context = EntelecheiaContext(
            purpose="formatting fix",
            is_cosmetic=True,
            liveness_gain=0.0
        )
        
        result = gate.check(context)
        
        self.assertFalse(result, "Should reject cosmetic without spec")
        print("✅ PROOF: Entelecheia gate rejects cosmetic changes without liveness gain")
    
    def test_homeostasis_blocks_growth_during_instability(self):
        """PROOF: Homeostasis blocks new features when entropy is high"""
        gate = HomeostasisGate()
        
        context = HomeostasisContext(
            entropy={"h_struct": 0.9, "h_doc": 0.8, "h_test": 0.9},
            threshold=0.7,
            is_stabilization_commit=False
        )
        
        result = gate.check(context)
        
        self.assertFalse(result, "Should block growth during instability")
        print("✅ PROOF: Homeostasis gate blocks growth when entropy > 0.7")
    
    def test_homeostasis_allows_stabilization_during_instability(self):
        """PROOF: Homeostasis allows stabilization commits during crisis"""
        gate = HomeostasisGate()
        
        context = HomeostasisContext(
            entropy={"h_struct": 0.9, "h_doc": 0.8},
            threshold=0.7,
            is_stabilization_commit=True
        )
        
        result = gate.check(context)
        
        self.assertTrue(result, "Should allow stabilization")
        print("✅ PROOF: Homeostasis allows stabilization commits during instability")


class TestRhizomeDAG(unittest.TestCase):
    """Test Rhizome (semantic DAG) operations."""
    
    def setUp(self):
        self.test_dir = tempfile.mkdtemp(prefix="rhizome_test_")
        Path(self.test_dir).mkdir(parents=True, exist_ok=True)
    
    def tearDown(self):
        shutil.rmtree(self.test_dir, ignore_errors=True)
    
    def test_rhizome_stores_etymology(self):
        """PROOF: Rhizome stores and retrieves etymology"""
        rhizom = RhizomDAG(Path(self.test_dir))
        
        node = RhizomNode(
            sha="abc123",
            etymology="Implementing user authentication module",
            cmp=0.75,
            entropy={"h_struct": 0.3}
        )
        
        rhizom.insert(node)
        
        # Retrieve
        retrieved = rhizom.get("abc123")
        
        self.assertIsNotNone(retrieved)
        self.assertEqual(retrieved.etymology, "Implementing user authentication module")
        print(f"✅ PROOF: Rhizome stores etymology: '{retrieved.etymology}'")
    
    def test_rhizome_queries_by_etymology(self):
        """PROOF: Rhizome semantic search works"""
        rhizom = RhizomDAG(Path(self.test_dir))
        
        # Insert multiple nodes
        rhizom.insert(RhizomNode(sha="a1", etymology="Authentication module"))
        rhizom.insert(RhizomNode(sha="a2", etymology="Database connection"))
        rhizom.insert(RhizomNode(sha="a3", etymology="Auth token validation"))
        
        # Query for auth-related commits
        results = rhizom.query_by_etymology("auth")
        
        self.assertEqual(len(results), 2, "Should find 2 auth-related commits")
        print(f"✅ PROOF: Rhizome semantic search found {len(results)} 'auth' commits")
    
    def test_rhizome_tracks_cmp_trajectory(self):
        """PROOF: Rhizome tracks CMP scores over lineage"""
        rhizom = RhizomDAG(Path(self.test_dir))
        
        # Create a lineage
        rhizom.insert(RhizomNode(sha="c1", cmp=0.5, parents=[]))
        rhizom.insert(RhizomNode(sha="c2", cmp=0.6, parents=["c1"]))
        rhizom.insert(RhizomNode(sha="c3", cmp=0.75, parents=["c2"]))
        
        tracker = LineageTracker(rhizom)
        lineage = tracker.get_lineage("c3", max_depth=10)
        
        self.assertEqual(len(lineage.cmp_trajectory), 3)
        self.assertEqual(lineage.cmp_trajectory, [0.75, 0.6, 0.5])
        print(f"✅ PROOF: Rhizome tracks CMP trajectory: {lineage.cmp_trajectory}")


class TestEtymologyExtractor(unittest.TestCase):
    """Test etymology extraction from real code."""
    
    def test_extracts_from_docstring(self):
        """PROOF: EtymologyExtractor reads module docstring"""
        extractor = EtymologyExtractor()
        
        code = '''"""
        User authentication and session management.
        
        Handles OAuth2 flows and JWT token validation.
        """
        
        def authenticate(user, password):
            pass
        '''
        
        etymology = extractor.extract_from_code(code, "auth.py")
        
        self.assertIn("authentication", etymology.primary.lower())
        self.assertIn("auth", etymology.keywords)
        print(f"✅ PROOF: Etymology extracted: '{etymology.primary}'")
    
    def test_classifies_domain(self):
        """PROOF: EtymologyExtractor classifies code domain"""
        extractor = EtymologyExtractor()
        
        evolution_code = '''
        """Evolutionary fitness calculation."""
        from clade import Clade
        def calculate_cmp(gene):
            return gene.fitness * gene.mutations
        '''
        
        etymology = extractor.extract_from_code(evolution_code, "evolution.py")
        
        self.assertEqual(etymology.domain, "evolution")
        print(f"✅ PROOF: Domain classified as '{etymology.domain}'")


class TestDistillationPipeline(unittest.TestCase):
    """Test the full distillation pipeline."""
    
    def setUp(self):
        self.source_dir = tempfile.mkdtemp(prefix="entropic_")
        self.target_dir = tempfile.mkdtemp(prefix="negentropic_")
        
        # Create some test files
        good_file = Path(self.source_dir) / "good_module.py"
        good_file.write_text('''"""
        Well-documented utility module.
        
        Provides helper functions for data processing.
        """
        
        from typing import List
        
        def process_data(items: List[str]) -> List[str]:
            """Process and filter data items."""
            return [item.strip() for item in items if item]
        ''')
        
        bad_file = Path(self.source_dir) / "bad_module.py"
        bad_file.write_text('''# TODO: fix this mess
        # FIXME: ugly hack
        # HACK: temporary workaround
        x=1
        y=2
        def f():
            if True:
                if True:
                    if True:
                        if True:
                            if True:
                                if True:
                                    pass
        ''')
    
    def tearDown(self):
        shutil.rmtree(self.source_dir, ignore_errors=True)
        shutil.rmtree(self.target_dir, ignore_errors=True)
    
    def test_distillation_accepts_good_code(self):
        """PROOF: Distillation accepts well-structured code"""
        pipeline = DistillationPipeline(self.source_dir, self.target_dir)
        report = pipeline.run(purpose="Test distillation")
        
        self.assertEqual(report.status, "completed")
        self.assertGreater(report.ingest_report.accepted_files, 0)
        
        # Verify file was copied
        target_good = Path(self.target_dir) / "good_module.py"
        self.assertTrue(target_good.exists(), "Good file should be copied to target")
        print(f"✅ PROOF: Distillation accepted {report.ingest_report.accepted_files} files")
    
    def test_distillation_creates_genes(self):
        """PROOF: Distillation creates Gene objects with fitness"""
        pipeline = DistillationPipeline(self.source_dir, self.target_dir)
        report = pipeline.run()
        
        self.assertGreater(report.genes_created, 0)
        self.assertGreater(report.total_cmp, 0)
        print(f"✅ PROOF: Created {report.genes_created} genes, CMP={report.total_cmp:.2f}")


class TestLTLVerification(unittest.TestCase):
    """Test LTL specification verification."""
    
    def test_verifies_gate_compliance(self):
        """PROOF: LTL verifier checks gate compliance"""
        spec = PluribusLTLSpec.core_spec()
        verifier = LTLVerifier(spec)
        
        # Good trace - all gates pass
        good_trace = [
            {"commit": True, "inertia_pass": True, "entelecheia_pass": True, 
             "homeostasis_pass": True, "stable": True}
        ]
        
        result = verifier.verify_trace(good_trace)
        
        self.assertTrue(result)
        print("✅ PROOF: LTL verifier validates compliant trace")
    
    def test_detects_gate_violations(self):
        """PROOF: LTL verifier detects gate violations"""
        spec = PluribusLTLSpec.core_spec()
        verifier = LTLVerifier(spec)
        
        # Bad trace - inertia fails
        bad_trace = [
            {"commit": True, "inertia_pass": False, "entelecheia_pass": True,
             "homeostasis_pass": True, "stable": True}
        ]
        
        result = verifier.verify_trace(bad_trace)
        
        self.assertFalse(result)
        self.assertGreater(len(verifier.violations), 0)
        print(f"✅ PROOF: LTL verifier detected violation: {verifier.violations[0]['formula']}")


class TestGrammarFilter(unittest.TestCase):
    """Test grammar-guided synthesis filter."""
    
    def test_rejects_god_functions(self):
        """PROOF: Grammar filter rejects overly long functions"""
        grammar = SynthesisGrammar.strict()
        grammar.max_function_lines = 10
        filter = GrammarFilter(grammar)
        
        # Create a 50-line function
        long_func = "def very_long_function():\n" + "    x = 1\n" * 50
        
        valid, violations = filter.validate(long_func)
        
        self.assertFalse(valid)
        self.assertTrue(any("lines" in v for v in violations))
        print(f"✅ PROOF: Grammar filter rejects god function: {violations[0]}")
    
    def test_rejects_anti_patterns(self):
        """PROOF: Grammar filter rejects anti-pattern names"""
        grammar = SynthesisGrammar()
        filter = GrammarFilter(grammar)
        
        bad_code = '''
        class AbstractManagerFactory:
            pass
        '''
        
        valid, violations = filter.validate(bad_code)
        
        self.assertFalse(valid)
        self.assertTrue(any("Anti-pattern" in v for v in violations))
        print(f"✅ PROOF: Grammar filter rejects anti-pattern: {violations[0]}")


class TestThompsonSampling(unittest.TestCase):
    """Test Thompson Sampling clade selection."""
    
    def test_sampling_prefers_successful_clades(self):
        """PROOF: Thompson Sampling learns from successes"""
        clade_a = Clade(name="feature-a", alpha=10, beta=2)  # 10 wins, 2 losses
        clade_b = Clade(name="feature-b", alpha=2, beta=10)  # 2 wins, 10 losses
        
        # Sample 100 times
        a_wins = 0
        for _ in range(100):
            if clade_a.sample_fitness() > clade_b.sample_fitness():
                a_wins += 1
        
        self.assertGreater(a_wins, 80, "Successful clade should win most samples")
        print(f"✅ PROOF: Thompson Sampling selected successful clade {a_wins}/100 times")


def run_all_tests():
    """Run all operational tests and summarize."""
    print("=" * 60)
    print("ARK OPERATIONAL ENTELECHEIA PROOF")
    print("=" * 60)
    print()
    
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()
    
    # Add all test classes
    suite.addTests(loader.loadTestsFromTestCase(TestArkRepository))
    suite.addTests(loader.loadTestsFromTestCase(TestDNAGates))
    suite.addTests(loader.loadTestsFromTestCase(TestRhizomeDAG))
    suite.addTests(loader.loadTestsFromTestCase(TestEtymologyExtractor))
    suite.addTests(loader.loadTestsFromTestCase(TestDistillationPipeline))
    suite.addTests(loader.loadTestsFromTestCase(TestLTLVerification))
    suite.addTests(loader.loadTestsFromTestCase(TestGrammarFilter))
    suite.addTests(loader.loadTestsFromTestCase(TestThompsonSampling))
    
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    print()
    print("=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"Tests run: {result.testsRun}")
    print(f"Failures: {len(result.failures)}")
    print(f"Errors: {len(result.errors)}")
    
    if result.wasSuccessful():
        print("\n✅ ALL PROOFS VERIFIED - ARK ENTELECHEIA CONFIRMED")
    else:
        print("\n❌ SOME PROOFS FAILED - REFINEMENT NEEDED")
        for test, trace in result.failures + result.errors:
            print(f"  - {test}: {trace[:100]}...")
    
    return result.wasSuccessful()


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)
