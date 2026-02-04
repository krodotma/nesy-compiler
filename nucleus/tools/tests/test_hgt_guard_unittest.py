#!/usr/bin/env python3
"""
Unit tests for hgt_guard.py - CMP fitness checks for RGMA HGT.
"""
from __future__ import annotations

import json
import tempfile
import uuid
from pathlib import Path
from unittest import TestCase, main

import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from hgt_guard import (
    PHI_INV,
    PHI_INV_3,
    CMPRegistry,
    Evidence,
    GuardResult,
    HGTProposal,
    OmegaMotif,
    approve_splice,
    check_donor_fitness,
    check_motif_viability,
    check_provenance,
    check_execution,
    check_reproducibility,
    sha256_content,
)


class TestConstants(TestCase):
    """Test golden ratio constants."""

    def test_phi_inv_value(self):
        """PHI_INV should be approximately 0.618."""
        self.assertAlmostEqual(PHI_INV, 0.618, places=3)

    def test_phi_inv_3_value(self):
        """PHI_INV_3 should be approximately 0.236."""
        self.assertAlmostEqual(PHI_INV_3, 0.236, places=3)


class TestCMPRegistry(TestCase):
    """Test CMP registry operations."""

    def test_set_and_get_cmp(self):
        """Can set and retrieve CMP scores."""
        registry = CMPRegistry()
        registry.set_cmp("lineage-a", 0.75)
        self.assertEqual(registry.get_cmp("lineage-a"), 0.75)

    def test_get_missing_lineage(self):
        """Missing lineage returns None."""
        registry = CMPRegistry()
        self.assertIsNone(registry.get_cmp("nonexistent"))

    def test_is_extinct(self):
        """Lineage with CMP below PHI_INV_3 is extinct."""
        registry = CMPRegistry()
        registry.set_cmp("healthy", 0.5)
        registry.set_cmp("dying", 0.2)

        self.assertFalse(registry.is_extinct("healthy"))
        self.assertTrue(registry.is_extinct("dying"))

    def test_save_and_load(self):
        """Can persist and reload registry."""
        registry = CMPRegistry()
        registry.set_cmp("lineage-a", 0.75)
        registry.set_cmp("lineage-b", 0.60)

        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "cmp.ndjson"
            registry.save(path)

            loaded = CMPRegistry.load(path)
            self.assertEqual(loaded.get_cmp("lineage-a"), 0.75)
            self.assertEqual(loaded.get_cmp("lineage-b"), 0.60)


class TestDonorFitness(TestCase):
    """Test check_donor_fitness function."""

    def test_donor_meets_threshold(self):
        """Donor with CMP >= 0.618 passes."""
        registry = CMPRegistry()
        registry.set_cmp("healthy-donor", 0.72)

        passed, cmp, reason = check_donor_fitness("healthy-donor", registry)

        self.assertTrue(passed)
        self.assertEqual(cmp, 0.72)
        self.assertIn("meets threshold", reason)

    def test_donor_below_threshold(self):
        """Donor with CMP < 0.618 fails."""
        registry = CMPRegistry()
        registry.set_cmp("weak-donor", 0.50)

        passed, cmp, reason = check_donor_fitness("weak-donor", registry)

        self.assertFalse(passed)
        self.assertEqual(cmp, 0.50)
        self.assertIn("below threshold", reason)

    def test_donor_not_found(self):
        """Missing donor fails with None CMP."""
        registry = CMPRegistry()

        passed, cmp, reason = check_donor_fitness("missing", registry)

        self.assertFalse(passed)
        self.assertIsNone(cmp)
        self.assertIn("not found", reason)

    def test_donor_extinct(self):
        """Extinct donor fails even if technically present."""
        registry = CMPRegistry()
        registry.set_cmp("extinct-donor", 0.1)

        passed, cmp, reason = check_donor_fitness("extinct-donor", registry)

        self.assertFalse(passed)
        self.assertIn("extinct", reason.lower())

    def test_custom_threshold(self):
        """Can use custom CMP threshold."""
        registry = CMPRegistry()
        registry.set_cmp("donor", 0.55)

        # Fails with default threshold
        passed1, _, _ = check_donor_fitness("donor", registry)
        self.assertFalse(passed1)

        # Passes with lower threshold
        passed2, _, _ = check_donor_fitness("donor", registry, min_cmp=0.5)
        self.assertTrue(passed2)


class TestMotifViability(TestCase):
    """Test check_motif_viability function."""

    def test_viable_motif(self):
        """Motif meeting thresholds is viable."""
        motif = OmegaMotif(
            id="motif-1",
            content_sha="abc123",
            occurrence_count=5,
            cmp_correlation=0.7,
        )

        passed, reason = check_motif_viability(motif)

        self.assertTrue(passed)
        self.assertIn("viable", reason.lower())

    def test_insufficient_occurrences(self):
        """Motif with too few occurrences is not viable."""
        motif = OmegaMotif(
            id="motif-1",
            content_sha="abc123",
            occurrence_count=2,
            cmp_correlation=0.7,
        )

        passed, reason = check_motif_viability(motif)

        self.assertFalse(passed)
        self.assertIn("occurrences", reason.lower())

    def test_low_correlation(self):
        """Motif with low CMP correlation is not viable."""
        motif = OmegaMotif(
            id="motif-1",
            content_sha="abc123",
            occurrence_count=5,
            cmp_correlation=0.3,
        )

        passed, reason = check_motif_viability(motif)

        self.assertFalse(passed)
        self.assertIn("correlation", reason.lower())

    def test_custom_thresholds(self):
        """Can use custom viability thresholds."""
        motif = OmegaMotif(
            id="motif-1",
            content_sha="abc123",
            occurrence_count=2,
            cmp_correlation=0.4,
        )

        # Fails with defaults
        passed1, _ = check_motif_viability(motif)
        self.assertFalse(passed1)

        # Passes with lowered thresholds
        passed2, _ = check_motif_viability(motif, min_occurrences=2, min_correlation=0.4)
        self.assertTrue(passed2)


class TestHGTProposal(TestCase):
    """Test HGTProposal dataclass."""

    def test_to_dict_serialization(self):
        """Proposal serializes to dict correctly."""
        proposal = HGTProposal(
            req_id="test-123",
            actor="test-agent",
            donor_lineage="core.strp",
            recipient_lineage="core.tools",
            motif_sha="sha256abc",
            donor_cmp=0.72,
            recipient_cmp=0.65,
            path="/tmp/test.py",
            reason="test transfer",
            evidence=[Evidence(kind="path", value="/tmp/test.py", note="test")],
        )

        d = proposal.to_dict()

        self.assertEqual(d["req_id"], "test-123")
        self.assertEqual(d["donor_cmp"], 0.72)
        self.assertEqual(d["recipient_cmp"], 0.65)
        self.assertEqual(len(d["evidence"]), 1)

    def test_from_dict_deserialization(self):
        """Proposal deserializes from dict correctly."""
        d = {
            "req_id": "test-456",
            "actor": "test-agent",
            "donor_lineage": "core.strp",
            "recipient_lineage": "core.tools",
            "motif_sha": "sha256def",
            "donor_cmp": 0.80,
            "recipient_cmp": 0.70,
            "path": "/tmp/test2.py",
            "reason": "",
            "evidence": [{"kind": "bus", "value": "task.complete", "note": ""}],
        }

        proposal = HGTProposal.from_dict(d)

        self.assertEqual(proposal.req_id, "test-456")
        self.assertEqual(proposal.donor_cmp, 0.80)
        self.assertEqual(len(proposal.evidence), 1)
        self.assertEqual(proposal.evidence[0].kind, "bus")


class TestGuardChecks(TestCase):
    """Test individual guard check functions."""

    def test_check_provenance_valid(self):
        """Valid provenance passes."""
        with tempfile.NamedTemporaryFile(suffix=".py", delete=False) as f:
            f.write(b"# test\n")
            test_path = f.name

        try:
            proposal = HGTProposal(
                req_id="test-123",
                actor="test",
                donor_lineage="a",
                recipient_lineage="b",
                motif_sha="sha",
                donor_cmp=0.7,
                recipient_cmp=0.6,
                path=test_path,
                evidence=[Evidence(kind="path", value=test_path)],
            )

            passed, reason = check_provenance(proposal)
            self.assertTrue(passed)
        finally:
            Path(test_path).unlink()

    def test_check_provenance_missing_evidence(self):
        """Missing evidence fails provenance."""
        proposal = HGTProposal(
            req_id="test-123",
            actor="test",
            donor_lineage="a",
            recipient_lineage="b",
            motif_sha="sha",
            donor_cmp=0.7,
            recipient_cmp=0.6,
            path="/tmp/nonexistent.py",
            evidence=[],
        )

        passed, reason = check_provenance(proposal)
        self.assertFalse(passed)
        self.assertIn("No evidence", reason)

    def test_check_execution_valid_python(self):
        """Valid Python file passes execution check."""
        with tempfile.NamedTemporaryFile(suffix=".py", delete=False) as f:
            f.write(b"def hello():\n    return 'world'\n")
            test_path = f.name

        try:
            proposal = HGTProposal(
                req_id="test",
                actor="test",
                donor_lineage="a",
                recipient_lineage="b",
                motif_sha="sha",
                donor_cmp=0.7,
                recipient_cmp=0.6,
                path=test_path,
            )

            passed, reason = check_execution(proposal)
            self.assertTrue(passed)
        finally:
            Path(test_path).unlink()

    def test_check_execution_invalid_python(self):
        """Invalid Python file fails execution check."""
        with tempfile.NamedTemporaryFile(suffix=".py", delete=False) as f:
            f.write(b"def broken(\n")  # Syntax error
            test_path = f.name

        try:
            proposal = HGTProposal(
                req_id="test",
                actor="test",
                donor_lineage="a",
                recipient_lineage="b",
                motif_sha="sha",
                donor_cmp=0.7,
                recipient_cmp=0.6,
                path=test_path,
            )

            passed, reason = check_execution(proposal)
            self.assertFalse(passed)
            self.assertIn("syntax", reason.lower())
        finally:
            Path(test_path).unlink()

    def test_check_reproducibility_valid(self):
        """Valid reproducibility info passes."""
        proposal = HGTProposal(
            req_id="test",
            actor="test",
            donor_lineage="a",
            recipient_lineage="b",
            motif_sha="sha256abc123",
            donor_cmp=0.7,
            recipient_cmp=0.6,
            path="/tmp/test.py",
        )

        passed, reason = check_reproducibility(proposal)
        self.assertTrue(passed)

    def test_check_reproducibility_missing_sha(self):
        """Missing motif_sha fails reproducibility."""
        proposal = HGTProposal(
            req_id="test",
            actor="test",
            donor_lineage="a",
            recipient_lineage="b",
            motif_sha="",
            donor_cmp=0.7,
            recipient_cmp=0.6,
            path="/tmp/test.py",
        )

        passed, reason = check_reproducibility(proposal)
        self.assertFalse(passed)
        self.assertIn("Missing motif_sha", reason)


class TestApproveSplice(TestCase):
    """Test the full approve_splice guard ladder."""

    def test_successful_splice(self):
        """Valid proposal passes all guards."""
        with tempfile.NamedTemporaryFile(suffix=".py", delete=False) as f:
            f.write(b"# Valid Python\ndef hello():\n    return 'world'\n")
            test_path = f.name

        try:
            registry = CMPRegistry()
            registry.set_cmp("donor", 0.72)
            registry.set_cmp("recipient", 0.65)

            proposal = HGTProposal(
                req_id=str(uuid.uuid4()),
                actor="test",
                donor_lineage="donor",
                recipient_lineage="recipient",
                motif_sha=sha256_content("test"),
                donor_cmp=0.72,
                recipient_cmp=0.65,
                path=test_path,
                evidence=[Evidence(kind="path", value=test_path)],
            )

            result = approve_splice(proposal, registry)

            self.assertTrue(result.passed)
            self.assertIsNone(result.gate_id)
        finally:
            Path(test_path).unlink()

    def test_rejected_for_low_donor_cmp(self):
        """Proposal with low donor CMP fails G-CMP."""
        with tempfile.NamedTemporaryFile(suffix=".py", delete=False) as f:
            f.write(b"# Valid Python\n")
            test_path = f.name

        try:
            registry = CMPRegistry()
            registry.set_cmp("weak-donor", 0.50)  # Below 0.618 threshold
            registry.set_cmp("recipient", 0.65)

            proposal = HGTProposal(
                req_id=str(uuid.uuid4()),
                actor="test",
                donor_lineage="weak-donor",
                recipient_lineage="recipient",
                motif_sha=sha256_content("test"),
                donor_cmp=0.50,
                recipient_cmp=0.65,
                path=test_path,
                evidence=[Evidence(kind="path", value=test_path)],
            )

            result = approve_splice(proposal, registry)

            self.assertFalse(result.passed)
            self.assertEqual(result.gate_id, "G-CMP")
            self.assertIn("below threshold", result.reason.lower())
        finally:
            Path(test_path).unlink()

    def test_rejected_for_extinct_recipient(self):
        """Proposal with extinct recipient fails G-CMP."""
        with tempfile.NamedTemporaryFile(suffix=".py", delete=False) as f:
            f.write(b"# Valid Python\n")
            test_path = f.name

        try:
            registry = CMPRegistry()
            registry.set_cmp("donor", 0.72)
            registry.set_cmp("extinct-recipient", 0.10)  # Below PHI_INV_3

            proposal = HGTProposal(
                req_id=str(uuid.uuid4()),
                actor="test",
                donor_lineage="donor",
                recipient_lineage="extinct-recipient",
                motif_sha=sha256_content("test"),
                donor_cmp=0.72,
                recipient_cmp=0.10,
                path=test_path,
                evidence=[Evidence(kind="path", value=test_path)],
            )

            result = approve_splice(proposal, registry)

            self.assertFalse(result.passed)
            self.assertEqual(result.gate_id, "G-CMP")
            self.assertIn("extinct", result.reason.lower())
        finally:
            Path(test_path).unlink()


class TestUtilityFunctions(TestCase):
    """Test utility functions."""

    def test_sha256_content_string(self):
        """SHA256 of string is computed correctly."""
        sha = sha256_content("hello world")
        self.assertEqual(len(sha), 64)
        self.assertEqual(sha[:8], "b94d27b9")

    def test_sha256_content_bytes(self):
        """SHA256 of bytes is computed correctly."""
        sha = sha256_content(b"hello world")
        self.assertEqual(sha[:8], "b94d27b9")


if __name__ == "__main__":
    main()
