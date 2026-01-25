#!/usr/bin/env python3
"""
repository.py - ArkRepository: The core wrapper around git operations

Implements DNA-aware version control with:
- Cell Cycle (G1-S-G2-M) commit flow
- Rhizom DAG integration for semantic lineage
- CMP scoring for fitness tracking
"""

import os
import json
import subprocess
import logging
from pathlib import Path
from dataclasses import dataclass, field
from typing import Optional, Dict, List, Any
from datetime import datetime

from nucleus.ark.core.context import ArkCommitContext
from nucleus.ark.gates.inertia import InertiaGate
from nucleus.ark.gates.entelecheia import EntelecheiaGate
from nucleus.ark.gates.homeostasis import HomeostasisGate
from nucleus.ark.neural.model import NeuralGate, HAS_TORCH
from nucleus.ark.neural.thrash_predictor import ThrashPredictor
from nucleus.ark.neural.thrash_predictor import ThrashPredictor
from nucleus.ark.meta.client import MetaLearnerClient, ExperienceRecord
from nucleus.tools.agent_bus import bus

logger = logging.getLogger("ARK.Repository")


@dataclass
class CellCycleResult:
    """Result of a Cell Cycle phase."""
    phase: str          # G1, S, G2, M
    passed: bool
    reason: str = ""
    data: Dict[str, Any] = field(default_factory=dict)


class ArkRepository:
    """
    ARK Repository wrapper providing DNA-aware git operations.
    
    Built on standard git (via subprocess) with ARK metadata extensions.
    Future: migrate to isomorphic-git for cross-platform purity.
    """
    
    def __init__(self, path: str):
        self.path = Path(path)
        self.ark_dir = self.path / ".ark"
        self.rhizom_path = self.ark_dir / "rhizom.json"
        self.config_path = self.ark_dir / "config.json"
        
        # DNA Gates
        self.gates = [
            InertiaGate(),
            EntelecheiaGate(),
            HomeostasisGate()
        ]
        
        # Rhizom DAG (lazy loaded)
        self._rhizom: Optional[Dict] = None
    
    @property
    def is_initialized(self) -> bool:
        """Check if this is an ARK repository."""
        return self.ark_dir.exists() and (self.path / ".git").exists()
    
    def init(self, initial_entropy: Optional[Dict[str, float]] = None) -> bool:
        """
        Initialize a new ARK repository.
        Creates .git and .ark directories with initial metadata.
        """
        if self.is_initialized:
            logger.warning(f"ARK repository already exists at {self.path}")
            return False
        
        # Initialize git
        result = subprocess.run(
            ["git", "init"],
            cwd=self.path,
            capture_output=True,
            text=True
        )
        if result.returncode != 0:
            logger.error(f"Git init failed: {result.stderr}")
            return False
        
        # Create .ark directory
        self.ark_dir.mkdir(exist_ok=True)
        
        # Initialize Rhizom
        self._rhizom = {
            "version": "1.0",
            "created": datetime.utcnow().isoformat(),
            "nodes": {}
        }
        self._save_rhizom()
        
        # Initialize config
        config = {
            "version": "1.0",
            "gates_enabled": True,
            "neural_enabled": False,
            "initial_entropy": initial_entropy or {
                "h_struct": 0.5,
                "h_doc": 0.5,
                "h_type": 0.5,
                "h_test": 0.5,
                "h_deps": 0.5,
                "h_churn": 0.5,
                "h_debt": 0.5,
                "h_align": 0.5
            }
        }
        self.config_path.write_text(json.dumps(config, indent=2))
        
        logger.info(f"ARK repository initialized at {self.path}")
        return True
    
    def commit(self, message: str, context: ArkCommitContext) -> Optional[str]:
        """
        Perform a DNA-gated commit using the Cell Cycle.
        
        Returns commit SHA on success, None on failure.
        """
        # Pre-check: Detect empty commit (nothing staged)
        staged = self._get_staged_files()
        if not staged:
            logger.warning("Nothing to commit: no staged files")
            return None
        
        # G1 Phase: Observation & Availability Check
        g1 = self._run_g1(context)
        if not g1.passed:
            logger.warning(f"G1 SLEEP: {g1.reason}")
            bus.publish("ark.gate.reject", {"phase": "G1", "reason": g1.reason})
            return None
        
        bus.publish("ark.commit.g1_passed", {"context": context.to_dict()})
        
        # S Phase: Synthesis (future: LTL-guided patch generation)
        s_result = self._run_s(context)
        
        # G2 Phase: Verification & Safety Check
        g2 = self._run_g2(context, s_result)
        if not g2.passed:
            logger.warning(f"G2 ABORT: {g2.reason}")
            bus.publish("ark.gate.reject", {"phase": "G2", "reason": g2.reason})
            return None
        
        bus.publish("ark.commit.g2_passed", {"context": context.to_dict()})
        
        # M Phase: Mitosis (actual commit)
        m_result = self._run_m(message, context)
        if not m_result.passed:
            logger.error(f"M FAIL: {m_result.reason}")
            return None
        
        sha = m_result.data.get("sha")
        
        # Post-commit: Update Rhizom
        self._update_rhizom(sha, context)

        # Post-commit: Record Experience for CL
        self._run_post_commit_actions(sha, context)
        
        bus.publish("ark.commit.mitosis", {"sha": sha, "cmp": context.cmp})
        bus.publish("ark.cmp.score", {"sha": sha, "score": context.cmp})
        
        logger.info(f"✅ ARK commit successful: {sha[:8]}")
        return sha
    
    def _run_g1(self, context: ArkCommitContext) -> CellCycleResult:
        """
        G1 Phase: Gap 1 - Observation
        Check aleatoric availability, hysteresis, and system stability.
        
        DNA Axiom 4 (Hysteresis): Past states influence present decisions.
        """
        # Check if system is in high-entropy state
        entropy = context.entropy or {}
        h_total = sum(entropy.values()) / max(len(entropy), 1)
        
        if h_total > 0.8:
            return CellCycleResult(
                phase="G1",
                passed=False,
                reason=f"System entropy too high ({h_total:.2f}). SLEEP until stable."
            )
        
        # Hysteresis Check (DNA Axiom 4): Past states influence present
        hysteresis_result = self._check_hysteresis(context)
        if not hysteresis_result.passed:
            return hysteresis_result
        
        # Witness Verification: Verify signatures if required
        if context.require_witness and context.witness:
            witness_result = self._verify_witness(context)
            if not witness_result.passed:
                return witness_result
        
        # Check Neural Gates (P2-042)
        if hasattr(context, 'entropy') and context.entropy:
            if HAS_TORCH:
                try:
                    # Neural Gate Check
                    ng = NeuralGate()
                    predictions = ng.predict(context.entropy)
                    
                    # Thrash Prediction
                    tp = ThrashPredictor()
                    thrash = tp.predict_from_entropy(context.entropy)
                    
                    if thrash.risk_level == "critical":
                        return CellCycleResult(
                            phase="G1",
                            passed=False,
                            reason=f"Neural Gate REJECT: Critical thrash risk ({thrash.thrash_probability:.1%})"
                        )
                        
                    # Log neural warnings
                    if thrash.risk_level == "high":
                        logger.warning(f"⚠️ Neural Gate Warning: High thrash risk ({thrash.thrash_probability:.1%})")
                        
                except Exception as e:
                    logger.warning(f"Neural gate check failed (non-blocking): {e}")

        # Check Homeostasis gate first
        for gate in self.gates:
            if gate.name() == "Homeostasis":
                from nucleus.ark.gates.homeostasis import HomeostasisContext
                hctx = HomeostasisContext(entropy=entropy)
                if not gate.check(hctx):
                    return CellCycleResult(
                        phase="G1", 
                        passed=False,
                        reason="Homeostasis gate rejected: system needs stabilization"
                    )
        
        return CellCycleResult(phase="G1", passed=True, reason="Environment stable")
    
    def _check_hysteresis(self, context: ArkCommitContext) -> CellCycleResult:
        """
        Check hysteresis: past commit patterns influence current decisions.
        
        DNA Axiom 4: "Past states influence present" (temporal continuity)
        
        Checks:
        - Recent commit frequency (prevent rapid-fire commits)
        - CMP trajectory stability
        - Entropy trend consistency
        """
        from nucleus.ark.core.hysteresis import HysteresisTracker
        
        try:
            tracker = HysteresisTracker(str(self.ark_dir / "hysteresis.json"))
            
            # Load recent history if available
            # Note: HysteresisTracker handles file not found gracefully
            
            # Check commit frequency (prevent >3 commits in 60 seconds)
            recent_count = len([
                s for s in tracker.state_history 
                if s.get('phase') == 'M'
            ][-3:])
            
            if recent_count >= 3:
                import time
                last_times = [s.get('timestamp', 0) for s in tracker.state_history[-3:]]
                if last_times and (time.time() - min(last_times)) < 60:
                    return CellCycleResult(
                        phase="G1",
                        passed=False,
                        reason="Hysteresis delay: too many commits in short period. SLEEP."
                    )
            
            # Track current state for future hysteresis
            state = tracker.create_state_snapshot(
                phase="G1",
                cmp=context.cmp,
                entropy=context.entropy or {},
                files_changed=len(context.files or [])
            )
            tracker.add_state(state)
            tracker.save()
            
        except Exception as e:
            # Hysteresis is advisory, not blocking on errors
            logger.debug("Hysteresis check skipped: %s", e)
        
        return CellCycleResult(phase="G1", passed=True, reason="Hysteresis check passed")
    
    def _verify_witness(self, context: ArkCommitContext) -> CellCycleResult:
        """
        Verify witness signature when required.
        
        P0-07: Reject unsigned witnesses when require_witness=True.
        """
        witness = context.witness
        
        if not witness:
            return CellCycleResult(
                phase="G1",
                passed=False,
                reason="Witness required but not provided"
            )
        
        if not witness.is_signed:
            return CellCycleResult(
                phase="G1",
                passed=False,
                reason="Witness not signed: cryptographic attestation required"
            )
        
        # Load secret key from config for verification
        config = {}
        if self.config_path.exists():
            config = json.loads(self.config_path.read_text())
        
        secret_key = config.get("witness_secret", "ark-default-key")
        
        if not witness.verify(secret_key):
            return CellCycleResult(
                phase="G1",
                passed=False,
                reason="Witness signature verification failed: invalid attestation"
            )
        
        logger.info("Witness %s verified successfully", witness.id)
        return CellCycleResult(phase="G1", passed=True, reason="Witness verified")
    
    def _run_s(self, context: ArkCommitContext) -> CellCycleResult:
        """
        S Phase: Synthesis
        LTL-guided verification of proposed changes with prescient analysis.
        
        Phase 2.1 Enhancement: MetaLearner prescient analysis predicts
        CMP and entropy before commit for early warning.
        """
        from nucleus.ark.synthesis.ltl_spec import PluribusLTLSpec, LTLVerifier
        
        # Compute actual entropy from staged files (Defense against spoofing)
        actual_entropy = self._compute_entropy_from_staged()
        if actual_entropy:
            # Override context entropy with verified values
            context.entropy = actual_entropy
        
        # Prescient Analysis (P2-008): Predict outcome before commit
        prescient_result = self._run_prescient_analysis(context)
        if prescient_result and prescient_result.get("risk_level") == "high":
            suggestions = prescient_result.get("suggestions", [])
            logger.warning(
                "Prescient analysis: HIGH RISK commit (predicted CMP=%.2f)",
                prescient_result.get("predicted_cmp", 0.0)
            )
            # Don't block yet, but log recommendation
            for suggestion in suggestions:
                logger.info("  → %s", suggestion)
        
        # Create LTL spec and verifier
        spec = PluribusLTLSpec.core_spec()
        verifier = LTLVerifier(spec)
        
        # Build verification trace
        h_total = context.total_entropy() if context.entropy else 0.5
        trace = [{
            "commit": True,
            "inertia_pass": True,  # Preliminary, G2 will verify
            "entelecheia_pass": bool(context.purpose),
            "homeostasis_pass": h_total < 0.7,
            "stable": h_total < 0.7
        }]
        
        # Verify against LTL specs
        if not verifier.verify_trace(trace):
            violations = [v['formula'] for v in verifier.violations]
            return CellCycleResult(
                phase="S",
                passed=False,
                reason=f"LTL pre-verification failed: {violations}",
                data={"violations": violations}
            )
        
        return CellCycleResult(
            phase="S",
            passed=True,
            reason="LTL-guided synthesis verified",
            data={
                "synthesized": True, 
                "spec": spec.name, 
                "entropy_verified": True,
                "prescient": prescient_result
            }
        )
    
    def _run_prescient_analysis(self, context: ArkCommitContext) -> Optional[Dict]:
        """
        Run MetaLearner prescient analysis.
        
        P2-008: Predict commit outcome before execution.
        """
        try:
            from nucleus.ark.meta.client import MetaLearnerClient
            
            client = MetaLearnerClient()
            if not client.available:
                return None
            
            # Build context for prediction
            files = context.files or []
            message = context.purpose or context.etymology
            entropy = context.entropy or {}
            
            # Get staged content for snippets
            snippets = []
            staged = self._get_staged_files()
            for filepath in staged[:5]:  # Limit to 5 files
                full_path = self.path / filepath
                if full_path.exists():
                    try:
                        content = full_path.read_text()[:500]  # First 500 chars
                        snippets.append(content)
                    except Exception:
                        pass
            
            prediction_context = {
                "files": files,
                "message": message,
                "entropy": entropy,
                "snippets": snippets
            }
            
            return client.prescient_analysis(prediction_context)
            
        except Exception as e:
            logger.debug("Prescient analysis skipped: %s", e)
            return None
    
    def _compute_entropy_from_staged(self) -> Optional[Dict[str, float]]:
        """
        Compute actual entropy from staged files.
        Defense against entropy spoofing attack.
        """
        # Get staged files
        result = subprocess.run(
            ["git", "diff", "--cached", "--name-only"],
            cwd=self.path,
            capture_output=True,
            text=True
        )
        
        if result.returncode != 0:
            return None
        
        staged_files = [f for f in result.stdout.strip().split("\n") if f]
        if not staged_files:
            return None
        
        entropies = []
        for filepath in staged_files:
            full_path = self.path / filepath
            if not full_path.exists() or not filepath.endswith('.py'):
                continue
            
            try:
                content = full_path.read_text()
                file_entropy = self._calculate_file_entropy(content)
                entropies.append(file_entropy)
            except Exception:
                continue
        
        if not entropies:
            return None
        
        # Average across files
        avg = {}
        for key in entropies[0]:
            avg[key] = sum(e[key] for e in entropies) / len(entropies)
        return avg
    
    def _calculate_file_entropy(self, content: str) -> Dict[str, float]:
        """Calculate entropy vector for a single file."""
        lines = content.split('\n')
        
        # h_struct: indentation variance
        indents = [len(l) - len(l.lstrip()) for l in lines if l.strip()]
        h_struct = min(len(set(indents)) / 10, 1.0) if indents else 0.5
        
        # h_doc: documentation coverage
        doc_lines = sum(1 for l in lines if l.strip().startswith('#') or '"""' in l)
        h_doc = 1.0 - min(doc_lines / max(len(lines), 1), 0.5) * 2
        
        # h_type: type hints
        type_hints = sum(1 for l in lines if 'def ' in l and ':' in l.split('def')[1])
        func_count = sum(1 for l in lines if 'def ' in l)
        h_type = 1.0 - (type_hints / max(func_count, 1)) if func_count else 0.5
        
        # h_test: assume no tests
        h_test = 0.7
        
        # h_deps: import sprawl
        import_count = sum(1 for l in lines if l.strip().startswith(('import ', 'from ')))
        h_deps = min(import_count / 20, 1.0)
        
        # h_churn: default
        h_churn = 0.5
        
        # h_debt: TODO/FIXME markers
        todo_count = sum(1 for l in lines if 'TODO' in l or 'FIXME' in l or 'HACK' in l)
        h_debt = min(todo_count / 5, 1.0)
        
        # h_align: default
        h_align = 0.5
        
        return {
            "h_struct": h_struct, "h_doc": h_doc, "h_type": h_type, "h_test": h_test,
            "h_deps": h_deps, "h_churn": h_churn, "h_debt": h_debt, "h_align": h_align
        }
    
    def _run_g2(self, context: ArkCommitContext, s_result: CellCycleResult) -> CellCycleResult:
        """
        G2 Phase: Gap 2 - Verification
        Check epistemic safety and DNA gate compliance.
        """
        # Run Inertia and Entelecheia gates
        for gate in self.gates:
            if gate.name() in ["Inertia", "Entelecheia"]:
                # Create appropriate context for each gate
                if gate.name() == "Inertia":
                    from nucleus.ark.gates.inertia import InertiaContext
                    gctx = InertiaContext(
                        files=context.files,
                        high_inertia_threshold=0.8
                    )
                else:
                    from nucleus.ark.gates.entelecheia import EntelecheiaContext
                    gctx = EntelecheiaContext(
                        purpose=context.purpose,
                        spec_ref=context.spec_ref
                    )
                
                if not gate.check(gctx):
                    return CellCycleResult(
                        phase="G2",
                        passed=False,
                        reason=f"{gate.name()} gate rejected commit"
                    )
        
        # Check witness requirement
        if context.require_witness and not context.witness:
            return CellCycleResult(
                phase="G2",
                passed=False,
                reason="Witness required but not provided"
            )
        
        return CellCycleResult(phase="G2", passed=True, reason="All gates passed")
    
    def _run_m(self, message: str, context: ArkCommitContext) -> CellCycleResult:
        """
        M Phase: Mitosis - Actual git commit
        """
        # Stage all changes if not already staged
        if context.stage_all:
            subprocess.run(["git", "add", "."], cwd=self.path, capture_output=True)
        
        # Encode ARK metadata in commit message
        full_message = self._encode_ark_message(message, context)
        
        # Perform commit
        result = subprocess.run(
            ["git", "commit", "-m", full_message],
            cwd=self.path,
            capture_output=True,
            text=True
        )
        
        if result.returncode != 0:
            return CellCycleResult(
                phase="M",
                passed=False,
                reason=f"Git commit failed: {result.stderr}"
            )
        
        # Get commit SHA
        sha_result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=self.path,
            capture_output=True,
            text=True
        )
        sha = sha_result.stdout.strip()
        
        return CellCycleResult(
            phase="M",
            passed=True,
            reason="Commit successful",
            data={"sha": sha}
        )
    
    def _encode_ark_message(self, message: str, context: ArkCommitContext) -> str:
        """Embed ARK metadata in commit message."""
        metadata = {
            "cmp": context.cmp,
            "entropy": context.entropy,
            "witness": context.witness.id if context.witness else None,
            "spec_ref": context.spec_ref,
            "etymology": context.etymology
        }
        
        return f"""{message}

---ARK-METADATA---
{json.dumps(metadata, indent=2)}
"""
    
    def _update_rhizom(self, sha: str, context: ArkCommitContext) -> None:
        """Update Rhizom DAG with new commit node."""
        if self._rhizom is None:
            self._load_rhizom()
        
        self._rhizom["nodes"][sha] = {
            "sha": sha,
            "etymology": context.etymology,
            "cmp": context.cmp,
            "entropy": context.entropy,
            "parent": context.parent_sha,
            "lineage_tags": context.lineage_tags,
            "timestamp": datetime.utcnow().isoformat(),
            "witness_id": context.witness.id if context.witness else None
        }
        
        self._save_rhizom()
    
    def _load_rhizom(self) -> None:
        """Load Rhizom DAG from disk."""
        if self.rhizom_path.exists():
            self._rhizom = json.loads(self.rhizom_path.read_text())
        else:
            self._rhizom = {"version": "1.0", "nodes": {}}
    
    def _save_rhizom(self) -> None:
        """Save Rhizom DAG to disk."""
        self.rhizom_path.write_text(json.dumps(self._rhizom, indent=2))

    def _run_post_commit_actions(self, sha: str, context: ArkCommitContext) -> None:
        """
        Run Continual Learning hooks after commit.
        Capture experience and update CL models.
        """
        try:
            client = MetaLearnerClient()
            if not client.available:
                return

            # Calculate reward (simplified: success = 1.0, high cmp is better)
            cmp_score = context.cmp or 0.5
            reward = cmp_score * 2.0  # simple scaling

            record = ExperienceRecord(
                commit_sha=sha,
                etymology=context.etymology,
                cmp_before=0.5, # Placeholder, in prod would be previous HEAD cmp
                cmp_after=cmp_score,
                entropy_before=context.entropy or {},
                entropy_after={}, # Would need re-calc, skipping for perf
                gate_results={"G1": True, "S": True, "G2": True},
                success=True,
                reward=reward
            )

            client.record_experience(record)
            logger.info("Recorded CL experience for %s", sha[:8])

        except Exception as e:
            logger.warning(f"CL post-commit hook failed: {e}")
    
    def status(self) -> Dict[str, Any]:
        """Get ARK repository status including DNA metrics."""
        if not self.is_initialized:
            return {"error": "Not an ARK repository"}
        
        # Get git status
        result = subprocess.run(
            ["git", "status", "--porcelain"],
            cwd=self.path,
            capture_output=True,
            text=True
        )
        
        modified = []
        staged = []
        for line in result.stdout.strip().split("\n"):
            if line:
                status = line[:2]
                filename = line[3:]
                if status[0] in "MADRC":
                    staged.append(filename)
                if status[1] in "MADRC":
                    modified.append(filename)
        
        # Load config for entropy
        config = {}
        if self.config_path.exists():
            config = json.loads(self.config_path.read_text())
        
        return {
            "path": str(self.path),
            "initialized": True,
            "modified": modified,
            "staged": staged,
            "gates_enabled": config.get("gates_enabled", True),
            "current_entropy": config.get("initial_entropy", {})
        }
    
    def log(self, limit: int = 10) -> List[Dict[str, Any]]:
        """Get ARK-enhanced commit log with CMP and entropy."""
        result = subprocess.run(
            ["git", "log", f"-{limit}", "--format=%H|%s|%ai"],
            cwd=self.path,
            capture_output=True,
            text=True
        )
        
        commits = []
        if self._rhizom is None:
            self._load_rhizom()
        
        for line in result.stdout.strip().split("\n"):
            if line:
                parts = line.split("|")
                sha = parts[0]
                subject = parts[1] if len(parts) > 1 else ""
                date = parts[2] if len(parts) > 2 else ""
                
                # Get Rhizom metadata
                rhizom_node = self._rhizom.get("nodes", {}).get(sha, {})
                
                commits.append({
                    "sha": sha,
                    "subject": subject,
                    "date": date,
                    "cmp": rhizom_node.get("cmp"),
                    "etymology": rhizom_node.get("etymology"),
                    "entropy": rhizom_node.get("entropy")
                })
        
        return commits
    
    def _get_staged_files(self) -> List[str]:
        """Get list of staged files."""
        result = subprocess.run(
            ["git", "diff", "--cached", "--name-only"],
            cwd=self.path,
            capture_output=True,
            text=True
        )
        
        if result.returncode != 0:
            return []
        
        return [f for f in result.stdout.strip().split("\n") if f]

