#!/usr/bin/env python3
"""
rollback.py - Homeostatic Rollback Automation

Monitors commits for CMP regression and auto-reverts if threshold exceeded.
Part of the ARK Self-Healing System.
"""

import time
import logging
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Callable

logger = logging.getLogger("ARK.Rollback")


@dataclass
class RollbackResult:
    """Result of a rollback operation."""
    triggered: bool
    original_sha: str
    reverted_to: Optional[str] = None
    reason: str = ""
    cmp_before: float = 0.0
    cmp_after: float = 0.0


class RollbackAutomation:
    """
    Monitors commits and auto-reverts on CMP regression.
    
    Implements homeostatic self-healing:
    - Watches CMP after commit
    - If regression > threshold, reverts
    - Emits ark.safety.rollback event
    """
    
    def __init__(
        self,
        threshold: float = 0.95,  # 5% regression triggers rollback
        watch_duration: int = 60,  # seconds to monitor before deciding
        bus_emit: Optional[Callable] = None,
    ):
        self.threshold = threshold
        self.watch_duration = watch_duration
        self.bus_emit = bus_emit
        self._cmp_cache: dict = {}
    
    def monitor(
        self,
        sha: str,
        cmp_before: float,
        blocking: bool = False,
    ) -> RollbackResult:
        """
        Monitor a commit for CMP regression.
        
        Args:
            sha: The commit SHA to monitor
            cmp_before: CMP score before this commit
            blocking: If True, block until watch_duration elapses
        
        Returns:
            RollbackResult with outcome
        """
        if blocking:
            time.sleep(self.watch_duration)
        
        # Get current CMP (after commit effects)
        cmp_after = self._get_current_cmp(sha)
        
        # Check for regression
        if cmp_after < cmp_before * self.threshold:
            logger.warning(
                f"CMP regression detected: {cmp_before:.3f} → {cmp_after:.3f}"
            )
            return self._execute_rollback(sha, cmp_before, cmp_after)
        
        return RollbackResult(
            triggered=False,
            original_sha=sha,
            cmp_before=cmp_before,
            cmp_after=cmp_after,
            reason="CMP stable or improved",
        )
    
    def _get_current_cmp(self, sha: str) -> float:
        """
        Get current CMP score.
        
        In a full implementation, this would:
        1. Run test suite
        2. Measure entropy
        3. Compute CMP from results
        
        For now, returns cached or baseline value.
        """
        if sha in self._cmp_cache:
            return self._cmp_cache[sha]
        
        # Default: assume stable
        return 0.5
    
    def set_cmp(self, sha: str, cmp: float) -> None:
        """Set CMP for a SHA (used by external monitors)."""
        self._cmp_cache[sha] = cmp
    
    def _execute_rollback(
        self,
        sha: str,
        cmp_before: float,
        cmp_after: float,
    ) -> RollbackResult:
        """Execute git revert and emit event."""
        try:
            # Get parent SHA
            parent = subprocess.check_output(
                ["git", "rev-parse", f"{sha}^"],
                text=True,
            ).strip()
            
            # Revert to parent
            subprocess.run(
                ["git", "revert", "--no-commit", sha],
                check=True,
            )
            subprocess.run(
                ["git", "commit", "-m", f"ARK Auto-Rollback: Reverted {sha[:8]} (CMP regression)"],
                check=True,
            )
            
            # Get new HEAD
            new_head = subprocess.check_output(
                ["git", "rev-parse", "HEAD"],
                text=True,
            ).strip()
            
            # Emit bus event
            if self.bus_emit:
                self.bus_emit("ark.safety.rollback", {
                    "original_sha": sha,
                    "reverted_to": new_head,
                    "cmp_before": cmp_before,
                    "cmp_after": cmp_after,
                    "reason": "CMP regression exceeded threshold",
                })
            
            logger.info(f"Rollback successful: {sha[:8]} → {new_head[:8]}")
            
            return RollbackResult(
                triggered=True,
                original_sha=sha,
                reverted_to=new_head,
                cmp_before=cmp_before,
                cmp_after=cmp_after,
                reason="CMP regression exceeded threshold",
            )
            
        except subprocess.CalledProcessError as e:
            logger.error(f"Rollback failed: {e}")
            return RollbackResult(
                triggered=False,
                original_sha=sha,
                cmp_before=cmp_before,
                cmp_after=cmp_after,
                reason=f"Rollback failed: {e}",
            )


class WatchDaemon:
    """
    Background daemon that watches for CMP regression.
    
    Usage:
        daemon = WatchDaemon()
        daemon.start()  # Runs in background thread
    """
    
    def __init__(self, automation: Optional[RollbackAutomation] = None):
        self.automation = automation or RollbackAutomation()
        self._running = False
        self._last_sha: Optional[str] = None
        self._last_cmp: float = 0.5
    
    def watch_commit(self, sha: str, cmp: float) -> RollbackResult:
        """Watch a specific commit (called after commit)."""
        result = self.automation.monitor(sha, cmp, blocking=False)
        self._last_sha = sha
        self._last_cmp = cmp
        return result
