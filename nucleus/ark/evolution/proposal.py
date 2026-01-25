#!/usr/bin/env python3
"""
proposal.py - Cross-Trunk Synthesis: Proposal Generator

The Evolution Trunk observes patterns and generates improvement proposals
that flow back to the Execution Trunk for application.

Part of the DNA Tri-Part Architecture.
"""

import json
import logging
import time
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any, Dict, List, Optional
from enum import Enum

logger = logging.getLogger("ARK.Proposal")


class ProposalType(Enum):
    REFACTOR = "refactor"
    OPTIMIZE = "optimize"
    FIX = "fix"
    ENHANCE = "enhance"
    DEPRECATE = "deprecate"


class ProposalStatus(Enum):
    PENDING = "pending"
    APPROVED = "approved"
    REJECTED = "rejected"
    APPLIED = "applied"


@dataclass
class Proposal:
    """A code improvement proposal from the Evolution Trunk."""
    id: str
    proposal_type: ProposalType
    title: str
    description: str
    target_files: List[str]
    patch: str  # Unified diff format
    rationale: str
    expected_cmp_delta: float
    status: ProposalStatus = ProposalStatus.PENDING
    created_at: float = field(default_factory=time.time)
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "proposal_type": self.proposal_type.value,
            "title": self.title,
            "description": self.description,
            "target_files": self.target_files,
            "patch": self.patch,
            "rationale": self.rationale,
            "expected_cmp_delta": self.expected_cmp_delta,
            "status": self.status.value,
            "created_at": self.created_at,
            "metadata": self.metadata,
        }
    
    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "Proposal":
        return cls(
            id=d["id"],
            proposal_type=ProposalType(d["proposal_type"]),
            title=d["title"],
            description=d["description"],
            target_files=d["target_files"],
            patch=d["patch"],
            rationale=d["rationale"],
            expected_cmp_delta=d["expected_cmp_delta"],
            status=ProposalStatus(d.get("status", "pending")),
            created_at=d.get("created_at", time.time()),
            metadata=d.get("metadata", {}),
        )


class ProposalGenerator:
    """
    Generates improvement proposals based on codebase analysis.
    
    This is the Evolution Trunk's output mechanism. It:
    1. Analyzes entropy patterns over time
    2. Identifies optimization opportunities
    3. Generates actionable proposals with patches
    """
    
    def __init__(self, ark_path: Optional[Path] = None):
        self.ark_path = ark_path or Path.cwd() / ".ark"
        self.proposals_path = self.ark_path / "proposals"
        self.proposals_path.mkdir(parents=True, exist_ok=True)
    
    def generate(
        self,
        proposal_type: ProposalType,
        title: str,
        description: str,
        target_files: List[str],
        patch: str,
        rationale: str,
        expected_cmp_delta: float = 0.05,
    ) -> Proposal:
        """Generate a new proposal."""
        proposal_id = f"prop-{int(time.time())}-{hash(title) % 10000:04d}"
        
        proposal = Proposal(
            id=proposal_id,
            proposal_type=proposal_type,
            title=title,
            description=description,
            target_files=target_files,
            patch=patch,
            rationale=rationale,
            expected_cmp_delta=expected_cmp_delta,
        )
        
        # Save to disk
        self._save_proposal(proposal)
        
        logger.info(f"Generated proposal: {proposal_id} - {title}")
        return proposal
    
    def _save_proposal(self, proposal: Proposal) -> None:
        """Save proposal to disk."""
        path = self.proposals_path / f"{proposal.id}.json"
        path.write_text(json.dumps(proposal.to_dict(), indent=2))
    
    def list_proposals(
        self,
        status: Optional[ProposalStatus] = None,
    ) -> List[Proposal]:
        """List all proposals, optionally filtered by status."""
        proposals = []
        for p in self.proposals_path.glob("*.json"):
            data = json.loads(p.read_text())
            proposal = Proposal.from_dict(data)
            if status is None or proposal.status == status:
                proposals.append(proposal)
        return sorted(proposals, key=lambda p: p.created_at, reverse=True)
    
    def get_proposal(self, proposal_id: str) -> Optional[Proposal]:
        """Get a specific proposal by ID."""
        path = self.proposals_path / f"{proposal_id}.json"
        if path.exists():
            return Proposal.from_dict(json.loads(path.read_text()))
        return None
    
    def update_status(
        self,
        proposal_id: str,
        status: ProposalStatus,
    ) -> Optional[Proposal]:
        """Update proposal status."""
        proposal = self.get_proposal(proposal_id)
        if proposal:
            proposal.status = status
            self._save_proposal(proposal)
            return proposal
        return None


class ProposalApplicator:
    """
    Applies proposals to the codebase.
    
    This is the bridge from Evolution → Execution.
    """
    
    def __init__(self, generator: Optional[ProposalGenerator] = None):
        self.generator = generator or ProposalGenerator()
    
    def apply(self, proposal_id: str) -> bool:
        """
        Apply a proposal's patch to the codebase.
        
        Returns True if successful.
        """
        import subprocess
        
        proposal = self.generator.get_proposal(proposal_id)
        if not proposal:
            logger.error(f"Proposal not found: {proposal_id}")
            return False
        
        if proposal.status != ProposalStatus.APPROVED:
            logger.warning(f"Proposal not approved: {proposal_id}")
            return False
        
        try:
            # Apply patch
            result = subprocess.run(
                ["git", "apply", "--check"],
                input=proposal.patch,
                capture_output=True,
                text=True,
            )
            
            if result.returncode != 0:
                logger.error(f"Patch check failed: {result.stderr}")
                return False
            
            # Actually apply
            subprocess.run(
                ["git", "apply"],
                input=proposal.patch,
                check=True,
                text=True,
            )
            
            # Update status
            self.generator.update_status(proposal_id, ProposalStatus.APPLIED)
            
            logger.info(f"Applied proposal: {proposal_id}")
            return True
            
        except subprocess.CalledProcessError as e:
            logger.error(f"Failed to apply proposal: {e}")
            return False
