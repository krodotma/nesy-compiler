#!/usr/bin/env python3
"""
context.py - ArkCommitContext: Metadata for DNA-aware commits

Carries all information needed for Cell Cycle processing:
- Etymology (semantic origin)
- CMP score (fitness)
- H* entropy vector
- Witness attestation
- LTL spec reference
"""

from dataclasses import dataclass, field, asdict
from typing import Optional, Dict, List, Any


@dataclass
class Witness:
    """Attestation witness for a commit with cryptographic signing."""
    id: str
    attester: str
    timestamp: str
    intent: str
    verification_method: str = "empirical"  # ltl, empirical, formal
    verification_result: str = "pass"
    signature: str = ""  # HMAC-SHA256 signature
    
    def sign(self, secret_key: str) -> "Witness":
        """
        Sign the witness attestation with HMAC-SHA256.
        
        Args:
            secret_key: Secret key for signing (agent-specific)
        
        Returns:
            Self with signature field populated
        """
        import hmac
        import hashlib
        
        payload = f"{self.attester}:{self.timestamp}:{self.intent}:{self.id}"
        self.signature = hmac.new(
            secret_key.encode(),
            payload.encode(),
            hashlib.sha256
        ).hexdigest()
        return self
    
    def verify(self, secret_key: str) -> bool:
        """
        Verify the witness signature.
        
        Args:
            secret_key: Secret key used for signing
        
        Returns:
            True if signature is valid
        """
        import hmac
        import hashlib
        
        if not self.signature:
            return False
        
        payload = f"{self.attester}:{self.timestamp}:{self.intent}:{self.id}"
        expected = hmac.new(
            secret_key.encode(),
            payload.encode(),
            hashlib.sha256
        ).hexdigest()
        
        return hmac.compare_digest(self.signature, expected)
    
    @property
    def is_signed(self) -> bool:
        """Check if witness has been signed."""
        return bool(self.signature)


@dataclass
class ArkCommitContext:
    """
    Context for an ARK commit operation.
    
    Carries all DNA-relevant metadata through the Cell Cycle.
    """
    # Semantic origin of the change
    etymology: str = ""
    
    # Fitness score (Cumulative Meta-Priority)
    cmp: float = 0.5
    
    # 8-dimensional entropy vector
    entropy: Dict[str, float] = field(default_factory=lambda: {
        "h_struct": 0.5,
        "h_doc": 0.5,
        "h_type": 0.5,
        "h_test": 0.5,
        "h_deps": 0.5,
        "h_churn": 0.5,
        "h_debt": 0.5,
        "h_align": 0.5
    })
    
    # Attestation witness
    witness: Optional[Witness] = None
    require_witness: bool = False
    
    # LTL spec reference
    spec_ref: Optional[str] = None
    
    # Purpose description (for Entelecheia gate)
    purpose: str = ""
    
    # Files being modified
    files: List[str] = field(default_factory=list)
    
    # Parent commit SHA
    parent_sha: Optional[str] = None
    
    # Clade/lineage tags
    lineage_tags: List[str] = field(default_factory=list)
    
    # Whether to stage all changes
    stage_all: bool = True
    
    # Author info
    author_name: Optional[str] = None
    author_email: Optional[str] = None
    
    def total_entropy(self) -> float:
        """Calculate total entropy as average of H* vector."""
        if not self.entropy:
            return 0.5
        return sum(self.entropy.values()) / len(self.entropy)
    
    def is_negentropic(self, threshold: float = 0.4) -> bool:
        """Check if this commit reduces entropy below threshold."""
        return self.total_entropy() < threshold
    
    def with_witness(self, attester: str, intent: str) -> "ArkCommitContext":
        """Create a new context with witness attestation."""
        import time
        self.witness = Witness(
            id=f"witness-{int(time.time())}",
            attester=attester,
            timestamp=time.strftime("%Y-%m-%dT%H:%M:%SZ"),
            intent=intent
        )
        return self

    def to_dict(self) -> Dict[str, Any]:
        """Serialize context to dictionary."""
        return asdict(self)
