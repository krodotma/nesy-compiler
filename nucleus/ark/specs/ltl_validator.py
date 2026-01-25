#!/usr/bin/env python3
"""
ltl_validator.py - LTL Specification Validation for ARK Gates

Provides:
- Spec loading from .ark/specs/*.tla (simplified)
- Invariant checking against commit diffs
- Purpose alignment verification
"""

import json
import re
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Any

logger = logging.getLogger("ARK.LTL")


@dataclass
class LTLSpec:
    """Parsed LTL specification."""
    name: str
    formula: str
    invariants: List[str]
    allowed_patterns: List[str]
    forbidden_patterns: List[str]
    description: str = ""
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "formula": self.formula,
            "invariants": self.invariants,
            "allowed_patterns": self.allowed_patterns,
            "forbidden_patterns": self.forbidden_patterns,
            "description": self.description,
        }


class SpecLoader:
    """
    Loads and parses LTL specifications from .ark/specs/.
    
    Supports:
    - .json specs (native format)
    - .tla specs (simplified TLA+ subset)
    """
    
    def __init__(self, ark_path: Optional[Path] = None):
        self.ark_path = ark_path or Path.cwd() / ".ark"
        self.specs_path = self.ark_path / "specs"
        self._cache: Dict[str, LTLSpec] = {}
    
    def load(self, spec_ref: str) -> Optional[LTLSpec]:
        """Load a spec by reference (name or path)."""
        if spec_ref in self._cache:
            return self._cache[spec_ref]
        
        # Try JSON first
        json_path = self.specs_path / f"{spec_ref}.json"
        if json_path.exists():
            spec = self._load_json(json_path)
            self._cache[spec_ref] = spec
            return spec
        
        # Try TLA+
        tla_path = self.specs_path / f"{spec_ref}.tla"
        if tla_path.exists():
            spec = self._load_tla(tla_path)
            self._cache[spec_ref] = spec
            return spec
        
        logger.warning(f"Spec not found: {spec_ref}")
        return None
    
    def _load_json(self, path: Path) -> LTLSpec:
        """Load JSON spec."""
        data = json.loads(path.read_text())
        return LTLSpec(
            name=data.get("name", path.stem),
            formula=data.get("formula", ""),
            invariants=data.get("invariants", []),
            allowed_patterns=data.get("allowed_patterns", []),
            forbidden_patterns=data.get("forbidden_patterns", []),
            description=data.get("description", ""),
        )
    
    def _load_tla(self, path: Path) -> LTLSpec:
        """
        Load simplified TLA+ spec.
        
        Extracts:
        - INVARIANT declarations
        - FORBIDDEN patterns
        - ALLOWED patterns
        """
        content = path.read_text()
        
        invariants = re.findall(r'INVARIANT\s+(.+?)(?:\n|$)', content)
        forbidden = re.findall(r'FORBIDDEN\s+"(.+?)"', content)
        allowed = re.findall(r'ALLOWED\s+"(.+?)"', content)
        formula_match = re.search(r'FORMULA\s+"(.+?)"', content)
        
        return LTLSpec(
            name=path.stem,
            formula=formula_match.group(1) if formula_match else "",
            invariants=invariants,
            allowed_patterns=allowed,
            forbidden_patterns=forbidden,
            description=f"Loaded from {path.name}",
        )
    
    def list_specs(self) -> List[str]:
        """List available spec names."""
        if not self.specs_path.exists():
            return []
        specs = []
        for p in self.specs_path.iterdir():
            if p.suffix in [".json", ".tla"]:
                specs.append(p.stem)
        return specs


class LTLValidator:
    """
    Validates commits against LTL specifications.
    
    This is a simplified validator that checks:
    1. Invariants (patterns that must NOT be violated)
    2. Allowed/Forbidden patterns in diffs
    3. Purpose alignment with spec intent
    """
    
    def __init__(self, loader: Optional[SpecLoader] = None):
        self.loader = loader or SpecLoader()
    
    def validate(
        self,
        spec_ref: str,
        diff: str,
        purpose: str,
    ) -> tuple[bool, str]:
        """
        Validate a diff against a spec.
        
        Returns:
            (passed, reason)
        """
        spec = self.loader.load(spec_ref)
        if not spec:
            return True, "No spec found (allowing by default)"
        
        # Check forbidden patterns
        for pattern in spec.forbidden_patterns:
            if re.search(pattern, diff, re.IGNORECASE):
                return False, f"Forbidden pattern detected: {pattern}"
        
        # Check invariants (simplified: treat as required patterns NOT in diff)
        for inv in spec.invariants:
            # If invariant looks like "NoDeleting X" and diff contains "delete X"
            if self._invariant_violated(inv, diff):
                return False, f"Invariant violated: {inv}"
        
        # Check purpose alignment (semantic, simplified)
        if spec.description and purpose:
            if not self._purposes_align(purpose, spec.description):
                logger.debug(f"Purpose may not align: {purpose} vs {spec.description}")
                # Soft warning, don't fail
        
        return True, "Spec validation passed"
    
    def _invariant_violated(self, invariant: str, diff: str) -> bool:
        """
        Check if an invariant is violated by the diff.
        
        Simplified logic:
        - "NoDelete<X>" invariant is violated if diff contains "-.*X"
        - "Preserve<X>" invariant is violated if diff contains "-.*X" without "+.*X"
        """
        diff_lower = diff.lower()
        
        # NoDelete pattern
        no_delete_match = re.match(r'NoDelete\s*[<"](.*?)[>"]', invariant)
        if no_delete_match:
            target = no_delete_match.group(1).lower()
            if re.search(rf'^-.*{re.escape(target)}', diff_lower, re.MULTILINE):
                return True
        
        # Preserve pattern
        preserve_match = re.match(r'Preserve\s*[<"](.*?)[>"]', invariant)
        if preserve_match:
            target = preserve_match.group(1).lower()
            deleted = re.search(rf'^-.*{re.escape(target)}', diff_lower, re.MULTILINE)
            added = re.search(rf'^\+.*{re.escape(target)}', diff_lower, re.MULTILINE)
            if deleted and not added:
                return True
        
        return False
    
    def _purposes_align(self, commit_purpose: str, spec_purpose: str) -> bool:
        """
        Check if commit purpose aligns with spec purpose.
        
        Simplified: keyword overlap check.
        """
        commit_words = set(commit_purpose.lower().split())
        spec_words = set(spec_purpose.lower().split())
        overlap = commit_words & spec_words
        return len(overlap) >= 2  # At least 2 common words
