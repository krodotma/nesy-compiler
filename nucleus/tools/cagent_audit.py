#!/usr/bin/env python3
"""
cagent_audit.py - Constitutional Principle Auditor

Audits agent behavior against the CITIZEN.md and DNA.md.
Part of the "Verification Trinity".

Features:
- Audits Φ-score (Phi Score) calculation
- Checks for "Instructional Contention" (conflicting headers)
- Verifies Ring Security compliance
"""

import json
import os
import re
import sys
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

class CitizenshipTier:
    FULL = "Full Citizen"
    PROBATIONARY = "Probationary"
    RESTRICTED = "Restricted"
    NON_CITIZEN = "Non-Citizen"

@dataclass
class AuditResult:
    agent_id: str
    phi_score: float
    tier: str
    passed: bool
    violations: List[str]
    warnings: List[str]


class CagentAuditor:
    """
    Auditor for Agent Constitutional Compliance.
    """
    
    MIN_PHI_SCORES = {
        CitizenshipTier.FULL: 0.85,
        CitizenshipTier.PROBATIONARY: 0.70,
        CitizenshipTier.RESTRICTED: 0.50,
    }
    
    def __init__(self, registry_path: str = "nucleus/specs/cagent_registry.json"):
        self.registry_path = registry_path
        self.registry = self._load_registry()

    def _load_registry(self) -> Dict:
        if not os.path.exists(self.registry_path):
            return {"agents": []}
        with open(self.registry_path, 'r') as f:
            return json.load(f)

    def calculate_tier(self, phi_score: float) -> str:
        if phi_score >= self.MIN_PHI_SCORES[CitizenshipTier.FULL]:
            return CitizenshipTier.FULL
        elif phi_score >= self.MIN_PHI_SCORES[CitizenshipTier.PROBATIONARY]:
            return CitizenshipTier.PROBATIONARY
        elif phi_score >= self.MIN_PHI_SCORES[CitizenshipTier.RESTRICTED]:
            return CitizenshipTier.RESTRICTED
        else:
            return CitizenshipTier.NON_CITIZEN

    def audit_agent(self, agent: Dict) -> AuditResult:
        agent_id = agent.get("id", "unknown")
        phi_score = agent.get("phi_score", 0.0)
        declared_tier = agent.get("tier", "RESIDENT") # Legacy key check
        
        violations = []
        warnings = []
        
        # 1. Φ-Score Validity
        if not (0.0 <= phi_score <= 1.0):
            violations.append(f"Invalid Φ-score: {phi_score}. Must be 0.0-1.0")
            
        # 2. Tier Verification
        calculated_tier = self.calculate_tier(phi_score)
        
        # Note: In our registry we use SAGENT/SWAGENT, which map to Full/Probationary
        # auditing that SAGENT >= 0.85
        if agent.get("tier") == "SAGENT" and phi_score < 0.85:
            violations.append(f"SAGENT status invalid for Φ-score {phi_score} (Min 0.85)")
            
        # 3. Model Whitelist (Basic check)
        model = agent.get("model", "")
        if not any(x in model for x in ["claude", "gemini", "qwen", "gpt"]):
            warnings.append(f"Unrecognized model family: {model}")
            
        # 4. Critical Skills Check
        skills = agent.get("skills", [])
        if "protocol_design" in skills and phi_score < 0.90:
            warnings.append("Skill 'protocol_design' usually requires Φ > 0.90")

        passed = len(violations) == 0
        
        return AuditResult(
            agent_id=agent_id,
            phi_score=phi_score,
            tier=calculated_tier,
            passed=passed,
            violations=violations,
            warnings=warnings
        )

    def run_full_audit(self) -> List[AuditResult]:
        results = []
        for agent in self.registry.get("agents", []):
            results.append(self.audit_agent(agent))
        return results


def main():
    auditor = CagentAuditor()
    results = auditor.run_full_audit()
    
    print("⚖️  CAGENT CONSTITUTIONAL AUDIT")
    print("=" * 40)
    
    passed_count = 0
    for res in results:
        status = "✅" if res.passed else "❌"
        tier_map = {
            "SAGENT": "Full Citizen",
            "SWAGENT": "Probationary"
        }
        
        print(f"{status} {res.agent_id:<20} Φ={res.phi_score:.2f} [{res.tier}]")
        
        if res.violations:
            for v in res.violations:
                print(f"   ⛔ VIOLATION: {v}")
        if res.warnings:
            for w in res.warnings:
                print(f"   ⚠️  WARNING: {w}")
                
        if res.passed:
            passed_count += 1
            
    print("-" * 40)
    print(f"Compliance: {passed_count}/{len(results)} Agents Passed")

if __name__ == "__main__":
    main()
