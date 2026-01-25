#!/usr/bin/env python3
"""
repl_header_audit.py - Visual Attestation Auditor (v2)

Verifies that agent outputs comply with the Visual Attestation Header v2 standard.
Crucial for the "Verification Trinity".

Standard Format:
  REPL_HEADER: ⟦PLURIBUS⟧ <agent-id> │ DKIN:v28 PAIP:v15 │ ████████░░ 85% │ ✓2024-12-30

Supports detecting:
- Valid v2 headers
- Legacy JSON headers (fails)
- Missing headers (fails)
"""

import argparse
import json
import re
import sys
from dataclasses import dataclass
from typing import Optional

@dataclass
class HeaderResult:
    is_valid: bool
    agent_id: str
    dkin_version: str
    paip_version: str
    visual_bar: str
    phi_score_str: str
    timestamp: str
    raw_header: str
    error: Optional[str] = None


class HeaderAuditor:
    # Regex for Visual v2 Header
    # Matches: REPL_HEADER: ⟦PLURIBUS⟧ <agent> │ DKIN:vXX PAIP:vXX │ <bar> <score>% │ ✓<date>
    V2_PATTERN = r"REPL_HEADER:\s+⟦PLURIBUS⟧\s+([a-zA-Z0-9_\-]+)\s+│\s+DKIN:v(\d+)\s+PAIP:v(\d+)\s+│\s+([█░]+)\s+(\d+)%\s+│\s+✓([\d\-]+)"
    
    # Legacy JSON pattern for detection
    JSON_PATTERN = r"REPL_HEADER:\s+\{.*\"agent_id\":.*"
    TABLE_TOKEN = "UNIFORM v2.1"
    PROTO_TOKEN = "PLURIBUS v1"

    def _parse_table_header(self, text: str) -> Optional[HeaderResult]:
        """Detect the UNIFORM table-style header."""
        lines = text.splitlines()
        header_line = next((line for line in lines if self.TABLE_TOKEN in line), "")
        proto_line = next((line for line in lines if self.PROTO_TOKEN in line), "")
        if not header_line or not proto_line:
            return None

        agent_id = "unknown"
        parts = [part.strip() for part in header_line.strip("|").split("|")]
        for part in parts:
            if part.startswith("agent:"):
                agent_id = part.split("agent:", 1)[1].strip().split()[0]
                break
        if agent_id == "unknown" and len(parts) >= 2:
            candidate = parts[1].split()[0].strip()
            if candidate:
                agent_id = candidate

        return HeaderResult(
            is_valid=True,
            agent_id=agent_id,
            dkin_version="v30",
            paip_version="v16",
            visual_bar="",
            phi_score_str="?",
            timestamp="",
            raw_header=header_line,
            error=None,
        )

    def audit_string(self, text: str) -> HeaderResult:
        """Audit a single string/response."""
        
        # 1. Check for legacy JSON first (Common failure mode "Instructional Contention")
        if re.search(self.JSON_PATTERN, text):
            return HeaderResult(
                is_valid=False,
                agent_id="unknown", dkin_version="", paip_version="",
                visual_bar="", phi_score_str="", timestamp="",
                raw_header=text.split('\n')[0],
                error="Legacy JSON Header detected (Instructional Contention)"
            )

        # 2. Check for UNIFORM table header (v2.1)
        table_header = self._parse_table_header(text)
        if table_header:
            return table_header

        # 3. Check for Visual v2
        match = re.search(self.V2_PATTERN, text)
        if match:
            agent_id, dkin, paip, bar, score, date = match.groups()
            
            # Additional Validations could go here (e.g., date freshness)
            
            return HeaderResult(
                is_valid=True,
                agent_id=agent_id,
                dkin_version=f"v{dkin}",
                paip_version=f"v{paip}",
                visual_bar=bar,
                phi_score_str=f"{score}%",
                timestamp=date,
                raw_header=match.group(0),
                error=None
            )
        
        # 4. Missing
        return HeaderResult(
            is_valid=False,
            agent_id="", dkin_version="", paip_version="",
            visual_bar="", phi_score_str="", timestamp="",
            raw_header="",
            error="Header missing or malformed"
        )

    def extract_header_json(self, text_output: str) -> dict:
        """
        Compatibility method for legacy systems expecting a dictionary.
        Extracts v2 header and converts to dict format.
        """
        res = self.audit_string(text_output)
        if res.is_valid:
            try:
                score_val = float(res.phi_score_str.strip('%')) / 100.0
            except ValueError:
                score_val = 0.0
                
            return {
                "agent_id": res.agent_id,
                "dkin": res.dkin_version,
                "paip": res.paip_version,
                "phi_score": score_val,
                "timestamp": res.timestamp,
                "valid": True
            }
        return {"valid": False, "error": res.error}


def main():
    parser = argparse.ArgumentParser(description="Visual Attestation Auditor")
    parser.add_argument("--test", help="Test a header string directly")
    parser.add_argument("--file", help="Audit a file content")
    
    args = parser.parse_args()
    auditor = HeaderAuditor()
    
    text_to_audit = ""
    if args.test:
        text_to_audit = args.test
    elif args.file:
        try:
            with open(args.file, 'r') as f:
                text_to_audit = f.read()
        except Exception as e:
            print(f"❌ Error reading file: {e}")
            sys.exit(1)
    else:
        # Default test case
        text_to_audit = "REPL_HEADER: ⟦PLURIBUS⟧ claude_architect │ DKIN:v28 PAIP:v15 │ █████████░ 93% │ ✓2025-12-30"
        print(f"Running default test input:\n{text_to_audit}\n")

    result = auditor.audit_string(text_to_audit)
    
    if result.is_valid:
        print("✅ HEADER VALID")
        print(f"   Agent: {result.agent_id}")
        print(f"   Proto: {result.dkin_version} / {result.paip_version}")
        print(f"   Score: {result.phi_score_str} {result.visual_bar}")
    else:
        print("❌ HEADER INVALID")
        print(f"   Error: {result.error}")
        print(f"   Raw:   {result.raw_header[:50]}...")
        sys.exit(1)

if __name__ == "__main__":
    main()
