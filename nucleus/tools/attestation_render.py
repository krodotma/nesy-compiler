#!/usr/bin/env python3
"""
Attestation Render - Visual Header Generator

Creates beautiful, low-token attestation headers for Pluribus agents.
Supports both box format and compact single-line format.
"""

import os
import time
import uuid
from datetime import datetime, timezone

# Protocol versions (canonical source)
DKIN_VERSION = "v28"
PAIP_VERSION = "v15"
CITIZEN_VERSION = "v1"


def render_score_bar(score: int, width: int = 10) -> str:
    """Render a visual progress bar for attestation score."""
    filled = int((score / 100) * width)
    empty = width - filled
    bar = "█" * filled + "░" * empty
    if score < 50:
        bar += " ⚠"
    return bar


def render_attestation_box(
    agent_id: str,
    score: int = 100,
    session_id: str = None,
    verified_at: str = None
) -> str:
    """Render full box-style attestation header."""
    if verified_at is None:
        verified_at = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    if session_id is None:
        session_id = str(uuid.uuid4())[:8]
    
    bar = render_score_bar(score)
    
    return f"""┌─ PLURIBUS CITIZEN ─────────────────────────────┐
│ Agent: {agent_id:<12} Protocol: DKIN {DKIN_VERSION:<5}  │
│ PAIP: {PAIP_VERSION:<4} CITIZEN: {CITIZEN_VERSION:<3}  Score: {bar} {score:>3}% │
│ Verified: {verified_at[:19]:<19} Session: {session_id:<8}│
└────────────────────────────────────────────────┘"""


def render_attestation_compact(
    agent_id: str,
    score: int = 100,
    verified_at: str = None
) -> str:
    """Render single-line compact attestation."""
    if verified_at is None:
        verified_at = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    
    bar = render_score_bar(score, width=10)
    check = "✓" if score >= 80 else "⚠" if score >= 50 else "✗"
    
    return f"⟦PLURIBUS⟧ {agent_id} │ DKIN:{DKIN_VERSION} PAIP:{PAIP_VERSION} │ {bar} {score}% │ {check}{verified_at}"


def render_attestation(
    agent_id: str,
    score: int = 100,
    compact: bool = False,
    session_id: str = None,
    verified_at: str = None
) -> str:
    """Main entry point - render attestation header."""
    if compact:
        return render_attestation_compact(agent_id, score, verified_at)
    return render_attestation_box(agent_id, score, session_id, verified_at)


def get_cached_attestation(agent_id: str) -> tuple[int, str]:
    """Get cached attestation score and timestamp from last audit."""
    cache_path = f"/pluribus/.pluribus/cache/attestation_{agent_id}.json"
    try:
        import json
        with open(cache_path) as f:
            data = json.load(f)
            return data.get("score", 100), data.get("verified_at")
    except:
        return 100, None


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Render attestation header")
    parser.add_argument("agent", help="Agent ID")
    parser.add_argument("--score", type=int, default=100, help="Attestation score (0-100)")
    parser.add_argument("--compact", action="store_true", help="Single-line format")
    parser.add_argument("--session", help="Session ID")
    args = parser.parse_args()
    
    print(render_attestation(args.agent, args.score, args.compact, args.session))
