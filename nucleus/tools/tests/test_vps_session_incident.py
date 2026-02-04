#!/usr/bin/env python3
from __future__ import annotations

import time
from pathlib import Path

from vps_session import VPSSessionManager


def write_session(root: Path, obj: dict) -> None:
    p = root / ".pluribus" / "vps_session.json"
    p.write_text(__import__("json").dumps(obj, indent=2), encoding="utf-8")


def test_determine_active_fallback_respects_cooldown(temp_rhizome_dir: Path, monkeypatch) -> None:
    # Use web-session provider names that match current VPSSession architecture
    root = temp_rhizome_dir
    now = time.time()
    write_session(
        root,
        {
            "flow_mode": "m",
            "providers": {
                "chatgpt-web": {"available": False, "last_check": "", "error": None, "quota_remaining": None, "model": None},
                "claude-web": {"available": True, "last_check": "", "error": None, "quota_remaining": None, "model": None},
                "gemini-web": {"available": True, "last_check": "", "error": None, "quota_remaining": None, "model": None},
            },
            "fallback_order": ["claude-web", "gemini-web"],
            "active_fallback": None,
            "provider_cooldowns": {"claude-web": now + 300},
        },
    )

    # claude-web is on cooldown, so gemini-web should be selected
    monkeypatch.setenv("PLURIBUS_PROVIDER_PROFILE", "web-only")
    mgr = VPSSessionManager(root)
    active = mgr.determine_active_fallback()
    assert active == "gemini-web"


def test_apply_provider_incident_sets_cooldown(temp_rhizome_dir: Path, monkeypatch) -> None:
    # Use web-session provider names that match current VPSSession architecture
    root = temp_rhizome_dir
    write_session(
        root,
        {
            "flow_mode": "m",
            "providers": {
                "chatgpt-web": {"available": False, "last_check": "", "error": None, "quota_remaining": None, "model": None},
                "claude-web": {"available": True, "last_check": "", "error": None, "quota_remaining": None, "model": None},
                "gemini-web": {"available": True, "last_check": "", "error": None, "quota_remaining": None, "model": None},
            },
            "fallback_order": ["claude-web", "gemini-web"],
            "active_fallback": None,
            "provider_cooldowns": {},
        },
    )

    mgr = VPSSessionManager(root)
    t0 = time.time()
    # apply_provider_incident should handle web-session names directly
    mgr.apply_provider_incident("claude-web", available=False, error="overloaded", cooldown_s=120)
    session = mgr.load()
    assert session.providers["claude-web"].available is False
    assert (session.providers["claude-web"].error or "").startswith("overloaded")
    assert session.provider_cooldowns["claude-web"] >= t0 + 100

