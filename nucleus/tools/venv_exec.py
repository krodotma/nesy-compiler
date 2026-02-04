#!/usr/bin/env python3
from __future__ import annotations

import os
import sys
from pathlib import Path


def _find_repo_root(start: Path) -> Path:
    cur = start.resolve()
    for cand in [cur, *cur.parents]:
        if (cand / "nucleus").exists():
            return cand
    return cur


def find_realagents_python() -> Path | None:
    override = (os.environ.get("PLURIBUS_REALAGENTS_PYTHON") or "").strip()
    if override:
        p = Path(override).expanduser()
        return p if p.exists() else None

    repo_root = _find_repo_root(Path(__file__).resolve())
    cand = repo_root / ".pluribus" / "venv" / "realagents" / "bin" / "python"
    return cand if cand.exists() else None


def maybe_reexec_in_realagents_venv() -> None:
    venv_python = find_realagents_python()
    if not venv_python:
        return
    try:
        if Path(sys.executable).resolve() == venv_python.resolve():
            return
    except Exception:
        if str(sys.executable) == str(venv_python):
            return
    os.execv(str(venv_python), [str(venv_python), *sys.argv])

