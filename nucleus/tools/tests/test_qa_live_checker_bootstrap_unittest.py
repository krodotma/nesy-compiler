import sys
from pathlib import Path

import qa_live_checker


def test_ensure_pluribus_root_on_path_prepends():
    original = list(sys.path)
    try:
        sys.path = [p for p in sys.path if p != "/pluribus"]
        qa_live_checker.ensure_pluribus_root_on_path(Path("/pluribus"))
        assert sys.path[0] == "/pluribus"
    finally:
        sys.path = original


def test_import_cagent_recovers_with_root_on_path(monkeypatch):
    original = list(sys.path)
    try:
        sys.path = [p for p in sys.path if p != "/pluribus"]
        monkeypatch.setenv("PLURIBUS_ROOT", "/pluribus")
        Cagent = qa_live_checker.import_cagent()
        assert Cagent.__name__ == "Cagent"
    finally:
        sys.path = original
