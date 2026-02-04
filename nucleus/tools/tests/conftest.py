#!/usr/bin/env python3
"""Pytest configuration and shared fixtures for nucleus tests."""

import os
import sys
import tempfile
from pathlib import Path

import pytest

# Add tools directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))


@pytest.fixture
def temp_bus_dir():
    """Create a temporary bus directory for test isolation."""
    with tempfile.TemporaryDirectory(prefix="pluribus_test_bus_") as tmpdir:
        bus_dir = Path(tmpdir)
        (bus_dir / "events.ndjson").touch()
        old_env = os.environ.get("PLURIBUS_BUS_DIR")
        os.environ["PLURIBUS_BUS_DIR"] = str(bus_dir)
        yield bus_dir
        if old_env is not None:
            os.environ["PLURIBUS_BUS_DIR"] = old_env
        else:
            os.environ.pop("PLURIBUS_BUS_DIR", None)


@pytest.fixture
def temp_rhizome_dir():
    """Create a temporary rhizome directory for test isolation."""
    with tempfile.TemporaryDirectory(prefix="pluribus_test_rhizome_") as tmpdir:
        root = Path(tmpdir)
        pluribus_dir = root / ".pluribus"
        pluribus_dir.mkdir()
        (pluribus_dir / "rhizome.json").write_text('{"name":"test","purpose":"testing"}')
        yield root


@pytest.fixture
def mock_provider_env(monkeypatch):
    """Set up mock provider environment variables."""
    monkeypatch.setenv("PLURIBUS_ACTOR", "test-actor")
    monkeypatch.setenv("PLURIBUS_FLOW_MODE", "m")


# Coverage configuration for pytest-cov
def pytest_configure(config):
    """Configure pytest with coverage thresholds."""
    config.addinivalue_line(
        "markers", "slow: marks tests as slow (deselect with '-m \"not slow\"')"
    )
    config.addinivalue_line(
        "markers", "integration: marks tests as integration tests"
    )
    config.addinivalue_line(
        "markers", "e2e: marks tests as end-to-end tests"
    )
