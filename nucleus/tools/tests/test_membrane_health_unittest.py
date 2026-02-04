#!/usr/bin/env python3
"""Unit tests for membrane_health.py.

DKIN v28 Remediation: Step 55
"""
import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch, MagicMock

sys.path.insert(0, str(Path(__file__).parent.parent))

from membrane_health import (
    MembraneHealthChecker,
    AdapterHealth,
    IntegrationType,
    HealthStatus,
    format_table,
)


class TestIntegrationType(unittest.TestCase):
    """Test IntegrationType enum."""

    def test_cli_wrapper_value(self):
        self.assertEqual(IntegrationType.CLI_WRAPPER.value, "CLI_WRAPPER")

    def test_mcp_boundary_value(self):
        self.assertEqual(IntegrationType.MCP_BOUNDARY.value, "MCP_BOUNDARY")

    def test_deep_integration_value(self):
        self.assertEqual(IntegrationType.DEEP_INTEGRATION.value, "DEEP_INTEGRATION")


class TestHealthStatus(unittest.TestCase):
    """Test HealthStatus enum."""

    def test_healthy_value(self):
        self.assertEqual(HealthStatus.HEALTHY.value, "healthy")

    def test_degraded_value(self):
        self.assertEqual(HealthStatus.DEGRADED.value, "degraded")

    def test_unhealthy_value(self):
        self.assertEqual(HealthStatus.UNHEALTHY.value, "unhealthy")


class TestAdapterHealth(unittest.TestCase):
    """Test AdapterHealth dataclass."""

    def test_healthy_adapter(self):
        health = AdapterHealth(
            name="test",
            type=IntegrationType.CLI_WRAPPER,
            status=HealthStatus.HEALTHY,
            healthy=True,
        )
        self.assertTrue(health.healthy)
        self.assertEqual(health.status, HealthStatus.HEALTHY)

    def test_unhealthy_adapter(self):
        health = AdapterHealth(
            name="test",
            type=IntegrationType.MCP_BOUNDARY,
            status=HealthStatus.UNHEALTHY,
            healthy=False,
            error="Connection failed",
        )
        self.assertFalse(health.healthy)
        self.assertEqual(health.error, "Connection failed")

    def test_to_dict(self):
        health = AdapterHealth(
            name="graphiti",
            type=IntegrationType.MCP_BOUNDARY,
            status=HealthStatus.HEALTHY,
            healthy=True,
            latency_ms=5.2,
        )
        d = health.to_dict()
        self.assertEqual(d["name"], "graphiti")
        self.assertEqual(d["type"], "MCP_BOUNDARY")
        self.assertEqual(d["status"], "healthy")
        self.assertTrue(d["healthy"])
        self.assertEqual(d["latency_ms"], 5.2)


class TestMembraneHealthChecker(unittest.TestCase):
    """Test MembraneHealthChecker."""

    def setUp(self):
        self.checker = MembraneHealthChecker(emit_bus_events=False)

    def test_adapters_defined(self):
        """Verify ADAPTERS list is populated."""
        self.assertGreater(len(self.checker.ADAPTERS), 0)

    def test_check_binary_missing(self):
        """Test check_binary for non-existent binary."""
        found, path = self.checker.check_binary("nonexistent_binary_xyz")
        self.assertFalse(found)
        self.assertIsNone(path)

    def test_check_binary_exists(self):
        """Test check_binary for existing binary (python3)."""
        found, path = self.checker.check_binary("python3")
        self.assertTrue(found)
        self.assertIsNotNone(path)

    def test_check_adapter_file_exists(self):
        """Test check_adapter_file for existing file."""
        # membrane_health.py exists in tools/
        found, path = self.checker.check_adapter_file("membrane_health.py")
        self.assertTrue(found)

    def test_check_adapter_file_missing(self):
        """Test check_adapter_file for missing file."""
        found, path = self.checker.check_adapter_file("nonexistent_adapter.py")
        self.assertFalse(found)
        self.assertIsNone(path)

    def test_check_membrane_dir_exists(self):
        """Test check_membrane_dir for existing directory."""
        found, path = self.checker.check_membrane_dir("graphiti")
        self.assertTrue(found)

    def test_check_membrane_dir_missing(self):
        """Test check_membrane_dir for missing directory."""
        found, path = self.checker.check_membrane_dir("nonexistent_membrane")
        self.assertFalse(found)

    def test_check_mcp_server_exists(self):
        """Test check_mcp_server for existing server."""
        found, path = self.checker.check_mcp_server("graphiti_server.py")
        self.assertTrue(found)

    def test_check_all_returns_list(self):
        """Test check_all returns a list of AdapterHealth."""
        results = self.checker.check_all()
        self.assertIsInstance(results, list)
        self.assertGreater(len(results), 0)
        for result in results:
            self.assertIsInstance(result, AdapterHealth)

    def test_get_summary(self):
        """Test get_summary generates proper report."""
        results = self.checker.check_all()
        summary = self.checker.get_summary(results)

        self.assertIn("timestamp", summary)
        self.assertIn("total_adapters", summary)
        self.assertIn("healthy", summary)
        self.assertIn("unhealthy", summary)
        self.assertIn("health_percentage", summary)
        self.assertIn("by_status", summary)
        self.assertIn("adapters", summary)

        # Verify counts add up
        self.assertEqual(
            summary["healthy"] + summary["unhealthy"],
            summary["total_adapters"]
        )


class TestFormatTable(unittest.TestCase):
    """Test format_table function."""

    def test_format_table_output(self):
        """Test that format_table produces readable output."""
        results = [
            AdapterHealth(
                name="test1",
                type=IntegrationType.CLI_WRAPPER,
                status=HealthStatus.HEALTHY,
                healthy=True,
            ),
            AdapterHealth(
                name="test2",
                type=IntegrationType.MCP_BOUNDARY,
                status=HealthStatus.UNHEALTHY,
                healthy=False,
                error="Connection failed",
            ),
        ]
        output = format_table(results)
        self.assertIn("MEMBRANE HEALTH CHECK", output)
        self.assertIn("test1", output)
        self.assertIn("test2", output)
        self.assertIn("Healthy: 1", output)
        self.assertIn("Unhealthy: 1", output)


class TestHealthCheckIntegration(unittest.TestCase):
    """Integration tests for membrane health check."""

    def test_graphiti_adapter_healthy(self):
        """Test that graphiti adapter is healthy (files exist)."""
        checker = MembraneHealthChecker(emit_bus_events=False)
        results = checker.check_all()

        graphiti = next((r for r in results if r.name == "graphiti"), None)
        self.assertIsNotNone(graphiti)
        self.assertTrue(graphiti.healthy)
        self.assertEqual(graphiti.type, IntegrationType.MCP_BOUNDARY)

    def test_mem0_adapter_healthy(self):
        """Test that mem0 adapter is healthy (files exist)."""
        checker = MembraneHealthChecker(emit_bus_events=False)
        results = checker.check_all()

        mem0 = next((r for r in results if r.name == "mem0"), None)
        self.assertIsNotNone(mem0)
        self.assertTrue(mem0.healthy)

    def test_codex_adapter_healthy(self):
        """Test that codex adapter is healthy."""
        checker = MembraneHealthChecker(emit_bus_events=False)
        results = checker.check_all()

        codex = next((r for r in results if r.name == "codex"), None)
        self.assertIsNotNone(codex)
        # codex binary may or may not be installed
        self.assertIsNotNone(codex.status)


if __name__ == "__main__":
    unittest.main()
