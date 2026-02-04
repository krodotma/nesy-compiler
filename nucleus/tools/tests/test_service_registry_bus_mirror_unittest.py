#!/usr/bin/env python3
from __future__ import annotations

import pathlib
import sys
import tempfile
import unittest

TOOLS_DIR = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(TOOLS_DIR))

import service_registry  # noqa: E402


class TestServiceRegistryBusMirror(unittest.TestCase):
    def test_builtin_contains_bus_mirror_daemon(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = pathlib.Path(tmp)
            reg = service_registry.ServiceRegistry(root)
            reg.init()
            reg.load()
            svc = reg.get_service("bus-mirror-daemon")
            self.assertIsNotNone(svc)
            self.assertEqual(svc.id, "bus-mirror-daemon")
            self.assertEqual(svc.kind, "process")
            self.assertEqual(svc.entry_point, "nucleus/tools/bus_mirror_daemon.py")

    def test_builtin_contains_reverse_bus_mirror_daemon(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = pathlib.Path(tmp)
            reg = service_registry.ServiceRegistry(root)
            reg.init()
            reg.load()
            svc = reg.get_service("bus-mirror-daemon-reverse")
            self.assertIsNotNone(svc)
            self.assertEqual(svc.id, "bus-mirror-daemon-reverse")
            self.assertEqual(svc.kind, "process")
            self.assertEqual(svc.entry_point, "nucleus/tools/bus_mirror_daemon.py")


if __name__ == "__main__":
    unittest.main()
