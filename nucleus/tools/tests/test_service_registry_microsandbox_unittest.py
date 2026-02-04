#!/usr/bin/env python3
from __future__ import annotations

import pathlib
import sys
import tempfile
import unittest

TOOLS_DIR = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(TOOLS_DIR))

import service_registry  # noqa: E402


class TestServiceRegistryMicrosandbox(unittest.TestCase):
    def test_builtin_contains_microsandbox(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = pathlib.Path(tmp)
            reg = service_registry.ServiceRegistry(root)
            reg.init()
            reg.load()
            svc = reg.get_service("microsandbox")
            self.assertIsNotNone(svc)
            self.assertEqual(svc.id, "microsandbox")
            self.assertEqual(svc.kind, "port")
            self.assertEqual(svc.port, 8300)


if __name__ == "__main__":
    unittest.main()
