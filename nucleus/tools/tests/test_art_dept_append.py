import json
import os
import pathlib
import sys
import tempfile
import unittest

TOOLS_DIR = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(TOOLS_DIR))

import art_dept_append  # noqa: E402


class TestArtDeptAppend(unittest.TestCase):
    def setUp(self):
        self._old_bus = os.environ.get("PLURIBUS_BUS_DIR")
        self._old_actor = os.environ.get("PLURIBUS_ACTOR")
        self._tmp_bus = tempfile.TemporaryDirectory(prefix="pluribus_test_bus_")
        bus_dir = pathlib.Path(self._tmp_bus.name)
        (bus_dir / "events.ndjson").touch()
        os.environ["PLURIBUS_BUS_DIR"] = str(bus_dir)
        os.environ["PLURIBUS_ACTOR"] = "test-actor"

    def tearDown(self):
        self._tmp_bus.cleanup()
        if self._old_bus is not None:
            os.environ["PLURIBUS_BUS_DIR"] = self._old_bus
        else:
            os.environ.pop("PLURIBUS_BUS_DIR", None)
        if self._old_actor is not None:
            os.environ["PLURIBUS_ACTOR"] = self._old_actor
        else:
            os.environ.pop("PLURIBUS_ACTOR", None)

    def test_append_source_writes_one_line(self):
        with tempfile.TemporaryDirectory(prefix="pluribus_art_dept_") as tmp:
            tmp_path = pathlib.Path(tmp)
            sources = tmp_path / "sources.ndjson"
            rc = art_dept_append.main(
                [
                    "--root",
                    str(tmp_path),
                    "--sources-path",
                    str(sources),
                    "source",
                    "--kind",
                    "reference",
                    "--title",
                    "Example Source",
                    "--url",
                    "https://example.invalid/",
                    "--license",
                    "test",
                    "--tags",
                    "a",
                    "b",
                    "--notes",
                    "hello",
                ]
            )
            self.assertEqual(rc, 0)
            lines = sources.read_text(encoding="utf-8").splitlines()
            self.assertEqual(len(lines), 1)
            obj = json.loads(lines[0])
            self.assertEqual(obj["kind"], "reference")
            self.assertEqual(obj["title"], "Example Source")
            self.assertEqual(obj["ingest"]["status"], "unfetched")

    def test_append_artifact_writes_one_line(self):
        with tempfile.TemporaryDirectory(prefix="pluribus_art_dept_") as tmp:
            tmp_path = pathlib.Path(tmp)
            artifacts = tmp_path / "genomes.ndjson"
            rc = art_dept_append.main(
                [
                    "--root",
                    str(tmp_path),
                    "--artifacts-path",
                    str(artifacts),
                    "artifact",
                    "--type",
                    "shader",
                    "--engine",
                    "webgl2",
                    "--name",
                    "test.shader",
                    "--source-refs",
                    "seed-1",
                    "seed-2",
                    "--code-path",
                    "x/y/z.tsx",
                    "--bus-bindings",
                    '{"entropy":"x","rate":"y"}',
                ]
            )
            self.assertEqual(rc, 0)
            lines = artifacts.read_text(encoding="utf-8").splitlines()
            self.assertEqual(len(lines), 1)
            obj = json.loads(lines[0])
            self.assertEqual(obj["type"], "shader")
            self.assertEqual(obj["engine"], "webgl2")
            self.assertEqual(obj["source_refs"], ["seed-1", "seed-2"])
            self.assertEqual(obj["bus_bindings"]["entropy"], "x")


if __name__ == "__main__":
    unittest.main()

