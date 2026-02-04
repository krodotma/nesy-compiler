import pathlib
import sys
import tempfile
import unittest

from pathlib import Path

TOOLS_DIR = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(TOOLS_DIR))

import ckin_report  # noqa: E402


class TestCKINReport(unittest.TestCase):
    def test_analyze_beam_parses_entry_headers_and_fields(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            beam = Path(td) / "beam.md"
            beam.write_text(
                "\n".join(
                    [
                        "# BEAM 10× Discourse Buffer (append-only)",
                        "",
                        "## Entry 0000 — Seed (Codex) — 2025-12-15T06:27:25Z",
                        "",
                        "iteration: 0",
                        "subagent_id: codex-root",
                        "actor: codex-cli",
                        "scope: other",
                        "tags: [I]",
                        "",
                        "## Entry 59b3068e-fa0d-4505-b968-36a808b2c8f7 — codex-cli — 2025-12-15T06:28:36Z",
                        "",
                        "iteration: 1",
                        "subagent_id: persona-4-topologist",
                        "actor: claude-opus",
                        "scope: iso_git",
                        "tags: [V, G]",
                        "",
                    ]
                ),
                encoding="utf-8",
            )

            stats = ckin_report.analyze_beam(beam)
            self.assertEqual(stats.total_entries, 2)
            self.assertEqual(stats.iteration_counts.get(0), 1)
            self.assertEqual(stats.iteration_counts.get(1), 1)
            self.assertEqual(stats.entries_by_agent.get("codex"), 1)
            self.assertEqual(stats.entries_by_agent.get("claude"), 1)
            self.assertEqual(stats.entries_by_tag.get("I"), 1)
            self.assertEqual(stats.entries_by_tag.get("V"), 1)
            self.assertEqual(stats.entries_by_tag.get("G"), 1)
            self.assertTrue(stats.recent_entries[-1]["id"].startswith("59b3068e-"))
            self.assertEqual(stats.recent_entries[-1]["scope"], "iso_git")

    def test_generate_ckin_report_includes_protocol_version(self) -> None:
        txt = ckin_report.generate_ckin_report(agent_name="test-agent", verbose=True)
        self.assertIn(f"CKIN DASHBOARD v{ckin_report.CKIN_PROTOCOL_VERSION}", txt)
        self.assertIn("FORENSICS INDEX (PBDEEP)", txt)

    def test_detect_mcp_official_interop_from_fake_repo(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            (root / "nucleus" / "mcp" / "compat").mkdir(parents=True)
            (root / "nucleus" / "tools" / "tests").mkdir(parents=True)
            (root / "nucleus" / "third_party").mkdir(parents=True)

            (root / "package.json").write_text(
                '{"devDependencies":{"@modelcontextprotocol/sdk":"1.25.0"}}\n', encoding="utf-8"
            )
            (root / "nucleus" / "mcp" / "compat" / "mcp_typescript_sdk_smoke.mjs").write_text(
                "// harness\n", encoding="utf-8"
            )
            (root / "nucleus" / "tools" / "tests" / "test_mcp_official_typescript_sdk_compat_unittest.py").write_text(
                "# test\n", encoding="utf-8"
            )
            (root / "nucleus" / "third_party" / "mcp_typescript_sdk.md").write_text(
                "# docs\n", encoding="utf-8"
            )

            bus = root / ".pluribus" / "bus" / "events.ndjson"
            bus.parent.mkdir(parents=True, exist_ok=True)
            bus.write_text(
                '{"ts":1,"iso":"1970-01-01T00:00:01Z","topic":"mcp.official_sdk.interop.announced","kind":"artifact","actor":"x","data":{"req_id":"r1"}}\n',
                encoding="utf-8",
            )

            status = ckin_report.detect_mcp_official_interop(root, bus)
            self.assertEqual(status["sdk"]["version"], "1.25.0")
            self.assertTrue(status["sdk"]["pinned"])
            self.assertTrue(status["artifacts"]["harness_exists"])
            self.assertTrue(status["artifacts"]["test_exists"])
            self.assertTrue(status["artifacts"]["docs_exists"])
            self.assertEqual(status["bus_evidence"]["req_id"], "r1")
