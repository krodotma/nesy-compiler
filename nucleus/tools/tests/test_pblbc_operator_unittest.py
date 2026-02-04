import unittest

from nucleus.tools import pblbc_operator


class TestPBLBCOperator(unittest.TestCase):
    def test_human_bytes(self) -> None:
        self.assertEqual(pblbc_operator.human_bytes(0), "0B")
        self.assertEqual(pblbc_operator.human_bytes(1024), "1KB")

    def test_render_report_sections(self) -> None:
        report = {
            "generated_iso": "2025-12-26T00:00:00Z",
            "totals": {"active_human": "1KB", "expected_reduction_human": "0B"},
            "policy": {"path": "/tmp/policy.json", "updated_iso": "2025-12-26T00:00:00Z"},
            "sizes": [
                {"label": "Test", "human": "1KB", "path": "/tmp/test.log"}
            ],
            "hygiene": {
                "recent": [
                    {"topic": "operator.pblbc.report", "iso": "2025-12-26T00:00:00Z", "actor": "tester"}
                ],
                "next": [
                    {"title": "QA backups keep_last=5", "expected_reduction_human": "0B"}
                ],
            },
        }
        rendered = pblbc_operator.render_report(report)
        self.assertIn("PBLBC report @ 2025-12-26T00:00:00Z", rendered)
        self.assertIn("SIZES", rendered)
        self.assertIn("HYGIENE_RECENT", rendered)
        self.assertIn("HYGIENE_NEXT", rendered)


if __name__ == "__main__":
    unittest.main()
