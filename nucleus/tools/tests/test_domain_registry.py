import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from domain_registry import normalize_domain, scan_workspace_domains  # noqa: E402


class TestDomainRegistry(unittest.TestCase):
    def test_normalize_domain_filters_versions_and_ips(self) -> None:
        self.assertIsNone(normalize_domain("0.0.0"))
        self.assertIsNone(normalize_domain("69.169.104.17"))
        self.assertIsNone(normalize_domain("127.0.0.1"))

        self.assertEqual(normalize_domain("avtr.you"), "avtr.you")
        self.assertEqual(normalize_domain("FITTW.IN"), "fittw.in")
        self.assertEqual(normalize_domain("*.example.com"), "example.com")

    def test_scan_reads_nucleus_ingresses_hosts(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            (root / ".pluribus").mkdir(parents=True, exist_ok=True)
            (root / "nucleus" / "docs" / "ingresses").mkdir(parents=True, exist_ok=True)
            (root / "nucleus" / "docs" / "ingresses" / "hosts.txt").write_text("a.example.com\nb.example.com\n", encoding="utf-8")

            found = scan_workspace_domains(root, include_bare=False, max_files=10, max_bytes=100_000)
            self.assertIn("a.example.com", found)
            self.assertIn("b.example.com", found)


if __name__ == "__main__":
    unittest.main()
