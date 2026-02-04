import json
import pathlib
import unittest


class TestSemopsToolPaths(unittest.TestCase):
    def test_semops_tool_paths_exist(self) -> None:
        root = pathlib.Path(__file__).resolve().parents[2]
        semops_path = root / "specs" / "semops.json"
        obj = json.loads(semops_path.read_text(encoding="utf-8"))
        missing = []

        for name, rel in (obj.get("tool_map") or {}).items():
            path = root.parent / rel
            if not path.exists():
                missing.append(f"tool_map:{name}:{rel}")

        for name, op in (obj.get("operators") or {}).items():
            rel = op.get("tool")
            if not rel:
                continue
            path = root.parent / rel
            if not path.exists():
                missing.append(f"operator:{name}:{rel}")

        if missing:
            msg = "Missing semops tool paths:\\n" + "\\n".join(sorted(missing))
            self.fail(msg)


if __name__ == "__main__":
    unittest.main()
