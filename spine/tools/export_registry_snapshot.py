#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Iterable


def iter_json_files(root: Path) -> Iterable[Path]:
    for p in root.rglob("*.json"):
        if p.name.endswith(".schema.json"):
            yield p
        else:
            yield p


def iter_ndjson_files(root: Path) -> Iterable[Path]:
    for p in root.rglob("*.ndjson"):
        yield p


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--registry", default=str(Path(__file__).resolve().parents[1] / "registry"))
    ap.add_argument("--out", default=str(Path.cwd() / "spine_registry_snapshot.ndjson"))
    args = ap.parse_args()

    reg_root = Path(args.registry)
    out_path = Path(args.out)

    if not reg_root.exists():
        raise SystemExit(f"registry root not found: {reg_root}")

    rows = 0
    with out_path.open("w", encoding="utf-8") as handle:
        for json_path in sorted(iter_json_files(reg_root)):
            payload = json.loads(json_path.read_text(encoding="utf-8"))
            record = {
                "kind": "registry.json",
                "path": str(json_path.relative_to(reg_root)),
                "payload": payload,
            }
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
            rows += 1

        for ndjson_path in sorted(iter_ndjson_files(reg_root)):
            for line in ndjson_path.read_text(encoding="utf-8").splitlines():
                if not line.strip():
                    continue
                try:
                    payload = json.loads(line)
                except json.JSONDecodeError:
                    payload = line
                record = {
                    "kind": "registry.ndjson",
                    "path": str(ndjson_path.relative_to(reg_root)),
                    "payload": payload,
                }
                handle.write(json.dumps(record, ensure_ascii=False) + "\n")
                rows += 1

    print(f"wrote {rows} records to {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
