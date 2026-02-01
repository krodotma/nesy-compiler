import json
from pathlib import Path

from spine_registry_client import index_spine_records, load_spine_records


def test_spine_registry_client_reads_snapshot(tmp_path: Path) -> None:
    snapshot = tmp_path / "spine_registry_snapshot.ndjson"
    entries = [
        {"kind": "registry.json", "path": "specs/subsystem_registry.json", "payload": {"services": []}},
        {"kind": "registry.ndjson", "path": "runtime/skills_registry.ndjson", "payload": {"name": "skill"}},
    ]
    snapshot.write_text("\n".join(json.dumps(e) for e in entries) + "\n", encoding="utf-8")

    records = load_spine_records(str(snapshot))
    assert len(records) == 2
    assert records[0].path == "specs/subsystem_registry.json"

    index = index_spine_records(str(snapshot))
    assert "specs/subsystem_registry.json" in index
    assert "runtime/skills_registry.ndjson" in index
