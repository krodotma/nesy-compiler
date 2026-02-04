import json
import os
import sqlite3
from pathlib import Path

from nucleus.tools.meta_repo import MetaRepo, resolve_paths


def test_meta_repo_appends_and_indexes(tmp_path, monkeypatch):
    monkeypatch.setenv("PLURIBUS_META_ROOT", str(tmp_path))
    repo = MetaRepo()

    prompt_info = repo.store_object({"prompt": "hello"})
    repo.append_event(
        {
            "kind": "prompt",
            "topic": "dialogos.submit",
            "actor": "tester",
            "req_id": "req-1",
            "payload_ref": prompt_info["ref"],
            "summary": "hello",
        }
    )

    response_info = repo.store_object({"content": "world"})
    repo.append_event(
        {
            "kind": "response",
            "topic": "dialogos.cell.end",
            "actor": "tester",
            "req_id": "req-1",
            "payload_ref": response_info["ref"],
            "summary": "world",
        }
    )

    paths = resolve_paths(Path(tmp_path))
    ledger = paths.ledger_path
    assert ledger.exists()
    lines = ledger.read_text(encoding="utf-8").strip().splitlines()
    assert len(lines) == 2
    first = json.loads(lines[0])
    assert first["kind"] == "prompt"

    db = sqlite3.connect(str(paths.index_path))
    try:
        edge_count = db.execute("SELECT COUNT(*) FROM meta_edges").fetchone()[0]
        assert edge_count == 1
    finally:
        db.close()
