#!/usr/bin/env python3
from __future__ import annotations

import argparse
import getpass
import hashlib
import json
import os
import sqlite3
import subprocess
import sys
import time
import uuid
from pathlib import Path

sys.dont_write_bytecode = True

from embeddings import DEFAULT_DIM, embed_text, maybe_add_pluribus_venv_site

maybe_add_pluribus_venv_site()

EMBEDDING_DIM = int(DEFAULT_DIM)

_SQLITE_VEC_AVAILABLE = False
_SQLITE_VEC = None
try:
    import sqlite_vec as _sqlite_vec  # type: ignore

    _SQLITE_VEC_AVAILABLE = True
    _SQLITE_VEC = _sqlite_vec
except Exception:
    _SQLITE_VEC_AVAILABLE = False
    _SQLITE_VEC = None


def try_load_vec_extension(con: sqlite3.Connection) -> bool:
    if not _SQLITE_VEC_AVAILABLE or _SQLITE_VEC is None:
        return False
    try:
        con.enable_load_extension(True)
        _SQLITE_VEC.load(con)
        return True
    except Exception:
        return False
    finally:
        try:
            con.enable_load_extension(False)
        except Exception:
            pass

def now_iso_utc() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def default_actor() -> str:
    return os.environ.get("PLURIBUS_ACTOR") or os.environ.get("USER") or getpass.getuser()


def find_rhizome_root(start: Path) -> Path | None:
    cur = start.resolve()
    for cand in [cur, *cur.parents]:
        if (cand / ".pluribus" / "rhizome.json").exists():
            return cand
    return None


def ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def db_path_for_root(root: Path) -> Path:
    return root / ".pluribus" / "index" / "rag.sqlite3"


def connect(db_path: Path) -> sqlite3.Connection:
    ensure_dir(db_path.parent)
    con = sqlite3.connect(str(db_path))
    con.execute("PRAGMA journal_mode=WAL;")
    con.execute("PRAGMA synchronous=NORMAL;")

    # Best-effort: enable vec0 for this connection (no-op if unavailable).
    _ = try_load_vec_extension(con)
    return con


def init_db(con: sqlite3.Connection) -> None:
    con.execute(
        """
        CREATE TABLE IF NOT EXISTS docs (
          doc_id TEXT PRIMARY KEY,
          doc_sha256 TEXT NOT NULL UNIQUE,
          ts REAL NOT NULL,
          iso TEXT NOT NULL,
          actor TEXT NOT NULL,
          title TEXT,
          source TEXT,
          meta_json TEXT,
          text TEXT NOT NULL
        );
        """
    )
    con.execute(
        """
        CREATE VIRTUAL TABLE IF NOT EXISTS docs_fts
        USING fts5(doc_id, title, source, text, content='docs', content_rowid='rowid');
        """
    )
    con.execute(
        """
        CREATE TABLE IF NOT EXISTS docs_vec_meta (
          doc_id TEXT PRIMARY KEY,
          embedding_mode TEXT,
          embedding_model TEXT,
          embedding_dim INTEGER,
          updated_iso TEXT
        );
        """
    )

    if try_load_vec_extension(con):
        try:
            con.execute(
                f"""
                CREATE VIRTUAL TABLE IF NOT EXISTS docs_vec
                USING vec0(doc_id TEXT PRIMARY KEY, embedding FLOAT[{EMBEDDING_DIM}]);
                """
            )
        except Exception:
            # If vec0 isn't available in this connection, skip semantic index.
            pass

    con.execute(
        """
        CREATE TRIGGER IF NOT EXISTS docs_ai AFTER INSERT ON docs BEGIN
          INSERT INTO docs_fts(rowid, doc_id, title, source, text)
          VALUES (new.rowid, new.doc_id, new.title, new.source, new.text);
        END;
        """
    )
    con.commit()


def emit_bus(*, bus_dir: str | None, topic: str, kind: str, level: str, actor: str, data: dict) -> None:
    if not bus_dir:
        return
    tool = Path(__file__).with_name("agent_bus.py")
    if not tool.exists():
        return
    subprocess.run(
        [
            sys.executable,
            str(tool),
            "--bus-dir",
            bus_dir,
            "pub",
            "--topic",
            topic,
            "--kind",
            kind,
            "--level",
            level,
            "--actor",
            actor,
            "--data",
            json.dumps(data, ensure_ascii=False),
        ],
        check=False,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        env={**os.environ, "PYTHONDONTWRITEBYTECODE": "1"},
    )


def try_extract_text(path: Path) -> str | None:
    if not path.exists() or not path.is_file():
        return None
    ext = path.suffix.lower().lstrip(".")
    if ext in {"md", "txt", "log"}:
        return path.read_text(encoding="utf-8", errors="replace")
    if ext in {"json"}:
        return path.read_text(encoding="utf-8", errors="replace")
    if ext in {"pdf"}:
        if not shutil_which("pdftotext"):
            return None
        try:
            p = subprocess.run(
                ["pdftotext", str(path), "-"],
                check=False,
                stdout=subprocess.PIPE,
                stderr=subprocess.DEVNULL,
            )
            if p.returncode != 0:
                return None
            return p.stdout.decode("utf-8", errors="replace")
        except Exception:
            return None
    return None


def shutil_which(cmd: str) -> str | None:
    for d in os.environ.get("PATH", "").split(os.pathsep):
        cand = Path(d) / cmd
        if cand.exists() and os.access(str(cand), os.X_OK):
            return str(cand)
    return None


def upsert_text(
    con: sqlite3.Connection,
    *,
    title: str | None,
    source: str | None,
    text: str,
    meta: dict | None,
    actor: str,
) -> str:
    vec_ok = try_load_vec_extension(con)
    data = text.encode("utf-8", errors="replace")
    doc_sha = sha256_bytes(data)
    doc_id = str(uuid.uuid4())
    con.execute(
        """
        INSERT OR IGNORE INTO docs (doc_id, doc_sha256, ts, iso, actor, title, source, meta_json, text)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            doc_id,
            doc_sha,
            time.time(),
            now_iso_utc(),
            actor,
            title,
            source,
            json.dumps(meta or {}, ensure_ascii=False),
            text,
        ),
    )
    # If it was a duplicate, return the existing doc_id.
    row = con.execute("SELECT doc_id FROM docs WHERE doc_sha256 = ?", (doc_sha,)).fetchone()
    final_doc_id = str(row[0]) if row else doc_id

    # Generate and store embedding if possible (best-effort, never blocks ingest).
    if vec_ok:
        try:
            emb, emb_meta = embed_text((text or "")[:8000], dim=EMBEDDING_DIM)
            if emb is not None:
                emb_json = json.dumps(emb, ensure_ascii=False)
                con.execute(
                    "INSERT OR REPLACE INTO docs_vec (doc_id, embedding) VALUES (?, ?)",
                    (final_doc_id, emb_json),
                )
                con.execute(
                    """
                    INSERT OR REPLACE INTO docs_vec_meta (doc_id, embedding_mode, embedding_model, embedding_dim, updated_iso)
                    VALUES (?, ?, ?, ?, ?)
                    """,
                    (
                        final_doc_id,
                        str(emb_meta.get("mode") or "unknown"),
                        str(emb_meta.get("model") or ""),
                        int(emb_meta.get("dim") or EMBEDDING_DIM),
                        now_iso_utc(),
                    ),
                )
        except Exception:
            pass

    con.commit()
    return final_doc_id


def cmd_init(args: argparse.Namespace) -> int:
    root = Path(args.root).expanduser().resolve() if args.root else (find_rhizome_root(Path.cwd()) or Path.cwd().resolve())
    db_path = db_path_for_root(root)
    con = connect(db_path)
    init_db(con)
    sys.stdout.write(str(db_path) + "\n")
    return 0


def cmd_add_text(args: argparse.Namespace) -> int:
    root = Path(args.root).expanduser().resolve() if args.root else (find_rhizome_root(Path.cwd()) or Path.cwd().resolve())
    db_path = db_path_for_root(root)
    con = connect(db_path)
    init_db(con)

    actor = default_actor()
    bus_dir = args.bus_dir or os.environ.get("PLURIBUS_BUS_DIR")
    meta = {"kind": args.kind, "tags": [t for t in (args.tag or []) if t.strip()]}
    doc_id = upsert_text(con, title=args.title, source=args.source, text=args.text, meta=meta, actor=actor)
    emit_bus(
        bus_dir=bus_dir,
        topic="rag.doc.added",
        kind="artifact",
        level="info",
        actor=actor,
        data={"doc_id": doc_id, "title": args.title, "source": args.source, "root": str(root)},
    )
    sys.stdout.write(doc_id + "\n")
    return 0


def iter_ndjson(path: Path):
    if not path.exists():
        return
    with path.open("r", encoding="utf-8", errors="replace") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except Exception:
                continue


def cmd_sync_from_rhizome(args: argparse.Namespace) -> int:
    root = Path(args.root).expanduser().resolve() if args.root else (find_rhizome_root(Path.cwd()) or Path.cwd().resolve())
    rhizome_manifest = root / ".pluribus" / "rhizome.json"
    if not rhizome_manifest.exists():
        sys.stderr.write("no rhizome manifest found; run: rhizome.py init\n")
        return 2

    db_path = db_path_for_root(root)
    con = connect(db_path)
    init_db(con)

    actor = default_actor()
    bus_dir = args.bus_dir or os.environ.get("PLURIBUS_BUS_DIR")

    artifacts = root / ".pluribus" / "index" / "artifacts.ndjson"
    added = 0
    skipped = 0

    for rec in iter_ndjson(artifacts):
        sources = rec.get("sources") or []
        if not sources or not isinstance(sources, list):
            continue
        source_path = Path(str(sources[0])).expanduser()
        text = try_extract_text(source_path)
        if not text:
            skipped += 1
            continue
        meta = {
            "kind": "rhizome_artifact",
            "artifact_id": rec.get("id"),
            "artifact_sha256": rec.get("sha256"),
            "artifact_tags": rec.get("tags") or [],
            "media_type": rec.get("media_type"),
        }
        before = con.execute("SELECT COUNT(1) FROM docs").fetchone()[0]
        _ = upsert_text(con, title=source_path.name, source=str(source_path), text=text, meta=meta, actor=actor)
        after = con.execute("SELECT COUNT(1) FROM docs").fetchone()[0]
        if after > before:
            added += 1
        else:
            skipped += 1

    emit_bus(
        bus_dir=bus_dir,
        topic="rag.sync.completed",
        kind="log",
        level="info",
        actor=actor,
        data={"root": str(root), "added": added, "skipped": skipped},
    )
    sys.stdout.write(f"added {added} skipped {skipped}\n")
    return 0


def cmd_query(args: argparse.Namespace) -> int:
    root = Path(args.root).expanduser().resolve() if args.root else (find_rhizome_root(Path.cwd()) or Path.cwd().resolve())
    db_path = db_path_for_root(root)
    if not db_path.exists():
        sys.stderr.write("missing RAG DB; run: rag_index.py init\n")
        return 2
    con = connect(db_path)
    init_db(con)

    q = args.query.strip()
    if not q:
        return 2

    # Semantic Vector Search
    if getattr(args, "semantic", False) and try_load_vec_extension(con):
        q_vec, q_meta = embed_text(q, dim=EMBEDDING_DIM)
        if q_vec is not None:
            q_json = json.dumps(q_vec, ensure_ascii=False)
            rows = con.execute(
                """
                SELECT d.doc_id, d.title, d.source, substr(d.text, 1, 240) AS snip, v.distance
                FROM docs_vec v
                JOIN docs d ON d.doc_id = v.doc_id
                WHERE v.embedding MATCH ?
                AND k = ?
                ORDER BY v.distance
                """,
                (q_json, max(1, int(args.k))),
            ).fetchall()
            for doc_id, title, source, snip, dist in rows:
                sys.stdout.write(
                    json.dumps(
                        {
                            "doc_id": doc_id,
                            "title": title,
                            "source": source,
                            "snippet": snip,
                            "distance": dist,
                            "method": "semantic",
                            "embedder": q_meta,
                        },
                        ensure_ascii=False,
                    )
                    + "\n"
                )
            return 0

    # FTS5 Fallback
    rows = con.execute(
        """
        SELECT d.doc_id, d.title, d.source, snippet(docs_fts, 3, '[', ']', '…', 12) AS snip
        FROM docs_fts
        JOIN docs d ON d.doc_id = docs_fts.doc_id
        WHERE docs_fts MATCH ?
        ORDER BY bm25(docs_fts)
        LIMIT ?
        """,
        (q, max(1, int(args.k))),
    ).fetchall()
    for doc_id, title, source, snip in rows:
        sys.stdout.write(json.dumps({"doc_id": doc_id, "title": title, "source": source, "snippet": snip, "method": "lexical"}, ensure_ascii=False) + "\n")
    return 0


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="rag_index.py", description="Local-first RAG index (SQLite FTS5 + vec0) integrated with STRp rhizomes.")
    p.add_argument("--root", default=None, help="Rhizome root (defaults: search upward for .pluribus/rhizome.json).")
    p.add_argument("--bus-dir", default=None, help="Bus dir (or set PLURIBUS_BUS_DIR).")
    sub = p.add_subparsers(dest="cmd", required=True)

    init = sub.add_parser("init", help="Create/initialize the RAG DB under .pluribus/index/")
    init.set_defaults(func=cmd_init)

    add = sub.add_parser("add-text", help="Add an arbitrary text doc.")
    add.add_argument("--title", default=None)
    add.add_argument("--source", default=None)
    add.add_argument("--kind", default="note")
    add.add_argument("--tag", action="append", default=[])
    add.add_argument("--text", required=True)
    add.set_defaults(func=cmd_add_text)

    sync = sub.add_parser("sync-from-rhizome", help="Index extractable text from rhizome artifacts.ndjson sources.")
    sync.set_defaults(func=cmd_sync_from_rhizome)

    q = sub.add_parser("query", help="Query the index.")
    q.add_argument("query")
    q.add_argument("-k", type=int, default=8)
    q.add_argument("--semantic", action="store_true", help="Use semantic vector search (requires sqlite-vec).")
    q.set_defaults(func=cmd_query)

    return p


def main(argv: list[str]) -> int:
    args = build_parser().parse_args(argv)
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
