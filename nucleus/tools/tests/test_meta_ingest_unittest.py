import tempfile
from pathlib import Path

from nucleus.tools.meta_ingest import FileCursor, _read_new_lines


def test_read_new_lines_max_bytes_chunking():
    with tempfile.TemporaryDirectory() as td:
        path = Path(td) / "events.ndjson"
        path.write_bytes(b"alpha\nbravo\ncharlie\n")
        cursor = FileCursor(path, offset=0, inode=path.stat().st_ino)

        lines = _read_new_lines(cursor, tail_bytes=0, max_bytes=8)
        assert lines == ["alpha"]

        lines = _read_new_lines(cursor, tail_bytes=0, max_bytes=8)
        assert lines == ["bravo"]

        lines = _read_new_lines(cursor, tail_bytes=0, max_bytes=8)
        assert lines == ["charlie"]


def test_read_new_lines_skips_oversized_line():
    with tempfile.TemporaryDirectory() as td:
        path = Path(td) / "events.ndjson"
        path.write_text("x" * 50 + "\nshort\n", encoding="utf-8")
        cursor = FileCursor(path, offset=0, inode=path.stat().st_ino)

        lines = _read_new_lines(
            cursor,
            tail_bytes=0,
            max_bytes=256,
            max_line_bytes=10,
        )
        assert lines == ["short"]
