import os
from pathlib import Path

from codex_global_doctor import detect_codex_on_path, which_all


def _write_exe(path: Path, text: str) -> None:
    path.write_text(text, encoding="utf-8")
    path.chmod(0o755)


def test_which_all_returns_in_path_order(tmp_path: Path) -> None:
    a = tmp_path / "a"
    b = tmp_path / "b"
    a.mkdir()
    b.mkdir()
    _write_exe(a / "codex", "#!/usr/bin/env sh\necho A\n")
    _write_exe(b / "codex", "#!/usr/bin/env sh\necho B\n")

    p = os.pathsep.join([str(b), str(a)])
    found = which_all("codex", path=p)
    assert found[0].endswith("/b/codex")
    assert found[1].endswith("/a/codex")


def test_detect_codex_on_path_collects_versions(tmp_path: Path) -> None:
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    _write_exe(bin_dir / "codex", "#!/usr/bin/env sh\necho codex-cli 0.75.0\n")
    found = detect_codex_on_path(path=str(bin_dir))
    assert len(found) == 1
    assert found[0].version == "codex-cli 0.75.0"

