import subprocess
from pathlib import Path


def test_zshrc_snippet_parses() -> None:
    root = Path(__file__).resolve().parents[3]
    snippet = root / "shell" / "zsh" / "zshrc.snippet.zsh"
    assert snippet.exists()

    p = subprocess.run(
        ["zsh", "-n", str(snippet)],
        check=False,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    assert p.returncode == 0, (p.stdout or "") + (p.stderr or "")

