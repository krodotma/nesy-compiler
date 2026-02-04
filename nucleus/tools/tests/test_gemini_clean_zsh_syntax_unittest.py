import subprocess
from pathlib import Path


def test_gemini_clean_zsh_parses() -> None:
    root = Path(__file__).resolve().parents[3]
    script = root / "gemini_clean.zsh"
    assert script.exists()

    p = subprocess.run(
        ["zsh", "-n", str(script)],
        check=False,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    assert p.returncode == 0, (p.stdout or "") + (p.stderr or "")

