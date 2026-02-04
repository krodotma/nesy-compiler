import subprocess
from pathlib import Path


def test_vertex_express_smoke_help() -> None:
    tools_root = Path(__file__).resolve().parents[1]
    tool = tools_root / "providers" / "vertex_express_gemini_smoke.py"
    assert tool.exists()
    p = subprocess.run(
        ["python3", str(tool), "--help"],
        check=False,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    assert p.returncode == 0, (p.stdout or "") + (p.stderr or "")
