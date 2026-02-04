import os
import subprocess
from pathlib import Path


def test_bus_codex_avoids_recursive_codex_resolution(tmp_path: Path) -> None:
    tools_dir = Path(__file__).resolve().parents[1]
    bus_codex = tools_dir / "bus-codex"
    assert bus_codex.exists()

    wrapper_bin = tmp_path / "wrapper_bin"
    prefix = tmp_path / "prefix"
    real_bin = prefix / "bin"
    wrapper_bin.mkdir()
    real_bin.mkdir(parents=True)

    (wrapper_bin / "codex").symlink_to(bus_codex)

    real_codex = real_bin / "codex"
    real_codex.write_text(
        "#!/usr/bin/env bash\n"
        "set -euo pipefail\n"
        "echo \"REAL_CODEX npm_config_prefix=${npm_config_prefix:-} actor=${PLURIBUS_ACTOR:-} tasks=${PLURIBUS_TASKS_FROM_BUS_RUN:-}\"\n",
        encoding="utf-8",
    )
    real_codex.chmod(0o755)

    env = dict(os.environ)
    env["PLURIBUS_STATUSLINE"] = "0"
    env["PLURIBUS_BUS_DIR"] = str(tmp_path / "bus")
    env["PLURIBUS_CODEX_HOME"] = str(tmp_path / "codex_home")
    env["PLURIBUS_CODEX_BIN"] = str(wrapper_bin / "codex")  # intentionally recursive
    env["PATH"] = os.pathsep.join([str(wrapper_bin), str(real_bin), "/usr/bin", "/bin"])

    p = subprocess.run(
        [str(wrapper_bin / "codex"), "--version"],
        check=False,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        env=env,
        timeout=30,
    )
    assert p.returncode == 0, (p.stdout or "") + (p.stderr or "")
    assert "REAL_CODEX" in p.stdout
    assert f"npm_config_prefix={prefix}" in p.stdout
    assert "actor=codex" in p.stdout
    assert "tasks=1" in p.stdout
