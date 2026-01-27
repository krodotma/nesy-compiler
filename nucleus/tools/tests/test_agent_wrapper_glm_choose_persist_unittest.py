import os
import subprocess
from pathlib import Path


def _run_wrapper(env: dict, *args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ["/pluribus/nucleus/tools/agent-wrapper", *args],
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )


def test_choose_mode_uses_last_key(tmp_path: Path) -> None:
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    glm_stub = bin_dir / "glm"
    glm_stub.write_text(
        "#!/usr/bin/env bash\n"
        "echo \"MINDLIKE_API_KEY=$MINDLIKE_API_KEY\"\n"
        "echo \"GLM_API_KEY=$GLM_API_KEY\"\n"
        "echo \"MINDLIKE_KEY_MODE=$MINDLIKE_KEY_MODE\"\n"
    )
    glm_stub.chmod(0o755)

    config_root = tmp_path / "config"
    keys_dir = config_root / "agent-wrapper"
    keys_dir.mkdir(parents=True)
    (keys_dir / "keys").write_text("mindlike_key\nlegacy_key\n")
    (keys_dir / "last_key").write_text("legacy_key\n")

    env = os.environ.copy()
    env.update(
        {
            "PATH": f"{bin_dir}:{env.get('PATH', '')}",
            "XDG_CONFIG_HOME": str(config_root),
            "HOME": str(tmp_path),
            "PLURIBUS_SUPPRESS_HEADER": "1",
            "PLURIBUS_MINDLIKE_ENV": str(tmp_path / "mindlike_env.conf"),
            "PLURIBUS_DISABLE_SEED_KEYS": "1",
        }
    )
    env.pop("MINDLIKE_API_KEY", None)
    env.pop("GLM_API_KEY", None)
    (tmp_path / "mindlike_env.conf").write_text("# empty\n")

    result = _run_wrapper(env, "glm", "--choose", "-p", "ping")
    assert result.returncode == 0, result.stderr
    assert "MINDLIKE_API_KEY=legacy_key" in result.stdout
    assert "GLM_API_KEY=legacy_key" in result.stdout


def test_choose_mode_persists_last_key(tmp_path: Path) -> None:
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    glm_stub = bin_dir / "glm"
    glm_stub.write_text(
        "#!/usr/bin/env bash\n"
        "echo \"MINDLIKE_API_KEY=$MINDLIKE_API_KEY\"\n"
        "echo \"GLM_API_KEY=$GLM_API_KEY\"\n"
        "echo \"MINDLIKE_KEY_MODE=$MINDLIKE_KEY_MODE\"\n"
    )
    glm_stub.chmod(0o755)

    config_root = tmp_path / "config"
    keys_dir = config_root / "agent-wrapper"
    keys_dir.mkdir(parents=True)
    (keys_dir / "keys").write_text("mindlike_key\nlegacy_key\n")
    (keys_dir / "last_key").write_text("")

    env = os.environ.copy()
    env.update(
        {
            "PATH": f"{bin_dir}:{env.get('PATH', '')}",
            "XDG_CONFIG_HOME": str(config_root),
            "HOME": str(tmp_path),
            "PLURIBUS_SUPPRESS_HEADER": "1",
            "PLURIBUS_MINDLIKE_ENV": str(tmp_path / "mindlike_env.conf"),
            "PLURIBUS_DISABLE_SEED_KEYS": "1",
        }
    )
    env.pop("MINDLIKE_API_KEY", None)
    env.pop("GLM_API_KEY", None)
    (tmp_path / "mindlike_env.conf").write_text("# empty\n")

    result = _run_wrapper(env, "glm", "--choose", "-p", "ping")
    assert result.returncode == 0, result.stderr
    assert "MINDLIKE_API_KEY=mindlike_key" in result.stdout

    last_key = (keys_dir / "last_key").read_text().strip()
    assert last_key == "mindlike_key"
