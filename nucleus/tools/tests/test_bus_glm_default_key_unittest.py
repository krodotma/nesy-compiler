import os
import subprocess
from pathlib import Path


def test_bus_glm_defaults_to_first_key_without_choose(tmp_path: Path) -> None:
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    glm_stub = bin_dir / "glm"
    glm_stub.write_text(
        "#!/usr/bin/env bash\n"
        "echo \"ANTHROPIC_API_KEY=$ANTHROPIC_API_KEY\"\n"
        "echo \"ANTHROPIC_BASE_URL=$ANTHROPIC_BASE_URL\"\n"
        "echo \"MINDLIKE_API_KEY=$MINDLIKE_API_KEY\"\n"
        "echo \"GLM_API_KEY=$GLM_API_KEY\"\n"
        "echo \"MINDLIKE_KEY_MODE=$MINDLIKE_KEY_MODE\"\n"
        "echo \"CLAUDE_CONFIG_DIR=$CLAUDE_CONFIG_DIR\"\n"
    )
    glm_stub.chmod(0o755)

    config_root = tmp_path / "config"
    keys_dir = config_root / "agent-wrapper"
    keys_dir.mkdir(parents=True)
    keys_file = keys_dir / "keys"
    keys_file.write_text("mindlike_key\nlegacy_key\n")

    mindlike_env = tmp_path / "mindlike_env.conf"
    mindlike_env.write_text("# empty for test\n")

    env = os.environ.copy()
    env.update(
        {
            "PATH": f"{bin_dir}:{env.get('PATH', '')}",
            "XDG_CONFIG_HOME": str(config_root),
            "HOME": str(tmp_path),
            "PLURIBUS_SUPPRESS_HEADER": "1",
            "PLURIBUS_MINDLIKE_ENV": str(mindlike_env),
        }
    )
    env.pop("MINDLIKE_API_KEY", None)
    env.pop("GLM_API_KEY", None)
    env.pop("ANTHROPIC_API_KEY", None)

    result = subprocess.run(
        ["/pluribus/nucleus/tools/bus-glm", "-p", "ping"],
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr

    seen = {}
    for line in result.stdout.strip().splitlines():
        if "=" in line:
            key, value = line.split("=", 1)
            seen[key] = value

    assert seen.get("MINDLIKE_API_KEY") == "mindlike_key"
    assert seen.get("GLM_API_KEY") == "mindlike_key"
    assert seen.get("ANTHROPIC_API_KEY") == "mindlike_key"
    assert seen.get("MINDLIKE_KEY_MODE") in {"default", ""}
