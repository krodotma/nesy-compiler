#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

sys.dont_write_bytecode = True


@dataclass(frozen=True)
class CodexOnPath:
    path: str
    realpath: str
    version: str | None


def _run_text(cmd: list[str], timeout_s: int = 8) -> tuple[int, str, str]:
    try:
        p = subprocess.run(cmd, check=False, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, timeout=timeout_s)
    except subprocess.TimeoutExpired:
        return 124, "", f"timeout after {timeout_s}s"
    return int(p.returncode), p.stdout.strip(), p.stderr.strip()


def which_all(program: str, path: str | None = None) -> list[str]:
    search = path if path is not None else os.environ.get("PATH") or ""
    out: list[str] = []
    for directory in search.split(os.pathsep):
        if not directory:
            directory = "."
        candidate = Path(directory) / program
        try:
            if candidate.exists() and os.access(candidate, os.X_OK):
                out.append(str(candidate))
        except OSError:
            continue
    return out


def read_package_version(pkg_dir: Path) -> str | None:
    pkg_json = pkg_dir / "package.json"
    if not pkg_json.exists():
        return None
    try:
        payload = json.loads(pkg_json.read_text(encoding="utf-8"))
    except Exception:
        return None
    v = payload.get("version")
    return str(v) if v else None


def detect_codex_on_path(path: str | None = None) -> list[CodexOnPath]:
    results: list[CodexOnPath] = []
    for p in which_all("codex", path=path):
        realp = os.path.realpath(p)
        rc, out, err = _run_text([p, "--version"], timeout_s=5)
        version = out or None
        if rc != 0 and not version:
            version = f"error(rc={rc}): {err or out}".strip() or None
        results.append(CodexOnPath(path=p, realpath=realp, version=version))
    return results


def npm_global_prefix() -> str | None:
    npm = shutil.which("npm")
    if not npm:
        return None
    rc, out, _ = _run_text([npm, "prefix", "-g"], timeout_s=8)
    return out if rc == 0 and out else None


def npm_global_root() -> str | None:
    npm = shutil.which("npm")
    if not npm:
        return None
    rc, out, _ = _run_text([npm, "root", "-g"], timeout_s=8)
    return out if rc == 0 and out else None


def fix_instructions(preferred_bin: Path, shadowing_bin: Path) -> list[str]:
    return [
        "Mismatch detected: you are running a different `codex` than npm updates.",
        "",
        "Most common fix (safe): repoint the higher-precedence binary to the npm-managed one:",
        f"  sudo ln -sf {preferred_bin} {shadowing_bin}",
        "  hash -r   # refresh shell command cache",
        "",
        "If you prefer to keep using the current binary instead, make npm install there (advanced):",
        "  npm config get prefix",
        "  sudo npm config set prefix /usr/local",
    ]


def guess_prefix_from_path(bin_path: Path) -> Path | None:
    parts = bin_path.parts
    if "lib" in parts:
        i = parts.index("lib")
        if i > 0 and i + 1 < len(parts) and parts[i + 1] == "node_modules":
            return Path(*parts[:i])
    if "bin" in parts:
        i = parts.index("bin")
        return Path(*parts[:i]) if i > 0 else None
    if "lib" in parts:
        i = parts.index("lib")
        return Path(*parts[:i]) if i > 0 else None
    return None


def main(argv: list[str]) -> int:
    p = argparse.ArgumentParser(prog="codex_global_doctor.py", description="Diagnose/fix npm global Codex upgrade loops (PATH vs npm prefix).")
    p.add_argument("--apply", action="store_true", help="Apply the safest fix when possible (may require sudo/root).")
    args = p.parse_args(argv)

    prefix = npm_global_prefix()
    root = npm_global_root()
    codexes = detect_codex_on_path()
    selected = shutil.which("codex")
    selected_real = os.path.realpath(selected) if selected else None

    sys.stdout.write("codex global doctor\n\n")
    sys.stdout.write(f"- npm prefix -g: {prefix or 'missing'}\n")
    sys.stdout.write(f"- npm root   -g: {root or 'missing'}\n")
    sys.stdout.write(f"- codex (selected): {selected or 'missing'}\n")
    if selected_real and selected_real != selected:
        sys.stdout.write(f"  realpath: {selected_real}\n")
    sys.stdout.write("\n")

    if codexes:
        sys.stdout.write("codex candidates on PATH (in precedence order):\n")
        for c in codexes:
            sys.stdout.write(f"- {c.path}")
            if c.realpath != c.path:
                sys.stdout.write(f" -> {c.realpath}")
            if c.version:
                sys.stdout.write(f"  ({c.version})")
            sys.stdout.write("\n")
        sys.stdout.write("\n")

    if root:
        npm_pkg = Path(root) / "@openai" / "codex"
        npm_pkg_version = read_package_version(npm_pkg)
        sys.stdout.write(f"- npm @openai/codex dir: {npm_pkg} ({'present' if npm_pkg.exists() else 'missing'})\n")
        sys.stdout.write(f"  version: {npm_pkg_version or 'unknown'}\n")

        if npm_pkg.exists() and not (npm_pkg / "bin" / "codex.js").exists():
            sys.stdout.write("  WARN: looks like a partial/broken install (missing bin/codex.js)\n")
        sys.stdout.write("\n")

    if not (prefix and selected):
        sys.stdout.write("No npm or codex found; nothing to fix.\n")
        return 2

    preferred_bin = Path(prefix) / "bin" / "codex"
    if not preferred_bin.exists():
        sys.stdout.write(f"No codex at npm prefix: {preferred_bin}\n\n")

        selected_exec_path = Path(selected)
        selected_prefix = guess_prefix_from_path(selected_exec_path)
        if selected_prefix and selected_prefix != Path(prefix):
            sys.stdout.write("This usually causes the upgrade loop:\n")
            sys.stdout.write("- You run `codex` from one prefix (often `/usr/local`).\n")
            sys.stdout.write("- npm installs/updates globals into another prefix (often `/usr`).\n\n")
            sys.stdout.write("Fix option A (recommended if you want the currently-selected `codex` to be npm-managed):\n")
            sys.stdout.write(f"  sudo npm config set prefix {selected_prefix}\n")
            sys.stdout.write("  sudo npm i -g @openai/codex@latest\n\n")
            sys.stdout.write("Fix option B (recommended if you want to keep npm prefix as-is):\n")
            sys.stdout.write("  # remove the higher-precedence Codex so PATH resolves to the npm-managed one\n")
            sys.stdout.write(f"  sudo rm -f {selected_exec_path}\n")
            sys.stdout.write("  sudo npm i -g @openai/codex@latest\n")
            sys.stdout.write("  hash -r\n")
            return 2

        sys.stdout.write("Fix: reinstall with npm, then rerun this doctor:\n")
        sys.stdout.write("  sudo npm i -g @openai/codex@latest\n")
        return 2

    selected_path = Path(selected_real or selected)
    if selected_path == preferred_bin:
        sys.stdout.write("OK: `codex` matches `npm prefix -g`.\n")
        return 0

    shadowing = codexes[0].path if codexes else str(selected_path)
    shadowing_path = Path(os.path.realpath(shadowing))

    sys.stdout.write("WARN: `codex` on PATH does not match npm-global `codex`.\n")
    for line in fix_instructions(preferred_bin=preferred_bin, shadowing_bin=shadowing_path):
        sys.stdout.write(line + "\n")

    if not args.apply:
        return 1

    try:
        if shadowing_path.exists():
            shadowing_path.unlink()
        shadowing_path.symlink_to(preferred_bin)
    except PermissionError as e:
        sys.stdout.write(f"\nFAILED to apply (permission): {e}\n")
        return 3
    except OSError as e:
        sys.stdout.write(f"\nFAILED to apply: {e}\n")
        return 3

    sys.stdout.write("\nApplied: repointed PATH-precedence `codex` to npm-managed `codex`.\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
