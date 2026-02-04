#!/usr/bin/env python3
from __future__ import annotations

import argparse
import getpass
import json
import os
import subprocess
import sys
import time
from pathlib import Path

sys.dont_write_bytecode = True


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


def which(cmd: str) -> str | None:
    for d in os.environ.get("PATH", "").split(os.pathsep):
        cand = Path(d) / cmd
        if cand.exists() and os.access(str(cand), os.X_OK):
            return str(cand)
    return None


def emit_bus(bus_dir: str | None, *, topic: str, kind: str, level: str, actor: str, data: dict) -> None:
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


def config_path(root: Path) -> Path:
    return root / ".pluribus" / "mounts.json"


def default_mountpoint(root: Path, name: str) -> Path:
    return root / ".pluribus" / "mnt" / name


def load_config(path: Path) -> dict:
    if not path.exists():
        return {"schema_version": 1, "mounts": []}
    return json.loads(path.read_text(encoding="utf-8"))


def save_config(path: Path, obj: dict) -> None:
    ensure_dir(path.parent)
    path.write_text(json.dumps(obj, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def resolve_root(raw_root: str | None) -> Path:
    if raw_root:
        return Path(raw_root).expanduser().resolve()
    return find_rhizome_root(Path.cwd()) or Path.cwd().resolve()


def mountpoint_status(mountpoint: Path) -> bool:
    mp = which("mountpoint")
    if not mp:
        return mountpoint.exists() and any(mountpoint.iterdir())
    p = subprocess.run([mp, "-q", str(mountpoint)], check=False)
    return p.returncode == 0


def get_mount(cfg: dict, name: str) -> dict | None:
    for m in cfg.get("mounts") or []:
        if isinstance(m, dict) and m.get("name") == name:
            return m
    return None


def cmd_init(args: argparse.Namespace) -> int:
    root = resolve_root(args.root)
    path = config_path(root)
    if path.exists() and not args.force:
        sys.stderr.write(f"exists: {path}\n")
        return 2

    example = {
        "schema_version": 1,
        "mounts": [
            {
                "name": "gdrive",
                "type": "rclone",
                "remote": "gdrive",
                "remote_path": "",
                "mountpoint": str(default_mountpoint(root, "gdrive")),
                "options": {"vfs_cache_mode": "writes"},
                "notes": "Requires: rclone config create gdrive drive (interactive OAuth).",
            },
            {
                "name": "sshfs_example",
                "type": "sshfs",
                "source": "user@example.com:/path",
                "mountpoint": str(default_mountpoint(root, "sshfs_example")),
                "options": {"reconnect": True},
                "notes": "Fill in source and ensure SSH keys are configured.",
            },
        ],
    }
    save_config(path, example)
    sys.stdout.write(str(path) + "\n")
    return 0


def cmd_list(args: argparse.Namespace) -> int:
    root = resolve_root(args.root)
    cfg = load_config(config_path(root))
    for m in cfg.get("mounts") or []:
        if not isinstance(m, dict):
            continue
        sys.stdout.write(f"{m.get('name','')}\t{m.get('type','')}\t{m.get('mountpoint','')}\n")
    return 0


def cmd_status(args: argparse.Namespace) -> int:
    root = resolve_root(args.root)
    cfg = load_config(config_path(root))
    actor = default_actor()
    bus_dir = args.bus_dir or os.environ.get("PLURIBUS_BUS_DIR")

    rows = []
    for m in cfg.get("mounts") or []:
        if not isinstance(m, dict):
            continue
        mp = Path(str(m.get("mountpoint") or "")).expanduser()
        mounted = mountpoint_status(mp) if mp else False
        row = {"name": m.get("name"), "type": m.get("type"), "mountpoint": str(mp), "mounted": mounted}
        rows.append(row)
        sys.stdout.write(json.dumps(row, ensure_ascii=False) + "\n")

    if args.emit_bus:
        emit_bus(
            bus_dir,
            topic="mount.status",
            kind="metric",
            level="info",
            actor=actor,
            data={"root": str(root), "mounts": rows, "iso": now_iso_utc()},
        )
    return 0


def cmd_doctor(args: argparse.Namespace) -> int:
    missing: list[str] = []
    if not which("fusermount") and not which("fusermount3"):
        missing.append("fusermount (fuse)")
    if not which("mountpoint"):
        missing.append("mountpoint (util-linux)")
    if args.check_rclone and not which("rclone"):
        missing.append("rclone")
    if args.check_sshfs and not which("sshfs"):
        missing.append("sshfs")

    if missing:
        sys.stderr.write("missing:\n")
        for m in missing:
            sys.stderr.write(f"- {m}\n")
        return 1
    sys.stdout.write("ok\n")
    return 0


def run_mount_command(argv: list[str]) -> int:
    return int(subprocess.run(argv, check=False).returncode)


def cmd_mount(args: argparse.Namespace) -> int:
    root = resolve_root(args.root)
    cfg_path = config_path(root)
    cfg = load_config(cfg_path)
    mount = get_mount(cfg, args.name)
    if not mount:
        sys.stderr.write(f"unknown mount: {args.name}\n")
        return 2

    actor = default_actor()
    bus_dir = args.bus_dir or os.environ.get("PLURIBUS_BUS_DIR")

    mp = Path(str(mount.get("mountpoint") or default_mountpoint(root, args.name))).expanduser()
    ensure_dir(mp)
    if mountpoint_status(mp):
        sys.stderr.write("already mounted\n")
        return 0

    mtype = mount.get("type")
    emit_bus(bus_dir, topic="mount.mount.start", kind="log", level="info", actor=actor, data={"name": args.name, "type": mtype, "mountpoint": str(mp)})

    rc = 1
    if mtype == "rclone":
        rclone = which("rclone")
        if not rclone:
            sys.stderr.write("missing rclone\n")
            rc = 2
        else:
            rclone_cfg = mount.get("rclone_config")
            remote = str(mount.get("remote") or "")
            remote_path = str(mount.get("remote_path") or "")
            target = f"{remote}:{remote_path}" if remote_path else f"{remote}:"
            opts = mount.get("options") or {}
            vfs_cache = str(opts.get("vfs_cache_mode") or "writes")
            extra_args = mount.get("rclone_args") or []
            if not isinstance(extra_args, list):
                extra_args = []
            cmd = [rclone]
            if rclone_cfg:
                cmd += ["--config", str(rclone_cfg)]
            cmd += ["mount", target, str(mp), "--daemon", "--vfs-cache-mode", vfs_cache, *[str(x) for x in extra_args]]
            rc = run_mount_command(cmd)
    elif mtype == "sshfs":
        sshfs = which("sshfs")
        if not sshfs:
            sys.stderr.write("missing sshfs\n")
            rc = 2
        else:
            src = str(mount.get("source") or "")
            if not src:
                sys.stderr.write("missing sshfs source\n")
                rc = 2
            else:
                opts = mount.get("options") or {}
                extra = []
                if opts.get("reconnect", True):
                    extra += ["-o", "reconnect"]
                rc = run_mount_command([sshfs, src, str(mp), *extra])
    else:
        sys.stderr.write(f"unsupported type: {mtype}\n")
        rc = 2

    mounted = mountpoint_status(mp)
    emit_bus(
        bus_dir,
        topic="mount.mount.end",
        kind="log",
        level="info" if rc == 0 and mounted else "error",
        actor=actor,
        data={"name": args.name, "type": mtype, "mountpoint": str(mp), "exit_code": rc, "mounted": mounted},
    )
    return 0 if mounted else int(rc)


def cmd_unmount(args: argparse.Namespace) -> int:
    root = resolve_root(args.root)
    cfg = load_config(config_path(root))
    mount = get_mount(cfg, args.name)
    if not mount:
        sys.stderr.write(f"unknown mount: {args.name}\n")
        return 2

    actor = default_actor()
    bus_dir = args.bus_dir or os.environ.get("PLURIBUS_BUS_DIR")
    mp = Path(str(mount.get("mountpoint") or default_mountpoint(root, args.name))).expanduser()
    if not mp.exists():
        return 0

    emit_bus(bus_dir, topic="mount.unmount.start", kind="log", level="info", actor=actor, data={"name": args.name, "mountpoint": str(mp)})
    fusermount = which("fusermount") or which("fusermount3")
    if not fusermount:
        sys.stderr.write("missing fusermount\n")
        return 2
    rc = run_mount_command([fusermount, "-u", str(mp)])
    mounted = mountpoint_status(mp)
    emit_bus(
        bus_dir,
        topic="mount.unmount.end",
        kind="log",
        level="info" if rc == 0 and not mounted else "error",
        actor=actor,
        data={"name": args.name, "mountpoint": str(mp), "exit_code": rc, "mounted": mounted},
    )
    return 0 if not mounted else int(rc)


def cmd_watch(args: argparse.Namespace) -> int:
    interval = max(1.0, float(args.interval))
    while True:
        _ = cmd_status(argparse.Namespace(root=args.root, bus_dir=args.bus_dir, emit_bus=args.emit_bus))
        time.sleep(interval)


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="mountctl.py", description="Managed mountpoints for STRp rhizomes (bus-instrumented).")
    p.add_argument("--root", default=None, help="Rhizome root (defaults: search upward for .pluribus/rhizome.json).")
    p.add_argument("--bus-dir", default=None, help="Bus dir (or set PLURIBUS_BUS_DIR).")
    sub = p.add_subparsers(dest="cmd", required=True)

    init = sub.add_parser("init", help="Create .pluribus/mounts.json template.")
    init.add_argument("--force", action="store_true")
    init.set_defaults(func=cmd_init)

    ls = sub.add_parser("list", help="List mount definitions.")
    ls.set_defaults(func=cmd_list)

    st = sub.add_parser("status", help="Print status for each mount (JSON lines).")
    st.add_argument("--emit-bus", action="store_true")
    st.set_defaults(func=cmd_status)

    doc = sub.add_parser("doctor", help="Check for required binaries.")
    doc.add_argument("--check-rclone", action="store_true")
    doc.add_argument("--check-sshfs", action="store_true")
    doc.set_defaults(func=cmd_doctor)

    m = sub.add_parser("mount", help="Mount one configured mount by name.")
    m.add_argument("name")
    m.set_defaults(func=cmd_mount)

    u = sub.add_parser("unmount", help="Unmount one configured mount by name.")
    u.add_argument("name")
    u.set_defaults(func=cmd_unmount)

    w = sub.add_parser("watch", help="Poll status periodically.")
    w.add_argument("--interval", default="10")
    w.add_argument("--emit-bus", action="store_true")
    w.set_defaults(func=cmd_watch)

    return p


def main(argv: list[str]) -> int:
    args = build_parser().parse_args(argv)
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
