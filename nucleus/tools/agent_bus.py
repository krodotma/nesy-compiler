#!/usr/bin/env python3
from __future__ import annotations

import argparse
import getpass
import json
import os
import socket
import sys
import time
import uuid
import zlib
from dataclasses import dataclass
from pathlib import Path

sys.dont_write_bytecode = True

try:
    import fcntl  # type: ignore
except Exception:  # pragma: no cover
    fcntl = None  # type: ignore

BUS_PARTITION_DEFAULT = "topics"
BUS_PARTITION_CONFIG_ENV = "PLURIBUS_BUS_PARTITION_CONFIG"
BUS_MIN_RETAIN_MB_DEFAULT = 100.0
BUS_RETAIN_MB_DEFAULT = 100.0
BUS_ROTATE_MB_DEFAULT = 100.0
BUS_ROTATE_HEADROOM_MB_DEFAULT = 0.0
_PARTITION_CONFIG_CACHE: dict | None = None


def _env_flag(name: str, default: bool = False) -> bool:
    value = os.environ.get(name)
    if value is None:
        return default
    return str(value).strip().lower() not in {"0", "false", "no", "off", ""}


def _env_list(name: str, default: str) -> list[str]:
    raw = os.environ.get(name)
    if raw is None:
        raw = default
    if not raw:
        return []
    return [part.strip().lower() for part in raw.split(",") if part.strip()]


def _env_int(name: str, default: int, *, minimum: int = 0) -> int:
    value = os.environ.get(name)
    try:
        parsed = int(value) if value is not None else default
    except (TypeError, ValueError):
        parsed = default
    return max(parsed, minimum)


def _partition_config() -> dict:
    global _PARTITION_CONFIG_CACHE
    if _PARTITION_CONFIG_CACHE is not None:
        return _PARTITION_CONFIG_CACHE
    path = (os.environ.get(BUS_PARTITION_CONFIG_ENV) or "").strip()
    if not path:
        _PARTITION_CONFIG_CACHE = {}
        return _PARTITION_CONFIG_CACHE
    try:
        with open(path, "r", encoding="utf-8") as handle:
            data = json.load(handle)
    except (OSError, ValueError):
        data = {}
    if not isinstance(data, dict):
        data = {}
    _PARTITION_CONFIG_CACHE = data
    return _PARTITION_CONFIG_CACHE


def _partition_config_section(section: str) -> dict:
    cfg = _partition_config()
    data = cfg.get(section) if isinstance(cfg, dict) else None
    return data if isinstance(data, dict) else {}


def _env_or_config_list(env_key: str, section: str, config_key: str, default: str) -> list[str]:
    raw = os.environ.get(env_key)
    if raw is None:
        raw = _partition_config_section(section).get(config_key, default)
    if isinstance(raw, list):
        parts = [str(part).strip() for part in raw]
        return [part.lower() for part in parts if part]
    raw = default if raw is None else str(raw)
    return [part.strip().lower() for part in raw.split(",") if part.strip()]


def _env_or_config_int(env_key: str, section: str, config_key: str, default: int, *, minimum: int = 0) -> int:
    raw = os.environ.get(env_key)
    if raw is None:
        raw = _partition_config_section(section).get(config_key, default)
    try:
        parsed = int(raw) if raw is not None else default
    except (TypeError, ValueError):
        parsed = default
    return max(parsed, minimum)


def _env_or_config_str(env_key: str, section: str, config_key: str, default: str) -> str:
    raw = os.environ.get(env_key)
    if raw is None:
        raw = _partition_config_section(section).get(config_key, default)
    return str(raw) if raw else default


def _rotation_limits():
    try:
        retain_mb = float(os.environ.get("PLURIBUS_BUS_RETAIN_MB") or BUS_RETAIN_MB_DEFAULT)
    except (TypeError, ValueError):
        retain_mb = BUS_RETAIN_MB_DEFAULT
    try:
        min_retain_mb = float(os.environ.get("PLURIBUS_BUS_MIN_RETAIN_MB") or BUS_MIN_RETAIN_MB_DEFAULT)
    except (TypeError, ValueError):
        min_retain_mb = BUS_MIN_RETAIN_MB_DEFAULT
    try:
        rotate_mb = float(os.environ.get("PLURIBUS_BUS_ROTATE_MB") or BUS_ROTATE_MB_DEFAULT)
    except (TypeError, ValueError):
        rotate_mb = BUS_ROTATE_MB_DEFAULT
    try:
        headroom_mb = float(os.environ.get("PLURIBUS_BUS_HEADROOM_MB") or BUS_ROTATE_HEADROOM_MB_DEFAULT)
    except (TypeError, ValueError):
        headroom_mb = BUS_ROTATE_HEADROOM_MB_DEFAULT

    retain_mb = max(retain_mb, min_retain_mb)
    rotate_floor = retain_mb + max(0.0, headroom_mb)
    rotate_mb = max(rotate_mb, rotate_floor)

    retain_bytes = int(retain_mb * 1024 * 1024)
    rotate_bytes = int(rotate_mb * 1024 * 1024)
    return rotate_bytes, retain_bytes


def _topic_bucket(topic: str) -> str:
    override = _topic_bucket_override(topic)
    if override:
        return override
    if topic.startswith("omega."):
        return "omega"
    if topic.startswith("qa.") or topic.startswith("qa_"):
        return "qa"
    if topic.startswith("telemetry."):
        return "telemetry"
    if topic.startswith("browser."):
        return "browser"
    if topic.startswith("dashboard."):
        return "dashboard"
    if topic.startswith("agent."):
        return "agent"
    if topic.startswith("operator."):
        return "operator"
    if topic.startswith("rd.tasks.") or topic.startswith("task.") or topic.startswith("task_ledger.") or topic.endswith(".task"):
        return "task"
    if topic.startswith("a2a."):
        return "a2a"
    if topic.startswith("lens.") or topic.startswith("collimator."):
        return "lens"
    if topic.startswith("dialogos."):
        return "dialogos"
    if topic.startswith("infer_sync."):
        return "infer_sync"
    if topic.startswith("providers.") or topic.startswith("provider."):
        return "providers"
    return "other"


def _topic_bucket_override(topic: str) -> str | None:
    overrides = _partition_config().get("bucket_overrides")
    if not isinstance(overrides, dict):
        return None
    cleaned = topic.strip(".")
    for prefix in sorted(overrides.keys(), key=len, reverse=True):
        if cleaned == prefix or cleaned.startswith(f"{prefix}."):
            return _sanitize_segment(str(overrides[prefix]))
    return None


def _sanitize_segment(segment: str) -> str:
    if not segment:
        return "unknown"
    cleaned = []
    for ch in segment.strip():
        if ch.isascii() and (ch.isalnum() or ch in "-_@+"):
            cleaned.append(ch.lower())
        else:
            cleaned.append("_")
    value = "".join(cleaned).strip("_")
    return value or "unknown"


def _topic_segments(topic: str, bucket: str, max_depth: int) -> list[str]:
    raw = [seg for seg in topic.strip(".").split(".") if seg]
    if raw and raw[0] == bucket:
        raw = raw[1:]
    segments = [_sanitize_segment(seg) for seg in raw]
    if not segments:
        segments = ["root"]
    if max_depth > 0 and len(segments) > max_depth:
        head = segments[: max_depth - 1]
        tail = "_".join(segments[max_depth - 1 :])
        segments = head + [tail]
    return segments


def _partition_shard(topic: str, bucket: str) -> str | None:
    shards = _env_or_config_int("PLURIBUS_BUS_PARTITION_SHARDS", "partition", "shards", 4, minimum=1)
    if shards <= 1:
        return None
    hot_buckets = _env_or_config_list(
        "PLURIBUS_BUS_PARTITION_HOT_BUCKETS",
        "partition",
        "hot_buckets",
        "dashboard,telemetry,omega,task,agent,operator,dialogos,providers,browser",
    )
    if hot_buckets and "*" not in hot_buckets and "all" not in hot_buckets and bucket not in hot_buckets:
        return None
    digest = zlib.crc32(topic.encode("utf-8")) & 0xFFFFFFFF
    return f"{digest % shards:02d}"

def _partition_frequency(bucket: str) -> str:
    hot_buckets = _env_or_config_list(
        "PLURIBUS_BUS_PARTITION_HOT_BUCKETS",
        "partition",
        "hot_buckets",
        "dashboard,telemetry,omega,task,agent,operator,dialogos,providers,browser",
    )
    if hot_buckets and ("*" in hot_buckets or "all" in hot_buckets or bucket in hot_buckets):
        return "hot"
    cold_buckets = _env_or_config_list("PLURIBUS_BUS_PARTITION_COLD_BUCKETS", "partition", "cold_buckets", "other")
    if cold_buckets and ("*" in cold_buckets or "all" in cold_buckets or bucket in cold_buckets):
        return "cold"
    return "warm"


def _partition_paths(base_dir: str, *, topic: str, kind: str, level: str, actor: str) -> list[str]:
    part_root = os.path.join(
        base_dir,
        _env_or_config_str("PLURIBUS_BUS_PARTITION_DIR", "partition", "dir", BUS_PARTITION_DEFAULT),
    )
    bucket = _sanitize_segment(_topic_bucket(topic))
    max_depth = _env_or_config_int("PLURIBUS_BUS_PARTITION_MAX_DEPTH", "partition", "max_depth", 8, minimum=0)
    segments = _topic_segments(topic, bucket, max_depth)
    shard = _partition_shard(topic, bucket)
    shard_seg = [shard] if shard else []

    fanout = _env_or_config_list(
        "PLURIBUS_BUS_PARTITION_FANOUT",
        "partition",
        "fanout",
        "topic,type,actor,frequency",
    )
    paths: list[str] = []
    for dim in fanout:
        if dim in {"topic", "topics"}:
            paths.append(os.path.join(part_root, "topic", bucket, *segments, *shard_seg, "events.ndjson"))
        elif dim in {"type", "types", "eventtype", "eventtypes"}:
            kind_seg = _sanitize_segment(kind or "unknown")
            level_seg = _sanitize_segment(level or "info")
            paths.append(
                os.path.join(part_root, "eventtypes", kind_seg, level_seg, bucket, *segments, *shard_seg, "events.ndjson")
            )
        elif dim in {"actor", "actors"}:
            actor_seg = _sanitize_segment(actor or "unknown")
            paths.append(os.path.join(part_root, "actors", actor_seg, bucket, *segments, *shard_seg, "events.ndjson"))
        elif dim in {"freq", "frequency", "rate", "hot"}:
            tier = _partition_frequency(bucket)
            paths.append(os.path.join(part_root, "frequency", tier, bucket, *segments, *shard_seg, "events.ndjson"))

    if _env_flag("PLURIBUS_BUS_PARTITION_LEGACY", False):
        paths.append(os.path.join(part_root, f"{bucket}.ndjson"))

    seen = set()
    ordered: list[str] = []
    for path in paths:
        if path in seen:
            continue
        seen.add(path)
        ordered.append(path)
    return ordered


def _partition_archive_dir(base_dir: str, part_path: str) -> str:
    part_root = Path(base_dir) / os.environ.get("PLURIBUS_BUS_PARTITION_DIR", BUS_PARTITION_DEFAULT)
    try:
        rel_dir = Path(part_path).resolve().parent.relative_to(part_root.resolve())
    except Exception:
        rel_dir = Path()
    return os.path.join(base_dir, "archive", "partitions", str(rel_dir))


def rotate_log_tail(path: str, *, retain_bytes: int, archive_dir: str, durable: bool = False) -> str | None:
    if retain_bytes <= 0:
        return None
    try:
        st = os.stat(path)
    except OSError:
        st = None
    try:
        size = os.path.getsize(path)
    except OSError:
        return None
    if size <= retain_bytes:
        return None

    ensure_dir(archive_dir)
    archive_name = f"{Path(path).stem}-{now_iso_utc().replace(':', '')}.ndjson.gz"
    archive_path = os.path.join(archive_dir, archive_name)

    fd = os.open(path, os.O_RDONLY)
    try:
        if fcntl is not None:
            fcntl.flock(fd, fcntl.LOCK_EX)
        with os.fdopen(fd, "rb", closefd=False) as f_in:
            import gzip
            import shutil

            f_in.seek(0)
            with gzip.open(archive_path, "wb") as f_out:
                shutil.copyfileobj(f_in, f_out)

            if size > retain_bytes:
                f_in.seek(size - retain_bytes)
            tail = f_in.read()
            tmp_path = f"{path}.tmp"
            with open(tmp_path, "wb") as f_tmp:
                f_tmp.write(tail)
                if durable:
                    f_tmp.flush()
                    os.fsync(f_tmp.fileno())
            os.replace(tmp_path, path)
            if st is not None:
                try:
                    os.chmod(path, st.st_mode & 0o777)
                except OSError:
                    pass
                if os.geteuid() == 0:
                    try:
                        os.chown(path, st.st_uid, st.st_gid)
                    except OSError:
                        pass
    finally:
        try:
            if fcntl is not None:
                fcntl.flock(fd, fcntl.LOCK_UN)
        finally:
            os.close(fd)

    return archive_path


def now_iso_utc() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def default_actor() -> str:
    return os.environ.get("PLURIBUS_ACTOR") or getpass.getuser()


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def json_load_maybe(value: str | None):
    if value is None:
        return None
    value = value.strip()
    if not value:
        return None
    if value == "-":
        return json.load(sys.stdin)
    if value.startswith("@"):
        file_path = Path(value[1:])
        if not file_path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")
        return json.loads(file_path.read_text(encoding="utf-8"))
    return json.loads(value)


@dataclass(frozen=True)
class BusPaths:
    # active_dir is where we write for sure (may be the fallback)
    active_dir: str
    events_path: str
    # primary_dir tracks the canonical bus (usually /pluribus/.pluribus/bus)
    primary_dir: str
    # fallback_dir is used when primary is not writable; kept for mirroring attempts
    fallback_dir: str | None = None

    # Back-compat alias: older callers expect `.bus_dir`.
    @property
    def bus_dir(self) -> str:
        return self.active_dir


def resolve_bus_paths(bus_dir: str | None) -> BusPaths:
    # Prefer the repo-local system bus if available; fall back to user state dir.
    default_bus = "/pluribus/.pluribus/bus"
    state_home = os.environ.get("XDG_STATE_HOME") or os.path.join(os.path.expanduser("~"), ".local", "state")
    if not os.path.exists(default_bus):
        default_bus = os.path.join(state_home, "nucleus", "bus")
    primary = bus_dir or os.environ.get("PLURIBUS_BUS_DIR") or default_bus
    primary = os.path.abspath(primary)
    workspace_root = Path(__file__).resolve().parents[2]
    fallback = os.path.abspath(os.environ.get("PLURIBUS_FALLBACK_BUS_DIR") or str(workspace_root / ".pluribus_local" / "bus"))
    active = primary

    # Attempt to use the primary path; fallback if not writable
    try:
        os.makedirs(primary, exist_ok=True)
        # Important: the bus may be a world-writable file inside a non-writable directory.
        # Test writability by opening the actual events file for append (not by creating a temp file).
        primary_events = os.path.join(primary, "events.ndjson")
        fd = os.open(primary_events, os.O_WRONLY | os.O_APPEND | os.O_CREAT, 0o644)
        os.close(fd)
    except (PermissionError, OSError):
        # Fallback to local workspace bus (avoids total failure in sandboxed contexts)
        if fallback != primary:
            sys.stderr.write(f"WARN: Bus path '{primary}' is not writable. Falling back to '{fallback}'.\n")
        active = fallback
        os.makedirs(active, exist_ok=True)

    return BusPaths(
        active_dir=active,
        events_path=os.path.join(active, "events.ndjson"),
        primary_dir=primary,
        fallback_dir=(fallback if active != primary else None),
    )


def append_line(path: str, line: str, durable: bool) -> None:
    ensure_dir(os.path.dirname(path))
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_APPEND, 0o644)
    try:
        if fcntl is not None:
            fcntl.flock(fd, fcntl.LOCK_EX)
        os.write(fd, line.encode("utf-8"))
        if durable:
            os.fsync(fd)
    finally:
        try:
            if fcntl is not None:
                fcntl.flock(fd, fcntl.LOCK_UN)
        finally:
            os.close(fd)


def emit_event(
    paths: BusPaths,
    *,
    topic: str,
    kind: str,
    level: str,
    actor: str,
    data,
    trace_id: str | None,
    run_id: str | None,
    durable: bool,
) -> str:
    event_id = str(uuid.uuid4())
    partition_enabled = _env_flag("PLURIBUS_BUS_PARTITION", True)
    rotate_enabled = _env_flag("PLURIBUS_BUS_ROTATE", True)
    rotate_bytes, retain_bytes = _rotation_limits()
    payload = {
        "id": event_id,
        "ts": time.time(),
        "iso": now_iso_utc(),
        "topic": topic,
        "kind": kind,
        "level": level,
        "actor": actor,
        "host": socket.gethostname(),
        "pid": os.getpid(),
        "trace_id": trace_id,
        "run_id": run_id,
        "data": data,
    }
    line = json.dumps(payload, ensure_ascii=False, separators=(",", ":")) + "\n"
    append_line(paths.events_path, line, durable=durable)
    if rotate_enabled:
        try:
            if os.path.getsize(paths.events_path) > rotate_bytes:
                rotate_log_tail(
                    paths.events_path,
                    retain_bytes=retain_bytes,
                    archive_dir=os.path.join(paths.active_dir, "archive"),
                    durable=False,
                )
        except OSError:
            pass

    # Best-effort mirror to the primary bus if we were forced onto a fallback.
    if paths.fallback_dir and paths.primary_dir != paths.active_dir:
        primary_events = os.path.join(paths.primary_dir, "events.ndjson")
        try:
            append_line(primary_events, line, durable=False)
            if rotate_enabled and os.path.getsize(primary_events) > rotate_bytes:
                rotate_log_tail(
                    primary_events,
                    retain_bytes=retain_bytes,
                    archive_dir=os.path.join(paths.primary_dir, "archive"),
                    durable=False,
                )
        except Exception:
            # Mirror failures are non-fatal; keep the fallback event.
            pass

    if partition_enabled:
        part_paths = _partition_paths(base_dir=paths.active_dir, topic=topic, kind=kind, level=level, actor=actor)
        for part_path in part_paths:
            append_line(part_path, line, durable=durable)
            if rotate_enabled:
                try:
                    if os.path.getsize(part_path) > rotate_bytes:
                        rotate_log_tail(
                            part_path,
                            retain_bytes=retain_bytes,
                            archive_dir=_partition_archive_dir(paths.active_dir, part_path),
                            durable=False,
                        )
                except OSError:
                    pass
        if paths.fallback_dir and paths.primary_dir != paths.active_dir:
            primary_parts = _partition_paths(
                base_dir=paths.primary_dir, topic=topic, kind=kind, level=level, actor=actor
            )
            for primary_part in primary_parts:
                try:
                    append_line(primary_part, line, durable=False)
                    if rotate_enabled and os.path.getsize(primary_part) > rotate_bytes:
                        rotate_log_tail(
                            primary_part,
                            retain_bytes=retain_bytes,
                            archive_dir=_partition_archive_dir(paths.primary_dir, primary_part),
                            durable=False,
                        )
                except Exception:
                    pass
    return event_id


def cmd_pub(args: argparse.Namespace) -> int:
    paths = resolve_bus_paths(args.bus_dir)
    data = json_load_maybe(args.data) if args.data is not None else None
    emit_event(
        paths,
        topic=args.topic,
        kind=args.kind,
        level=args.level,
        actor=args.actor,
        data=data,
        trace_id=args.trace_id,
        run_id=args.run_id,
        durable=args.durable,
    )
    return 0


def iter_lines_follow(path: str, poll_s: float = 0.25):
    ensure_dir(os.path.dirname(path))
    if not os.path.exists(path):
        open(path, "a", encoding="utf-8").close()
    with open(path, "r", encoding="utf-8", errors="replace") as f:
        f.seek(0, os.SEEK_END)
        while True:
            line = f.readline()
            if line:
                yield line
                continue
            time.sleep(poll_s)


def matches_filters(obj: dict, *, topic: str | None, actor: str | None, run_id: str | None) -> bool:
    if topic and obj.get("topic") != topic:
        return False
    if actor and obj.get("actor") != actor:
        return False
    if run_id and obj.get("run_id") != run_id:
        return False
    return True


def cmd_tail(args: argparse.Namespace) -> int:
    paths = resolve_bus_paths(args.bus_dir)
    for line in iter_lines_follow(paths.events_path, poll_s=args.poll):
        try:
            obj = json.loads(line)
        except Exception:
            continue
        if not matches_filters(obj, topic=args.topic, actor=args.actor, run_id=args.run_id):
            continue
        sys.stdout.write(line if args.raw else json.dumps(obj, indent=2, ensure_ascii=False) + "\n")
        sys.stdout.flush()
    return 0


def cmd_mk_run_id(_: argparse.Namespace) -> int:
    sys.stdout.write(str(uuid.uuid4()) + "\n")
    return 0


def cmd_resolve(args: argparse.Namespace) -> int:
    paths = resolve_bus_paths(args.bus_dir)
    if args.json:
        sys.stdout.write(
            json.dumps(
                {
                    "active_dir": paths.active_dir,
                    "events_path": paths.events_path,
                    "primary_dir": paths.primary_dir,
                    "fallback_dir": paths.fallback_dir,
                },
                ensure_ascii=False,
            )
            + "\n"
        )
        return 0
    if args.events_path:
        sys.stdout.write(paths.events_path + "\n")
        return 0
    sys.stdout.write(paths.active_dir + "\n")
    return 0


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="agent_bus.py", description="Append-only agent IPC bus (NDJSON).")
    p.add_argument("--bus-dir", default=None, help="Bus directory (default: $PLURIBUS_BUS_DIR or ./.pluribus/bus)")
    sub = p.add_subparsers(dest="cmd", required=True)

    pub = sub.add_parser("pub", help="Publish one event.")
    pub.add_argument("--topic", required=True, help="Stable dot-name topic, e.g. tests.unit.start")
    pub.add_argument("--kind", default="log", choices=["log", "request", "response", "artifact", "metric"])
    pub.add_argument("--level", default="info", choices=["debug", "info", "warn", "error"])
    pub.add_argument("--actor", default=default_actor())
    pub.add_argument("--trace-id", default=None)
    pub.add_argument("--run-id", default=None)
    pub.add_argument("--data", default=None, help="JSON string, or '-' to read JSON from stdin")
    pub.add_argument("--durable", action="store_true", help="fsync after append")
    pub.set_defaults(func=cmd_pub)

    tail = sub.add_parser("tail", help="Tail events (follow).")
    tail.add_argument("--topic", default=None)
    tail.add_argument("--actor", default=None)
    tail.add_argument("--run-id", default=None)
    tail.add_argument("--raw", action="store_true", help="Print raw NDJSON lines")
    tail.add_argument("--poll", type=float, default=0.25, help="Poll interval seconds")
    tail.set_defaults(func=cmd_tail)

    mk = sub.add_parser("mk-run-id", help="Print a new run id (uuid4).")
    mk.set_defaults(func=cmd_mk_run_id)

    resolve = sub.add_parser("resolve", help="Resolve and print the active bus path.")
    resolve.add_argument("--json", action="store_true", help="Print JSON with active/primary/fallback and events_path.")
    resolve.add_argument("--events-path", action="store_true", help="Print the resolved events.ndjson path.")
    resolve.set_defaults(func=cmd_resolve)

    return p


def main(argv: list[str]) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
