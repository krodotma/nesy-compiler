#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import socket
import time
import uuid
from pathlib import Path

from .code_analyzer import CodeAnalyzer
from .drift_detector import DriftDetector
from .vector_profiler import VectorProfiler


def now_iso_utc() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def default_bus_dir() -> Path:
    return Path(os.environ.get("PLURIBUS_BUS_DIR") or "/pluribus/.pluribus/bus").expanduser().resolve()


def default_actor() -> str:
    return os.environ.get("PLURIBUS_ACTOR") or "pluribus-evolution"


def append_event(bus_dir: Path, *, topic: str, kind: str, level: str, actor: str, data: dict) -> None:
    bus_dir.mkdir(parents=True, exist_ok=True)
    path = bus_dir / "events.ndjson"
    event = {
        "id": str(uuid.uuid4()),
        "ts": time.time(),
        "iso": now_iso_utc(),
        "topic": topic,
        "kind": kind,
        "level": level,
        "actor": actor,
        "host": socket.gethostname(),
        "data": data,
    }
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(event, ensure_ascii=False, separators=(",", ":")) + "\n")


def run_once(*, analysis_root: str, genotype_root: str, phenotype_root: str, bus_dir: Path, actor: str) -> None:
    analyzer = CodeAnalyzer(primary_root=analysis_root)
    analysis_results = analyzer.analyze_directory(analysis_root)
    analysis_event = analyzer.to_bus_event(analysis_results)
    append_event(bus_dir, topic=analysis_event["topic"], kind=analysis_event["kind"], level=analysis_event["level"], actor=actor, data=analysis_event["data"])

    detector = DriftDetector(genotype_root=genotype_root, phenotype_root=phenotype_root)
    drift_report = detector.detect_all()
    drift_event = detector.to_bus_event(drift_report)
    append_event(bus_dir, topic=drift_event["topic"], kind=drift_event["kind"], level=drift_event["level"], actor=actor, data=drift_event["data"])

    profiler = VectorProfiler(root_path=analysis_root)
    snapshot = profiler.profile_directory()
    profile_event = profiler.to_bus_event(snapshot)
    append_event(bus_dir, topic=profile_event["topic"], kind=profile_event["kind"], level=profile_event["level"], actor=actor, data=profile_event["data"])


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Pluribus Evolution Observer Daemon")
    ap.add_argument("--analysis-root", default=os.environ.get("PLURIBUS_EVOLUTION_ANALYSIS_ROOT") or "/pluribus/nucleus/tools")
    ap.add_argument("--genotype-root", default=os.environ.get("PLURIBUS_EVOLUTION_GENOTYPE_ROOT") or "/pluribus/nucleus/specs")
    ap.add_argument("--phenotype-root", default=os.environ.get("PLURIBUS_EVOLUTION_PHENOTYPE_ROOT") or "/pluribus/nucleus/tools")
    ap.add_argument("--bus-dir", default=os.environ.get("PLURIBUS_BUS_DIR") or "/pluribus/.pluribus/bus")
    ap.add_argument("--actor", default=os.environ.get("PLURIBUS_ACTOR") or "pluribus-evolution")
    ap.add_argument("--interval-s", type=int, default=int(os.environ.get("PLURIBUS_EVOLUTION_INTERVAL_S") or "1800"))
    ap.add_argument("--once", action="store_true", help="Run a single observation pass and exit")
    return ap.parse_args()


def main() -> int:
    args = parse_args()
    bus_dir = Path(args.bus_dir).expanduser().resolve()
    interval_s = max(60, int(args.interval_s))

    # Loop to avoid systemd restart storms and keep resource use predictable.
    while True:
        try:
            run_once(
                analysis_root=args.analysis_root,
                genotype_root=args.genotype_root,
                phenotype_root=args.phenotype_root,
                bus_dir=bus_dir,
                actor=args.actor,
            )
        except Exception as exc:
            append_event(
                bus_dir,
                topic="evolution.observer.error",
                kind="log",
                level="error",
                actor=args.actor,
                data={"error": str(exc)},
            )

        if args.once:
            break
        time.sleep(interval_s)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
