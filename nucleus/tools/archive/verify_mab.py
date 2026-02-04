#!/usr/bin/env python3
"""
MAB Verification Tool (Multi-Agent Bus)
=======================================

Verifies the integrity, schema compliance, and liveness of the Pluribus Bus.

Usage:
    python3 verify_mab.py --window 60
"""

import argparse
import json
import time
from pathlib import Path
from collections import Counter

sys_dont_write_bytecode = True

def verify_schema(event: dict) -> list[str]:
    errors = []
    required = ["id", "ts", "iso", "topic", "kind", "level", "actor", "data"]
    for field in required:
        if field not in event:
            errors.append(f"missing_{field}")
    
    if "kind" in event and event["kind"] not in ["metric", "request", "response", "artifact", "log", "signal"]:
        errors.append(f"invalid_kind:{event['kind']}")
        
    return errors

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--bus-dir", default="/pluribus/.pluribus/bus")
    parser.add_argument("--window", type=int, default=60, help="Analysis window in seconds")
    args = parser.parse_args()
    
    bus_path = Path(args.bus_dir) / "events.ndjson"
    
    if not bus_path.exists():
        print(json.dumps({"status": "fail", "error": "bus_not_found"}))
        return

    now = time.time()
    cutoff = now - args.window
    
    stats = {
        "total_events": 0,
        "valid_events": 0,
        "malformed_lines": 0,
        "schema_violations": 0,
        "schema_errors": Counter(),
        "recent_count": 0,
        "kinds": Counter(),
        "topics": Counter(),
        "actors": Counter()
    }
    
    try:
        # Read file (best effort, handling potential races roughly by just reading)
        with open(bus_path, "r", encoding="utf-8", errors="replace") as f:
            for line in f:
                stats["total_events"] += 1
                line = line.strip()
                if not line: continue
                
                try:
                    event = json.loads(line)
                except json.JSONDecodeError:
                    stats["malformed_lines"] += 1
                    continue
                
                errors = verify_schema(event)
                if errors:
                    stats["schema_violations"] += 1
                    for e in errors:
                        stats["schema_errors"][e] += 1
                else:
                    stats["valid_events"] += 1
                    
                # Analyze content
                stats["kinds"][event.get("kind", "unknown")] += 1
                stats["topics"][event.get("topic", "unknown")] += 1
                stats["actors"][event.get("actor", "unknown")] += 1
                
                # Time window
                try:
                    ts = float(event.get("ts", 0))
                    if ts >= cutoff:
                        stats["recent_count"] += 1
                except:
                    pass

    except Exception as e:
        print(json.dumps({"status": "error", "error": str(e)}))
        return

    # Rate calculation
    rate_mps = stats["recent_count"] / (args.window / 60.0)
    
    report = {
        "status": "ok" if stats["malformed_lines"] == 0 else "degraded",
        "mab_integrity": {
            "total_lines": stats["total_events"],
            "valid_json": stats["total_events"] - stats["malformed_lines"],
            "schema_compliant": stats["valid_events"],
            "compliance_rate": stats["valid_events"] / max(1, stats["total_events"])
        },
        "liveness": {
            "window_s": args.window,
            "events_in_window": stats["recent_count"],
            "rate_events_per_min": round(rate_mps, 2)
        },
        "distribution": {
            "kinds": dict(stats["kinds"].most_common(5)),
            "top_topics": dict(stats["topics"].most_common(5)),
            "active_actors": dict(stats["actors"].most_common(5))
        },
        "violations": dict(stats["schema_errors"])
    }
    
    print(json.dumps(report, indent=2))

if __name__ == "__main__":
    main()
