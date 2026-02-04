#!/usr/bin/env python3
"""
CAGENT Audit Tool - Compliance Verification

Checks agent compliance with DKIN v28 CAGENT protocol.
Emits pbclitest.verdict events to bus.

Usage:
    cagent_audit.py check              # Run compliance check
    cagent_audit.py report             # Generate detailed report
    cagent_audit.py --actor claude     # Check specific actor
"""

import argparse
import json
import os
import sys
import time
import uuid
from pathlib import Path
from typing import Tuple
try:
    import repl_header_audit
except ImportError:
    repl_header_audit = None

PLURIBUS_ROOT = os.environ.get("PLURIBUS_ROOT", "/pluribus")
BUS_DIR = os.environ.get("PLURIBUS_BUS_DIR", f"{PLURIBUS_ROOT}/.pluribus/bus")
SPECS_DIR = f"{PLURIBUS_ROOT}/nucleus/specs"


def emit_bus_event(topic: str, kind: str, data: dict, level: str = "info"):
    """Emit event to Pluribus bus."""
    event = {
        "id": str(uuid.uuid4()),
        "ts": time.time(),
        "iso": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "topic": topic,
        "kind": kind,
        "level": level,
        "actor": os.environ.get("PLURIBUS_ACTOR", "cagent_audit"),
        "host": os.uname().nodename,
        "pid": os.getpid(),
        "data": data,
    }
    bus_file = f"{BUS_DIR}/events.ndjson"
    try:
        with open(bus_file, "a") as f:
            f.write(json.dumps(event) + "\n")
    except Exception:
        pass
    return event["id"]


def check_file_exists(path: str) -> Tuple[bool, str]:
    """Check if file exists and return status."""
    exists = os.path.exists(path)
    return exists, "PASS" if exists else "FAIL"


def check_json_valid(path: str) -> Tuple[bool, str]:
    """Check if JSON file is valid."""
    try:
        with open(path) as f:
            json.load(f)
        return True, "PASS"
    except (FileNotFoundError, json.JSONDecodeError):
        return False, "FAIL"


def check_constitution_principles() -> Tuple[int, int]:
    """Check CITIZEN.md has all 10 principles."""
    principles = [
        "Append-Only Evidence",
        "Non-Blocking IPC",
        "Tests-First",
        "No Secrets Emission",
        "Conservation of Work",
        "Replisome Compliance",
        "Protocol Versioning",
        "Deterministic Behavior",
        "Self-Verification",
        "Graceful Degradation"
    ]
    found = 0
    try:
        with open(f"{SPECS_DIR}/CITIZEN.md") as f:
            content = f.read()
            for p in principles:
                if p in content:
                    found += 1
    except FileNotFoundError:
        pass
    return found, len(principles)


def check_version_consistency() -> Tuple[int, int, list]:
    """Check DKIN version is v28 across all files."""
    checks = []
    files_to_check = [
        (f"{SPECS_DIR}/CITIZEN.md", "DKIN v28"),
        (f"{SPECS_DIR}/cagent_adaptations.json", '"dkin_version": "v28"'),
        (f"{SPECS_DIR}/cagent_protocol_v1.md", "DKIN Version:** v28"),
    ]
    passed = 0
    for path, pattern in files_to_check:
        try:
            with open(path) as f:
                content = f.read()
                if pattern in content:
                    passed += 1
                    checks.append((os.path.basename(path), "PASS"))
                else:
                    checks.append((os.path.basename(path), "FAIL"))
        except FileNotFoundError:
            checks.append((os.path.basename(path), "MISSING"))
    return passed, len(files_to_check), checks


def check_registry_taxonomy() -> Tuple[int, int]:
    """Check registry has proper taxonomy."""
    checks = 0
    total = 4
    try:
        with open(f"{SPECS_DIR}/cagent_registry.json") as f:
            registry = json.load(f)

        if "class_aliases" in registry:
            checks += 1
        if "actors" in registry:
            checks += 1
        aliases = registry.get("class_aliases", {})
        if aliases.get("sagent") == "superagent":
            checks += 1
        if aliases.get("swagent") == "superworker":
            checks += 1
    except (FileNotFoundError, json.JSONDecodeError):
        pass
    return checks, total


def check_canonical_paths() -> Tuple[int, int]:
    """Check paths registry has all sections."""
    sections = ["roots", "bus", "state", "specs", "tools"]
    found = 0
    try:
        with open(f"{SPECS_DIR}/cagent_paths.json") as f:
            paths = json.load(f)
            for s in sections:
                if s in paths:
                    found += 1
    except (FileNotFoundError, json.JSONDecodeError):
        pass
    return found, len(sections)


def run_audit() -> dict:
    """Run full CAGENT compliance audit."""
    results = {
        "categories": {},
        "total_passed": 0,
        "total_checks": 0,
        "score_percent": 0,
        "verdict": "FAIL"
    }

    # 1. File Existence
    required_files = [
        f"{SPECS_DIR}/CITIZEN.md",
        f"{SPECS_DIR}/cagent_protocol_v1.md",
        f"{SPECS_DIR}/cagent_adaptations.json",
        f"{SPECS_DIR}/cagent_paths.json",
        f"{SPECS_DIR}/cagent_registry.json",
        f"{SPECS_DIR}/dkin_protocol_v26_replisome.md",
        f"{SPECS_DIR}/dkin_protocol_v28_cagent.md",
    ]
    file_passed = sum(1 for f in required_files if os.path.exists(f))
    results["categories"]["file_existence"] = {
        "passed": file_passed,
        "total": len(required_files),
        "status": "PASS" if file_passed == len(required_files) else "FAIL"
    }

    # 2. JSON Validity
    json_files = [
        f"{SPECS_DIR}/cagent_adaptations.json",
        f"{SPECS_DIR}/cagent_paths.json",
        f"{SPECS_DIR}/cagent_registry.json",
    ]
    json_passed = sum(1 for f in json_files if check_json_valid(f)[0])
    results["categories"]["json_validity"] = {
        "passed": json_passed,
        "total": len(json_files),
        "status": "PASS" if json_passed == len(json_files) else "FAIL"
    }

    # 3. Constitution Principles
    principles_found, principles_total = check_constitution_principles()
    results["categories"]["constitution_principles"] = {
        "passed": principles_found,
        "total": principles_total,
        "status": "PASS" if principles_found == principles_total else "FAIL"
    }

    # 4. Version Consistency
    version_passed, version_total, _ = check_version_consistency()
    results["categories"]["version_consistency"] = {
        "passed": version_passed,
        "total": version_total,
        "status": "PASS" if version_passed == version_total else "FAIL"
    }

    # 5. Registry Taxonomy
    taxonomy_passed, taxonomy_total = check_registry_taxonomy()
    results["categories"]["registry_taxonomy"] = {
        "passed": taxonomy_passed,
        "total": taxonomy_total,
        "status": "PASS" if taxonomy_passed == taxonomy_total else "FAIL"
    }

    # 6. Canonical Paths
    paths_passed, paths_total = check_canonical_paths()
    results["categories"]["canonical_paths"] = {
        "passed": paths_passed,
        "total": paths_total,
        "status": "PASS" if paths_passed == paths_total else "FAIL"
    }

    # 7. REPL Header Contract
    if repl_header_audit:
        repl_results = repl_header_audit.run_audit()
        results["categories"]["repl_header_contract"] = {
            "passed": repl_results["total_passed"],
            "total": repl_results["total_checks"],
            "status": "PASS" if repl_results["verdict"] == "PASS" else "FAIL"
        }

    # Calculate totals
    for cat in results["categories"].values():
        results["total_passed"] += cat["passed"]
        results["total_checks"] += cat["total"]

    if results["total_checks"] > 0:
        results["score_percent"] = round(
            (results["total_passed"] / results["total_checks"]) * 100, 1
        )

    # Determine verdict
    if results["score_percent"] >= 90:
        results["verdict"] = "PASS"
    elif results["score_percent"] >= 70:
        results["verdict"] = "WARN"
    else:
        results["verdict"] = "FAIL"

    return results


def print_report(results: dict):
    """Print formatted audit report."""
    print("=" * 50)
    print("CAGENT COMPLIANCE AUDIT REPORT")
    print("=" * 50)
    print()

    for cat_name, cat_data in results["categories"].items():
        status_icon = "✓" if cat_data["status"] == "PASS" else "✗"
        print(f"{status_icon} {cat_name.replace('_', ' ').title()}: "
              f"{cat_data['passed']}/{cat_data['total']} {cat_data['status']}")

    print()
    print("-" * 50)
    print(f"Total: {results['total_passed']}/{results['total_checks']} "
          f"({results['score_percent']}%)")
    print(f"Verdict: {results['verdict']}")
    print("=" * 50)


def main():
    parser = argparse.ArgumentParser(description="CAGENT Compliance Audit Tool")
    parser.add_argument("command", nargs="?", default="check",
                        choices=["check", "report"],
                        help="Command to run")
    parser.add_argument("--actor", help="Specific actor to check")
    parser.add_argument("--json", action="store_true", help="Output JSON")
    parser.add_argument("--emit", action="store_true", help="Emit to bus")

    args = parser.parse_args()

    results = run_audit()

    if args.emit:
        emit_bus_event("pbclitest.verdict", "artifact", {
            "scope": "CAGENT Full System",
            "mode": "integration",
            "intent": "CAGENT compliance audit",
            "categories": {k: f"{v['passed']}/{v['total']} {v['status']}"
                          for k, v in results["categories"].items()},
            "score": f"{results['total_passed']}/{results['total_checks']} "
                     f"({results['score_percent']}%)",
            "verdict": results["verdict"],
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
        })

    if args.json:
        print(json.dumps(results, indent=2))
    else:
        print_report(results)

    # Exit code based on verdict
    sys.exit(0 if results["verdict"] == "PASS" else 1)


if __name__ == "__main__":
    main()
