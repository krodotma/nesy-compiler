#!/usr/bin/env python3
"""
METATEST Collector - Comprehensive Test Inventory & Execution Tool

Part of the METATEST dashboard system. Collects test inventory across:
- Unit tests (nucleus/tools/tests/)
- E2E tests (nucleus/dashboard/e2e/)
- Service tests (systemd services)
- Coverage analysis

Protocol: DKIN v29 | PAIP v15 | CITIZEN v1

Bus Topics:
- metatest.inventory.update: Publishes inventory data
- metatest.execute.start: Test execution started
- metatest.execute.result: Individual test result
- metatest.execute.complete: Execution finished
- metatest.coverage.update: Coverage stats updated
- metatest.alert.critical: Critical gap/failure alert

Usage:
    python3 metatest_collector.py --inventory    # Collect full inventory
    python3 metatest_collector.py --status       # Current test status
    python3 metatest_collector.py --gaps         # Critical gaps report
    python3 metatest_collector.py --json         # Output as JSON
"""

import argparse
import datetime
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any

# Paths
NUCLEUS_TOOLS = Path("/pluribus/nucleus/tools")
NUCLEUS_TESTS = NUCLEUS_TOOLS / "tests"
NUCLEUS_ARCHIVE = NUCLEUS_TOOLS / "archive"
DASHBOARD_E2E = Path("/pluribus/nucleus/dashboard/e2e")
SYSTEMD_SERVICES = Path("/pluribus/nucleus/deploy/systemd")
PLURIBUS_NEXT = Path("/pluribus/pluribus_next/tools")
BUS_PATH = Path("/pluribus/.pluribus/bus/events.ndjson")

# Critical tools that MUST have tests
CRITICAL_TOOLS = {
    "omega_guardian.py": "Buchi/Rabin/Streett verification",
    "codemaster_agent.py": "Merge authority for protected branches",
    "codemaster_confirm.py": "Merge confirmation",
    "cagent_bootstrap.py": "Citizen Agent initialization",
    "paip_isolation.py": "Parallel agent isolation",
    "paip_mitosis.py": "Clone management",
    "dkin_amber.py": "Dead Man's Switch snapshots",
    "dkin_flow_monitor.py": "DKIN flow state tracking",
    "pbtest_operator.py": "Test execution framework",
    "pbcmaster_operator.py": "Merge request operator",
    "pbaudit_operator.py": "Audit log operations",
    "pbverify_operator.py": "Verification operations",
    "pbreconcile_operator.py": "Reconciliation",
    "pbwua_operator.py": "WUA protocol operator",
    "verify_system.py": "System verification",
    "verify_3pillar.py": "3-pillar lifecycle verification",
    "verify_inception.py": "Inception phase verification",
    "process_guardian.py": "Process lifecycle guardrail",
}


def emit_event(topic: str, data: dict, actor: str = "metatest") -> None:
    """Emit event to bus."""
    try:
        event = {
            "topic": topic,
            "actor": actor,
            "iso": datetime.datetime.now(datetime.timezone.utc).isoformat(),
            "data": data,
        }
        with open(BUS_PATH, "a", encoding="utf-8") as f:
            f.write(json.dumps(event) + "\n")
    except Exception as e:
        print(f"[WARN] Bus emit failed: {e}", file=sys.stderr)


def collect_tools() -> list[dict]:
    """Collect all Python tools in nucleus/tools."""
    tools = []
    if NUCLEUS_TOOLS.exists():
        for py_file in sorted(NUCLEUS_TOOLS.glob("*.py")):
            if py_file.name.startswith("_"):
                continue
            tools.append({
                "name": py_file.name,
                "path": str(py_file),
                "type": "tool",
                "status": "active",
            })
    return tools


def collect_tests() -> list[dict]:
    """Collect all test files."""
    tests = []
    if NUCLEUS_TESTS.exists():
        for test_file in sorted(NUCLEUS_TESTS.glob("test_*.py")):
            tests.append({
                "name": test_file.name,
                "path": str(test_file),
                "type": "unit",
                "target": test_file.name.replace("test_", "").replace("_unittest", "").replace(".py", ""),
            })
    return tests


def collect_e2e_tests() -> list[dict]:
    """Collect Playwright E2E tests."""
    tests = []
    if DASHBOARD_E2E.exists():
        for spec_file in sorted(DASHBOARD_E2E.glob("*.spec.ts")):
            tests.append({
                "name": spec_file.name,
                "path": str(spec_file),
                "type": "e2e",
            })
    return tests


def collect_services() -> list[dict]:
    """Collect systemd services."""
    services = []
    if SYSTEMD_SERVICES.exists():
        for svc_file in sorted(SYSTEMD_SERVICES.glob("*.service")):
            services.append({
                "name": svc_file.name.replace(".service", ""),
                "path": str(svc_file),
                "type": "service",
            })
    return services


def check_service_status(service_name: str) -> str:
    """Check if a systemd service is running."""
    try:
        result = subprocess.run(
            ["systemctl", "is-active", f"{service_name}.service"],
            capture_output=True,
            text=True,
            timeout=5,
        )
        return result.stdout.strip()
    except Exception:
        return "unknown"


def map_test_coverage(tools: list[dict], tests: list[dict]) -> list[dict]:
    """Map tools to their test files and calculate coverage."""
    test_targets = {t["target"]: t for t in tests if "target" in t}

    coverage_map = []
    for tool in tools:
        tool_base = tool["name"].replace(".py", "")
        test_file = test_targets.get(tool_base)

        is_critical = tool["name"] in CRITICAL_TOOLS
        has_test = test_file is not None

        tier = "critical" if is_critical else "medium"
        if tool["name"].startswith("pb"):
            tier = "high" if not has_test else tier

        coverage_map.append({
            "tool": tool["name"],
            "path": tool["path"],
            "has_test": has_test,
            "test_file": test_file["name"] if test_file else None,
            "tier": tier,
            "status": "tested" if has_test else "missing",
            "critical": is_critical,
            "critical_reason": CRITICAL_TOOLS.get(tool["name"]),
        })

    return coverage_map


def get_inventory() -> dict:
    """Get full test inventory."""
    tools = collect_tools()
    tests = collect_tests()
    e2e_tests = collect_e2e_tests()
    services = collect_services()
    coverage_map = map_test_coverage(tools, tests)

    tested_count = sum(1 for c in coverage_map if c["has_test"])
    total_count = len(coverage_map)
    coverage_pct = round((tested_count / total_count * 100), 1) if total_count > 0 else 0

    critical_gaps = [c for c in coverage_map if c["critical"] and not c["has_test"]]

    return {
        "version": "1.0.0",
        "generated": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "summary": {
            "total_tools": len(tools),
            "total_tests": len(tests),
            "total_e2e": len(e2e_tests),
            "total_services": len(services),
            "coverage_percent": coverage_pct,
            "tested_count": tested_count,
            "untested_count": total_count - tested_count,
            "critical_gaps": len(critical_gaps),
        },
        "categories": [
            {
                "id": "unit",
                "name": "Unit Tests",
                "icon": "🧪",
                "count": len(tests),
                "items": tests[:50],  # Limit for performance
            },
            {
                "id": "e2e",
                "name": "E2E Tests",
                "icon": "🎭",
                "count": len(e2e_tests),
                "items": e2e_tests,
            },
            {
                "id": "services",
                "name": "Services",
                "icon": "⚙️",
                "count": len(services),
                "items": services[:30],  # Limit for performance
            },
            {
                "id": "coverage",
                "name": "Coverage Map",
                "icon": "📊",
                "count": total_count,
                "coverage_percent": coverage_pct,
                "items": coverage_map[:100],  # Limit for performance
            },
        ],
        "critical_gaps": critical_gaps,
        "protocol": "DKIN v29 | PAIP v15 | CITIZEN v1",
    }


def get_gaps() -> dict:
    """Get critical test gaps."""
    inventory = get_inventory()
    return {
        "generated": inventory["generated"],
        "critical_gaps": inventory["critical_gaps"],
        "total_gaps": inventory["summary"]["untested_count"],
        "coverage_percent": inventory["summary"]["coverage_percent"],
    }


def main():
    parser = argparse.ArgumentParser(description="METATEST Collector")
    parser.add_argument("--inventory", action="store_true", help="Collect full inventory")
    parser.add_argument("--status", action="store_true", help="Current test status")
    parser.add_argument("--gaps", action="store_true", help="Critical gaps report")
    parser.add_argument("--json", action="store_true", help="Output as JSON")
    parser.add_argument("--emit", action="store_true", help="Emit to bus")
    args = parser.parse_args()

    if args.inventory or args.status:
        data = get_inventory()
        if args.emit:
            emit_event("metatest.inventory.update", data)
        if args.json:
            print(json.dumps(data, indent=2))
        else:
            print(f"METATEST Inventory ({data['generated']})")
            print("=" * 60)
            s = data["summary"]
            print(f"Tools:     {s['total_tools']}")
            print(f"Tests:     {s['total_tests']} unit + {s['total_e2e']} e2e")
            print(f"Services:  {s['total_services']}")
            print(f"Coverage:  {s['coverage_percent']}% ({s['tested_count']}/{s['total_tools']})")
            print(f"Critical Gaps: {s['critical_gaps']}")
            print()
            if data["critical_gaps"]:
                print("CRITICAL UNTESTED TOOLS:")
                for gap in data["critical_gaps"]:
                    print(f"  - {gap['tool']}: {gap['critical_reason']}")

    elif args.gaps:
        data = get_gaps()
        if args.emit:
            emit_event("metatest.alert.critical", data)
        if args.json:
            print(json.dumps(data, indent=2))
        else:
            print(f"METATEST Critical Gaps ({data['generated']})")
            print("=" * 60)
            print(f"Total Gaps: {data['total_gaps']}")
            print(f"Coverage:   {data['coverage_percent']}%")
            print()
            for gap in data["critical_gaps"]:
                print(f"  ❌ {gap['tool']}: {gap['critical_reason']}")

    else:
        parser.print_help()


if __name__ == "__main__":
    main()
