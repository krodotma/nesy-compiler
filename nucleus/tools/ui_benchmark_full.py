#!/usr/bin/env python3
"""
Full UI Benchmark - Measures complete page interactivity, not just initial load.

Captures:
- Initial load time
- Hydration time (Qwik ready)
- WebSocket connection time
- Full interactivity (all components rendered)
- View switching latency

Usage:
    python3 ui_benchmark_full.py --url https://kroma.live --runs 3
"""
from __future__ import annotations

import argparse
import asyncio
import json
import os
import statistics
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def now_iso() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def emit_bus_event(bus_dir: str, topic: str, kind: str, level: str, data: dict) -> None:
    """Emit event to bus."""
    try:
        events_path = Path(bus_dir) / "events.ndjson"
        events_path.parent.mkdir(parents=True, exist_ok=True)
        event = {
            "id": f"fullbench-{int(time.time()*1000)}",
            "ts": time.time(),
            "iso": now_iso(),
            "topic": topic,
            "kind": kind,
            "level": level,
            "actor": "ui-benchmark-full",
            "host": os.uname().nodename,
            "pid": os.getpid(),
            "data": data,
        }
        with open(events_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(event, ensure_ascii=False, separators=(",", ":")) + "\n")
    except Exception as e:
        print(f"[WARN] Failed to emit bus event: {e}", file=sys.stderr)


async def run_full_benchmark(url: str, runs: int, bus_dir: str) -> dict[str, Any]:
    """Run comprehensive page load benchmarks."""
    try:
        from playwright.async_api import async_playwright
    except ImportError:
        print("[ERROR] Playwright not installed.", file=sys.stderr)
        return {"error": "playwright_missing"}

    results: list[dict[str, Any]] = []
    print(f"\n{'='*70}")
    print(f"FULL UI BENCHMARK: {url}")
    print(f"Runs: {runs} | Measuring: Load -> Hydration -> Interactive")
    print(f"{'='*70}\n")

    emit_bus_event(bus_dir, "benchmark.ui.full.start", "metric", "info", {
        "url": url,
        "runs": runs,
        "timestamp": now_iso(),
    })

    async with async_playwright() as p:
        browser = await p.chromium.launch(headless=True)

        for i in range(runs):
            context = await browser.new_context()
            page = await context.new_page()

            timing: dict[str, Any] = {
                "run": i + 1,
                "url": url,
                "phases": {},
            }

            # Capture console output for debugging
            console_logs: list[str] = []
            page.on("console", lambda msg: console_logs.append(f"[{msg.type}] {msg.text}"))

            try:
                overall_start = time.perf_counter()

                # Phase 1: Initial HTML load
                response = await page.goto(url, wait_until="commit", timeout=60000)
                phase1_end = time.perf_counter()
                timing["phases"]["html_commit"] = round((phase1_end - overall_start) * 1000, 2)

                # Phase 2: DOM Content Loaded
                await page.wait_for_load_state("domcontentloaded", timeout=30000)
                phase2_end = time.perf_counter()
                timing["phases"]["dom_content_loaded"] = round((phase2_end - overall_start) * 1000, 2)

                # Phase 3: Full load event
                await page.wait_for_load_state("load", timeout=30000)
                phase3_end = time.perf_counter()
                timing["phases"]["load_event"] = round((phase3_end - overall_start) * 1000, 2)

                # Phase 4: Wait for Qwik hydration marker
                try:
                    await page.wait_for_selector('[data-qwik-ready="1"]', timeout=15000)
                    phase4_end = time.perf_counter()
                    timing["phases"]["qwik_ready"] = round((phase4_end - overall_start) * 1000, 2)
                except Exception:
                    timing["phases"]["qwik_ready"] = None

                # Phase 5: Wait for main content (nav + drawer)
                try:
                    # Wait for navigation to be visible
                    await page.wait_for_selector('[class*="nav"], [class*="Nav"], nav', timeout=10000)
                    phase5_end = time.perf_counter()
                    timing["phases"]["nav_visible"] = round((phase5_end - overall_start) * 1000, 2)
                except Exception:
                    timing["phases"]["nav_visible"] = None

                # Phase 6: Wait for WebSocket connection (look for bus client)
                try:
                    ws_connected = await page.evaluate("""() => {
                        return new Promise((resolve) => {
                            // Check if already connected
                            if (window.__pluribusWsConnected) {
                                resolve(true);
                                return;
                            }
                            // Listen for connection event
                            const listener = () => {
                                resolve(true);
                                window.removeEventListener('pluribus:ws-connected', listener);
                            };
                            window.addEventListener('pluribus:ws-connected', listener);
                            // Timeout fallback
                            setTimeout(() => resolve(false), 5000);
                        });
                    }""")
                    phase6_end = time.perf_counter()
                    timing["phases"]["ws_connected"] = round((phase6_end - overall_start) * 1000, 2) if ws_connected else None
                except Exception:
                    timing["phases"]["ws_connected"] = None

                # Phase 7: Network idle (no pending requests for 500ms)
                try:
                    await page.wait_for_load_state("networkidle", timeout=30000)
                    phase7_end = time.perf_counter()
                    timing["phases"]["network_idle"] = round((phase7_end - overall_start) * 1000, 2)
                except Exception:
                    timing["phases"]["network_idle"] = None

                # Total time to interactive
                total_end = time.perf_counter()
                timing["total_ms"] = round((total_end - overall_start) * 1000, 2)

                # Get resource breakdown
                timing["resources"] = await page.evaluate("""() => {
                    const resources = performance.getEntriesByType('resource');
                    let scripts = 0, scriptBytes = 0;
                    let styles = 0, styleBytes = 0;
                    let fonts = 0, fontBytes = 0;
                    let images = 0, imgBytes = 0;
                    let other = 0, otherBytes = 0;

                    for (const r of resources) {
                        const size = r.transferSize || r.encodedBodySize || 0;
                        if (r.initiatorType === 'script' || r.name.endsWith('.js')) {
                            scripts++;
                            scriptBytes += size;
                        } else if (r.initiatorType === 'css' || r.name.endsWith('.css')) {
                            styles++;
                            styleBytes += size;
                        } else if (r.name.includes('font') || /\\.(woff|woff2|ttf|otf)/.test(r.name)) {
                            fonts++;
                            fontBytes += size;
                        } else if (r.initiatorType === 'img' || /\\.(png|jpg|svg|gif|webp)/.test(r.name)) {
                            images++;
                            imgBytes += size;
                        } else {
                            other++;
                            otherBytes += size;
                        }
                    }

                    return {
                        total: resources.length,
                        scripts: { count: scripts, bytes: scriptBytes },
                        styles: { count: styles, bytes: styleBytes },
                        fonts: { count: fonts, bytes: fontBytes },
                        images: { count: images, bytes: imgBytes },
                        other: { count: other, bytes: otherBytes },
                    };
                }""")

                # Check for console errors
                timing["console_errors"] = len([l for l in console_logs if l.startswith("[error]")])

                # Determine grade based on total time
                grade = "FAST" if timing["total_ms"] < 2000 else "MEDIUM" if timing["total_ms"] < 5000 else "SLOW"
                timing["grade"] = grade

                results.append(timing)

                # Print progress with phase breakdown
                color = "\033[92m" if grade == "FAST" else "\033[93m" if grade == "MEDIUM" else "\033[91m"
                reset = "\033[0m"

                print(f"  Run {i+1}/{runs}: {color}Total: {timing['total_ms']:.0f}ms ({grade}){reset}")
                print(f"      Phases: HTML={timing['phases'].get('html_commit', 'n/a')}ms"
                      f" | DCL={timing['phases'].get('dom_content_loaded', 'n/a')}ms"
                      f" | Load={timing['phases'].get('load_event', 'n/a')}ms"
                      f" | Qwik={timing['phases'].get('qwik_ready', 'n/a')}ms"
                      f" | Idle={timing['phases'].get('network_idle', 'n/a')}ms")
                print(f"      Resources: {timing['resources']['total']} files"
                      f" | JS: {timing['resources']['scripts']['count']} ({timing['resources']['scripts']['bytes']//1024}KB)"
                      f" | CSS: {timing['resources']['styles']['count']}")

            except Exception as e:
                timing["error"] = str(e)
                results.append(timing)
                print(f"  Run {i+1}/{runs}: ERROR - {e}", file=sys.stderr)

            await context.close()

            if i < runs - 1:
                await asyncio.sleep(2)  # Longer pause between full runs

        await browser.close()

    # Calculate statistics
    total_times = [r["total_ms"] for r in results if "total_ms" in r and "error" not in r]
    phases_data: dict[str, list[float]] = {}
    for r in results:
        if "phases" in r and "error" not in r:
            for phase, value in r["phases"].items():
                if value is not None:
                    phases_data.setdefault(phase, []).append(value)

    summary: dict[str, Any] = {
        "url": url,
        "runs": runs,
        "successful_runs": len(total_times),
        "timestamp": now_iso(),
    }

    if total_times:
        summary["total_time"] = {
            "min": round(min(total_times), 2),
            "max": round(max(total_times), 2),
            "mean": round(statistics.mean(total_times), 2),
            "median": round(statistics.median(total_times), 2),
            "p90": round(sorted(total_times)[int(len(total_times) * 0.9)] if total_times else 0, 2),
        }

    summary["phases"] = {}
    for phase, values in phases_data.items():
        if values:
            summary["phases"][phase] = {
                "mean": round(statistics.mean(values), 2),
                "max": round(max(values), 2),
            }

    # Resource summary
    if results and "resources" in results[-1]:
        summary["resources"] = results[-1]["resources"]

    # Grade
    if total_times:
        median = statistics.median(total_times)
        summary["overall_grade"] = "FAST" if median < 2000 else "MEDIUM" if median < 5000 else "SLOW"
        summary["target_met"] = median < 2000

    emit_bus_event(bus_dir, "benchmark.ui.full.complete", "metric", "info", {
        "summary": summary,
        "runs": results,
    })

    return summary


def print_summary(summary: dict[str, Any]) -> None:
    """Print formatted summary."""
    print(f"\n{'='*70}")
    print("FULL BENCHMARK SUMMARY")
    print(f"{'='*70}\n")

    print(f"  URL: {summary.get('url')}")
    print(f"  Runs: {summary.get('successful_runs')}/{summary.get('runs')}")

    if "total_time" in summary:
        tt = summary["total_time"]
        grade = summary.get("overall_grade", "UNKNOWN")
        color = "\033[92m" if grade == "FAST" else "\033[93m" if grade == "MEDIUM" else "\033[91m"
        reset = "\033[0m"

        print(f"\n  Total Time to Interactive:")
        print(f"    Min:    {tt['min']:.0f}ms")
        print(f"    Max:    {tt['max']:.0f}ms")
        print(f"    Mean:   {tt['mean']:.0f}ms")
        print(f"    Median: {color}{tt['median']:.0f}ms{reset}")
        print(f"    P90:    {tt['p90']:.0f}ms")

    if "phases" in summary:
        print(f"\n  Phase Breakdown (mean):")
        for phase, data in summary["phases"].items():
            phase_label = phase.replace("_", " ").title()
            print(f"    {phase_label:.<25} {data['mean']:.0f}ms (max: {data['max']:.0f}ms)")

    if "resources" in summary:
        res = summary["resources"]
        total_bytes = sum(v.get("bytes", 0) for v in res.values() if isinstance(v, dict))
        print(f"\n  Resources Loaded:")
        print(f"    Total Files: {res.get('total', 0)}")
        print(f"    Total Size:  {total_bytes // 1024}KB")
        print(f"    Scripts:     {res['scripts']['count']} ({res['scripts']['bytes'] // 1024}KB)")
        print(f"    Styles:      {res['styles']['count']} ({res['styles']['bytes'] // 1024}KB)")
        print(f"    Fonts:       {res['fonts']['count']} ({res['fonts']['bytes'] // 1024}KB)")
        print(f"    Images:      {res['images']['count']} ({res['images']['bytes'] // 1024}KB)")

    grade = summary.get("overall_grade", "UNKNOWN")
    target = summary.get("target_met", False)
    color = "\033[92m" if target else "\033[91m"
    reset = "\033[0m"

    print(f"\n  {'-'*50}")
    print(f"  Overall Grade: {color}{grade}{reset}")
    print(f"  Target (<2s):  {color}{'MET' if target else 'NOT MET'}{reset}")
    print(f"{'='*70}\n")


def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(
        prog="ui_benchmark_full.py",
        description="Full UI benchmark measuring time to interactive.",
    )
    parser.add_argument(
        "--url",
        default=os.environ.get("E2E_BASE_URL") or "https://kroma.live",
        help="URL to benchmark",
    )
    parser.add_argument(
        "--runs",
        type=int,
        default=3,
        help="Number of runs (default: 3)",
    )
    parser.add_argument(
        "--bus-dir",
        default=os.environ.get("PLURIBUS_BUS_DIR") or "/pluribus/.pluribus/bus",
        help="Bus directory",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Output JSON",
    )

    args = parser.parse_args(argv)

    summary = asyncio.run(run_full_benchmark(args.url, args.runs, args.bus_dir))

    if args.json:
        print(json.dumps(summary, indent=2))
    else:
        print_summary(summary)

    return 0 if summary.get("target_met", False) else 1


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
