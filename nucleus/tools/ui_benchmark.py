#!/usr/bin/env python3
"""
UI Benchmark Tool - Measures Pluribus Dashboard load times.

Runs multiple page loads via Playwright, captures timing metrics,
and emits results to the bus for analysis.

Usage:
    python3 ui_benchmark.py --url https://kroma.live --runs 5

DKIN v25 compliant - emits structured bus events.
"""
from __future__ import annotations

import argparse
import asyncio
import json
import os
import statistics
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any

# Ensure we can import from nucleus tools
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def now_iso() -> str:
    return datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ")


def emit_bus_event(bus_dir: str, topic: str, kind: str, level: str, data: dict) -> None:
    """Emit event to bus."""
    try:
        events_path = Path(bus_dir) / "events.ndjson"
        events_path.parent.mkdir(parents=True, exist_ok=True)
        event = {
            "id": f"bench-{int(time.time()*1000)}",
            "ts": time.time(),
            "iso": now_iso(),
            "topic": topic,
            "kind": kind,
            "level": level,
            "actor": "ui-benchmark",
            "host": os.uname().nodename,
            "pid": os.getpid(),
            "data": data,
        }
        with open(events_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(event, ensure_ascii=False, separators=(",", ":")) + "\n")
    except Exception as e:
        print(f"[WARN] Failed to emit bus event: {e}", file=sys.stderr)


async def run_benchmark(url: str, runs: int, bus_dir: str) -> dict[str, Any]:
    """Run page load benchmarks using Playwright."""
    try:
        from playwright.async_api import async_playwright
    except ImportError:
        print("[ERROR] Playwright not installed. Run: pip install playwright && playwright install chromium", file=sys.stderr)
        return {"error": "playwright_missing"}

    results: list[dict[str, Any]] = []
    print(f"\n{'='*60}")
    print(f"UI BENCHMARK: {url}")
    print(f"Runs: {runs}")
    print(f"{'='*60}\n")

    emit_bus_event(bus_dir, "benchmark.ui.start", "metric", "info", {
        "url": url,
        "runs": runs,
        "timestamp": now_iso(),
    })

    async with async_playwright() as p:
        browser = await p.chromium.launch(headless=True)

        for i in range(runs):
            context = await browser.new_context()
            page = await context.new_page()

            # Capture performance timing
            timing_data: dict[str, Any] = {
                "run": i + 1,
                "url": url,
            }

            try:
                start = time.perf_counter()

                # Navigate and wait for load
                response = await page.goto(url, wait_until="load", timeout=60000)
                load_time = (time.perf_counter() - start) * 1000  # ms

                timing_data["load_time_ms"] = round(load_time, 2)
                timing_data["http_status"] = response.status if response else None

                # Extract Navigation Timing from page
                nav_timing = await page.evaluate("""() => {
                    const perf = performance.getEntriesByType('navigation')[0];
                    if (!perf) return null;
                    return {
                        ttfb: Math.round(perf.responseStart - perf.requestStart),
                        domContentLoaded: Math.round(perf.domContentLoadedEventEnd - perf.startTime),
                        domComplete: Math.round(perf.domComplete - perf.startTime),
                        loadEvent: Math.round(perf.loadEventEnd - perf.startTime),
                        transferSize: perf.transferSize,
                        encodedBodySize: perf.encodedBodySize,
                    };
                }""")

                if nav_timing:
                    timing_data.update(nav_timing)

                # Get resource counts
                resource_stats = await page.evaluate("""() => {
                    const resources = performance.getEntriesByType('resource');
                    let scriptCount = 0, scriptBytes = 0;
                    let styleCount = 0, styleBytes = 0;
                    let imgCount = 0, imgBytes = 0;

                    for (const r of resources) {
                        const size = r.transferSize || r.encodedBodySize || 0;
                        if (r.initiatorType === 'script' || r.name.endsWith('.js')) {
                            scriptCount++;
                            scriptBytes += size;
                        } else if (r.initiatorType === 'css' || r.name.endsWith('.css')) {
                            styleCount++;
                            styleBytes += size;
                        } else if (r.initiatorType === 'img' || /\\.(png|jpg|svg|gif|webp)/.test(r.name)) {
                            imgCount++;
                            imgBytes += size;
                        }
                    }

                    return {
                        totalResources: resources.length,
                        scriptCount, scriptBytes,
                        styleCount, styleBytes,
                        imgCount, imgBytes,
                    };
                }""")

                if resource_stats:
                    timing_data.update(resource_stats)

                # Check for console errors
                console_errors = []
                page.on("console", lambda msg: console_errors.append(msg.text) if msg.type == "error" else None)

                # Wait a bit for any late console errors
                await asyncio.sleep(0.5)
                timing_data["console_errors"] = len(console_errors)

                # Grade the load time
                grade = "FAST" if load_time < 2000 else "MEDIUM" if load_time < 5000 else "SLOW"
                timing_data["grade"] = grade

                results.append(timing_data)

                # Print progress
                color = "\033[92m" if grade == "FAST" else "\033[93m" if grade == "MEDIUM" else "\033[91m"
                reset = "\033[0m"
                print(f"  Run {i+1}/{runs}: {color}{load_time:.0f}ms ({grade}){reset}"
                      f" | TTFB: {timing_data.get('ttfb', 'n/a')}ms"
                      f" | DCL: {timing_data.get('domContentLoaded', 'n/a')}ms"
                      f" | Scripts: {timing_data.get('scriptCount', 0)}"
                      f" ({timing_data.get('scriptBytes', 0) // 1024}KB)")

            except Exception as e:
                timing_data["error"] = str(e)
                results.append(timing_data)
                print(f"  Run {i+1}/{runs}: ERROR - {e}", file=sys.stderr)

            await context.close()

            # Brief pause between runs
            if i < runs - 1:
                await asyncio.sleep(1)

        await browser.close()

    # Calculate statistics
    load_times = [r["load_time_ms"] for r in results if "load_time_ms" in r and "error" not in r]
    ttfbs = [r["ttfb"] for r in results if "ttfb" in r and r["ttfb"]]
    dcls = [r["domContentLoaded"] for r in results if "domContentLoaded" in r]

    summary: dict[str, Any] = {
        "url": url,
        "runs": runs,
        "successful_runs": len(load_times),
        "timestamp": now_iso(),
    }

    if load_times:
        summary["load_time"] = {
            "min": round(min(load_times), 2),
            "max": round(max(load_times), 2),
            "mean": round(statistics.mean(load_times), 2),
            "median": round(statistics.median(load_times), 2),
            "stdev": round(statistics.stdev(load_times), 2) if len(load_times) > 1 else 0,
            "p90": round(sorted(load_times)[int(len(load_times) * 0.9)] if load_times else 0, 2),
        }

    if ttfbs:
        summary["ttfb"] = {
            "min": min(ttfbs),
            "max": max(ttfbs),
            "mean": round(statistics.mean(ttfbs), 2),
        }

    if dcls:
        summary["dom_content_loaded"] = {
            "min": min(dcls),
            "max": max(dcls),
            "mean": round(statistics.mean(dcls), 2),
        }

    # Resource summary from last run
    if results and "scriptCount" in results[-1]:
        summary["resources"] = {
            "scripts": results[-1].get("scriptCount", 0),
            "script_bytes": results[-1].get("scriptBytes", 0),
            "styles": results[-1].get("styleCount", 0),
            "style_bytes": results[-1].get("styleBytes", 0),
        }

    # Determine overall grade
    if load_times:
        median = statistics.median(load_times)
        summary["overall_grade"] = "FAST" if median < 2000 else "MEDIUM" if median < 5000 else "SLOW"
        summary["target_met"] = median < 2000  # Target: <2s

    # Emit results to bus
    emit_bus_event(bus_dir, "benchmark.ui.complete", "metric", "info", {
        "summary": summary,
        "runs": results,
    })

    return summary


def print_summary(summary: dict[str, Any]) -> None:
    """Print formatted summary report."""
    print(f"\n{'='*60}")
    print("BENCHMARK SUMMARY")
    print(f"{'='*60}\n")

    print(f"  URL: {summary.get('url')}")
    print(f"  Runs: {summary.get('successful_runs')}/{summary.get('runs')}")

    if "load_time" in summary:
        lt = summary["load_time"]
        grade = summary.get("overall_grade", "UNKNOWN")
        color = "\033[92m" if grade == "FAST" else "\033[93m" if grade == "MEDIUM" else "\033[91m"
        reset = "\033[0m"

        print(f"\n  Load Time:")
        print(f"    Min:    {lt['min']:.0f}ms")
        print(f"    Max:    {lt['max']:.0f}ms")
        print(f"    Mean:   {lt['mean']:.0f}ms")
        print(f"    Median: {color}{lt['median']:.0f}ms{reset}")
        print(f"    P90:    {lt['p90']:.0f}ms")
        print(f"    StdDev: {lt['stdev']:.0f}ms")

    if "ttfb" in summary:
        ttfb = summary["ttfb"]
        print(f"\n  Time to First Byte (TTFB):")
        print(f"    Min:  {ttfb['min']}ms")
        print(f"    Max:  {ttfb['max']}ms")
        print(f"    Mean: {ttfb['mean']:.0f}ms")

    if "dom_content_loaded" in summary:
        dcl = summary["dom_content_loaded"]
        print(f"\n  DOM Content Loaded:")
        print(f"    Min:  {dcl['min']}ms")
        print(f"    Max:  {dcl['max']}ms")
        print(f"    Mean: {dcl['mean']:.0f}ms")

    if "resources" in summary:
        res = summary["resources"]
        print(f"\n  Resources Loaded:")
        print(f"    Scripts: {res['scripts']} ({res['script_bytes'] // 1024}KB)")
        print(f"    Styles:  {res['styles']} ({res['style_bytes'] // 1024}KB)")

    grade = summary.get("overall_grade", "UNKNOWN")
    target = summary.get("target_met", False)
    color = "\033[92m" if target else "\033[91m"
    reset = "\033[0m"

    print(f"\n  {'-'*40}")
    print(f"  Overall Grade: {color}{grade}{reset}")
    print(f"  Target (<2s):  {color}{'MET' if target else 'NOT MET'}{reset}")
    print(f"{'='*60}\n")


def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(
        prog="ui_benchmark.py",
        description="Benchmark Pluribus Dashboard load times.",
    )
    parser.add_argument(
        "--url",
        default=os.environ.get("E2E_BASE_URL") or "https://kroma.live",
        help="URL to benchmark (default: https://kroma.live)",
    )
    parser.add_argument(
        "--runs",
        type=int,
        default=5,
        help="Number of page load runs (default: 5)",
    )
    parser.add_argument(
        "--bus-dir",
        default=os.environ.get("PLURIBUS_BUS_DIR") or "/pluribus/.pluribus/bus",
        help="Bus directory for event emission",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Output raw JSON instead of formatted report",
    )

    args = parser.parse_args(argv)

    summary = asyncio.run(run_benchmark(args.url, args.runs, args.bus_dir))

    if args.json:
        print(json.dumps(summary, indent=2))
    else:
        print_summary(summary)

    # Exit code based on target
    return 0 if summary.get("target_met", False) else 1


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
