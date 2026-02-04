#!/usr/bin/env python3
"""
QA Live Checker - Proactive kroma.live verification daemon with AUTO-REMEDIATION

Periodically checks the live production site for:
- Console errors
- Network errors
- DOM issues
- Visual regressions

AUTO-REMEDIATION CAPABILITIES:
- Retry transient failures (confirm before action)
- Restart dashboard service on persistent failures
- Rollback to last known good commit if restart fails (via Replisome)
- Emit alerts and detailed telemetry
- Track failure patterns and apply cooldowns
- Maintain last known good state

Emits qa.verdict.* events and telemetry.client.* for errors found.
Responds to qa.stack.lane.* bus patterns.

Protocol: DKIN v28 (CAGENT), Replisome v1, PBTEST
"""
from __future__ import annotations

import argparse
import asyncio
import json
import os
import subprocess
import sys
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional
from dataclasses import dataclass, field

TOOLS_DIR = Path(__file__).resolve().parent

# Add tools dir for local imports if needed.
sys.path.insert(0, str(TOOLS_DIR))


def ensure_pluribus_root_on_path(root: Path) -> None:
    root_path = str(root.resolve())
    if root_path not in sys.path:
        sys.path.insert(0, root_path)


def import_cagent():
    try:
        from nucleus.sdk.cagent import Cagent
        return Cagent
    except ImportError:
        root = Path(os.environ.get("PLURIBUS_ROOT", str(TOOLS_DIR.parent.parent)))
        ensure_pluribus_root_on_path(root)
        from nucleus.sdk.cagent import Cagent
        return Cagent

# Configuration
LIVE_URL = os.environ.get("QA_LIVE_URL", "https://kroma.live")
CHECK_INTERVAL = int(os.environ.get("QA_CHECK_INTERVAL_S", "60"))
PLURIBUS_ROOT = os.environ.get("PLURIBUS_ROOT", "/pluribus")
DASHBOARD_DIR = os.path.join(PLURIBUS_ROOT, "nucleus/dashboard")
STATE_FILE = os.path.join(PLURIBUS_ROOT, ".pluribus/qa/state.json")
DASHBOARD_SERVICE = os.environ.get("QA_DASHBOARD_SERVICE", "pluribus-dashboard")
DASHBOARD_PM2_NAME = os.environ.get("QA_DASHBOARD_PM2_NAME", "dashboard")
ROLLBACK_REQUIRE_APPROVAL = os.environ.get("QA_ROLLBACK_REQUIRE_APPROVAL", "1").lower() in ("1", "true", "yes")
ROLLBACK_APPROVAL_TTL_S = int(os.environ.get("QA_ROLLBACK_APPROVAL_TTL_S", "900"))

# Remediation settings
RETRY_COUNT = 2
RETRY_DELAY_S = 5
RESTART_COOLDOWN_S = 300
ROLLBACK_COOLDOWN_S = 1800
MAX_CONSECUTIVE_FAILURES = 3
HEALTH_CHECK_DELAY_S = 15

agent = None


def init_agent():
    global agent
    if agent is None:
        Cagent = import_cagent()
        agent = Cagent(actor="qa-live-checker", citizen_class="swagent")
    return agent

@dataclass
class LiveCheckResult:
    passed: bool = True
    console_errors: List[Dict[str, Any]] = field(default_factory=list)
    network_errors: List[Dict[str, Any]] = field(default_factory=list)
    dom_issues: List[str] = field(default_factory=list)
    screenshot_path: Optional[str] = None
    load_time_ms: float = 0
    timestamp: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"))
    check_id: str = field(default_factory=lambda: str(uuid.uuid4())[:8])
    error_signature: str = ""

    def compute_signature(self):
        errors = sorted([e.get("text", "")[:50] for e in self.console_errors])
        errors += sorted(self.dom_issues)
        self.error_signature = hash(tuple(errors)) if errors else ""
        return self.error_signature


@dataclass
class RemediationState:
    last_restart_time: float = 0
    last_rollback_time: float = 0
    last_known_good_commit: str = ""
    consecutive_failures: int = 0
    failure_history: List[Dict[str, Any]] = field(default_factory=list)
    total_restarts: int = 0
    total_rollbacks: int = 0
    last_check_passed: bool = True
    rollback_approval_until: float = 0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "last_restart_time": self.last_restart_time,
            "last_rollback_time": self.last_rollback_time,
            "last_known_good_commit": self.last_known_good_commit,
            "consecutive_failures": self.consecutive_failures,
            "failure_history": self.failure_history[-50:],
            "total_restarts": self.total_restarts,
            "total_rollbacks": self.total_rollbacks,
            "last_check_passed": self.last_check_passed,
            "rollback_approval_until": self.rollback_approval_until,
        }

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "RemediationState":
        return cls(
            last_restart_time=d.get("last_restart_time", 0),
            last_rollback_time=d.get("last_rollback_time", 0),
            last_known_good_commit=d.get("last_known_good_commit", ""),
            consecutive_failures=d.get("consecutive_failures", 0),
            failure_history=d.get("failure_history", []),
            total_restarts=d.get("total_restarts", 0),
            total_rollbacks=d.get("total_rollbacks", 0),
            last_check_passed=d.get("last_check_passed", True),
            rollback_approval_until=d.get("rollback_approval_until", 0),
        )

    def record_failure(self, result: LiveCheckResult):
        self.consecutive_failures += 1
        self.last_check_passed = False
        self.failure_history.append({
            "timestamp": result.timestamp,
            "check_id": result.check_id,
            "console_errors": len(result.console_errors),
            "dom_issues": len(result.dom_issues),
            "signature": result.error_signature,
        })

    def record_success(self, commit: str = ""):
        self.consecutive_failures = 0
        self.last_check_passed = True
        if commit:
            self.last_known_good_commit = commit

    def can_restart(self) -> bool:
        return time.time() - self.last_restart_time > RESTART_COOLDOWN_S

    def can_rollback(self) -> bool:
        return (time.time() - self.last_rollback_time > ROLLBACK_COOLDOWN_S
                and self.last_known_good_commit)

    def approve_rollback(self, ttl_s: int):
        self.rollback_approval_until = time.time() + ttl_s

    def rollback_approved(self) -> bool:
        return time.time() <= self.rollback_approval_until


def load_state() -> RemediationState:
    try:
        if os.path.exists(STATE_FILE):
            with open(STATE_FILE, "r") as f:
                return RemediationState.from_dict(json.load(f))
    except Exception as e:
        print(f"[QA] Warning: Could not load state: {e}")
    return RemediationState()


def save_state(state: RemediationState):
    try:
        os.makedirs(os.path.dirname(STATE_FILE), exist_ok=True)
        with open(STATE_FILE, "w") as f:
            json.dump(state.to_dict(), f, indent=2)
    except Exception as e:
        print(f"[QA] Warning: Could not save state: {e}")


def get_current_commit() -> str:
    try:
        result = subprocess.run(
            ["git", "-C", PLURIBUS_ROOT, "rev-parse", "HEAD"],
            capture_output=True, text=True, timeout=10
        )
        return result.stdout.strip()[:12] if result.returncode == 0 else ""
    except Exception:
        return ""


def pm2_process_exists(name: str) -> tuple[bool, str]:
    try:
        result = subprocess.run(
            ["pm2", "jlist"],
            capture_output=True, text=True, timeout=20
        )
    except Exception as e:
        return False, f"pm2_jlist_error: {e}"
    if result.returncode != 0:
        return False, f"pm2_jlist_failed: {result.stderr.strip()[:120]}"
    try:
        processes = json.loads(result.stdout or "[]")
    except json.JSONDecodeError:
        return False, "pm2_jlist_invalid_json"
    for proc in processes:
        if proc.get("name") == name:
            return True, ""
    return False, "pm2_process_missing"


def emit_alert(level: str, title: str, message: str, data: Dict[str, Any] = None):
    """Emit an alert event via CAGENT."""
    agent.emit(
        topic=f"qa.alert.{level}",
        kind="alert",
        level=level,
        data={
            "title": title,
            "message": message,
            "timestamp": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
            **(data or {})
        }
    )
    print(f"[QA ALERT {level.upper()}] {title}: {message}", file=sys.stderr)


async def check_live_site(url: str, headless: bool = True) -> LiveCheckResult:
    result = LiveCheckResult()
    try:
        from playwright.async_api import async_playwright
    except ImportError:
        result.passed = False
        result.dom_issues.append("Playwright not installed")
        return result

    async with async_playwright() as p:
        browser = await p.chromium.launch(headless=headless)
        context = await browser.new_context()
        page = await context.new_page()

        IGNORE_PATTERNS = ["WebGL", "GL Driver Message", "GroupMarkerNotSet", "swiftshader", "GPU stall", "favicon.ico"]

        def on_console(msg):
            if msg.type in ("error", "warning"):
                text = msg.text
                if any(pattern in text for pattern in IGNORE_PATTERNS): return
                result.console_errors.append({"type": msg.type, "text": text, "location": str(msg.location) if msg.location else None})
                if msg.type == "error": result.passed = False

        page.on("console", on_console)

        def on_pageerror(exc):
            result.console_errors.append({"type": "pageerror", "text": str(exc), "location": None})
            result.passed = False

        page.on("pageerror", on_pageerror)

        def on_requestfailed(request):
            result.network_errors.append({"url": request.url, "method": request.method, "failure": request.failure})
            if "kroma.live" in request.url: result.passed = False

        page.on("requestfailed", on_requestfailed)

        try:
            start = time.time()
            response = await page.goto(url, timeout=60000, wait_until="domcontentloaded")
            result.load_time_ms = (time.time() - start) * 1000

            if not response or not response.ok:
                result.passed = False
                result.dom_issues.append(f"HTTP {response.status if response else 'No response'}")

            await page.wait_for_timeout(5000)
            content = await page.content()

            if "Failed to fetch dynamically imported module" in content:
                result.passed = False
                result.dom_issues.append("Vite HMR error overlay detected")
            if "Internal Server Error" in content:
                result.passed = False
                result.dom_issues.append("Internal Server Error detected")
            if "Cannot read properties of undefined" in content:
                result.passed = False
                result.dom_issues.append("React undefined property error detected")

            if not result.passed:
                screenshot_dir = "/pluribus/.pluribus/qa/screenshots"
                os.makedirs(screenshot_dir, exist_ok=True)
                result.screenshot_path = f"{screenshot_dir}/failure-{result.check_id}.png"
                await page.screenshot(path=result.screenshot_path, full_page=True)

        except Exception as e:
            result.passed = False
            result.dom_issues.append(f"Navigation error: {str(e)}")
        finally:
            await browser.close()

    result.compute_signature()
    return result


async def check_with_retry(url: str, retries: int = RETRY_COUNT) -> LiveCheckResult:
    result = await check_live_site(url)
    if result.passed: return result
    for i in range(retries):
        print(f"[QA] Retry {i+1}/{retries} after failure...")
        await asyncio.sleep(RETRY_DELAY_S)
        result = await check_live_site(url)
        if result.passed:
            print(f"[QA] Transient failure recovered on retry {i+1}")
            return result
    print(f"[QA] Persistent failure confirmed after {retries} retries")
    return result


def restart_dashboard() -> tuple[bool, str]:
    print("[QA] Attempting dashboard restart...")
    agent.emit(topic="qa.remediation.restart.started", kind="action", level="warn", data={"service": DASHBOARD_SERVICE, "reason": "persistent_failure"})

    try:
        result = subprocess.run(["systemctl", "restart", DASHBOARD_SERVICE], capture_output=True, text=True, timeout=60)
        if result.returncode == 0:
            print("[QA] Dashboard restart successful (systemctl)")
            agent.emit(topic="qa.remediation.restart.completed", kind="response", level="info", data={"service": DASHBOARD_SERVICE, "method": "systemctl", "success": True})
            return True, "systemctl"

        systemctl_error = result.stderr.strip()
        error_kind = "systemctl_service_missing" if "could not be found" in systemctl_error.lower() else "systemctl_failed"

        pm2_ok, pm2_reason = pm2_process_exists(DASHBOARD_PM2_NAME)
        if not pm2_ok:
            agent.emit(topic="qa.remediation.restart.skipped", kind="response", level="warn", data={"service": DASHBOARD_SERVICE, "method": "pm2", "reason": pm2_reason})
            return False, pm2_reason or error_kind

        result = subprocess.run(["pm2", "restart", DASHBOARD_PM2_NAME], capture_output=True, text=True, timeout=60)
        if result.returncode == 0:
            print("[QA] Dashboard restart successful (pm2)")
            agent.emit(topic="qa.remediation.restart.completed", kind="response", level="info", data={"service": DASHBOARD_SERVICE, "method": "pm2", "success": True})
            return True, "pm2"

        print(f"[QA] Dashboard restart failed: {result.stderr}")
        agent.emit(topic="qa.remediation.restart.failed", kind="response", level="error", data={"service": DASHBOARD_SERVICE, "method": "pm2", "error": result.stderr[:200]})
        return False, "pm2_restart_failed"

    except Exception as e:
        print(f"[QA] Dashboard restart error: {e}")
        agent.emit(topic="qa.remediation.restart.failed", kind="response", level="error", data={"service": DASHBOARD_SERVICE, "error": str(e)})
        return False, "restart_exception"


def rollback_to_commit(commit: str) -> bool:
    """Request rollback via Replisome (Protocol v28 Compliance)."""
    print(f"[QA] Requesting Replisome rollback to {commit}...")
    
    # Use CAGENT helper to request rollback
    req_id = agent.request_rollback(target_commit=commit, reason="qa-auto-remediation-failed")
    
    # Wait for completion (Replisome will emit replisome.rollback.completed)
    # We poll the bus manually here since agent_bus is simple
    print(f"[QA] Waiting for Replisome completion (ReqID: {req_id})...")
    
    start_wait = time.time()
    while time.time() - start_wait < 300: # 5 min timeout
        # In a real event loop we would subscribe, but here we scan recent events
        # We need to read the bus file
        try:
            with open(agent.paths.events_path, "r") as f:
                # Naive scan from end - simplified
                lines = f.readlines()[-50:] # Check last 50 events
                for line in lines:
                    try:
                        evt = json.loads(line)
                        if evt.get("topic") == "replisome.rollback.completed" and \
                           evt.get("data", {}).get("target_commit") == commit:
                            print(f"[QA] Replisome confirmed rollback to {commit}!")
                            agent.emit("qa.remediation.rollback.completed", data={"target_commit": commit, "success": True})
                            return True
                        if evt.get("topic") == "replisome.rollback.rejected" and \
                           evt.get("data", {}).get("req_id") == req_id:
                            reason = evt.get("data", {}).get("reason", "unknown")
                            print(f"[QA] Replisome rejected rollback: {reason}")
                            agent.emit("qa.remediation.rollback.failed", level="error", data={"target_commit": commit, "error": f"Replisome rejected: {reason}"})
                            return False
                    except: pass
        except Exception as e:
            print(f"[QA] Error polling bus: {e}")
        
        time.sleep(2)
        
    print(f"[QA] Timed out waiting for Replisome.")
    agent.emit("qa.remediation.rollback.failed", level="error", data={"target_commit": commit, "error": "Replisome timeout"})
    return False


async def perform_remediation(result: LiveCheckResult, state: RemediationState, url: str) -> bool:
    if state.consecutive_failures < MAX_CONSECUTIVE_FAILURES:
        print(f"[QA] Failure {state.consecutive_failures}/{MAX_CONSECUTIVE_FAILURES}, monitoring...")
        emit_alert("warn", "Site Check Failed", f"Failure #{state.consecutive_failures}, monitoring", {"check_id": result.check_id})
        return False

    if state.can_restart():
        emit_alert("error", "Persistent Failure - Restarting", f"Restarting dashboard after {state.consecutive_failures} failures", {"check_id": result.check_id})
        state.last_restart_time = time.time()
        state.total_restarts += 1
        save_state(state)

        restart_ok, restart_reason = restart_dashboard()
        if restart_ok:
            print(f"[QA] Waiting {HEALTH_CHECK_DELAY_S}s for stabilization...")
            await asyncio.sleep(HEALTH_CHECK_DELAY_S)
            health_result = await check_live_site(url)
            if health_result.passed:
                emit_alert("info", "Restart Successful", "Dashboard restart resolved the issue", {"check_id": health_result.check_id})
                state.record_success(get_current_commit())
                save_state(state)
                return True
            else:
                print("[QA] Health check failed after restart")
        else:
            emit_alert("warn", "Dashboard Restart Failed", f"Restart failed: {restart_reason}", {"reason": restart_reason})
    else:
        print(f"[QA] Restart on cooldown")

    if state.can_rollback():
        if ROLLBACK_REQUIRE_APPROVAL and not state.rollback_approved():
            agent.emit(topic="qa.remediation.rollback.blocked", kind="response", level="warn", data={"target_commit": state.last_known_good_commit, "reason": "approval_required", "cooldown_s": ROLLBACK_COOLDOWN_S})
            emit_alert("warn", "Rollback Blocked", "Rollback requires approval; emit qa.remediation.approval to proceed.", {"target_commit": state.last_known_good_commit})
            return False

        emit_alert("error", "Restart Failed - Rolling Back", f"Rolling back to: {state.last_known_good_commit}", {"target_commit": state.last_known_good_commit})
        state.last_rollback_time = time.time()
        state.total_rollbacks += 1
        save_state(state)

        if rollback_to_commit(state.last_known_good_commit):
            print(f"[QA] Waiting {HEALTH_CHECK_DELAY_S}s for stabilization...")
            await asyncio.sleep(HEALTH_CHECK_DELAY_S)
            health_result = await check_live_site(url)
            if health_result.passed:
                emit_alert("info", "Rollback Successful", f"Rolled back to {state.last_known_good_commit}", {"check_id": health_result.check_id})
                state.record_success()
                save_state(state)
                return True
            else:
                print("[QA] Health check failed after rollback")
    elif state.last_known_good_commit:
        print(f"[QA] Rollback on cooldown")
    else:
        print("[QA] No known good commit")

    emit_alert("critical", "Auto-Remediation Failed", "All attempts failed. Manual intervention required.", {"check_id": result.check_id})
    return False


def emit_verdict(result: LiveCheckResult, url: str, remediation_attempted: bool = False):
    topic = "qa.verdict.pass" if result.passed else "qa.verdict.fail"
    agent.emit(topic, kind="response", level="info" if result.passed else "error",
        data={
            "check_id": result.check_id,
            "url": url,
            "passed": result.passed,
            "load_time_ms": round(result.load_time_ms, 2),
            "console_errors": len(result.console_errors),
            "network_errors": len(result.network_errors),
            "dom_issues": len(result.dom_issues),
            "screenshot": result.screenshot_path,
            "timestamp": result.timestamp,
            "remediation_attempted": remediation_attempted,
        }
    )
    for err in result.console_errors:
        agent.emit("telemetry.client.error", kind="request", level="error", data={"check_id": result.check_id, "url": url, "type": err.get("type"), "message": err.get("text", "")[:500], "source": "qa-live-checker"})
    for err in result.dom_issues:
        agent.emit("telemetry.client.error", kind="request", level="error", data={"check_id": result.check_id, "url": url, "type": "dom_issue", "message": err[:500], "source": "qa-live-checker"})


async def handle_check_result(result: LiveCheckResult, state: RemediationState, url: str):
    remediation_attempted = False
    if result.passed:
        if not state.last_check_passed:
            emit_alert("info", "Site Recovered", f"Site healthy after {state.consecutive_failures} failures", {"check_id": result.check_id})
        state.record_success(get_current_commit())
        save_state(state)
    else:
        state.record_failure(result)
        save_state(state)
        remediation_attempted = await perform_remediation(result, state, url)
    emit_verdict(result, url, remediation_attempted)
    return result


async def watch_bus_and_check(bus_dir: str, url: str, interval_s: int):
    # Ensure events file exists
    if not os.path.exists(agent.paths.events_path):
        with open(agent.paths.events_path, "w") as f: pass

    # Load state
    state = load_state()
    if not state.last_known_good_commit:
        state.last_known_good_commit = get_current_commit()
        save_state(state)

    agent.emit("qa.live-checker.started", kind="artifact", level="info", data={"url": url, "features": ["cagent_v1", "replisome_v1"]})
    last_check = time.time()

    with open(agent.paths.events_path, "r") as f:
        f.seek(0, 2)
        while True:
            line = f.readline()
            if line:
                try:
                    event = json.loads(line)
                    topic = event.get("topic", "")
                    
                    if topic == "qa.remediation.approval":
                        action = event.get("data", {}).get("action")
                        if action == "rollback":
                            ttl = int(event.get("data", {}).get("ttl_s", ROLLBACK_APPROVAL_TTL_S))
                            state.approve_rollback(ttl)
                            save_state(state)
                            agent.emit("qa.remediation.approval.accepted", data={"approved_until": state.rollback_approval_until})
                            
                    # Other handlers (lanes, pbtest) can be added here
                except: pass
            else:
                time.sleep(0.1)

            now = time.time()
            if (now - last_check) >= interval_s:
                print(f"[QA] Periodic check...")
                result = await check_with_retry(url)
                await handle_check_result(result, state, url)
                last_check = now


def run_single_check(bus_dir: str, url: str) -> int:
    async def do_check():
        state = load_state()
        print(f"[QA] Checking {url}...")
        result = await check_with_retry(url)
        await handle_check_result(result, state, url)
        return 0 if result.passed else 1
    return asyncio.run(do_check())


def main():
    parser = argparse.ArgumentParser(description="QA Live Checker - Proactive site verification with auto-remediation")
    parser.add_argument("--url", default=LIVE_URL)
    parser.add_argument("--bus-dir", default=os.environ.get("PLURIBUS_BUS_DIR"))
    parser.add_argument("--interval", type=int, default=CHECK_INTERVAL)
    parser.add_argument("--daemon", action="store_true")
    args = parser.parse_args()

    init_agent()

    if args.daemon:
        asyncio.run(watch_bus_and_check(args.bus_dir, args.url, args.interval))
    else:
        sys.exit(run_single_check(args.bus_dir, args.url))

if __name__ == "__main__":
    main()
