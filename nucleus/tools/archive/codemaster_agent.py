#!/usr/bin/env python3
"""
CODEMASTER AGENT - E Pluribus Unum Singleton Gatekeeper
========================================================

"From Many, One" - The Codemaster Agent is the singular authority for all writes
to critical path branches (main, staging, dev). Multiple agents work in parallel
on PAIP clones and feature branches, but only Codemaster controls the critical path.

Architecture:
- Singleton daemon (one instance per repository)
- Watch bus for merge requests
- Validate with QA Omega and Hygiene
- Execute merges with full audit trail
- Conserve orphan work (no data loss)
- Rollback on test failures

Bus Topics Watched:
- codemaster.merge.request      - Incoming merge requests
- codemaster.rollback.request   - Rollback requests
- codemaster.status.request     - Status queries
- codemaster.conservation.request - Conservation requests

Bus Topics Emitted:
- codemaster.merge.accepted     - Request queued
- codemaster.merge.rejected     - Request rejected
- codemaster.merge.complete     - Merge successful
- codemaster.merge.conflict     - Conflict detected
- codemaster.rollback.complete  - Rollback done
- codemaster.status.report      - Status response
- codemaster.conservation.complete - Conservation done
- codemaster.audit.*            - Full audit trail
- codemaster.health.*           - Health metrics

Reference: nucleus/specs/codemaster_protocol_v2.md
DKIN Version: v28
"""
from __future__ import annotations

import atexit
import collections
import fcntl
import json
import os
import re
import signal
import socket
import subprocess
import sys
import threading
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Deque, Dict, Iterator, List, Optional, Tuple

sys.dont_write_bytecode = True

TOOLS_DIR = Path(__file__).resolve().parent
if str(TOOLS_DIR) not in sys.path:
    sys.path.insert(0, str(TOOLS_DIR))

import cagent_registry

# =============================================================================
# Constants
# =============================================================================

VERSION = "1.0.0"
PROTOCOL_VERSION = "codemaster-v2"
DKIN_VERSION = "v28"

# Default configuration
DEFAULT_INTERVAL_S = 5.0
DEFAULT_BUS_DIR = "/pluribus/.pluribus/bus"
DEFAULT_REPO_DIR = "/pluribus"
DEFAULT_CAGENT_REGISTRY = "/pluribus/nucleus/specs/cagent_registry.json"

# Critical branches that require Codemaster
CRITICAL_BRANCHES = {"main", "staging", "dev"}

# Agent-owned patterns (don't require Codemaster)
AGENT_OWNED_PATTERNS = [
    r"^krodotma/.*_.*$",       # Agent branches
    r"^feature/.*$",           # Feature branches
    r"^/tmp/pluribus_.*$",     # PAIP clones
]

# State machine states
class State:
    IDLE = "idle"
    VALIDATING = "validating"
    MERGING = "merging"
    RESOLVING = "resolving"
    PUSHING = "pushing"
    ROLLING_BACK = "rolling_back"

# Queue priorities
MAX_PRIORITY = 10
DEFAULT_PRIORITY = 5


# =============================================================================
# Time Utilities
# =============================================================================

def load_cagent_registry(path: Path) -> Dict[str, Any]:
    defaults = {
        "defaults": {
            "citizen_class": "superworker",
            "citizen_tier": "limited",
            "bootstrap_profile": "minimal",
            "scope_allowlist": [],
        },
        "actors": [],
        "class_aliases": {},
        "tier_aliases": {},
    }
    if not path.exists():
        return defaults
    try:
        registry = cagent_registry.load_registry(path)
        ok, _errs = cagent_registry.validate_registry(registry)
        return registry if ok else defaults
    except Exception:
        return defaults

def now_ts() -> float:
    return time.time()


def now_iso() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def format_duration(seconds: float) -> str:
    if seconds < 0.001:
        return f"{seconds * 1_000_000:.0f}μs"
    if seconds < 1:
        return f"{seconds * 1000:.1f}ms"
    if seconds < 60:
        return f"{seconds:.2f}s"
    if seconds < 3600:
        return f"{seconds / 60:.1f}m"
    return f"{seconds / 3600:.1f}h"


# =============================================================================
# File Tailer (from omega_heartbeat.py)
# =============================================================================

@dataclass
class FileTailer:
    """Efficient incremental file tailer with position tracking."""
    path: Path
    _position: int = 0
    _inode: int = 0

    def __post_init__(self):
        self._update_inode()

    def _update_inode(self) -> None:
        try:
            stat = self.path.stat()
            self._inode = stat.st_ino
        except FileNotFoundError:
            self._inode = 0

    def _check_rotation(self) -> bool:
        try:
            stat = self.path.stat()
            if stat.st_ino != self._inode:
                self._position = 0
                self._inode = stat.st_ino
                return True
            if stat.st_size < self._position:
                self._position = 0
                return True
        except FileNotFoundError:
            self._position = 0
            self._inode = 0
        return False

    def read_new_events(self) -> Iterator[dict]:
        if not self.path.exists():
            return

        self._check_rotation()

        try:
            with self.path.open("r", encoding="utf-8", errors="replace") as f:
                f.seek(self._position)

                while True:
                    line = f.readline()
                    if not line:
                        break
                    if line.endswith("\n"):
                        try:
                            obj = json.loads(line.rstrip("\n"))
                            if isinstance(obj, dict):
                                yield obj
                        except json.JSONDecodeError:
                            continue
                    else:
                        break

                self._position = f.tell()
        except Exception:
            pass


# =============================================================================
# Bus Emitter
# =============================================================================

class BusEmitter:
    """Direct NDJSON bus emission with file locking."""

    def __init__(self, bus_dir: str | Path, actor: str = "codemaster"):
        self.bus_dir = Path(bus_dir)
        self.events_path = self.bus_dir / "events.ndjson"
        self.actor = actor
        self.host = socket.gethostname()
        self.pid = os.getpid()
        self._emit_count = 0

    def _ensure_dir(self) -> None:
        self.bus_dir.mkdir(parents=True, exist_ok=True)

    def emit(
        self,
        topic: str,
        kind: str,
        level: str,
        data: dict,
        *,
        trace_id: Optional[str] = None,
    ) -> str:
        """Emit event to bus with atomic file locking."""
        self._ensure_dir()

        event_id = str(uuid.uuid4())
        ts = now_ts()

        event = {
            "id": event_id,
            "ts": ts,
            "iso": now_iso(),
            "topic": topic,
            "kind": kind,
            "level": level,
            "actor": self.actor,
            "host": self.host,
            "pid": self.pid,
            "data": data,
        }

        if trace_id:
            event["trace_id"] = trace_id

        line = json.dumps(event, ensure_ascii=False, separators=(",", ":")) + "\n"

        try:
            with self.events_path.open("a", encoding="utf-8") as f:
                fcntl.flock(f.fileno(), fcntl.LOCK_EX)
                try:
                    f.write(line)
                finally:
                    fcntl.flock(f.fileno(), fcntl.LOCK_UN)
            self._emit_count += 1
        except Exception as e:
            print(f"[codemaster] emit error: {e}", file=sys.stderr)

        return event_id

    @property
    def emit_count(self) -> int:
        return self._emit_count


# =============================================================================
# Merge Request
# =============================================================================

@dataclass
class MergeRequest:
    """Merge request in the queue."""
    id: str
    source_branch: str
    target_branch: str
    actor: str
    description: str
    citizen_class: str = ""
    citizen_tier: str = ""
    bootstrap_profile: str = ""
    scope_allowlist: List[str] = field(default_factory=list)
    commit_range: Optional[str] = None
    tests_passed: bool = True
    pbtest_verdict: str = "pass"
    paip_clone: Optional[str] = None
    priority: int = DEFAULT_PRIORITY
    dependencies: List[str] = field(default_factory=list)
    conservation_policy: str = "always"  # always | on_conflict | never
    created_at: float = field(default_factory=now_ts)
    trace_id: Optional[str] = None

    @classmethod
    def from_event(cls, event: dict) -> "MergeRequest":
        data = event.get("data", {})
        return cls(
            id=event.get("id", str(uuid.uuid4())),
            source_branch=data.get("source_branch", ""),
            target_branch=data.get("target_branch", ""),
            actor=event.get("actor", "unknown"),
            description=data.get("description", ""),
            citizen_class=data.get("citizen_class", ""),
            citizen_tier=data.get("citizen_tier", ""),
            bootstrap_profile=data.get("bootstrap_profile", ""),
            scope_allowlist=data.get("scope_allowlist", []) or [],
            commit_range=data.get("commit_range"),
            tests_passed=data.get("tests_passed", True),
            pbtest_verdict=data.get("pbtest_verdict", "pass"),
            paip_clone=data.get("paip_clone"),
            priority=data.get("priority", DEFAULT_PRIORITY),
            dependencies=data.get("dependencies", []),
            conservation_policy=data.get("conservation_policy", "always"),
            trace_id=event.get("trace_id"),
        )

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "source_branch": self.source_branch,
            "target_branch": self.target_branch,
            "actor": self.actor,
            "description": self.description,
            "citizen_class": self.citizen_class,
            "citizen_tier": self.citizen_tier,
            "bootstrap_profile": self.bootstrap_profile,
            "scope_allowlist": self.scope_allowlist,
            "commit_range": self.commit_range,
            "tests_passed": self.tests_passed,
            "pbtest_verdict": self.pbtest_verdict,
            "paip_clone": self.paip_clone,
            "priority": self.priority,
            "dependencies": self.dependencies,
            "conservation_policy": self.conservation_policy,
            "created_at": self.created_at,
        }


# =============================================================================
# Git Operations
# =============================================================================

class GitOperations:
    """Git operations for merge management."""

    def __init__(self, repo_dir: str | Path):
        self.repo_dir = Path(repo_dir)

    def _run(self, *args: str, check: bool = True) -> subprocess.CompletedProcess:
        """Run git command."""
        cmd = ["git", "-C", str(self.repo_dir)] + list(args)
        return subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            check=check,
        )

    def fetch_all(self) -> bool:
        """Fetch all remotes."""
        try:
            self._run("fetch", "--all", "--prune")
            return True
        except subprocess.CalledProcessError:
            return False

    def branch_exists(self, branch: str) -> bool:
        """Check if branch exists."""
        try:
            result = self._run("rev-parse", "--verify", branch, check=False)
            return result.returncode == 0
        except Exception:
            return False

    def current_branch(self) -> str:
        """Get current branch name."""
        result = self._run("rev-parse", "--abbrev-ref", "HEAD")
        return result.stdout.strip()

    def checkout(self, branch: str) -> bool:
        """Checkout branch."""
        try:
            self._run("checkout", branch)
            return True
        except subprocess.CalledProcessError:
            return False

    def get_head_sha(self, branch: Optional[str] = None) -> str:
        """Get HEAD SHA of branch."""
        ref = branch or "HEAD"
        result = self._run("rev-parse", ref)
        return result.stdout.strip()[:12]

    def get_merge_base(self, branch1: str, branch2: str) -> str:
        """Get merge base of two branches."""
        result = self._run("merge-base", branch1, branch2)
        return result.stdout.strip()[:12]

    def has_uncommitted_changes(self) -> bool:
        """Check for uncommitted changes."""
        result = self._run("status", "--porcelain")
        return bool(result.stdout.strip())

    def dry_run_merge(self, source: str) -> Tuple[bool, List[str]]:
        """
        Dry-run merge to detect conflicts.
        Returns (success, list of conflicting files).
        """
        try:
            # Try merge with --no-commit --no-ff
            result = self._run("merge", "--no-commit", "--no-ff", source, check=False)

            # Abort the merge to restore state
            self._run("merge", "--abort", check=False)

            if result.returncode != 0:
                # Parse conflict output
                conflicts = []
                for line in result.stdout.split("\n") + result.stderr.split("\n"):
                    if "CONFLICT" in line:
                        # Extract filename
                        match = re.search(r"CONFLICT.*:\s*(.+)$", line)
                        if match:
                            conflicts.append(match.group(1).strip())
                return False, conflicts

            return True, []
        except Exception as e:
            return False, [str(e)]

    def merge(
        self,
        source: str,
        target: str,
        message: str,
    ) -> Tuple[bool, str, List[str]]:
        """
        Execute merge.
        Returns (success, commit_sha, conflicts).
        """
        # Checkout target
        if not self.checkout(target):
            return False, "", ["Failed to checkout target branch"]

        try:
            # Pull latest
            self._run("pull", "--ff-only", check=False)

            # Merge
            result = self._run(
                "merge", source,
                "--no-ff",
                "-m", message,
                check=False,
            )

            if result.returncode != 0:
                # Parse conflicts
                conflicts = []
                for line in result.stdout.split("\n") + result.stderr.split("\n"):
                    if "CONFLICT" in line:
                        match = re.search(r"Merge conflict in (.+)$", line)
                        if match:
                            conflicts.append(match.group(1).strip())
                return False, "", conflicts

            # Get new commit SHA
            sha = self.get_head_sha()
            return True, sha, []

        except subprocess.CalledProcessError as e:
            return False, "", [str(e)]

    def push(self, branch: str, remote: str = "origin") -> Tuple[bool, str]:
        """Push branch to remote."""
        try:
            result = self._run("push", remote, branch)
            return True, ""
        except subprocess.CalledProcessError as e:
            return False, e.stderr

    def reset_hard(self, ref: str) -> bool:
        """Hard reset to ref."""
        try:
            self._run("reset", "--hard", ref)
            return True
        except subprocess.CalledProcessError:
            return False

    def create_amber_ref(
        self,
        source: str,
        agent: str,
        description: str,
    ) -> str:
        """
        Create Amber ref for conservation (DKIN v22).
        Returns ref name.
        """
        ts = time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())
        safe_desc = re.sub(r"[^a-zA-Z0-9_-]", "_", description)[:50]
        ref_name = f"refs/amber/{agent}/{ts}/{safe_desc}"

        try:
            sha = self.get_head_sha(source)
            self._run("update-ref", ref_name, sha)
            return ref_name
        except Exception as e:
            return f"error: {e}"

    def get_recent_commits(self, branch: str, limit: int = 10) -> List[dict]:
        """Get recent commits from branch."""
        try:
            result = self._run(
                "log", branch, f"-{limit}",
                "--format=%H|%an|%ae|%s|%ci",
            )
            commits = []
            for line in result.stdout.strip().split("\n"):
                if not line:
                    continue
                parts = line.split("|", 4)
                if len(parts) >= 5:
                    commits.append({
                        "sha": parts[0][:12],
                        "author_name": parts[1],
                        "author_email": parts[2],
                        "subject": parts[3],
                        "date": parts[4],
                    })
            return commits
        except Exception:
            return []


# =============================================================================
# QA Validation
# =============================================================================

class QAValidator:
    """Validate merge requests with QA Omega."""

    def __init__(self, bus_dir: Path):
        self.bus_dir = bus_dir
        self._qa_verdicts: Dict[str, dict] = {}

    def process_qa_event(self, event: dict) -> None:
        """Process QA verdict event."""
        topic = event.get("topic", "")
        if topic.startswith("qa.verdict."):
            data = event.get("data", {})
            scope = data.get("scope", "")
            if scope:
                self._qa_verdicts[scope] = {
                    "verdict": topic.split(".")[-1],  # pass/fail
                    "ts": event.get("ts", now_ts()),
                    "data": data,
                }

    def check_qa_approval(self, request: MergeRequest) -> Tuple[bool, str]:
        """
        Check if merge request has QA approval.
        Returns (approved, reason).
        """
        # Check pbtest_verdict from request
        if request.pbtest_verdict == "fail":
            return False, "PBTEST verdict is 'fail'"

        if not request.tests_passed:
            return False, "tests_passed is False"

        # Check for recent QA verdict on source branch
        scope = request.source_branch
        if scope in self._qa_verdicts:
            verdict = self._qa_verdicts[scope]
            if verdict["verdict"] == "fail":
                return False, f"QA verdict 'fail' for {scope}"

            # Check age - verdicts older than 1 hour are stale
            age = now_ts() - verdict["ts"]
            if age > 3600:
                return False, f"QA verdict stale ({format_duration(age)} old)"

        return True, "QA approved"


# =============================================================================
# Hygiene Validator
# =============================================================================

class HygieneValidator:
    """Validate merge requests for hygiene issues."""

    # Patterns for secrets
    SECRET_PATTERNS = [
        r"(?i)(api[_-]?key|apikey)\s*[=:]\s*['\"]?[a-zA-Z0-9_-]{20,}",
        r"(?i)(password|passwd|pwd)\s*[=:]\s*['\"]?[^\s'\"]{8,}",
        r"(?i)(secret|token)\s*[=:]\s*['\"]?[a-zA-Z0-9_-]{20,}",
        r"-----BEGIN (RSA |DSA |EC |OPENSSH )?PRIVATE KEY-----",
    ]

    # Max file sizes (bytes)
    MAX_BINARY_SIZE = 10 * 1024 * 1024  # 10MB

    def __init__(self, git: GitOperations):
        self.git = git

    def check_hygiene(self, request: MergeRequest) -> Tuple[bool, List[str]]:
        """
        Check merge request for hygiene issues.
        Returns (clean, list of issues).
        """
        issues = []

        # Get diff
        try:
            result = self.git._run(
                "diff", "--name-only",
                f"{request.target_branch}...{request.source_branch}",
            )
            changed_files = result.stdout.strip().split("\n")
        except Exception as e:
            issues.append(f"Failed to get diff: {e}")
            return False, issues

        # Check each file
        for filepath in changed_files:
            if not filepath:
                continue

            # Check for sensitive files
            if filepath.endswith((".env", ".pem", ".key", "credentials.json")):
                issues.append(f"Sensitive file in diff: {filepath}")

            # Check file content for secrets
            try:
                result = self.git._run("show", f"{request.source_branch}:{filepath}", check=False)
                if result.returncode == 0:
                    content = result.stdout
                    for pattern in self.SECRET_PATTERNS:
                        if re.search(pattern, content):
                            issues.append(f"Potential secret in {filepath}")
                            break
            except Exception:
                pass

        return len(issues) == 0, issues


# =============================================================================
# Merge Statistics
# =============================================================================

@dataclass
class MergeStats:
    """Statistics for merge operations."""
    merges_total: int = 0
    merges_success: int = 0
    merges_conflict: int = 0
    merges_rejected: int = 0
    rollbacks: int = 0
    conservations: int = 0

    # By actor
    by_actor: Dict[str, int] = field(default_factory=lambda: collections.defaultdict(int))

    # By target branch
    by_branch: Dict[str, int] = field(default_factory=lambda: collections.defaultdict(int))

    def record_merge(self, request: MergeRequest, success: bool, conflict: bool = False) -> None:
        self.merges_total += 1
        if success:
            self.merges_success += 1
        elif conflict:
            self.merges_conflict += 1
        else:
            self.merges_rejected += 1

        self.by_actor[request.actor] += 1
        self.by_branch[request.target_branch] += 1

    def to_dict(self) -> dict:
        return {
            "merges_total": self.merges_total,
            "merges_success": self.merges_success,
            "merges_conflict": self.merges_conflict,
            "merges_rejected": self.merges_rejected,
            "rollbacks": self.rollbacks,
            "conservations": self.conservations,
            "by_actor": dict(self.by_actor),
            "by_branch": dict(self.by_branch),
        }


# =============================================================================
# Codemaster Agent
# =============================================================================

class CodemasterAgent:
    """
    Main Codemaster Agent daemon.

    Implements the E Pluribus Unum pattern:
    - Many agents work in parallel on feature branches and PAIP clones
    - One Codemaster controls all writes to critical branches
    """

    def __init__(
        self,
        bus_dir: str | Path,
        repo_dir: str | Path,
        interval_s: float = DEFAULT_INTERVAL_S,
        quiet: bool = False,
    ):
        self.bus_dir = Path(bus_dir)
        self.repo_dir = Path(repo_dir)
        self.interval_s = interval_s
        self.quiet = quiet

        # Core components
        self.emitter = BusEmitter(bus_dir, actor="codemaster")
        self.tailer = FileTailer(self.bus_dir / "events.ndjson")
        self.git = GitOperations(repo_dir)
        self.qa_validator = QAValidator(self.bus_dir)
        self.hygiene_validator = HygieneValidator(self.git)
        self.cagent_registry_path = Path(
            os.environ.get("PLURIBUS_CAGENT_REGISTRY", DEFAULT_CAGENT_REGISTRY)
        )
        self.cagent_registry = load_cagent_registry(self.cagent_registry_path)

        # State
        self.state = State.IDLE
        self.stats = MergeStats()
        self.merge_queue: List[MergeRequest] = []
        self.current_request: Optional[MergeRequest] = None

        # Self-monitoring
        self.start_ts = now_ts()
        self.cycle_count = 0

        # Shutdown handling
        self._shutdown = threading.Event()
        self._lock = threading.Lock()
        self._setup_signal_handlers()

    def _setup_signal_handlers(self) -> None:
        """Setup graceful shutdown handlers."""
        def handle_shutdown(signum, frame):
            if not self.quiet:
                print(f"\n[codemaster] Received signal {signum}, shutting down...")
            self._shutdown.set()

        signal.signal(signal.SIGTERM, handle_shutdown)
        signal.signal(signal.SIGINT, handle_shutdown)
        atexit.register(self._on_exit)

    def _on_exit(self) -> None:
        """Emit shutdown event."""
        self.emitter.emit(
            topic="codemaster.shutdown",
            kind="log",
            level="info",
            data={
                "message": "Codemaster agent stopping",
                "uptime_s": now_ts() - self.start_ts,
                "total_cycles": self.cycle_count,
                "stats": self.stats.to_dict(),
            },
        )

    def _log(self, msg: str) -> None:
        if not self.quiet:
            print(f"[codemaster] {msg}")

    # -------------------------------------------------------------------------
    # Event Processing
    # -------------------------------------------------------------------------

    def _process_event(self, event: dict) -> None:
        """Process a single bus event."""
        topic = event.get("topic", "")

        # Skip our own events
        if event.get("actor") == "codemaster":
            return

        # Process QA events for validator
        if topic.startswith("qa.verdict."):
            self.qa_validator.process_qa_event(event)
            return

        # Process Codemaster requests
        if topic == "codemaster.merge.request":
            self._handle_merge_request(event)
        elif topic == "codemaster.rollback.request":
            self._handle_rollback_request(event)
        elif topic == "codemaster.status.request":
            self._handle_status_request(event)
        elif topic == "codemaster.conservation.request":
            self._handle_conservation_request(event)

    def _check_citizen_gate(self, request: MergeRequest) -> Tuple[bool, str, Dict[str, Any]]:
        profile = cagent_registry.resolve_actor(
            request.actor,
            self.cagent_registry,
            overrides=None,
            allow_override=False,
        )
        requested_class = ""
        if request.citizen_class:
            requested_class = cagent_registry.normalize_class(
                request.citizen_class,
                self.cagent_registry,
                profile.citizen_class,
            )
            if requested_class != profile.citizen_class:
                reason = (
                    f"Citizen class mismatch: request={requested_class}, registry={profile.citizen_class}"
                )
                return False, reason, {
                    "citizen_class": profile.citizen_class,
                    "citizen_tier": profile.citizen_tier,
                    "bootstrap_profile": profile.bootstrap_profile,
                    "scope_allowlist": profile.scope_allowlist,
                    "registry_source": profile.source,
                }

        if profile.citizen_class != "superagent":
            reason = f"SAGENT required (registry class={profile.citizen_class})"
            return False, reason, {
                "citizen_class": profile.citizen_class,
                "citizen_tier": profile.citizen_tier,
                "bootstrap_profile": profile.bootstrap_profile,
                "scope_allowlist": profile.scope_allowlist,
                "registry_source": profile.source,
            }

        return True, "", {
            "citizen_class": profile.citizen_class,
            "citizen_tier": profile.citizen_tier,
            "bootstrap_profile": profile.bootstrap_profile,
            "scope_allowlist": profile.scope_allowlist,
            "registry_source": profile.source,
        }

    def _handle_merge_request(self, event: dict) -> None:
        """Handle incoming merge request."""
        request = MergeRequest.from_event(event)

        # Validate target branch is critical
        if request.target_branch not in CRITICAL_BRANCHES:
            self.emitter.emit(
                topic="codemaster.merge.rejected",
                kind="response",
                level="warn",
                data={
                    "request_id": request.id,
                    "reason": f"Target '{request.target_branch}' is not a critical branch",
                    "note": "Non-critical branches don't require Codemaster",
                },
                trace_id=request.trace_id,
            )
            return

        # Validate source branch exists
        if not self.git.branch_exists(request.source_branch):
            self.emitter.emit(
                topic="codemaster.merge.rejected",
                kind="response",
                level="error",
                data={
                    "request_id": request.id,
                    "reason": f"Source branch '{request.source_branch}' does not exist",
                },
                trace_id=request.trace_id,
            )
            return

        # Citizen class gate
        ok, reason, citizen_meta = self._check_citizen_gate(request)
        if not ok:
            self.emitter.emit(
                topic="codemaster.merge.rejected",
                kind="response",
                level="warn",
                data={
                    "request_id": request.id,
                    "reason": reason,
                    "citizen_meta": citizen_meta,
                    "requested_class": request.citizen_class,
                },
                trace_id=request.trace_id,
            )
            return

        # Add to queue
        with self._lock:
            self.merge_queue.append(request)
            # Sort by priority (higher first)
            self.merge_queue.sort(key=lambda r: -r.priority)

        self._log(f"Queued merge: {request.source_branch} -> {request.target_branch} (priority {request.priority})")

        self.emitter.emit(
            topic="codemaster.merge.accepted",
            kind="response",
            level="info",
            data={
                "request_id": request.id,
                "queue_position": len(self.merge_queue),
                "source_branch": request.source_branch,
                "target_branch": request.target_branch,
                "citizen_meta": citizen_meta,
            },
            trace_id=request.trace_id,
        )

    def _handle_rollback_request(self, event: dict) -> None:
        """Handle rollback request."""
        data = event.get("data", {})
        commit = data.get("commit")
        branch = data.get("branch", "main")
        reason = data.get("reason", "Rollback requested")

        if not commit:
            self.emitter.emit(
                topic="codemaster.rollback.rejected",
                kind="response",
                level="error",
                data={"reason": "No commit specified for rollback"},
                trace_id=event.get("trace_id"),
            )
            return

        # Execute rollback
        self.state = State.ROLLING_BACK
        try:
            # Checkout target branch
            if not self.git.checkout(branch):
                raise Exception(f"Failed to checkout {branch}")

            # Record current HEAD for conservation
            current_head = self.git.get_head_sha()

            # Reset to commit
            if not self.git.reset_hard(commit):
                raise Exception(f"Failed to reset to {commit}")

            # Force push
            success, err = self.git.push(branch, "--force")
            if not success:
                raise Exception(f"Failed to push: {err}")

            self.stats.rollbacks += 1

            self.emitter.emit(
                topic="codemaster.rollback.complete",
                kind="artifact",
                level="warn",
                data={
                    "branch": branch,
                    "from_commit": current_head,
                    "to_commit": commit,
                    "reason": reason,
                },
                trace_id=event.get("trace_id"),
            )

            self._log(f"Rollback complete: {branch} -> {commit}")

        except Exception as e:
            self.emitter.emit(
                topic="codemaster.rollback.failed",
                kind="response",
                level="error",
                data={"error": str(e)},
                trace_id=event.get("trace_id"),
            )
        finally:
            self.state = State.IDLE

    def _handle_status_request(self, event: dict) -> None:
        """Handle status request."""
        # Get branch states
        branches = {}
        for branch in CRITICAL_BRANCHES:
            if self.git.branch_exists(branch):
                branches[branch] = {
                    "head": self.git.get_head_sha(branch),
                    "clean": True,  # Would need more checks
                }

        self.emitter.emit(
            topic="codemaster.status.report",
            kind="metric",
            level="info",
            data={
                "state": self.state,
                "uptime_s": now_ts() - self.start_ts,
                "queue_length": len(self.merge_queue),
                "current_request": self.current_request.to_dict() if self.current_request else None,
                "stats": self.stats.to_dict(),
                "branches": branches,
                "version": VERSION,
                "protocol": PROTOCOL_VERSION,
            },
            trace_id=event.get("trace_id"),
        )

    def _handle_conservation_request(self, event: dict) -> None:
        """Handle conservation request."""
        data = event.get("data", {})
        source = data.get("source", "")
        description = data.get("description", "Manual conservation")
        actor = event.get("actor", "unknown")

        if not source:
            self.emitter.emit(
                topic="codemaster.conservation.rejected",
                kind="response",
                level="error",
                data={"reason": "No source specified"},
                trace_id=event.get("trace_id"),
            )
            return

        # Create Amber ref
        ref = self.git.create_amber_ref(source, actor, description)
        self.stats.conservations += 1

        self.emitter.emit(
            topic="codemaster.conservation.complete",
            kind="artifact",
            level="info",
            data={
                "source": source,
                "preserved_at": ref,
                "description": description,
                "actor": actor,
            },
            trace_id=event.get("trace_id"),
        )

        self._log(f"Conserved: {source} -> {ref}")

    # -------------------------------------------------------------------------
    # Merge Processing
    # -------------------------------------------------------------------------

    def _process_queue(self) -> None:
        """Process next item in merge queue."""
        if self.state != State.IDLE:
            return

        with self._lock:
            if not self.merge_queue:
                return
            self.current_request = self.merge_queue.pop(0)

        request = self.current_request
        self._log(f"Processing merge: {request.source_branch} -> {request.target_branch}")

        # Emit audit start
        self.emitter.emit(
            topic="codemaster.audit.merge.start",
            kind="artifact",
            level="info",
            data={
                "request": request.to_dict(),
                "timestamp": now_iso(),
            },
            trace_id=request.trace_id,
        )

        try:
            # Phase 1: Validation
            self.state = State.VALIDATING

            # QA validation
            qa_ok, qa_reason = self.qa_validator.check_qa_approval(request)
            if not qa_ok:
                self._reject_merge(request, f"QA validation failed: {qa_reason}")
                return

            # Hygiene validation
            hygiene_ok, hygiene_issues = self.hygiene_validator.check_hygiene(request)
            if not hygiene_ok:
                self._reject_merge(request, f"Hygiene issues: {', '.join(hygiene_issues)}")
                return

            # Dry-run merge to check for conflicts
            merge_ok, conflicts = self.git.dry_run_merge(request.source_branch)
            if not merge_ok:
                self._handle_conflict(request, conflicts)
                return

            # Phase 2: Merge
            self.state = State.MERGING

            # Build merge message
            msg = f"Merge {request.source_branch} into {request.target_branch}\n\n"
            msg += f"{request.description}\n\n"
            msg += f"Requested-by: {request.actor}\n"
            msg += f"PBTEST-verdict: {request.pbtest_verdict}\n"
            msg += f"Codemaster-approved: true\n"
            msg += f"Request-id: {request.id}\n"

            success, sha, conflicts = self.git.merge(
                source=request.source_branch,
                target=request.target_branch,
                message=msg,
            )

            if not success:
                if conflicts:
                    self._handle_conflict(request, conflicts)
                else:
                    self._reject_merge(request, "Merge failed")
                return

            # Phase 3: Push
            self.state = State.PUSHING

            push_ok, push_err = self.git.push(request.target_branch)
            if not push_ok:
                # Rollback the merge
                self.git.reset_hard("HEAD~1")
                self._reject_merge(request, f"Push failed: {push_err}")
                return

            # Success!
            self.stats.record_merge(request, success=True)

            self.emitter.emit(
                topic="codemaster.merge.complete",
                kind="artifact",
                level="info",
                data={
                    "request_id": request.id,
                    "source_branch": request.source_branch,
                    "target_branch": request.target_branch,
                    "commit_sha": sha,
                    "actor": request.actor,
                },
                trace_id=request.trace_id,
            )

            # Emit audit complete
            self.emitter.emit(
                topic="codemaster.audit.merge.complete",
                kind="artifact",
                level="info",
                data={
                    "request": request.to_dict(),
                    "result": "success",
                    "commit_sha": sha,
                    "timestamp": now_iso(),
                },
                trace_id=request.trace_id,
            )

            self._log(f"Merge complete: {sha}")

        except Exception as e:
            self._reject_merge(request, f"Exception: {e}")
        finally:
            self.state = State.IDLE
            self.current_request = None

    def _reject_merge(self, request: MergeRequest, reason: str) -> None:
        """Reject a merge request."""
        self.stats.record_merge(request, success=False)

        # Conserve if policy allows
        if request.conservation_policy in ("always", "on_conflict"):
            ref = self.git.create_amber_ref(
                request.source_branch,
                request.actor,
                f"rejected-{request.id[:8]}",
            )
            self.stats.conservations += 1
        else:
            ref = None

        self.emitter.emit(
            topic="codemaster.merge.rejected",
            kind="response",
            level="warn",
            data={
                "request_id": request.id,
                "source_branch": request.source_branch,
                "target_branch": request.target_branch,
                "reason": reason,
                "conservation_ref": ref,
            },
            trace_id=request.trace_id,
        )

        self._log(f"Merge rejected: {reason}")

    def _handle_conflict(self, request: MergeRequest, conflicts: List[str]) -> None:
        """Handle merge conflict."""
        self.stats.record_merge(request, success=False, conflict=True)

        # Always conserve on conflict
        ref = self.git.create_amber_ref(
            request.source_branch,
            request.actor,
            f"conflict-{request.id[:8]}",
        )
        self.stats.conservations += 1

        self.emitter.emit(
            topic="codemaster.merge.conflict",
            kind="request",  # Request for human intervention
            level="warn",
            data={
                "request_id": request.id,
                "source_branch": request.source_branch,
                "target_branch": request.target_branch,
                "conflicts": conflicts,
                "conservation_ref": ref,
            },
            trace_id=request.trace_id,
        )

        # Emit audit
        self.emitter.emit(
            topic="codemaster.audit.conflict",
            kind="artifact",
            level="warn",
            data={
                "request": request.to_dict(),
                "conflicts": conflicts,
                "conservation_ref": ref,
                "timestamp": now_iso(),
            },
            trace_id=request.trace_id,
        )

        self._log(f"Conflict detected: {len(conflicts)} files")

    # -------------------------------------------------------------------------
    # Main Loop
    # -------------------------------------------------------------------------

    def run_cycle(self) -> None:
        """Run a single monitoring cycle."""
        self.cycle_count += 1

        # Process new events
        for event in self.tailer.read_new_events():
            self._process_event(event)

        # Process merge queue
        self._process_queue()

        # Emit health metrics every 10 cycles
        if self.cycle_count % 10 == 0:
            self.emitter.emit(
                topic="codemaster.health.heartbeat",
                kind="metric",
                level="info",
                data={
                    "state": self.state,
                    "queue_length": len(self.merge_queue),
                    "cycle": self.cycle_count,
                    "uptime_s": now_ts() - self.start_ts,
                    "stats": self.stats.to_dict(),
                },
            )

    def run(self) -> int:
        """Main run loop."""
        self._log(f"Codemaster Agent v{VERSION} starting")
        self._log(f"  Bus: {self.bus_dir}")
        self._log(f"  Repo: {self.repo_dir}")
        self._log(f"  Interval: {self.interval_s}s")
        self._log(f"  Protocol: {PROTOCOL_VERSION}")
        self._log(f"  DKIN: {DKIN_VERSION}")
        self._log(f"  Critical branches: {', '.join(CRITICAL_BRANCHES)}")

        # Emit startup event
        self.emitter.emit(
            topic="codemaster.startup",
            kind="log",
            level="info",
            data={
                "message": "Codemaster agent online - E Pluribus Unum",
                "version": VERSION,
                "protocol": PROTOCOL_VERSION,
                "dkin_version": DKIN_VERSION,
                "interval_s": self.interval_s,
                "pid": os.getpid(),
                "host": socket.gethostname(),
                "critical_branches": list(CRITICAL_BRANCHES),
            },
        )

        # Fetch latest from remotes
        self.git.fetch_all()

        # Main loop
        while not self._shutdown.is_set():
            try:
                self.run_cycle()
            except Exception as e:
                self._log(f"Cycle error: {e}")
                self.emitter.emit(
                    topic="codemaster.error",
                    kind="log",
                    level="error",
                    data={"error": str(e), "cycle": self.cycle_count},
                )

            self._shutdown.wait(timeout=self.interval_s)

        self._log("Codemaster Agent stopped")
        return 0


# =============================================================================
# CLI
# =============================================================================

def main() -> int:
    """Main entry point."""
    import argparse

    parser = argparse.ArgumentParser(
        description="Codemaster Agent - E Pluribus Unum Singleton Gatekeeper",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=f"""
Version: {VERSION}
Protocol: {PROTOCOL_VERSION}
DKIN: {DKIN_VERSION}

Philosophy: "From Many, One"
- Many agents work in parallel on feature branches and PAIP clones
- One Codemaster controls all writes to critical branches (main, staging, dev)

Environment Variables:
  PLURIBUS_BUS_DIR    Bus directory (default: {DEFAULT_BUS_DIR})
  PLURIBUS_ROOT       Repository root (default: {DEFAULT_REPO_DIR})

Examples:
  # Run with defaults
  python3 codemaster_agent.py

  # Custom interval
  python3 codemaster_agent.py --interval 10

  # Quiet mode
  python3 codemaster_agent.py --quiet

Reference: nucleus/specs/codemaster_protocol_v2.md
""",
    )

    parser.add_argument(
        "--bus-dir",
        default=os.environ.get("PLURIBUS_BUS_DIR", DEFAULT_BUS_DIR),
        help=f"Bus directory (default: {DEFAULT_BUS_DIR})",
    )
    parser.add_argument(
        "--repo-dir",
        default=os.environ.get("PLURIBUS_ROOT", DEFAULT_REPO_DIR),
        help=f"Repository root (default: {DEFAULT_REPO_DIR})",
    )
    parser.add_argument(
        "--interval", "-i",
        type=float,
        default=float(os.environ.get("CODEMASTER_INTERVAL_S", DEFAULT_INTERVAL_S)),
        help=f"Polling interval in seconds (default: {DEFAULT_INTERVAL_S})",
    )
    parser.add_argument(
        "--quiet", "-q",
        action="store_true",
        help="Quiet mode - no stdout logging",
    )
    parser.add_argument(
        "--version", "-v",
        action="version",
        version=f"codemaster-agent {VERSION}",
    )

    args = parser.parse_args()

    agent = CodemasterAgent(
        bus_dir=args.bus_dir,
        repo_dir=args.repo_dir,
        interval_s=args.interval,
        quiet=args.quiet,
    )

    return agent.run()


if __name__ == "__main__":
    sys.exit(main())
