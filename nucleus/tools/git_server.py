#!/usr/bin/env python3
"""
Git & FS Server for Pluribus Dashboard
======================================
Serves file system content and Isomorphic-Git state to the dashboard.

Endpoints:
  GET /fs/<path>      - Browse file system (JSON for dirs, raw for files)
  GET /health         - Fast health check (JSON)
  GET /git/log        - Get git log (via iso_git.mjs)
  GET /git/status     - Get git status (via iso_git.mjs)
  GET /git/graph      - Get commit graph (JSON)

Usage:
  python3 git_server.py --port 9300 --root /path/to/repo
"""

import argparse
import hashlib
import json
import os
import shlex
import shutil
import socket
import re
import subprocess
import sys
import mimetypes
import time
import uuid
from contextlib import closing
from http.server import HTTPServer, BaseHTTPRequestHandler
from pathlib import Path
from urllib.parse import unquote, urlparse, parse_qs

sys.dont_write_bytecode = True

DEFAULT_PORT = 9300

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

try:
    # Namespace import (repo root on sys.path). If unavailable, SemOps mutations will still work without bus emission.
    from nucleus.tools import agent_bus  # type: ignore
except Exception:  # pragma: no cover
    agent_bus = None  # type: ignore

try:
    from nucleus.tools import task_ledger as task_ledger_mod  # type: ignore
except Exception:  # pragma: no cover
    task_ledger_mod = None  # type: ignore

try:
    from nucleus.tools import recovery_snapshot as recovery_snapshot_mod  # type: ignore
except Exception:  # pragma: no cover
    recovery_snapshot_mod = None  # type: ignore

try:
    from nucleus.tools.semops_actions import derive_ui_actions, infer_flow_hints  # type: ignore
except Exception:  # pragma: no cover
    derive_ui_actions = None  # type: ignore
    infer_flow_hints = None  # type: ignore

class GitFSHandler(BaseHTTPRequestHandler):
    
    def __init__(self, *args, root: Path, tools_dir: Path, **kwargs):
        self.root = root
        self.tools_dir = tools_dir
        self.iso_git_tool = self.tools_dir / "iso_git.mjs"
        super().__init__(*args, **kwargs)

    def do_GET(self):
        try:
            if self.path.startswith('/fs'):
                self.handle_fs()
            elif self.path.startswith('/health'):
                self.handle_health()
            elif self.path.startswith('/git/health'):
                self.handle_health()
            elif self.path.startswith('/git/log'):
                self.handle_git_log()
            elif self.path.startswith('/git/status'):
                self.handle_git_status()
            elif self.path.startswith('/git/task_ledger'):
                self.handle_git_task_ledger()
            elif self.path.startswith('/git/recovery/snapshots'):
                self.handle_git_recovery_snapshots()
            elif self.path.startswith('/git/recovery'):
                self.handle_git_recovery()
            elif self.path.startswith('/git/branches'):
                self.handle_git_branches()
            elif self.path.startswith('/git/show/'):
                self.handle_git_show()
            elif self.path.startswith('/git/evo'):
                self.handle_git_evo()
            elif self.path.startswith('/git/lineage'):
                self.handle_git_lineage()
            elif self.path.startswith('/sota'):
                self.handle_sota()
            elif self.path.startswith('/metatest'):
                self.handle_metatest()
            elif self.path.startswith('/module/inspect'):
                self.handle_module_inspect()
            elif self.path.startswith('/module/verify'):
                self.handle_module_verify()
            elif self.path.startswith('/module/test'):
                self.handle_module_test()
            elif self.path.startswith('/semops'):
                if self.path.startswith('/semops/suggestions'):
                    self.handle_semops_suggestions()
                else:
                    self.handle_semops()
            elif self.path.startswith('/supermotd'):
                self.handle_supermotd()
            elif self.path.startswith('/browser/status') or self.path.startswith('/api/browser/status'):
                self.handle_browser_status()
            elif self.path.startswith('/browser/gemini_clean/status') or self.path.startswith('/ops/gemini_clean/status') or self.path.startswith('/api/browser/gemini_clean/status'):
                self.handle_gemini_clean_status()
            else:
                self.send_error(404, "Not Found")
        except (BrokenPipeError, ConnectionResetError):
            # Client disconnected; nothing to do and we should not emit a 500.
            return
        except Exception as e:
            self.send_error(500, str(e))

    def do_OPTIONS(self):
        self.send_response(200)
        self.send_header('Access-Control-Allow-Origin', '*')
        self.send_header('Access-Control-Allow-Methods', 'GET, POST, OPTIONS')
        self.send_header('Access-Control-Allow-Headers', 'Content-Type')
        self.end_headers()

    def do_POST(self):
        try:
            content_length = int(self.headers.get('Content-Length', 0))
            body = self.rfile.read(content_length).decode('utf-8') if content_length > 0 else '{}'
            try:
                data = json.loads(body) if body else {}
            except json.JSONDecodeError:
                data = {}

            if self.path.startswith('/git/hgt'):
                self.handle_git_hgt(data)
            elif self.path.startswith('/git/push'):
                self.handle_git_push(data)
            elif self.path.startswith('/git/fetch'):
                self.handle_git_fetch(data)
            elif self.path.startswith('/git/commit'):
                self.handle_git_commit(data)
            elif self.path.startswith('/semops/user_ops/define'):
                self.handle_semops_define(data)
            elif self.path.startswith('/semops/user_ops/undefine'):
                self.handle_semops_undefine(data)
            elif self.path.startswith('/semops/invoke'):
                self.handle_semops_invoke(data)
            elif self.path.startswith('/browser/vnc/navigate_login') or self.path.startswith('/api/browser/vnc/navigate_login'):
                self.handle_browser_vnc_navigate_login(data)
            elif self.path.startswith('/browser/vnc/check_login') or self.path.startswith('/api/browser/vnc/check_login'):
                self.handle_browser_vnc_check_login(data)
            elif self.path.startswith('/browser/vnc/focus_tab') or self.path.startswith('/api/browser/vnc/focus_tab'):
                self.handle_browser_vnc_focus_tab(data)
            elif self.path.startswith('/browser/vnc/enable') or self.path.startswith('/api/browser/vnc/enable'):
                self.handle_browser_vnc_enable(data)
            elif self.path.startswith('/browser/vnc/disable') or self.path.startswith('/api/browser/vnc/disable'):
                self.handle_browser_vnc_disable(data)
            elif self.path.startswith('/browser/vnc/status') or self.path.startswith('/api/browser/vnc/status'):
                self.handle_browser_vnc_status(data)
            elif self.path.startswith('/browser/bootstrap') or self.path.startswith('/api/browser/bootstrap'):
                self.handle_browser_bootstrap(data)
            elif self.path.startswith('/git/recovery/snapshot'):
                self.handle_git_recovery_snapshot(data)
            elif self.path.startswith('/git/rhizome/export'):
                self.handle_git_rhizome_export(data)
            else:
                self.send_error(404, "Not Found")
        except (BrokenPipeError, ConnectionResetError):
            return
        except Exception as e:
            self.send_error(500, str(e))

    def _parse_query(self) -> dict:
        return parse_qs(urlparse(self.path).query)

    def _collect_git_status_entries(self) -> tuple[list[dict], str | None]:
        try:
            result = subprocess.run(
                ["git", "status", "--porcelain"],
                capture_output=True,
                text=True,
                timeout=10,
                cwd=str(self.root),
            )
            entries = []
            for line in result.stdout.split('\n'):
                if not line.strip():
                    continue
                if len(line) > 3:
                    entries.append({"status": line[:2], "path": line[3:]})
            return entries, None
        except subprocess.TimeoutExpired:
            return [], "Status check timed out"
        except Exception as e:
            return [], str(e)

    def _emit_bus(self, *, topic: str, kind: str, level: str, actor: str, data: dict) -> None:
        """Best-effort bus emission (never blocks HTTP responses)."""
        if agent_bus is None:
            return
        try:
            paths = agent_bus.resolve_bus_paths(os.environ.get("PLURIBUS_BUS_DIR"))
            agent_bus.emit_event(
                paths,
                topic=topic,
                kind=kind,
                level=level,
                actor=actor,
                data=data,
                trace_id=None,
                run_id=None,
                durable=False,
            )
        except Exception:
            return

    def _bus_paths(self):
        if agent_bus is None:
            return None
        return agent_bus.resolve_bus_paths(os.environ.get("PLURIBUS_BUS_DIR") or str(self.root / ".pluribus" / "bus"))

    def _append_bus_request(self, *, topic: str, actor: str, data: dict) -> str:
        """Append a request event to the bus, returning req_id (caller-supplied or generated)."""
        req_id = str(data.get("req_id") or uuid.uuid4())
        data = {**data, "req_id": req_id}
        paths = self._bus_paths()
        if paths and agent_bus is not None:
            agent_bus.emit_event(
                paths,
                topic=topic,
                kind="request",
                level="info",
                actor=actor,
                data=data,
                trace_id=None,
                run_id=None,
                durable=False,
            )
            return req_id

        # Fallback: minimal NDJSON append (best-effort).
        events_path = Path(str(self.root / ".pluribus" / "bus" / "events.ndjson"))
        events_path.parent.mkdir(parents=True, exist_ok=True)
        payload = {"id": str(uuid.uuid4()), "ts": time.time(), "iso": self._iso_now(), "topic": topic, "actor": actor, "data": data}
        with events_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(payload, ensure_ascii=False, separators=(",", ":")) + "\n")
        return req_id

    def _wait_for_bus_response(self, *, response_topic: str, req_id: str, timeout_s: float) -> dict | None:
        paths = self._bus_paths()
        events_path = Path(paths.events_path) if paths else (self.root / ".pluribus" / "bus" / "events.ndjson")
        deadline = time.time() + max(0.1, float(timeout_s))

        pos = 0
        # Backfill scan: fast responses can land before we start tailing.
        if events_path.exists():
            try:
                backfill_bytes = 512 * 1024
                with events_path.open("rb") as f:
                    f.seek(0, os.SEEK_END)
                    end = f.tell()
                    start = max(0, end - backfill_bytes)
                    f.seek(start)
                    chunk = f.read(end - start)
                for raw in chunk.splitlines()[-2000:]:
                    try:
                        obj = json.loads(raw.decode("utf-8", errors="replace"))
                    except Exception:
                        continue
                    if (obj.get("topic") or "") != response_topic:
                        continue
                    data = obj.get("data") or {}
                    if not isinstance(data, dict):
                        continue
                    if str(data.get("req_id") or "") != req_id:
                        continue
                    return data
                pos = end
            except Exception:
                pos = events_path.stat().st_size

        while time.time() < deadline:
            if not events_path.exists():
                time.sleep(0.05)
                continue

            with events_path.open("rb") as f:
                f.seek(pos)
                chunk = f.read()
                pos = f.tell()

            if not chunk:
                time.sleep(0.05)
                continue

            for raw in chunk.splitlines():
                try:
                    obj = json.loads(raw.decode("utf-8", errors="replace"))
                except Exception:
                    continue
                if (obj.get("topic") or "") != response_topic:
                    continue
                data = obj.get("data") or {}
                if not isinstance(data, dict):
                    continue
                if str(data.get("req_id") or "") != req_id:
                    continue
                return data

        return None

    def _resolve_user_ops_paths(self) -> tuple[Path, Path | None]:
        """Return (active_path, primary_path_if_different)."""
        primary = (self.root / ".pluribus" / "user_operators.json").resolve()
        fallback = (self.root / ".pluribus_local" / "user_operators.json").resolve()
        active = primary
        try:
            active.parent.mkdir(parents=True, exist_ok=True)
            with open(active, "a", encoding="utf-8"):
                pass
        except (PermissionError, OSError):
            active = fallback
            active.parent.mkdir(parents=True, exist_ok=True)
        return active, (primary if active != primary else None)

    def _load_user_ops(self) -> dict:
        active, _primary = self._resolve_user_ops_paths()
        if not active.exists():
            return {"operators": {}}
        try:
            with open(active, encoding="utf-8") as f:
                obj = json.load(f) or {}
            if isinstance(obj, dict) and isinstance(obj.get("operators"), dict):
                return obj
        except Exception:
            pass
        return {"operators": {}}

    def _atomic_write_json(self, path: Path, obj: dict) -> None:
        tmp = path.with_suffix(path.suffix + ".tmp")
        data = json.dumps(obj, indent=2, ensure_ascii=False) + "\n"
        tmp.write_text(data, encoding="utf-8")
        tmp.replace(path)

    def _save_user_ops(self, obj: dict) -> None:
        active, primary_if_different = self._resolve_user_ops_paths()
        self._atomic_write_json(active, obj)
        if primary_if_different:
            try:
                self._atomic_write_json(primary_if_different, obj)
            except Exception:
                pass

    def handle_fs(self):
        # /fs/path/to/file
        rel_path = unquote(self.path[4:]) # strip /fs/
        if rel_path.startswith('/'):
            rel_path = rel_path[1:]
        
        # Security check: ensure we don't escape root.
        # Use abspath() instead of resolve() to avoid following symlinks (like .pluribus)
        # before the prefix check, while still neutralizing '..' segments.
        requested_path = Path(os.path.abspath(self.root / rel_path))

        if not str(requested_path).startswith(str(self.root)):
            self.send_error(403, "Forbidden")
            return

        abs_path = requested_path

        if not abs_path.exists():
            self.send_error(404, "File not found")
            return

        if abs_path.is_dir():
            entries = []
            for item in sorted(abs_path.iterdir()):
                if item.name.startswith('.'): continue # Hide dotfiles for now
                try:
                    stat = item.stat()
                    entries.append({
                        "name": item.name,
                        "type": "dir" if item.is_dir() else "file",
                        "path": str(item.relative_to(self.root)),
                        "size": stat.st_size if item.is_file() else 0,
                    })
                except:
                    entries.append({
                        "name": item.name,
                        "type": "dir" if item.is_dir() else "file",
                        "path": str(item.relative_to(self.root))
                    })

            self.send_json({"entries": entries, "path": rel_path})

        else:
            # Serve file content
            mime, _ = mimetypes.guess_type(abs_path)
            try:
                self.send_response(200)
                self.send_header('Content-Type', mime or 'text/plain')
                self.send_header('Access-Control-Allow-Origin', '*')
                self.end_headers()
                with abs_path.open('rb') as f:
                    self.wfile.write(f.read())
            except PermissionError:
                # Avoid "blank" UI: make unreadable paths explicit (e.g., mode 0600 under non-root).
                self.send_error(403, "Permission denied")
            except OSError as e:
                self.send_error(500, f"Failed to read file: {e}")

    def handle_health(self):
        self.send_json({"ok": True})

    def run_iso_git(self, cmd):
        # Run node iso_git.mjs <command> [dir] [args...]
        # Split cmd to support multi-word commands like "evo list" or "show <sha>".
        try:
            cmd_parts = cmd.split() if isinstance(cmd, str) else [cmd]
            if not cmd_parts:
                return ""

            # iso_git.mjs expects the repo dir immediately after the command, except for `clone`.
            if cmd_parts[0] == "clone":
                # clone <url> [target-dir]
                args = ["node", str(self.iso_git_tool), "clone"] + cmd_parts[1:]
            elif cmd_parts[0] == "evo":
                # evo <dir> <subcmd> [slug]
                args = ["node", str(self.iso_git_tool), "evo", str(self.root)] + cmd_parts[1:]
            else:
                args = ["node", str(self.iso_git_tool), cmd_parts[0], str(self.root)] + cmd_parts[1:]

            res = subprocess.run(
                args,
                capture_output=True,
                text=True,
                timeout=30,
                cwd=str(self.root),
                env={**os.environ, "PLURIBUS_BUS_DIR": str(self.root / ".pluribus" / "bus")},
            )
            if res.returncode == 0:
                return res.stdout
            return res.stderr or res.stdout
        except Exception as e:
            return str(e)

    def handle_git_log(self):
        # iso_git.mjs log returns JSON (commits[]).
        raw = self.run_iso_git("log") or ""
        try:
            data = json.loads(raw)
            if isinstance(data, dict) and isinstance(data.get("commits"), list):
                self.send_json(data)
                return
        except Exception:
            pass
        self.send_json({"commits": [], "error": "failed to parse iso_git log output"})

    def handle_git_status(self):
        # Use native git for status (much faster than iso_git on large repos)
        entries, error = self._collect_git_status_entries()
        payload = {"status": entries}
        if error:
            payload["error"] = error
        self.send_json(payload)

    def _resolve_recent_iterations_dir(self) -> Path | None:
        root = self.root / "agent_reports"
        if not root.exists():
            return None
        candidates = [p for p in root.iterdir() if p.is_dir() and p.name.startswith("recent_iterations_")]
        if not candidates:
            return None
        return max(candidates, key=lambda p: p.stat().st_mtime)

    def _list_wip_bundles(self, recent_dir: Path | None) -> list[dict]:
        bundles: list[dict] = []
        if not recent_dir or not recent_dir.exists():
            return bundles
        for entry in sorted(recent_dir.iterdir()):
            if not entry.is_dir() or not entry.name.startswith("wip_"):
                continue
            try:
                entries = sum(1 for _ in entry.iterdir())
                mtime = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(entry.stat().st_mtime))
            except Exception:
                entries = 0
                mtime = None
            bundles.append({
                "name": entry.name,
                "path": str(entry.relative_to(self.root)),
                "entries": entries,
                "mtime": mtime,
            })
        return bundles

    def _list_recovery_reports(self, recent_dir: Path | None) -> list[dict]:
        reports: list[dict] = []
        sources = [self.root / "agent_recovery_reports"]
        if recent_dir:
            sources.append(recent_dir)
        for src in sources:
            if not src.exists():
                continue
            for entry in sorted(src.glob("*.md")):
                name = entry.name.lower()
                if "recovery" not in name and "crash" not in name and "resilience" not in name:
                    continue
                try:
                    mtime = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(entry.stat().st_mtime))
                except Exception:
                    mtime = None
                reports.append({
                    "name": entry.name,
                    "path": str(entry.relative_to(self.root)),
                    "mtime": mtime,
                })
        return sorted(reports, key=lambda r: r.get("mtime") or "", reverse=True)

    def _read_interrupted_tasks(self, *, limit: int = 30) -> dict:
        path = self.root / "tasks" / "reconstructed_interrupted_tasks.md"
        if not path.exists():
            return {"exists": False, "path": str(path.relative_to(self.root))}
        preview = ""
        try:
            lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
            preview = "\n".join(lines[:limit])
        except Exception:
            preview = ""
        try:
            mtime = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(path.stat().st_mtime))
        except Exception:
            mtime = None
        return {"exists": True, "path": str(path.relative_to(self.root)), "preview": preview, "mtime": mtime}

    def _list_task_ledger_entries(self, limit: int, filters: dict) -> dict:
        if task_ledger_mod is None:
            return {"entries": [], "error": "task_ledger unavailable"}
        ledger_path = task_ledger_mod.default_ledger_path()
        if hasattr(task_ledger_mod, "tail_entries"):
            entries = task_ledger_mod.tail_entries(
                ledger_path,
                actor=filters.get("actor"),
                topic=filters.get("topic"),
                status=filters.get("status"),
                req_id=filters.get("req_id"),
                limit=limit,
            )
        else:
            entries = task_ledger_mod.read_entries(
                ledger_path,
                actor=filters.get("actor"),
                topic=filters.get("topic"),
                status=filters.get("status"),
                req_id=filters.get("req_id"),
                limit=limit,
            )
        return {"entries": entries, "path": str(ledger_path), "count": len(entries)}

    def _list_recovery_snapshots(self, limit: int) -> dict:
        snapshot_dirs: list[Path] = []
        if task_ledger_mod is not None:
            snapshot_dirs.append(task_ledger_mod.resolve_state_dir(for_write=False) / "recovery_snapshots")
        snapshot_dirs.extend([
            self.root / ".pluribus" / "index" / "recovery_snapshots",
            self.root / ".pluribus_local" / "index" / "recovery_snapshots",
        ])

        seen = set()
        items: list[dict] = []
        for directory in snapshot_dirs:
            if not directory.exists():
                continue
            for path in directory.glob("recovery_snapshot_*.json"):
                if str(path) in seen:
                    continue
                seen.add(str(path))
                try:
                    payload = json.loads(path.read_text(encoding="utf-8", errors="replace"))
                except Exception:
                    payload = {}
                try:
                    mtime = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(path.stat().st_mtime))
                except Exception:
                    mtime = None
                items.append({
                    "path": str(path.relative_to(self.root)),
                    "created_iso": payload.get("created_iso") or mtime,
                    "summary": payload.get("summary") or {},
                    "run_id": payload.get("run_id"),
                })
        items.sort(key=lambda item: item.get("created_iso") or "", reverse=True)
        return {"snapshots": items[:limit], "count": len(items)}

    def handle_git_task_ledger(self):
        query = self._parse_query()
        limit = int((query.get("limit") or ["50"])[0])
        filters = {
            "actor": (query.get("actor") or [None])[0],
            "topic": (query.get("topic") or [None])[0],
            "status": (query.get("status") or [None])[0],
            "req_id": (query.get("req_id") or [None])[0],
        }
        payload = self._list_task_ledger_entries(limit, filters)
        self.send_json(payload)

    def handle_git_recovery_snapshots(self):
        query = self._parse_query()
        limit = int((query.get("limit") or ["10"])[0])
        payload = self._list_recovery_snapshots(limit)
        self.send_json(payload)

    def handle_git_recovery(self):
        query = self._parse_query()
        limit = int((query.get("limit") or ["10"])[0])
        ledger_limit = int((query.get("ledger_limit") or [str(limit)])[0])
        snapshot_limit = int((query.get("snapshot_limit") or [str(limit)])[0])

        entries, error = self._collect_git_status_entries()
        recent_dir = self._resolve_recent_iterations_dir()
        payload = {
            "status": {
                "entries": entries,
                "count": len(entries),
                "error": error,
            },
            "recent_iterations": str(recent_dir.relative_to(self.root)) if recent_dir else None,
            "wip_bundles": self._list_wip_bundles(recent_dir),
            "recovery_reports": self._list_recovery_reports(recent_dir),
            "interrupted_tasks": self._read_interrupted_tasks(),
            "task_ledger": self._list_task_ledger_entries(ledger_limit, {}),
            "recovery_snapshots": self._list_recovery_snapshots(snapshot_limit),
        }
        self.send_json(payload)

    def send_json(self, data):
        try:
            self.send_response(200)
            self.send_header('Content-Type', 'application/json')
            self.send_header('Access-Control-Allow-Origin', '*')
            self.end_headers()
            self.wfile.write(json.dumps(data).encode('utf-8'))
        except (BrokenPipeError, ConnectionResetError):
            return

    def handle_git_branches(self):
        """Get list of branches using isomorphic-git"""
        raw = self.run_iso_git("branch")
        branches = []
        current = None
        for line in raw.split('\n'):
            line = line.strip()
            if not line: continue
            if line.startswith('* '):
                current = line[2:].strip()
                branches.append({"name": current, "current": True})
            else:
                branches.append({"name": line, "current": False})

        # If no branches found, try to get current from HEAD
        if not branches:
            # Fallback: assume main branch exists
            branches = [{"name": "main", "current": True}]
            current = "main"

        self.send_json({"branches": branches, "current": current or (branches[0]["name"] if branches else None)})

    def handle_git_show(self):
        """Show commit details - /git/show/<sha>"""
        sha = unquote(self.path.split('/git/show/')[-1])
        if not sha or len(sha) < 4:
            self.send_json({"error": "Invalid SHA"})
            return

        # Get commit details via iso_git
        raw = self.run_iso_git(f"show {sha}")

        # Parse the output - iso_git show returns JSON
        try:
            data = json.loads(raw)
            self.send_json(data)
        except:
            self.send_json({"sha": sha, "raw": raw})

    def handle_git_recovery_snapshot(self, data: dict) -> None:
        if recovery_snapshot_mod is None or agent_bus is None:
            self.send_json({"error": "recovery snapshot unavailable"})
            return
        actor = data.get("actor") or agent_bus.default_actor()
        run_id = data.get("run_id")
        bus_dir = data.get("bus_dir") or os.environ.get("PLURIBUS_BUS_DIR")

        snapshot = recovery_snapshot_mod.collect_snapshot(
            repo_dir=self.root,
            bus_limit=int(data.get("bus_limit") or 200),
            ledger_limit=int(data.get("ledger_limit") or 200),
            log_limit=int(data.get("log_limit") or 20),
            actor=actor,
            run_id=run_id,
            bus_dir=bus_dir,
        )
        path = recovery_snapshot_mod.write_snapshot(snapshot, recovery_snapshot_mod.snapshot_dir())

        try:
            paths = agent_bus.resolve_bus_paths(bus_dir)
            agent_bus.emit_event(
                paths,
                topic="recovery.snapshot.created",
                kind="artifact",
                level="info",
                actor=actor,
                data={"path": str(path), "summary": snapshot.get("summary"), "run_id": run_id},
                trace_id=None,
                run_id=run_id,
                durable=False,
            )
        except Exception:
            pass

        self.send_json({"path": str(path), "summary": snapshot.get("summary"), "run_id": run_id})

    def handle_git_rhizome_export(self, data: dict) -> None:
        sha = (data.get("sha") or data.get("ref") or "").strip()
        if len(sha) < 4:
            self.send_json({"error": "Missing or invalid sha"})
            return

        raw = self.run_iso_git(f"show {sha}") or ""
        try:
            commit = json.loads(raw)
        except Exception:
            self.send_json({"error": "Failed to parse commit details", "raw": raw})
            return

        export_dir = self.root / "rhizome_exports"
        export_dir.mkdir(parents=True, exist_ok=True)
        export_path = export_dir / f"git_commit_{commit.get('sha', sha)[:12]}.md"

        note = (data.get("note") or "").strip()
        tags = [t for t in (data.get("tags") or []) if isinstance(t, str) and t.strip()]
        for tag in ("git", "commit", "snapshot"):
            if tag not in tags:
                tags.append(tag)

        diff = commit.get("diff") or []
        diff_lines = [f"- [{d.get('status', '?')}] {d.get('path', '')}" for d in diff[:200]]
        content = [
            "# Git Commit Snapshot",
            "",
            f"SHA: {commit.get('sha', sha)}",
            f"Author: {commit.get('author', {}).get('name', 'unknown')} <{commit.get('author', {}).get('email', 'unknown')}>",
            f"Date: {commit.get('author', {}).get('date', 'unknown')}",
            "",
            "## Message",
            commit.get("message", "").strip() or "(empty message)",
            "",
            "## Diff Summary",
            f"Files changed: {commit.get('diffCount', len(diff))}",
            *diff_lines,
        ]
        if note:
            content.extend(["", "## Operator Note", note])
        export_path.write_text("\n".join(content) + "\n", encoding="utf-8")

        sha256 = hashlib.sha256(export_path.read_bytes()).hexdigest()
        cmd = [
            sys.executable,
            str(self.tools_dir / "rhizome.py"),
            "--root",
            str(self.root),
            "ingest",
            str(export_path),
            "--store",
            "--emit-bus",
        ]
        for tag in tags:
            cmd.extend(["--tag", tag])
        result = subprocess.run(cmd, capture_output=True, text=True, env={**os.environ, "PYTHONDONTWRITEBYTECODE": "1"})

        if result.returncode != 0:
            self.send_json({"error": result.stderr.strip() or result.stdout.strip(), "path": str(export_path)})
            return

        self._emit_bus(
            topic="git.rhizome.exported",
            kind="artifact",
            level="info",
            actor="git-server",
            data={"sha": commit.get("sha", sha), "artifact_sha256": sha256, "path": str(export_path), "tags": tags},
        )

        self.send_json({
            "ok": True,
            "sha": commit.get("sha", sha),
            "artifact_sha256": sha256,
            "path": str(export_path),
            "tags": tags,
            "ingest_output": result.stdout.strip(),
        })

    def handle_git_evo(self):
        """List evolutionary branches"""
        raw = self.run_iso_git("evo list")
        branches = []
        for line in raw.split('\n'):
            line = line.strip()
            if not line or line.startswith('No ') or line.startswith('Evolutionary'):
                continue
            current = line.startswith('* ')
            name = line[2:] if current else line
            if name.startswith('evo/'):
                # Parse date and slug from branch name
                parts = name[4:].split('-', 1)  # Skip 'evo/'
                date = parts[0] if len(parts) > 0 else ''
                slug = parts[1] if len(parts) > 1 else ''
                branches.append({
                    "name": name,
                    "current": current,
                    "date": date,
                    "slug": slug,
                })
        self.send_json({"branches": branches, "count": len(branches)})

    def handle_git_lineage(self):
        """Get current lineage info"""
        lineage_file = self.root / ".pluribus" / "lineage.json"
        if lineage_file.exists():
            try:
                with open(lineage_file) as f:
                    data = json.load(f)
                self.send_json({"lineage": data, "initialized": True})
            except Exception as e:
                self.send_json({"error": str(e), "initialized": False})
        else:
            self.send_json({"lineage": None, "initialized": False})

    def _sota_index_path(self) -> Path:
        return self.root / ".pluribus" / "index" / "sota.ndjson"

    def _rebuild_sota_index(self) -> tuple[bool, str]:
        manager = self.tools_dir / "sota_manager.py"
        if not manager.exists():
            return False, "sota_manager.py not found"
        try:
            result = subprocess.run(
                [sys.executable, str(manager), "rebuild", "--root", str(self.root)],
                capture_output=True,
                text=True,
                timeout=30,
                cwd=str(self.root),
            )
            if result.returncode != 0:
                return False, result.stderr.strip() or result.stdout.strip() or "sota_manager rebuild failed"
            return True, result.stdout.strip()
        except subprocess.TimeoutExpired:
            return False, "sota_manager rebuild timed out"
        except Exception as e:
            return False, str(e)

    def _metatest_cache_path(self) -> Path:
        return self.root / ".pluribus" / "index" / "metatest_inventory.json"

    def _load_json_cache(self, path: Path) -> dict | None:
        if not path.exists():
            return None
        try:
            with path.open("r", encoding="utf-8", errors="replace") as handle:
                return json.load(handle)
        except Exception:
            return None

    def _write_json_cache(self, path: Path, payload: dict) -> None:
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            with path.open("w", encoding="utf-8") as handle:
                json.dump(payload, handle)
        except Exception:
            return

    def handle_sota(self):
        """Get SOTA items from index - serves to WebUI/TUI"""
        sota_file = self._sota_index_path()
        rebuild_requested = (self._get_query_param("rebuild") or "").lower() in ("1", "true", "yes")
        rebuild_attempted = False
        rebuild_reason = None
        rebuild_output = None

        if rebuild_requested or not sota_file.exists():
            rebuild_attempted = True
            rebuild_reason = "requested" if rebuild_requested else "missing"
            ok, output = self._rebuild_sota_index()
            if not ok:
                self.send_json({
                    "items": [],
                    "count": 0,
                    "error": output,
                    "rebuilt": False,
                    "rebuild_reason": rebuild_reason,
                })
                return
            rebuild_output = output

        items = []
        try:
            with open(sota_file, 'r') as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        item = json.loads(line)
                        # Transform to match WebUI SOTAItem interface
                        items.append({
                            "id": item.get("id", ""),
                            "url": item.get("url", ""),
                            "title": item.get("title", ""),
                            "org": item.get("org", ""),
                            "region": item.get("region", ""),
                            "type": item.get("type", ""),
                            "priority": item.get("priority", 3),
                            "cadence_days": item.get("cadence_days", 7),
                            "tags": item.get("tags", []),
                            "notes": item.get("notes", ""),
                        })
                    except json.JSONDecodeError:
                        continue

            if not items and not rebuild_attempted:
                rebuild_attempted = True
                rebuild_reason = "empty"
                ok, output = self._rebuild_sota_index()
                if ok:
                    rebuild_output = output
                    with open(sota_file, 'r') as f:
                        for line in f:
                            line = line.strip()
                            if not line:
                                continue
                            try:
                                item = json.loads(line)
                                items.append({
                                    "id": item.get("id", ""),
                                    "url": item.get("url", ""),
                                    "title": item.get("title", ""),
                                    "org": item.get("org", ""),
                                    "region": item.get("region", ""),
                                    "type": item.get("type", ""),
                                    "priority": item.get("priority", 3),
                                    "cadence_days": item.get("cadence_days", 7),
                                    "tags": item.get("tags", []),
                                    "notes": item.get("notes", ""),
                                })
                            except json.JSONDecodeError:
                                continue

            # Sort by priority (lower = more important)
            items.sort(key=lambda x: (x.get("priority", 99), x.get("title", "")))
            payload = {"items": items, "count": len(items)}
            if rebuild_attempted:
                payload["rebuilt"] = True
                payload["rebuild_reason"] = rebuild_reason
                if rebuild_output:
                    payload["rebuild_output"] = rebuild_output
            self.send_json(payload)
        except Exception as e:
            self.send_json({"items": [], "count": 0, "error": str(e)})

    def handle_metatest(self):
        """Serve MetaTest inventory JSON."""
        collector = self.tools_dir / "metatest_collector.py"
        if not collector.exists():
            self.send_json({"error": "metatest_collector.py not found"})
            return
        cache_path = self._metatest_cache_path()
        cache_ttl_s = 300
        refresh = (self._get_query_param("refresh") or "").lower() in ("1", "true", "yes")
        if not refresh and cache_path.exists():
            age_s = time.time() - cache_path.stat().st_mtime
            if age_s <= cache_ttl_s:
                cached = self._load_json_cache(cache_path)
                if cached is not None:
                    cached["cache"] = {"used": True, "age_s": int(age_s), "stale": False}
                    self.send_json(cached)
                    return
        try:
            result = subprocess.run(
                [sys.executable, str(collector), "--inventory", "--json"],
                capture_output=True,
                text=True,
                timeout=30,
                cwd=str(self.root),
            )
            if result.returncode != 0:
                cached = self._load_json_cache(cache_path)
                if cached is not None:
                    cached["cache"] = {"used": True, "stale": True, "error": result.stderr.strip()}
                    self.send_json(cached)
                    return
                self.send_json({
                    "error": "metatest collector failed",
                    "status": result.returncode,
                    "stderr": result.stderr.strip(),
                })
                return
            try:
                payload = json.loads(result.stdout)
            except json.JSONDecodeError as e:
                cached = self._load_json_cache(cache_path)
                if cached is not None:
                    cached["cache"] = {"used": True, "stale": True, "error": str(e)}
                    self.send_json(cached)
                    return
                self.send_json({
                    "error": f"metatest collector returned invalid JSON: {e}",
                    "stdout": result.stdout.strip(),
                })
                return
            payload["cache"] = {"used": False, "stale": False}
            self._write_json_cache(cache_path, payload)
            self.send_json(payload)
        except subprocess.TimeoutExpired:
            cached = self._load_json_cache(cache_path)
            if cached is not None:
                cached["cache"] = {"used": True, "stale": True, "error": "timeout"}
                self.send_json(cached)
                return
            self.send_json({"error": "metatest collector timed out"})
        except Exception as e:
            cached = self._load_json_cache(cache_path)
            if cached is not None:
                cached["cache"] = {"used": True, "stale": True, "error": str(e)}
                self.send_json(cached)
                return
            self.send_json({"error": str(e)})

    def _get_query_param(self, param: str) -> str | None:
        """Extract query parameter from path."""
        from urllib.parse import urlparse, parse_qs
        parsed = urlparse(self.path)
        params = parse_qs(parsed.query)
        values = params.get(param, [])
        return values[0] if values else None

    def _find_module_path(self, filename: str) -> Path | None:
        """Find module by filename or relative path."""
        # Handle full relative path from root
        full_path = self.root / filename
        if full_path.exists():
            return full_path

        # Just filename - search in tools directory
        tools_dir = self.tools_dir
        basename = Path(filename).name

        # Direct match
        direct = tools_dir / basename
        if direct.exists():
            return direct
        # Search recursively
        for p in tools_dir.rglob(basename):
            return p
        # Try mjs in tools
        if basename.endswith('.py'):
            mjs_name = basename.replace('.py', '.mjs')
            mjs_path = tools_dir / mjs_name
            if mjs_path.exists():
                return mjs_path
        return None

    def handle_module_inspect(self):
        """Inspect a module - return metadata, exports, line count."""
        filename = self._get_query_param('file')
        if not filename:
            self.send_json({"error": "Missing 'file' parameter"})
            return

        mod_path = self._find_module_path(filename)
        if not mod_path or not mod_path.exists():
            self.send_json({
                "loaded": False,
                "error": f"Module not found: {filename}",
                "lineCount": 0,
                "exports": [],
                "imports": []
            })
            return

        try:
            content = mod_path.read_text(errors='replace')
            lines = content.split('\n')
            line_count = len(lines)

            # Extract exports (Python: def/class, JS: export)
            exports = []
            imports = []

            if mod_path.suffix == '.py':
                for line in lines:
                    line = line.strip()
                    if line.startswith('def ') and '(' in line:
                        name = line[4:line.index('(')]
                        if not name.startswith('_'):
                            exports.append(name)
                    elif line.startswith('class ') and ':' in line:
                        name = line[6:].split('(')[0].split(':')[0].strip()
                        exports.append(name)
                    elif line.startswith('import ') or line.startswith('from '):
                        imports.append(line.split()[1].split('.')[0])
            elif mod_path.suffix in ('.js', '.mjs', '.ts', '.tsx'):
                for line in lines:
                    if 'export ' in line:
                        if 'function ' in line:
                            parts = line.split('function ')
                            if len(parts) > 1:
                                name = parts[1].split('(')[0].strip()
                                exports.append(name)
                        elif 'const ' in line or 'let ' in line:
                            parts = line.split('const ' if 'const ' in line else 'let ')
                            if len(parts) > 1:
                                name = parts[1].split('=')[0].strip()
                                exports.append(name)

            self.send_json({
                "loaded": True,
                "lastCheck": self._iso_now(),
                "lineCount": line_count,
                "exports": exports[:20],  # Limit to 20
                "imports": list(set(imports))[:20],
                "path": str(mod_path.relative_to(self.root)),
                "size": mod_path.stat().st_size
            })
        except Exception as e:
            self.send_json({"loaded": False, "error": str(e)})

    def handle_module_verify(self):
        """Verify a module - check syntax, imports."""
        filename = self._get_query_param('file')
        if not filename:
            self.send_json({"error": "Missing 'file' parameter"})
            return

        mod_path = self._find_module_path(filename)
        if not mod_path or not mod_path.exists():
            self.send_json({"success": False, "loaded": False, "error": f"Module not found: {filename}"})
            return

        errors = []

        try:
            if mod_path.suffix == '.py':
                # Python syntax check
                result = subprocess.run(
                    ["python3", "-m", "py_compile", str(mod_path)],
                    capture_output=True,
                    text=True,
                    timeout=10
                )
                if result.returncode != 0:
                    errors.append(result.stderr.strip())
            elif mod_path.suffix in ('.js', '.mjs'):
                # Node syntax check
                result = subprocess.run(
                    ["node", "--check", str(mod_path)],
                    capture_output=True,
                    text=True,
                    timeout=10
                )
                if result.returncode != 0:
                    errors.append(result.stderr.strip())

            self.send_json({
                "success": len(errors) == 0,
                "loaded": True,
                "lastCheck": self._iso_now(),
                "errors": errors,
                "testStatus": "pass" if len(errors) == 0 else "fail"
            })
        except subprocess.TimeoutExpired:
            self.send_json({"success": False, "loaded": True, "errors": ["Verification timed out"]})
        except Exception as e:
            self.send_json({"success": False, "loaded": False, "errors": [str(e)]})

    def handle_module_test(self):
        """Run tests for a module if they exist."""
        filename = self._get_query_param('file')
        if not filename:
            self.send_json({"error": "Missing 'file' parameter"})
            return

        mod_path = self._find_module_path(filename)
        if not mod_path or not mod_path.exists():
            self.send_json({"success": False, "error": f"Module not found: {filename}"})
            return

        # Look for corresponding test file
        test_dir = self.root / "nucleus" / "tools" / "tests"
        test_name = f"test_{mod_path.stem}.py"
        test_path = test_dir / test_name

        if not test_path.exists():
            # Try to run module's internal tests if it has if __name__ == "__main__"
            if mod_path.suffix == '.py':
                try:
                    result = subprocess.run(
                        ["python3", str(mod_path), "--help"],
                        capture_output=True,
                        text=True,
                        timeout=5,
                        cwd=str(self.root)
                    )
                    self.send_json({
                        "success": result.returncode == 0,
                        "testStatus": "pass" if result.returncode == 0 else "skip",
                        "output": result.stdout[:500] if result.stdout else "",
                        "errors": [result.stderr[:200]] if result.stderr and result.returncode != 0 else [],
                        "note": "No dedicated test file found, ran --help check"
                    })
                    return
                except:
                    pass

            self.send_json({
                "success": True,
                "testStatus": "skip",
                "note": f"No test file found: {test_name}"
            })
            return

        try:
            result = subprocess.run(
                ["python3", "-m", "pytest", str(test_path), "-v", "--tb=short"],
                capture_output=True,
                text=True,
                timeout=60,
                cwd=str(self.root),
                env={**os.environ, "PYTHONPATH": str(self.root)}
            )

            success = result.returncode == 0
            self.send_json({
                "success": success,
                "testStatus": "pass" if success else "fail",
                "output": result.stdout[-1000:] if result.stdout else "",
                "errors": [result.stderr[-500:]] if result.stderr and not success else []
            })
        except subprocess.TimeoutExpired:
            self.send_json({"success": False, "testStatus": "fail", "errors": ["Test timed out after 60s"]})
        except Exception as e:
            self.send_json({"success": False, "testStatus": "fail", "errors": [str(e)]})

    def handle_semops(self):
        """Serve semantic operators schema for WebUI lexer."""
        semops_path = self.root / "nucleus" / "specs" / "semops.json"
        user_ops_path, _ = self._resolve_user_ops_paths()

        try:
            # Load base semops
            operators = {}
            alias_map = {}
            commands = []
            tool_map = {}
            bus_topics = {}

            if semops_path.exists():
                with open(semops_path) as f:
                    data = json.load(f)

                for op_id, op_data in data.get("operators", {}).items():
                    operators[op_id] = {
                        "id": op_data.get("id", op_id),
                        "name": op_data.get("name", op_id.upper()),
                        "domain": op_data.get("domain", "unknown"),
                        "category": op_data.get("category", "unknown"),
                        "description": op_data.get("description", ""),
                        "aliases": op_data.get("aliases", []),
                        "tool": op_data.get("tool"),
                        "bus_topic": op_data.get("bus_topic"),
                        "bus_kind": op_data.get("bus_kind"),
                        "secondary_topic": op_data.get("secondary_topic"),
                        "options": op_data.get("options", {}),
                        "invocation": op_data.get("invocation", {}),
                        "guarantees": op_data.get("guarantees", []),
                        "targets": op_data.get("targets", []),
                        "ui": op_data.get("ui", {}),
                        "agents": op_data.get("agents", []),
                        "apps": op_data.get("apps", []),
                        "effects": op_data.get("effects", "none"),
                        "user_defined": False,
                    }
                    if derive_ui_actions is not None:
                        operators[op_id]["ui_actions"] = derive_ui_actions(operator_key=op_id, op=operators[op_id])
                    if infer_flow_hints is not None:
                        operators[op_id]["flow_hints"] = infer_flow_hints(operator_key=op_id, op=operators[op_id])
                    for alias in op_data.get("aliases", []):
                        alias_map[alias.lower()] = op_id

                # Extract commands from grammar
                grammar = data.get("grammar", {})
                slash_pattern = grammar.get("slash_command_pattern", "")
                if slash_pattern:
                    match = re.search(r'\/([\w|]+)', slash_pattern)
                    if match:
                        commands = match.group(1).split('|')
                tool_map = data.get("tool_map") or {}
                bus_topics = data.get("bus_topics") or {}

            # Load user operators
            user_data = self._load_user_ops()
            for op_id, op_data in (user_data.get("operators") or {}).items():
                operators[op_id] = {
                    "id": op_data.get("id", op_id),
                    "name": op_data.get("name", op_id.upper()),
                    "domain": op_data.get("domain", "user"),
                    "category": op_data.get("category", "custom"),
                    "description": op_data.get("description", ""),
                    "aliases": op_data.get("aliases", []),
                    "tool": op_data.get("tool"),
                    "bus_topic": op_data.get("bus_topic"),
                    "bus_kind": op_data.get("bus_kind"),
                    "secondary_topic": op_data.get("secondary_topic"),
                    "options": op_data.get("options", {}),
                    "invocation": op_data.get("invocation", {}),
                    "guarantees": op_data.get("guarantees", []),
                    "targets": op_data.get("targets", []),
                    "ui": op_data.get("ui", {}),
                    "agents": op_data.get("agents", []),
                    "apps": op_data.get("apps", []),
                    "effects": op_data.get("effects", "none"),
                    "user_defined": True,
                }
                if derive_ui_actions is not None:
                    operators[op_id]["ui_actions"] = derive_ui_actions(operator_key=op_id, op=operators[op_id])
                if infer_flow_hints is not None:
                    operators[op_id]["flow_hints"] = infer_flow_hints(operator_key=op_id, op=operators[op_id])
                for alias in op_data.get("aliases", []):
                    alias_map[alias.lower()] = op_id

            self.send_json({
                "operators": operators,
                "commands": commands,
                "alias_map": alias_map,
                "tool_map": tool_map,
                "bus_topics": bus_topics,
                "user_ops_path": str(user_ops_path),
            })

        except Exception as e:
            self.send_json({"error": str(e), "operators": {}, "commands": [], "alias_map": {}})

    def handle_semops_suggestions(self):
        """Serve dynamic suggestions for SemOps editor (tools, actors, topics, UI components, SOTA candidates)."""
        # Tools (local)
        tools_root = self.root / "nucleus" / "tools"
        tool_paths: list[str] = []
        if tools_root.exists():
            for p in sorted(tools_root.glob("*.py")):
                if p.name.startswith("_"):
                    continue
                tool_paths.append(str(p.relative_to(self.root)))
            for p in sorted(tools_root.glob("*.mjs")):
                tool_paths.append(str(p.relative_to(self.root)))
        tool_paths = tool_paths[:500]

        # UI components (dashboard)
        ui_components: list[str] = []
        components_root = self.root / "nucleus" / "dashboard" / "src" / "components"
        if components_root.exists():
            for p in sorted(components_root.glob("*.tsx")):
                ui_components.append(p.stem)
        ui_components = ui_components[:500]

        # Recent bus actors + topics (tail only, bounded)
        actors: dict[str, int] = {}
        topics: dict[str, int] = {}
        bus_paths = None
        events_path = None
        if agent_bus is not None:
            try:
                bus_paths = agent_bus.resolve_bus_paths(os.environ.get("PLURIBUS_BUS_DIR"))
                events_path = Path(bus_paths.events_path)
            except Exception:
                events_path = None
        if events_path and events_path.exists():
            try:
                max_lines = 2500
                with open(events_path, "rb") as f:
                    f.seek(0, os.SEEK_END)
                    end = f.tell()
                    # Read up to last ~4MB for speed
                    start = max(0, end - (4 * 1024 * 1024))
                    f.seek(start)
                    chunk = f.read().decode("utf-8", errors="replace")
                lines = chunk.splitlines()[-max_lines:]
                for line in lines:
                    try:
                        obj = json.loads(line)
                        actor = obj.get("actor")
                        topic = obj.get("topic")
                        if isinstance(actor, str) and actor:
                            actors[actor] = actors.get(actor, 0) + 1
                        if isinstance(topic, str) and topic:
                            topics[topic] = topics.get(topic, 0) + 1
                    except Exception:
                        continue
            except Exception:
                pass

        # SOTA tool candidates from catalog (best-effort heuristic parse)
        sota_candidates: list[str] = []
        sota_path = self.root / "nucleus" / "docs" / "sota_tools_catalog.md"
        if sota_path.exists():
            try:
                text = sota_path.read_text(encoding="utf-8", errors="replace")
                # Table tool names look like: | **vLLM** | 100.0 | P1 | ...
                for m in re.finditer(r"\|\s*\*\*([^*]+)\*\*\s*\|\s*([0-9.]+)\s*\|\s*(P[1-4])\s*\|", text):
                    tool, score, prio = m.group(1).strip(), m.group(2).strip(), m.group(3).strip()
                    if tool:
                        sota_candidates.append(f"{tool} (prio={prio}, score={score})")
                # Deduplicate, keep most important first (P1..P4)
                def prio_key(s: str) -> tuple[int, str]:
                    m2 = re.search(r"prio=(P[1-4])", s)
                    pr = m2.group(1) if m2 else "P9"
                    return (int(pr[1:]) if len(pr) == 2 and pr[1].isdigit() else 9, s)
                sota_candidates = sorted(set(sota_candidates), key=prio_key)[:200]
            except Exception:
                pass

        self.send_json({
            "tool_paths": tool_paths,
            "ui_components": ui_components,
            "recent_actors": sorted(actors.items(), key=lambda kv: (-kv[1], kv[0]))[:50],
            "recent_topics": sorted(topics.items(), key=lambda kv: (-kv[1], kv[0]))[:80],
            "sota_candidates": sota_candidates,
            "bus": {
                "active_dir": getattr(bus_paths, "active_dir", None),
                "primary_dir": getattr(bus_paths, "primary_dir", None),
                "events_path": str(events_path) if events_path else None,
            },
        })

    def handle_semops_define(self, data: dict):
        """Define or update a user-defined semantic operator."""
        op = data.get("operator") if isinstance(data, dict) else None
        if not isinstance(op, dict):
            self.send_json({"success": False, "error": "Missing JSON body: {operator:{...}}"})
            return

        op_key = str(op.get("key") or op.get("op_id") or op.get("name") or op.get("id") or "").strip()
        if not op_key:
            self.send_json({"success": False, "error": "operator.key is required"})
            return
        if not re.match(r"^[A-Za-z0-9_\\-]+$", op_key):
            self.send_json({"success": False, "error": "operator.key must match [A-Za-z0-9_-]+"})
            return
        op_key = op_key.upper()

        # Prevent overriding built-ins.
        semops_path = self.root / "nucleus" / "specs" / "semops.json"
        try:
            if semops_path.exists():
                builtins = json.loads(semops_path.read_text(encoding="utf-8")).get("operators", {})
                if isinstance(builtins, dict) and op_key in builtins:
                    self.send_json({"success": False, "error": f"Cannot override built-in operator: {op_key}"})
                    return
        except Exception:
            pass

        user_data = self._load_user_ops()
        operators = user_data.get("operators") if isinstance(user_data.get("operators"), dict) else {}

        # Normalize payload; keep unknown fields for futureproofing.
        operator_payload = dict(op)
        operator_payload.pop("user_defined", None)
        operator_payload.pop("op_key", None)
        operator_payload["key"] = op_key
        operator_payload.setdefault("id", operator_payload.get("id") or op_key.lower())
        operator_payload.setdefault("name", operator_payload.get("name") or op_key)
        operator_payload.setdefault("domain", operator_payload.get("domain") or "user")
        operator_payload.setdefault("category", operator_payload.get("category") or "custom")
        operator_payload.setdefault("aliases", operator_payload.get("aliases") or [operator_payload["id"], op_key])
        effects_raw = operator_payload.get("effects")
        if not isinstance(effects_raw, str) or not effects_raw.strip():
            operator_payload["effects"] = "none"
        else:
            operator_payload["effects"] = effects_raw.strip().lower()

        operators[op_key] = operator_payload
        user_data["operators"] = operators
        user_data["updated_iso"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
        user_data.setdefault("schema_version", 1)

        self._save_user_ops(user_data)

        actor = str(data.get("actor") or "git-server")
        req_id = str(data.get("req_id") or data.get("data", {}).get("req_id") or "")
        self._emit_bus(
            topic="semops.user_ops.defined",
            kind="artifact",
            level="info",
            actor=actor,
            data={"req_id": req_id or None, "operator_key": op_key, "operator": operator_payload},
        )
        self.send_json({"success": True, "operator_key": op_key, "operator": operator_payload})

    def handle_semops_undefine(self, data: dict):
        """Remove a user-defined semantic operator."""
        op_key = str((data or {}).get("operator_key") or (data or {}).get("key") or (data or {}).get("op_id") or "").strip()
        if not op_key:
            self.send_json({"success": False, "error": "operator_key is required"})
            return
        op_key = op_key.upper()

        user_data = self._load_user_ops()
        operators = user_data.get("operators") if isinstance(user_data.get("operators"), dict) else {}
        if op_key not in operators:
            self.send_json({"success": False, "error": f"Operator not found: {op_key}"})
            return

        removed = operators.pop(op_key)
        user_data["operators"] = operators
        user_data["updated_iso"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
        user_data.setdefault("schema_version", 1)
        self._save_user_ops(user_data)

        actor = str(data.get("actor") or "git-server")
        req_id = str(data.get("req_id") or data.get("data", {}).get("req_id") or "")
        self._emit_bus(
            topic="semops.user_ops.undefined",
            kind="artifact",
            level="info",
            actor=actor,
            data={"req_id": req_id or None, "operator_key": op_key, "operator": removed},
        )
        self.send_json({"success": True, "operator_key": op_key})

    def handle_semops_invoke(self, data: dict):
        """Emit a non-blocking SemOps invocation request to the bus (no execution)."""
        if not isinstance(data, dict):
            self.send_json({"success": False, "error": "Missing JSON body"})
            return
        actor = str(data.get("actor") or "dashboard")
        req_id = str(data.get("req_id") or "").strip()
        operator_key = str(data.get("operator_key") or data.get("key") or data.get("op_key") or "").strip().upper()
        if not operator_key:
            self.send_json({"success": False, "error": "operator_key is required"})
            return
        mode = str(data.get("mode") or "").strip().lower() or "auto"
        payload = data.get("payload") if isinstance(data.get("payload"), dict) else {}
        effects = str(data.get("effects") or "").strip().lower() or None
        wait = bool(data.get("wait"))
        timeout_s = float(data.get("timeout_s") or 1.0)
        timeout_s = max(0.1, min(timeout_s, 20.0))

        if wait and not req_id:
            req_id = str(uuid.uuid4())

        operator_snapshot = None
        try:
            semops_path = self.root / "nucleus" / "specs" / "semops.json"
            if semops_path.exists():
                base = json.loads(semops_path.read_text(encoding="utf-8", errors="replace"))
                ops = base.get("operators") if isinstance(base.get("operators"), dict) else {}
                if isinstance(ops, dict):
                    op_data = ops.get(operator_key) or ops.get(operator_key.upper()) or None
                    if isinstance(op_data, dict):
                        operator_snapshot = {
                            "id": op_data.get("id", operator_key),
                            "name": op_data.get("name", operator_key),
                            "domain": op_data.get("domain", "unknown"),
                            "category": op_data.get("category", "unknown"),
                            "description": op_data.get("description", ""),
                            "aliases": op_data.get("aliases", []),
                            "tool": op_data.get("tool"),
                            "bus_topic": op_data.get("bus_topic"),
                            "bus_kind": op_data.get("bus_kind"),
                            "secondary_topic": op_data.get("secondary_topic"),
                            "targets": op_data.get("targets", []),
                            "ui": op_data.get("ui", {}),
                            "agents": op_data.get("agents", []),
                            "apps": op_data.get("apps", []),
                            "effects": op_data.get("effects", "none"),
                            "user_defined": False,
                        }
        except Exception:
            operator_snapshot = None

        if operator_snapshot is None:
            try:
                user_data = self._load_user_ops()
                ops = user_data.get("operators") if isinstance(user_data.get("operators"), dict) else {}
                op_data = ops.get(operator_key) if isinstance(ops, dict) else None
                if isinstance(op_data, dict):
                    operator_snapshot = {
                        "id": op_data.get("id", operator_key),
                        "name": op_data.get("name", operator_key),
                        "domain": op_data.get("domain", "user"),
                        "category": op_data.get("category", "custom"),
                        "description": op_data.get("description", ""),
                        "aliases": op_data.get("aliases", []),
                        "tool": op_data.get("tool"),
                        "bus_topic": op_data.get("bus_topic"),
                        "bus_kind": op_data.get("bus_kind"),
                        "secondary_topic": op_data.get("secondary_topic"),
                        "targets": op_data.get("targets", []),
                        "ui": op_data.get("ui", {}),
                        "agents": op_data.get("agents", []),
                        "apps": op_data.get("apps", []),
                        "effects": op_data.get("effects", "none"),
                        "user_defined": True,
                    }
            except Exception:
                operator_snapshot = None

        if effects is None and isinstance(operator_snapshot, dict):
            try:
                eff = operator_snapshot.get("effects")
                effects = str(eff).strip().lower() if eff is not None else None
            except Exception:
                effects = None

        if isinstance(operator_snapshot, dict):
            if derive_ui_actions is not None and "ui_actions" not in operator_snapshot:
                operator_snapshot["ui_actions"] = derive_ui_actions(operator_key=operator_key, op=operator_snapshot)
            if infer_flow_hints is not None and "flow_hints" not in operator_snapshot:
                operator_snapshot["flow_hints"] = infer_flow_hints(operator_key=operator_key, op=operator_snapshot)

        self._emit_bus(
            topic="semops.invoke.request",
            kind="request",
            level="info",
            actor=actor,
            data={
                "req_id": req_id or None,
                "operator_key": operator_key,
                "mode": mode,
                "effects": effects,
                "operator": operator_snapshot,
                "payload": payload,
            },
        )
        resp = None
        if wait and req_id:
            resp = self._wait_for_bus_response(response_topic="semops.invoke.response", req_id=req_id, timeout_s=timeout_s)
        self.send_json({"success": True, "operator_key": operator_key, "req_id": req_id or None, "response": resp})

    def handle_supermotd(self):
        """Serve SUPERMOTD data for dashboard footer."""
        try:
            compact = 'compact=true' in self.path.lower()
            supermotd_script = self.tools_dir / "supermotd.py"

            if not supermotd_script.exists():
                self.send_json({
                    "hostname": "pluribus",
                    "uptime_s": 0,
                    "rings": {
                        "ring0": {"status": "sealed", "pqc_algorithm": "ML-DSA-65"},
                        "ring1": {"lineage_id": "genesis", "generation": 0, "transfer_type": "VGT"},
                        "ring2": {"infercells_active": 0},
                        "ring3": {"omega_healthy": True, "omega_cycle": 0, "providers_available": [], "providers_total": 0},
                    },
                    "bus": {"events": 0, "last_event": ""},
                    "insights": ["SUPERMOTD generator not found - using defaults"],
                })
                return

            if compact:
                result = subprocess.run(
                    ["python3", str(supermotd_script), "--compact"],
                    capture_output=True,
                    text=True,
                    timeout=5,
                    cwd=str(self.root)
                )
                self.send_response(200)
                self.send_header('Content-Type', 'text/plain')
                self.send_header('Access-Control-Allow-Origin', '*')
                self.end_headers()
                self.wfile.write(result.stdout.encode())
            else:
                result = subprocess.run(
                    ["python3", str(supermotd_script), "--json"],
                    capture_output=True,
                    text=True,
                    timeout=5,
                    cwd=str(self.root)
                )
                if result.returncode == 0:
                    data = json.loads(result.stdout)
                    self.send_json(data)
                else:
                    self.send_json({"error": result.stderr or "Failed to generate SUPERMOTD"})
        except subprocess.TimeoutExpired:
            self.send_json({"error": "SUPERMOTD generation timed out"})
        except Exception as e:
            self.send_json({"error": str(e)})

    def handle_browser_status(self):
        """Return browser daemon status as JSON for VNCAuthPanel."""
        def autostart_enabled() -> bool:
            v = (os.environ.get("PLURIBUS_BROWSER_DAEMON_AUTOSTART") or "1").strip().lower()
            return v in {"1", "true", "yes", "on"}

        def port_open(host: str, port: int, timeout_s: float = 0.2) -> bool:
            try:
                with closing(socket.socket(socket.AF_INET, socket.SOCK_STREAM)) as s:
                    s.settimeout(timeout_s)
                    return s.connect_ex((host, port)) == 0
            except Exception:
                return False

        def infer_vnc_info() -> dict | None:
            # Canonical defaults for the kroma.live TigerVNC + websockify stack.
            vnc_port = 5901
            websockify_port = 6080

            vnc_ok = port_open("127.0.0.1", vnc_port)
            ws_ok = port_open("127.0.0.1", websockify_port)
            if not (vnc_ok or ws_ok):
                return None

            host = os.environ.get("PLURIBUS_PUBLIC_HOST") or socket.gethostname()
            return {
                "display": ":1",
                "vnc_port": vnc_port if vnc_ok else None,
                "websockify_port": websockify_port if ws_ok else None,
                "connection_string": f"{host}:1",
                "novnc_url": "/vnc/vnc.html?resize=scale&path=vnc/websockify" if ws_ok else None,
            }

        try:
            daemon_script = self.tools_dir / "browser_session_daemon.py"
            if not daemon_script.exists():
                base = {
                    "running": False,
                    "pid": None,
                    "started_at": None,
                    "tabs": {},
                    "vnc_mode": False,
                    "error": "Browser daemon script not found"
                }
                base["vnc_info"] = infer_vnc_info()
                self.send_json(base)
                return

            # Best-effort autostart for autonomy: if the daemon isn't running, start it in the background
            # so subsequent status polls and provider routing can proceed without manual ops.
            if autostart_enabled() and not self._browser_daemon_running():
                started = False
                autostart_pid = None
                autostart_error = None

                # Prefer systemd-managed daemon: avoids orphan/zombie processes and makes liveness observable.
                if shutil.which("systemctl") is not None:
                    try:
                        res = subprocess.run(
                            ["systemctl", "start", "pluribus-browser-session-daemon.service"],
                            capture_output=True,
                            text=True,
                            timeout=8,
                            cwd=str(self.root),
                            env={**os.environ, "PYTHONDONTWRITEBYTECODE": "1"},
                            check=False,
                        )
                        started = res.returncode == 0
                        if not started:
                            autostart_error = (res.stderr or res.stdout or "").strip()[:200] or "systemctl start failed"
                    except Exception as e:
                        autostart_error = str(e)[:200]
                else:
                    # Fallback for non-systemd environments: spawn the daemon directly.
                    try:
                        log_path = self.root / ".pluribus" / "logs" / "browser_daemon_autostart.log"
                        log_path.parent.mkdir(parents=True, exist_ok=True)
                        log = log_path.open("a", encoding="utf-8")
                    except Exception:
                        log = None
                    try:
                        bus_dir = Path(os.environ.get("PLURIBUS_BUS_DIR") or (self.root / ".pluribus" / "bus"))
                        env = dict(os.environ)
                        env.setdefault("PLURIBUS_BROWSER_AUTOLOGIN", "1")
                        env.setdefault("PLURIBUS_BROWSER_VNC", "1")
                        if env.get("PLURIBUS_BROWSER_VNC", "").strip().lower() in {"1", "true", "yes", "on"} and not env.get("DISPLAY"):
                            env["DISPLAY"] = (env.get("PLURIBUS_BROWSER_DISPLAY") or ":1").strip() or ":1"
                        proc = subprocess.Popen(
                            ["python3", str(daemon_script), "--root", str(self.root), "--bus-dir", str(bus_dir), "start"],
                            cwd=str(self.root),
                            env={**env, "PYTHONDONTWRITEBYTECODE": "1"},
                            stdout=log or subprocess.DEVNULL,
                            stderr=log or subprocess.DEVNULL,
                            start_new_session=True,
                        )
                        started = True
                        autostart_pid = int(getattr(proc, "pid", 0) or 0) or None
                    finally:
                        try:
                            if log:
                                log.close()
                        except Exception:
                            pass

                base = {
                    "running": False,
                    "pid": None,
                    "started_at": None,
                    "tabs": {},
                    "vnc_mode": False,
                    "starting": bool(started),
                    "autostart_pid": autostart_pid,
                    "autostart_error": autostart_error,
                }
                base["vnc_info"] = infer_vnc_info()
                self.send_json(base)
                return

            # Fast path: the daemon continuously writes a JSON status file; serve it directly to
            # keep the dashboard polling lightweight (avoid spawning a deep status probe on every refresh).
            state_path = self.root / ".pluribus" / "browser_daemon.json"
            try:
                if state_path.exists():
                    data = json.loads(state_path.read_text(encoding="utf-8") or "{}")
                    if isinstance(data, dict) and data:
                        if data.get("running") and not self._browser_daemon_running():
                            raise RuntimeError("stale browser daemon state")
                        # Never expose secrets from daemon state.
                        if "user_auths" in data:
                            data.pop("user_auths", None)
                        # Some older daemon versions stored tokens under user_auths; keep only meta if present.
                        if "user_auths_meta" in data and not isinstance(data.get("user_auths_meta"), dict):
                            data.pop("user_auths_meta", None)
                        if not data.get("vnc_info"):
                            data["vnc_info"] = infer_vnc_info()
                        self.send_json(data)
                        return
            except Exception:
                # Fall back to the slower `status` command below.
                pass

            env = dict(os.environ)
            env.setdefault("PLURIBUS_BROWSER_AUTOLOGIN", "1")
            env.setdefault("PLURIBUS_BROWSER_VNC", "1")
            if env.get("PLURIBUS_BROWSER_VNC", "").strip().lower() in {"1", "true", "yes", "on"} and not env.get("DISPLAY"):
                env["DISPLAY"] = (env.get("PLURIBUS_BROWSER_DISPLAY") or ":1").strip() or ":1"
            env["PYTHONDONTWRITEBYTECODE"] = "1"

            result = subprocess.run(
                ["python3", str(daemon_script), "--root", str(self.root),
                 "--bus-dir", str(self.root / ".pluribus" / "bus"), "status", "--json"],
                capture_output=True,
                text=True,
                timeout=10,
                cwd=str(self.root),
                env=env,
            )

            if result.returncode == 0 and result.stdout.strip():
                data = json.loads(result.stdout.strip())
                if isinstance(data, dict) and "user_auths" in data:
                    data.pop("user_auths", None)
                if not data.get("vnc_info"):
                    data["vnc_info"] = infer_vnc_info()
                self.send_json(data)
            else:
                base = {
                    "running": False,
                    "pid": None,
                    "started_at": None,
                    "tabs": {},
                    "vnc_mode": False,
                    "error": result.stderr or "Failed to get daemon status"
                }
                base["vnc_info"] = infer_vnc_info()
                self.send_json(base)
        except subprocess.TimeoutExpired:
            base = {"running": False, "error": "Status check timed out"}
            base["vnc_info"] = infer_vnc_info()
            self.send_json(base)
        except json.JSONDecodeError as e:
            base = {"running": False, "error": f"Invalid JSON response: {e}"}
            base["vnc_info"] = infer_vnc_info()
            self.send_json(base)
        except Exception as e:
            base = {"running": False, "error": str(e)}
            base["vnc_info"] = infer_vnc_info()
            self.send_json(base)

    def handle_gemini_clean_status(self):
        """Run `gemini_clean.zsh --mode status` and return its output (no secrets)."""
        try:
            script = self.root / "gemini_clean.zsh"
            if not script.exists():
                self.send_error(404, "gemini_clean.zsh not found")
                return
            if shutil.which("zsh") is None:
                self.send_json({"ok": False, "error": "zsh not installed"})
                return

            cmd = ["zsh", "-lc", f"source {shlex.quote(str(script))} --mode status --no-test --no-list"]
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=15,
                cwd=str(self.root),
                env={**os.environ, "PYTHONDONTWRITEBYTECODE": "1"},
            )
            text_out = (result.stdout or "").strip()
            text_err = (result.stderr or "").strip()
            self.send_response(200)
            self.send_header('Content-Type', 'application/json')
            self.send_header('Access-Control-Allow-Origin', '*')
            self.end_headers()
            self.wfile.write(json.dumps({
                "ok": result.returncode == 0,
                "exit_code": result.returncode,
                "stdout": text_out,
                "stderr": text_err,
            }, ensure_ascii=False).encode("utf-8"))
        except subprocess.TimeoutExpired:
            self.send_json({"ok": False, "error": "gemini_clean status timed out"})
        except Exception as e:
            self.send_json({"ok": False, "error": str(e)})

    def _browser_daemon_running(self) -> bool:
        state_path = self.root / ".pluribus" / "browser_daemon.json"
        try:
            if not state_path.exists():
                return False
            state = json.loads(state_path.read_text(encoding="utf-8") or "{}")
            if not isinstance(state, dict):
                return False
            if not state.get("running"):
                return False
            pid = int(state.get("pid") or 0)
            if pid <= 0:
                return False
            proc_dir = Path(f"/proc/{pid}")
            if not proc_dir.exists():
                return False
            try:
                status_raw = (proc_dir / "status").read_text(encoding="utf-8", errors="replace")
                for line in status_raw.splitlines():
                    if not line.startswith("State:"):
                        continue
                    code = line.split(":", 1)[1].strip().split(None, 1)[0].strip()
                    if code == "Z":
                        return False
                    break
            except Exception:
                pass
            try:
                cmd = (proc_dir / "cmdline").read_bytes().replace(b"\x00", b" ").decode("utf-8", errors="replace").strip()
            except Exception:
                cmd = ""
            return "browser_session_daemon.py" in cmd
        except Exception:
            return False

    def handle_browser_vnc_navigate_login(self, data: dict):
        provider = str((data or {}).get("provider") or "").strip()
        actor = str((data or {}).get("actor") or "dashboard")
        timeout_s = float((data or {}).get("timeout_s") or 8.0)
        wait = bool((data or {}).get("wait", True))

        if not provider:
            self.send_json({"success": False, "error": "provider is required"})
            return
        if not self._browser_daemon_running():
            self.send_json({"success": False, "error": "Browser daemon not running", "hint": "Start: python3 nucleus/tools/browser_session_daemon.py start"})
            return

        req_id = self._append_bus_request(topic="browser.vnc.navigate_login", actor=actor, data={"provider": provider, "req_id": (data or {}).get("req_id")})
        if not wait:
            self.send_json({"success": True, "req_id": req_id, "queued": True})
            return

        resp = self._wait_for_bus_response(response_topic="browser.vnc.navigate_login.response", req_id=req_id, timeout_s=timeout_s)
        if resp is None:
            self.send_json({"success": False, "req_id": req_id, "error": "timeout waiting for browser.vnc.navigate_login.response"})
            return
        self.send_json(resp)

    def handle_browser_vnc_check_login(self, data: dict):
        provider = str((data or {}).get("provider") or "").strip()
        actor = str((data or {}).get("actor") or "dashboard")
        timeout_s = float((data or {}).get("timeout_s") or 8.0)
        wait = bool((data or {}).get("wait", True))

        if not provider:
            self.send_json({"success": False, "error": "provider is required"})
            return
        if not self._browser_daemon_running():
            self.send_json({"success": False, "error": "Browser daemon not running"})
            return

        req_id = self._append_bus_request(topic="browser.vnc.check_login", actor=actor, data={"provider": provider, "req_id": (data or {}).get("req_id")})
        if not wait:
            self.send_json({"success": True, "req_id": req_id, "queued": True})
            return

        resp = self._wait_for_bus_response(response_topic="browser.vnc.check_login.response", req_id=req_id, timeout_s=timeout_s)
        if resp is None:
            self.send_json({"success": False, "req_id": req_id, "error": "timeout waiting for browser.vnc.check_login.response"})
            return
        self.send_json(resp)

    def handle_browser_vnc_focus_tab(self, data: dict):
        provider = str((data or {}).get("provider") or "").strip()
        actor = str((data or {}).get("actor") or "dashboard")
        timeout_s = float((data or {}).get("timeout_s") or 8.0)
        wait = bool((data or {}).get("wait", True))

        if not provider:
            self.send_json({"success": False, "error": "provider is required"})
            return
        if not self._browser_daemon_running():
            self.send_json({"success": False, "error": "Browser daemon not running"})
            return

        req_id = self._append_bus_request(topic="browser.vnc.focus_tab", actor=actor, data={"provider": provider, "req_id": (data or {}).get("req_id")})
        if not wait:
            self.send_json({"success": True, "req_id": req_id, "queued": True})
            return

        resp = self._wait_for_bus_response(response_topic="browser.vnc.focus_tab.response", req_id=req_id, timeout_s=timeout_s)
        if resp is None:
            self.send_json({"success": False, "req_id": req_id, "error": "timeout waiting for browser.vnc.focus_tab.response"})
            return
        self.send_json(resp)

    def handle_browser_vnc_enable(self, data: dict):
        actor = str((data or {}).get("actor") or "dashboard")
        timeout_s = float((data or {}).get("timeout_s") or 8.0)
        wait = bool((data or {}).get("wait", True))

        if not self._browser_daemon_running():
            self.send_json({"success": False, "error": "Browser daemon not running"})
            return

        req_id = self._append_bus_request(
            topic="browser.vnc.enable",
            actor=actor,
            data={"req_id": (data or {}).get("req_id")},
        )
        if not wait:
            self.send_json({"success": True, "req_id": req_id, "queued": True})
            return

        resp = self._wait_for_bus_response(response_topic="browser.vnc.enable.response", req_id=req_id, timeout_s=timeout_s)
        if resp is None:
            self.send_json({"success": False, "req_id": req_id, "error": "timeout waiting for browser.vnc.enable.response"})
            return
        self.send_json(resp)

    def handle_browser_vnc_disable(self, data: dict):
        actor = str((data or {}).get("actor") or "dashboard")
        timeout_s = float((data or {}).get("timeout_s") or 8.0)
        wait = bool((data or {}).get("wait", True))

        if not self._browser_daemon_running():
            self.send_json({"success": False, "error": "Browser daemon not running"})
            return

        req_id = self._append_bus_request(
            topic="browser.vnc.disable",
            actor=actor,
            data={"req_id": (data or {}).get("req_id")},
        )
        if not wait:
            self.send_json({"success": True, "req_id": req_id, "queued": True})
            return

        resp = self._wait_for_bus_response(response_topic="browser.vnc.disable.response", req_id=req_id, timeout_s=timeout_s)
        if resp is None:
            self.send_json({"success": False, "req_id": req_id, "error": "timeout waiting for browser.vnc.disable.response"})
            return
        self.send_json(resp)

    def handle_browser_vnc_status(self, data: dict):
        actor = str((data or {}).get("actor") or "dashboard")
        timeout_s = float((data or {}).get("timeout_s") or 8.0)
        wait = bool((data or {}).get("wait", True))

        if not self._browser_daemon_running():
            self.send_json({"success": False, "error": "Browser daemon not running"})
            return

        req_id = self._append_bus_request(
            topic="browser.vnc.status",
            actor=actor,
            data={"req_id": (data or {}).get("req_id")},
        )
        if not wait:
            self.send_json({"success": True, "req_id": req_id, "queued": True})
            return

        resp = self._wait_for_bus_response(response_topic="browser.vnc.status.response", req_id=req_id, timeout_s=timeout_s)
        if resp is None:
            self.send_json({"success": False, "req_id": req_id, "error": "timeout waiting for browser.vnc.status.response"})
            return
        self.send_json(resp)

    def handle_browser_bootstrap(self, data: dict):
        """
        Bootstrap web-provider auth state for autonomy:
        - best-effort autostart browser_session_daemon if not running
        - best-effort autostart noVNC/VNC bridge (TigerVNC + websockify) so OTP entry is always available
        - enqueue non-blocking check_login requests for web providers
        - refresh vps_session control plane (best-effort)
        """
        if not isinstance(data, dict):
            data = {}
        actor = str(data.get("actor") or "dashboard")
        providers = data.get("providers")
        if not isinstance(providers, list) or not providers:
            providers = ["chatgpt-web", "claude-web", "gemini-web"]
        providers = [str(p).strip() for p in providers if str(p).strip()]

        daemon_script = self.tools_dir / "browser_session_daemon.py"
        if not daemon_script.exists():
            self.send_json({"success": False, "error": "Browser daemon script not found"})
            return

        def port_open(host: str, port: int, timeout_s: float = 0.2) -> bool:
            try:
                with closing(socket.socket(socket.AF_INET, socket.SOCK_STREAM)) as s:
                    s.settimeout(timeout_s)
                    return s.connect_ex((host, port)) == 0
            except Exception:
                return False

        # Best-effort VNC/noVNC autostart so the human can always enter OTP codes.
        vnc_autostart_enabled = (os.environ.get("PLURIBUS_NOVNC_AUTOSTART") or "1").strip().lower() in {"1", "true", "yes", "on"}
        vnc_started = False
        vnc_autostart_pid = None
        try:
            if vnc_autostart_enabled:
                # Canonical defaults: VNC :1 (5901) + websockify (6080).
                vnc_ok = port_open("127.0.0.1", 5901)
                ws_ok = port_open("127.0.0.1", 6080)
                if not (vnc_ok and ws_ok):
                    novnc_script = self.tools_dir / "novnc_start.sh"
                    if novnc_script.exists() and shutil.which("bash") is not None:
                        try:
                            log_path = self.root / ".pluribus" / "logs" / "novnc_autostart.log"
                            log_path.parent.mkdir(parents=True, exist_ok=True)
                            log = log_path.open("a", encoding="utf-8")
                        except Exception:
                            log = None
                        try:
                            proc = subprocess.Popen(
                                ["bash", str(novnc_script), "start"],
                                cwd=str(self.root),
                                env={**os.environ, "PYTHONDONTWRITEBYTECODE": "1"},
                                stdout=log or subprocess.DEVNULL,
                                stderr=log or subprocess.DEVNULL,
                                start_new_session=True,
                            )
                            vnc_started = True
                            vnc_autostart_pid = int(getattr(proc, "pid", 0) or 0) or None
                        finally:
                            try:
                                if log:
                                    log.close()
                            except Exception:
                                pass
        except Exception:
            # Non-blocking; VNC can still be started manually.
            vnc_started = False

        started = False
        autostart_pid = None
        autostart_error = None
        if not self._browser_daemon_running():
            if shutil.which("systemctl") is not None:
                try:
                    res = subprocess.run(
                        ["systemctl", "start", "pluribus-browser-session-daemon.service"],
                        capture_output=True,
                        text=True,
                        timeout=8,
                        cwd=str(self.root),
                        env={**os.environ, "PYTHONDONTWRITEBYTECODE": "1"},
                        check=False,
                    )
                    started = res.returncode == 0
                    if not started:
                        autostart_error = (res.stderr or res.stdout or "").strip()[:200] or "systemctl start failed"
                except Exception as e:
                    autostart_error = str(e)[:200]
            else:
                # Fallback for non-systemd environments: spawn the daemon directly.
                try:
                    log_path = self.root / ".pluribus" / "logs" / "browser_daemon_autostart.log"
                    log_path.parent.mkdir(parents=True, exist_ok=True)
                    log = log_path.open("a", encoding="utf-8")
                except Exception:
                    log = None
                try:
                    bus_dir = Path(os.environ.get("PLURIBUS_BUS_DIR") or (self.root / ".pluribus" / "bus"))
                    env = dict(os.environ)
                    env.setdefault("PLURIBUS_BROWSER_AUTOLOGIN", "1")
                    env.setdefault("PLURIBUS_BROWSER_VNC", "1")
                    if env.get("PLURIBUS_BROWSER_VNC", "").strip().lower() in {"1", "true", "yes", "on"} and not env.get("DISPLAY"):
                        env["DISPLAY"] = (env.get("PLURIBUS_BROWSER_DISPLAY") or ":1").strip() or ":1"
                    proc = subprocess.Popen(
                        ["python3", str(daemon_script), "--root", str(self.root), "--bus-dir", str(bus_dir), "start"],
                        cwd=str(self.root),
                        env={**env, "PYTHONDONTWRITEBYTECODE": "1"},
                        stdout=log or subprocess.DEVNULL,
                        stderr=log or subprocess.DEVNULL,
                        start_new_session=True,
                    )
                    started = True
                    autostart_pid = int(getattr(proc, "pid", 0) or 0) or None
                finally:
                    try:
                        if log:
                            log.close()
                    except Exception:
                        pass

        queued: list[dict] = []
        for p in providers:
            try:
                req_id = self._append_bus_request(topic="browser.vnc.check_login", actor=actor, data={"provider": p})
                queued.append({"provider": p, "req_id": req_id})
            except Exception:
                continue

        refreshed = False
        try:
            vps_session_script = self.tools_dir / "vps_session.py"
            if vps_session_script.exists():
                subprocess.run(
                    ["python3", str(vps_session_script), "refresh", "--root", str(self.root), "--provider", "web"],
                    capture_output=True,
                    text=True,
                    timeout=8,
                    cwd=str(self.root),
                    env={**os.environ, "PYTHONDONTWRITEBYTECODE": "1"},
                    check=False,
                )
                refreshed = True
        except Exception:
            refreshed = False

        self._emit_bus(
            topic="dashboard.browser.bootstrap",
            kind="request",
            level="info",
            actor=actor,
            data={
                "started": started,
                "autostart_pid": autostart_pid,
                "autostart_error": autostart_error,
                "vnc_autostart_enabled": vnc_autostart_enabled,
                "vnc_started": vnc_started,
                "vnc_autostart_pid": vnc_autostart_pid,
                "queued": queued,
                "refreshed": refreshed,
            },
        )

        self.send_json({
            "success": True,
            "started": started,
            "autostart_pid": autostart_pid,
            "autostart_error": autostart_error,
            "vnc_autostart_enabled": vnc_autostart_enabled,
            "vnc_started": vnc_started,
            "vnc_autostart_pid": vnc_autostart_pid,
            "queued": queued,
            "refreshed": refreshed,
        })

    def _iso_now(self) -> str:
        """Return current time in ISO format."""
        from datetime import datetime, timezone
        return datetime.now(timezone.utc).strftime('%Y-%m-%dT%H:%M:%SZ')

    # --- POST Handlers for Git Operations ---

    def handle_git_hgt(self, data: dict):
        """Execute HGT (Horizontal Gene Transfer) splice operation."""
        source_sha = data.get('source_sha') or data.get('sha')
        if not source_sha:
            self.send_json({"success": False, "error": "Missing source_sha parameter"})
            return

        # Run iso_git.mjs evo hgt <sha>
        try:
            result = subprocess.run(
                ["node", str(self.iso_git_tool), "evo", str(self.root), "hgt", source_sha],
                capture_output=True,
                text=True,
                timeout=60,
                cwd=str(self.root),
                env={**os.environ, "PLURIBUS_BUS_DIR": str(self.root / ".pluribus" / "bus")}
            )

            if result.returncode == 0:
                self.send_json({
                    "success": True,
                    "message": "HGT splice applied successfully",
                    "source_sha": source_sha,
                    "output": result.stdout,
                })
            else:
                # Parse guard ladder results from stderr
                error_msg = result.stderr or result.stdout or "HGT failed"
                self.send_json({
                    "success": False,
                    "error": error_msg,
                    "source_sha": source_sha,
                })
        except subprocess.TimeoutExpired:
            self.send_json({"success": False, "error": "HGT operation timed out (60s)"})
        except Exception as e:
            self.send_json({"success": False, "error": str(e)})

    def handle_git_push(self, data: dict):
        """Execute git push (guarded boundary operation)."""
        remote = data.get('remote', 'origin')
        branch = data.get('branch')
        force = data.get('force', False)
        set_upstream = data.get('set_upstream', False)

        # Build iso_git.mjs push command
        args = ["node", str(self.iso_git_tool), "push", str(self.root), remote]
        if branch:
            args.append(branch)
        if force:
            args.append('--force')
        if set_upstream:
            args.append('--set-upstream')

        try:
            result = subprocess.run(
                args,
                capture_output=True,
                text=True,
                timeout=120,
                cwd=str(self.root),
                env={**os.environ, "PLURIBUS_BUS_DIR": str(self.root / ".pluribus" / "bus")}
            )

            if result.returncode == 0:
                self.send_json({
                    "success": True,
                    "message": f"Pushed to {remote}" + (f"/{branch}" if branch else ""),
                    "output": result.stdout,
                })
            else:
                self.send_json({
                    "success": False,
                    "error": result.stderr or result.stdout or "Push failed",
                    "remote": remote,
                })
        except subprocess.TimeoutExpired:
            self.send_json({"success": False, "error": "Push operation timed out (120s)"})
        except Exception as e:
            self.send_json({"success": False, "error": str(e)})

    def handle_git_fetch(self, data: dict):
        """Execute git fetch (guarded boundary operation)."""
        remote = data.get('remote', 'origin')
        all_remotes = data.get('all', False)
        prune = data.get('prune', False)

        args = ["node", str(self.iso_git_tool), "fetch", str(self.root), remote]
        if all_remotes:
            args.append('--all')
        if prune:
            args.append('--prune')

        try:
            result = subprocess.run(
                args,
                capture_output=True,
                text=True,
                timeout=120,
                cwd=str(self.root),
                env={**os.environ, "PLURIBUS_BUS_DIR": str(self.root / ".pluribus" / "bus")}
            )

            if result.returncode == 0:
                self.send_json({
                    "success": True,
                    "message": f"Fetched from {remote}",
                    "output": result.stdout,
                })
            else:
                self.send_json({
                    "success": False,
                    "error": result.stderr or result.stdout or "Fetch failed",
                    "remote": remote,
                })
        except subprocess.TimeoutExpired:
            self.send_json({"success": False, "error": "Fetch operation timed out (120s)"})
        except Exception as e:
            self.send_json({"success": False, "error": str(e)})

    def handle_git_commit(self, data: dict):
        """Execute git commit via iso_git.mjs."""
        message = data.get('message', 'update')
        paths = data.get('paths')

        if paths and isinstance(paths, list):
            # Commit specific paths
            args = ["node", str(self.iso_git_tool), "commit-paths", str(self.root), message] + paths
        else:
            # Commit all
            args = ["node", str(self.iso_git_tool), "commit", str(self.root), message]

        try:
            result = subprocess.run(
                args,
                capture_output=True,
                text=True,
                timeout=60,
                cwd=str(self.root),
                env={**os.environ, "PLURIBUS_BUS_DIR": str(self.root / ".pluribus" / "bus")}
            )

            if result.returncode == 0:
                # Parse SHA from output like "[abc1234] message"
                sha = None
                for line in result.stdout.split('\n'):
                    if line.startswith('[') and ']' in line:
                        sha = line[1:line.index(']')]
                        break

                self.send_json({
                    "success": True,
                    "message": "Commit created",
                    "sha": sha,
                    "output": result.stdout,
                })
            else:
                self.send_json({
                    "success": False,
                    "error": result.stderr or result.stdout or "Commit failed",
                })
        except subprocess.TimeoutExpired:
            self.send_json({"success": False, "error": "Commit operation timed out"})
        except Exception as e:
            self.send_json({"success": False, "error": str(e)})


def run_server(root: Path, port: int):
    # Tools live alongside this server script; keep independent of the served root.
    tools_dir = Path(__file__).resolve().parent
    
    # Factory for handler
    def handler_factory(*args, **kwargs):
        return GitFSHandler(*args, root=root, tools_dir=tools_dir, **kwargs)

    server = HTTPServer(('0.0.0.0', port), handler_factory)
    print(f"Git/FS Server running on http://0.0.0.0:{port}")
    server.serve_forever()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", required=True)
    parser.add_argument("--port", type=int, default=DEFAULT_PORT)
    args = parser.parse_args()
    
    run_server(Path(args.root).resolve(), args.port)
