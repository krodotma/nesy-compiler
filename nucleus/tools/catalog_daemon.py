#!/usr/bin/env python3
"""Catalog Daemon.

Continuously publishes SOTA and Services data to the event bus so the dashboard
can render them immediately. Also handles catalog-related actions (e.g., updates, instantiation).

Role: Backend (Codex)
"""
from __future__ import annotations

import argparse
import getpass
import json
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

sys.dont_write_bytecode = True

# Import tools (assuming they are in the same dir)
try:
    from sota import SotaManager, SotaItem
    from sota import build_sota_distill_request
    from sota_distillations import materialize_sota_distillation, snippet_from_markdown
    from sota_kg import append_sota_kg_node, build_sota_kg_node
    from service_registry import ServiceRegistry, ServiceDef, ServiceInstance
    import agent_bus
except ImportError:
    # Fallback pathing
    # This block is for when running this script directly without proper package installation.
    # It assumes sota.py and service_registry.py are siblings.
    _parent_dir = str(Path(__file__).resolve().parent)
    if _parent_dir not in sys.path:
        sys.path.insert(0, _parent_dir)
    from sota import SotaManager, SotaItem
    from sota import build_sota_distill_request
    from sota_distillations import materialize_sota_distillation, snippet_from_markdown
    from sota_kg import append_sota_kg_node, build_sota_kg_node
    from service_registry import ServiceRegistry, ServiceDef, ServiceInstance
    import agent_bus


def now_iso_utc() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def default_actor() -> str:
    return os.environ.get("PLURIBUS_ACTOR") or os.environ.get("USER") or getpass.getuser()


def find_rhizome_root(start: Path) -> Path | None:
    cur = start.resolve()
    for cand in [cur, *cur.parents]:
        if (cand / ".pluribus" / "rhizome.json").exists():
            return cand
    return None


def emit_bus(bus_dir: Path, topic: str, kind: str, level: str, data: dict) -> None:
    # Use agent_bus tool directly via subprocess.
    # For robustness, we use the subprocess approach matching other tools.
    tool = Path(__file__).with_name("agent_bus.py")
    if not tool.exists():
        return
    subprocess.run(
        [
            sys.executable,
            str(tool),
            "--bus-dir",
            str(bus_dir),
            "pub",
            "--topic", topic,
            "--kind", kind,
            "--level", level,
            "--actor", "catalog-daemon",
            "--data", json.dumps(data, ensure_ascii=False),
        ],
        check=False,
        stdout=subprocess.DEVNULL,
        # stderr=subprocess.DEVNULL, # Keep suppressed for daemon behavior
        env={**os.environ, "PYTHONDONTWRITEBYTECODE": "1"},
    )


def _exit_code(obj: dict[str, Any], *, default: int = 1) -> int:
    v = obj.get("exit_code")
    try:
        return int(v)
    except Exception:
        return default


class CatalogDaemon:
    def __init__(self, root: Path, bus_dir: Path, interval_s: float):
        self.root = root
        self.bus_dir = bus_dir
        self.interval_s = interval_s
        self.sota_mgr = SotaManager(root)
        self.svc_reg = ServiceRegistry(root)
        self.events_path = bus_dir / "events.ndjson"
        self.running = True
        self._responses_offset = 0
        self._sota_pending: dict[str, str] = {}  # req_id -> sota_item_id

    def publish_snapshot(self) -> None:
        """Publish full snapshot of SOTA and Services."""
        try:
            # Latest distillation artifacts (best-effort join so UI can show results after reload)
            latest_distill_any: dict[str, dict] = {}
            latest_distill_success: dict[str, dict] = {}
            artifacts_path = self.root / ".pluribus" / "index" / "artifacts.ndjson"
            if artifacts_path.exists():
                try:
                    for line in artifacts_path.read_text(encoding="utf-8", errors="replace").splitlines():
                        line = line.strip()
                        if not line:
                            continue
                        try:
                            obj = json.loads(line)
                        except Exception:
                            continue
                        if obj.get("type") != "sota_distillation":
                            continue
                        sota_item_id = obj.get("sota_item_id")
                        if not isinstance(sota_item_id, str) or not sota_item_id:
                            continue
                        ts = float(obj.get("ts") or 0.0)
                        prev_any = latest_distill_any.get(sota_item_id)
                        if not prev_any or ts > float(prev_any.get("ts") or 0.0):
                            latest_distill_any[sota_item_id] = obj
                        # Only surface successful distillations in the dashboard snapshot to avoid
                        # "cached stderr" spam from unauthenticated/over-quota providers.
                        if _exit_code(obj) == 0:
                            prev_ok = latest_distill_success.get(sota_item_id)
                            if not prev_ok or ts > float(prev_ok.get("ts") or 0.0):
                                latest_distill_success[sota_item_id] = obj
                except Exception:
                    latest_distill_any = {}
                    latest_distill_success = {}

            # Load SOTA
            items = self.sota_mgr.list_items()
            # Attach the latest distillation pointers (path/snippet) when available.
            merged: list[dict] = []
            for item in items:
                d = item.__dict__.copy()
                last_any = latest_distill_any.get(item.id)
                if isinstance(last_any, dict):
                    d["distill_status"] = "completed" if _exit_code(last_any) == 0 else "failed"
                    d["distill_last_iso"] = last_any.get("iso") or None
                    d["distill_req_id"] = last_any.get("req_id") or None

                last_ok = latest_distill_success.get(item.id)
                if isinstance(last_ok, dict):
                    d["distill_artifact_path"] = last_ok.get("path") or None
                    if d.get("distill_artifact_path"):
                        try:
                            d["distill_snippet"] = snippet_from_markdown(Path(str(d["distill_artifact_path"])), max_chars=360)
                        except Exception:
                            d["distill_snippet"] = None
                merged.append(d)
            emit_bus(
                self.bus_dir,
                "sota.list",
                "artifact", # Changed from "metric" to "artifact"
                "info",
                {"items": merged, "count": len(items)}
            )

            # Load Services
            self.svc_reg.load()
            self.svc_reg.refresh_instances()
            services = [s.__dict__ for s in self.svc_reg.list_services()]
            instances = [i.__dict__ for i in self.svc_reg.list_instances()]
            
            emit_bus(
                self.bus_dir,
                "services.list",
                "artifact", # Changed from "metric" to "artifact"
                "info",
                {"services": services, "instances": instances}
            )
            
        except Exception as e:
            sys.stderr.write(f"Snapshot failed: {e}\n")

    def _responses_path(self) -> Path:
        return self.root / ".pluribus" / "index" / "responses.ndjson"

    def _requests_path(self) -> Path:
        return self.root / ".pluribus" / "index" / "requests.ndjson"

    def _lookup_sota_item_for_req(self, req_id: str) -> str | None:
        # Fast path: remember requests queued by the daemon.
        if req_id in self._sota_pending:
            return self._sota_pending.get(req_id)

        # Slow path: scan the requests index for a matching SOTA distill request.
        rp = self._requests_path()
        if not rp.exists():
            return None
        try:
            for line in rp.read_text(encoding="utf-8", errors="replace").splitlines():
                line = line.strip()
                if not line:
                    continue
                try:
                    obj = json.loads(line)
                except Exception:
                    continue
                if obj.get("req_id") != req_id:
                    continue
                inputs = obj.get("inputs") if isinstance(obj.get("inputs"), dict) else {}
                sota_item_id = inputs.get("sota_item_id") if isinstance(inputs, dict) else None
                if isinstance(sota_item_id, str) and sota_item_id:
                    return sota_item_id
        except Exception:
            return None
        return None

    def check_new_responses(self) -> None:
        path = self._responses_path()
        if not path.exists():
            return
        try:
            data = path.read_bytes()
        except Exception:
            return
        if self._responses_offset >= len(data):
            return
        chunk = data[self._responses_offset :]
        self._responses_offset = len(data)
        for raw_line in chunk.splitlines():
            line = raw_line.decode("utf-8", errors="replace").strip()
            if not line:
                continue
            try:
                resp = json.loads(line)
            except Exception:
                continue
            if resp.get("kind") != "strp_response":
                continue
            req_id = resp.get("req_id")
            if not isinstance(req_id, str) or not req_id:
                continue

            sota_item_id = self._lookup_sota_item_for_req(req_id)
            if not sota_item_id:
                continue

            artifact_path = self.root / ".pluribus" / "index" / "distillations" / "sota" / sota_item_id / f"{req_id}.md"
            if artifact_path.exists():
                # Idempotency: avoid duplicating artifacts on daemon restart.
                continue

            try:
                record = materialize_sota_distillation(root=self.root, sota_item_id=sota_item_id, response=resp)
                status = "completed" if _exit_code(resp) == 0 else "failed"
                status_payload: dict[str, Any] = {"item_id": sota_item_id, "status": status, "req_id": req_id}
                if status == "completed":
                    status_payload["path"] = record.get("path")
                else:
                    err = resp.get("stderr")
                    if isinstance(err, str) and err.strip():
                        status_payload["error"] = err.strip()[:500]
                emit_bus(self.bus_dir, "sota.distill.status", "metric", "info", status_payload)
                # Only publish artifact/snippet for successful distillations; failed runs still
                # materialize append-only artifacts on disk for debugging/provenance.
                if status == "completed":
                    snippet = snippet_from_markdown(Path(record.get("path") or ""), max_chars=360)
                    emit_bus(self.bus_dir, "sota.distill.artifact", "artifact", "info", {"item_id": sota_item_id, "req_id": req_id, "path": record.get("path"), "snippet": snippet})
                # Stop tracking this req_id.
                self._sota_pending.pop(req_id, None)
            except Exception as e:
                emit_bus(self.bus_dir, "sota.distill.status", "metric", "error", {"item_id": sota_item_id, "status": "failed", "req_id": req_id, "error": str(e)})

    def handle_action(self, action: dict) -> None:
        """Handle catalog actions."""
        kind = action.get("kind")
        if kind == "instantiate_service":
            svc_id = action.get("service_id")
            if svc_id:
                emit_bus(self.bus_dir, "catalog.log", "log", "info", {"msg": f"Starting service {svc_id}..."})
                inst = self.svc_reg.start_service(svc_id)
                if inst:
                    emit_bus(self.bus_dir, "catalog.log", "log", "info", {"msg": f"Started {svc_id} ({inst.instance_id})"})
                    self.publish_snapshot()  # Immediate update
                else:
                    emit_bus(self.bus_dir, "catalog.log", "log", "error", {"msg": f"Failed to start {svc_id}"})

        elif kind == "stop_service":
            inst_id = action.get("instance_id")
            if inst_id:
                emit_bus(self.bus_dir, "catalog.log", "log", "info", {"msg": f"Stopping instance {inst_id}..."})
                if self.svc_reg.stop_service(inst_id):
                    self.publish_snapshot()
                else:
                    emit_bus(self.bus_dir, "catalog.log", "log", "error", {"msg": f"Failed to stop {inst_id}"})

        elif kind == "sota.distill":
            item_id = action.get("item_id")
            provider_hint = action.get("provider_hint") or "auto"
            if isinstance(item_id, str) and item_id:
                try:
                    items = self.sota_mgr.list_items()
                    item = next((i for i in items if i.id == item_id), None)
                    if not item:
                        emit_bus(self.bus_dir, "catalog.log", "log", "error", {"msg": f"SOTA item not found: {item_id}"})
                        return

                    # Write STRp request for worker consumption.
                    idx = self.root / ".pluribus" / "index"
                    idx.mkdir(parents=True, exist_ok=True)
                    requests_path = idx / "requests.ndjson"
                    payload = build_sota_distill_request(
                        root=self.root,
                        actor=default_actor(),
                        provider=str(provider_hint),
                        item={"id": item.id, "url": item.url, "title": item.title},
                    )
                    with requests_path.open("a", encoding="utf-8") as f:
                        f.write(json.dumps(payload, ensure_ascii=False, separators=(",", ":")) + "\n")

                    self._sota_pending[str(payload.get("req_id") or "")] = item_id
                    emit_bus(self.bus_dir, "sota.distill.status", "metric", "info", {"item_id": item_id, "status": "queued", "req_id": payload.get("req_id")})
                    emit_bus(self.bus_dir, "strp.request.distill", "request", "info", payload)
                except Exception as e:
                    emit_bus(self.bus_dir, "sota.distill.status", "metric", "error", {"item_id": item_id, "status": "failed", "error": str(e)})
            else:
                emit_bus(self.bus_dir, "catalog.log", "log", "error", {"msg": "sota.distill missing item_id"})

        elif kind == "sota.kg.add":
            item_id = action.get("item_id")
            ref = action.get("ref")
            if isinstance(item_id, str) and item_id and isinstance(ref, str) and ref:
                try:
                    items = self.sota_mgr.list_items()
                    item = next((i for i in items if i.id == item_id), None)
                    if not item:
                        emit_bus(self.bus_dir, "catalog.log", "log", "error", {"msg": f"SOTA item not found: {item_id}"})
                        return
                    idx = self.root / ".pluribus" / "index"
                    idx.mkdir(parents=True, exist_ok=True)
                    node = build_sota_kg_node(
                        sota_item=item.__dict__,
                        ref=ref,
                        actor=default_actor(),
                        context="catalog.action:sota.kg.add",
                        extra_tags=["from:dashboard"],
                    )
                    append_sota_kg_node(root=self.root, node=node)
                    emit_bus(self.bus_dir, "kg.node.added", "artifact", "info", node)
                    emit_bus(self.bus_dir, "sota.kg.status", "metric", "info", {"item_id": item_id, "status": "linked", "node_id": node.get("id"), "ref": ref})
                except Exception as e:
                    emit_bus(self.bus_dir, "sota.kg.status", "metric", "error", {"item_id": item_id, "status": "failed", "error": str(e)})
            else:
                emit_bus(self.bus_dir, "catalog.log", "log", "error", {"msg": "sota.kg.add missing item_id/ref"})

    def run(self) -> None:
        emit_bus(self.bus_dir, "catalog.daemon.start", "metric", "info", {"root": str(self.root)})
        
        # Initial publish
        self.svc_reg.init()
        self.publish_snapshot()
        
        next_publish = time.time() + self.interval_s
        next_responses_check = time.time() + 1.0
        
        # Follow bus for actions
        if not self.events_path.exists():
            self.events_path.parent.mkdir(parents=True, exist_ok=True)
            self.events_path.touch()

        with self.events_path.open("r", encoding="utf-8", errors="replace") as f:
            f.seek(0, os.SEEK_END)
            
            while self.running:
                now = time.time()
                if now >= next_publish:
                    self.publish_snapshot()
                    next_publish = now + self.interval_s
                if now >= next_responses_check:
                    self.check_new_responses()
                    next_responses_check = now + 1.0

                line = f.readline()
                if line:
                    try:
                        obj = json.loads(line)
                        topic = obj.get("topic")
                        if topic == "catalog.action":
                            self.handle_action(obj.get("data", {}))
                        elif topic == "dashboard.refresh":
                            self.publish_snapshot()
                    except Exception:
                        pass
                else:
                    time.sleep(0.1)

def main() -> int:
    parser = argparse.ArgumentParser(description="Catalog Daemon")
    parser.add_argument("--root", default=None)
    parser.add_argument("--bus-dir", default=None)
    parser.add_argument("--interval-s", type=float, default=10.0)
    args = parser.parse_args()

    root = Path(args.root).expanduser().resolve() if args.root else (find_rhizome_root(Path.cwd()) or Path.cwd())
    bus_dir = Path(args.bus_dir).resolve() if args.bus_dir else (root / ".pluribus" / "bus")

    daemon = CatalogDaemon(root, bus_dir, args.interval_s)
    try:
        daemon.run()
    except KeyboardInterrupt:
        return 0
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
