/**
 * SUPERWORKER INJECTION (The Hands)
 * =================================
 * Bridges the "Naive" Web Chat UI to the "Savant" Backend Worker.
 *
 * Protocol:
 * 1. UI (OmegaWorker) emits 'worker.spawn' -> Bus.
 * 2. This daemon hears 'worker.spawn'.
 * 3. It spawns a 'strp_worker.py' instance (or dedicated container).
 * 4. It replies with 'worker.ready' { worker_id, capacity }.
 */

import json
import logging
import subprocess
import sys
import uuid
from pathlib import Path

# Adjust path for nucleus import
sys.path.append(str(Path(__file__).resolve().parents[2]))

from nucleus.tools.agent_bus import BusPaths, emit_event, resolve_bus_paths, iter_lines_follow

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

def spawn_worker(provider: str, context_mode: str) -> str:
    """Spawns a background STRp worker process."""
    worker_id = str(uuid.uuid4())
    cmd = [
        sys.executable,
        "nucleus/tools/strp_worker.py",
        "--worker-id", worker_id,
        "--provider", provider,
        "--mode", "daemon"
    ]
    
    # In a real system, we'd use PM2 or Systemd here.
    # For now, we use Popen (detached).
    logging.info(f"Spawning worker {worker_id} for {provider}...")
    subprocess.Popen(cmd, start_new_session=True)
    return worker_id

def main():
    paths = resolve_bus_paths(None)
    logging.info(f"Superworker Injection Daemon listening on {paths.events_path}...")

    # Emit boot event
    emit_event(paths, topic="superworker.daemon.start", kind="metric", level="info", actor="superworker-d", data={}, trace_id=None, run_id=None, durable=False)

    for line in iter_lines_follow(paths.events_path):
        try:
            event = json.loads(line)
            topic = event.get("topic", "")
            
            if topic == "worker.spawn":
                data = event.get("data", {})
                provider = data.get("provider", "openai")
                context_mode = data.get("context_mode", "lite")
                
                try:
                    worker_id = spawn_worker(provider, context_mode)
                    
                    emit_event(paths, 
                        topic="worker.ready", 
                        kind="metric", 
                        level="info", 
                        actor="superworker-d",
                        data={"worker_id": worker_id, "provider": provider},
                        trace_id=event.get("trace_id"),
                        run_id=None,
                        durable=False
                    )
                except Exception as e:
                    logging.error(f"Failed to spawn worker: {e}")
                    emit_event(paths,
                        topic="worker.fail",
                        kind="error",
                        level="error",
                        actor="superworker-d",
                        data={"error": str(e)},
                        trace_id=event.get("trace_id"),
                        run_id=None,
                        durable=False
                    )

        except Exception:
            continue

if __name__ == "__main__":
    main()