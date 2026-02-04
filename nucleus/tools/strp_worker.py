#!/usr/bin/env python3
from __future__ import annotations

import argparse
import concurrent.futures
import contextlib
import getpass
import json
import os
import subprocess
import sys
import time
import uuid
from pathlib import Path

sys.dont_write_bytecode = True

from topology_policy import approx_tokens_from_text, choose_topology
from grounding import verify_grounded_output
try:
    from nucleus.tools.liveness import TimeBoundMonitor
except ImportError:
    # Fallback for when running from root without package install
    sys.path.append(str(Path(__file__).resolve().parents[2]))
    from nucleus.tools.liveness import TimeBoundMonitor

try:
    from iso_executor import IsoExecutor
except ImportError:
    sys.path.append(str(Path(__file__).resolve().parent))
    from iso_executor import IsoExecutor

try:
    from infercell_manager import InferCellManager
except ImportError:
    sys.path.append(str(Path(__file__).resolve().parent))
    from infercell_manager import InferCellManager


def now_iso_utc() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def default_actor() -> str:
    return os.environ.get("PLURIBUS_ACTOR") or os.environ.get("USER") or getpass.getuser()


def ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def append_ndjson(path: Path, obj: dict) -> None:
    ensure_dir(path.parent)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(obj, ensure_ascii=False, separators=(",", ":")) + "\n")


def iter_ndjson(path: Path):
    if not path.exists():
        return
    with path.open("r", encoding="utf-8", errors="replace") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except Exception:
                continue


def find_rhizome_root(start: Path) -> Path | None:
    cur = start.resolve()
    for cand in [cur, *cur.parents]:
        if (cand / ".pluribus" / "rhizome.json").exists():
            return cand
    return None


def emit_bus(bus_dir: str | None, *, topic: str, kind: str, level: str, actor: str, data: dict, trace_id: str | None = None) -> None:
    if not bus_dir:
        return
    tool = Path(__file__).with_name("agent_bus.py")
    if not tool.exists():
        return
    cmd = [
        sys.executable,
        str(tool),
        "--bus-dir",
        bus_dir,
        "pub",
        "--topic",
        topic,
        "--kind",
        kind,
        "--level",
        level,
        "--actor",
        actor,
        "--data",
        json.dumps(data, ensure_ascii=False),
    ]
    if trace_id:
        cmd.extend(["--trace-id", trace_id])
    
    subprocess.run(
        cmd,
        check=False,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        env={**os.environ, "PYTHONDONTWRITEBYTECODE": "1"},
    )


def extract_json_object(text: str) -> dict | None:
    t = (text or "").strip()
    if not t:
        return None
    try:
        obj = json.loads(t)
        return obj if isinstance(obj, dict) else None
    except Exception:
        pass
    # Heuristic fallback: find the outermost JSON object.
    i = t.find("{")
    j = t.rfind("}")
    if i < 0 or j < 0 or j <= i:
        return None
    try:
        obj = json.loads(t[i : j + 1])
        return obj if isinstance(obj, dict) else None
    except Exception:
        return None


def build_prompt(req: dict) -> str:
    kind = req.get("kind") or "distill"
    goal = req.get("goal") or ""
    inputs = req.get("inputs")
    constraints = req.get("constraints")
    sextet = req.get("sextet")

    require_citations = False
    if isinstance(constraints, dict):
        if bool(constraints.get("require_citations")):
            require_citations = True
        grounding_cfg = constraints.get("grounding")
        if isinstance(grounding_cfg, dict) and bool(grounding_cfg.get("require_citations")):
            require_citations = True

    schema_hint = {
        "distill": "Return JSON: {summary, claims[], gaps[], definitions[], counterexamples[], citations[], next_actions[]}.",
        "hypothesize": "Return JSON: {hypotheses[], falsifiers[], tests[], metrics[], next_actions[]}.",
        "apply": "Return JSON: {applied_theory, invariants[], interfaces[], data_requirements[], next_actions[]}.",
        "implement": "Return JSON: {tasks[], files[], tests[], risks[], next_actions[]}.",
        "verify": "Return JSON: {checks[], counterexamples[], gates[], next_actions[]}.",
    }.get(str(kind), "Return JSON with your best structured output and next_actions.")

    if require_citations and str(kind) in {"distill", "apply", "verify", "implement", "hypothesize"}:
        schema_hint = schema_hint.rstrip(".") + " citations[] is REQUIRED; use refs like rag:<doc_id>, sha256:<hex64>, or https://... ."

    payload = {
        "kind": kind,
        "goal": goal,
        "inputs": inputs,
        "constraints": constraints,
        "sextet": sextet,
        "output_contract": schema_hint,
        "rules": [
            "Prefer concise JSON.",
            "If unsure, add a 'gaps' entry instead of guessing.",
            "If you cite, use stable URLs or file paths provided in inputs.",
        ],
    }
    return "STRp request\n\n" + json.dumps(payload, ensure_ascii=False, indent=2)


def build_variant_prompt(req: dict, *, variant: str) -> str:
    base = build_prompt(req)
    return base + "\n\n# Variant\n" + variant + "\n"


def build_aggregate_prompt(req: dict, *, candidates: list[dict]) -> str:
    kind = req.get("kind") or "distill"
    goal = req.get("goal") or ""
    schema_hint = {
        "distill": "Return JSON: {summary, claims[], gaps[], definitions[], counterexamples[], citations[], next_actions[]}.",
        "hypothesize": "Return JSON: {hypotheses[], falsifiers[], tests[], metrics[], next_actions[]}.",
        "apply": "Return JSON: {applied_theory, invariants[], interfaces[], data_requirements[], next_actions[]}.",
        "implement": "Return JSON: {tasks[], files[], tests[], risks[], next_actions[]}.",
        "verify": "Return JSON: {checks[], counterexamples[], gates[], next_actions[]}.",
    }.get(str(kind), "Return JSON with your best structured output and next_actions.")

    payload = {
        "kind": kind,
        "goal": goal,
        "candidates": candidates,
        "output_contract": schema_hint,
        "rules": [
            "Prefer concise JSON.",
            "If candidates disagree, surface the disagreement under 'gaps' or 'counterexamples' rather than inventing certainty.",
            "Do not hallucinate citations; only pass through citations present in candidates or inputs.",
        ],
    }
    return "STRp aggregation\n\n" + json.dumps(payload, ensure_ascii=False, indent=2)


def run_router(*, prompt: str, provider: str, model: str | None, timeout: float | None = None) -> tuple[int, str, str]:
    tool = Path(__file__).resolve().parent / "providers" / "router.py"
    argv = [sys.executable, str(tool), "--provider", provider, "--prompt", prompt]
    if model:
        argv += ["--model", model]
    try:
        p = subprocess.run(argv, check=False, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, timeout=timeout)
        return int(p.returncode), p.stdout, p.stderr
    except subprocess.TimeoutExpired:
        return 124, "", f"timed out after {timeout}s"

@contextlib.contextmanager
def infercell_execution_context(
    *,
    root: Path,
    bus_dir: str | None,
    actor: str,
    req_id: str,
    trace_id: str | None,
    parent_trace_id: str | None = None,
):
    """
    If trace_id is present, bind execution into the corresponding InferCell workspace.

    This makes trace scopes physically isolate their filesystem side-effects under:
      .pluribus/infercells/<cell_id>/workspace
    """
    if not trace_id:
        yield None
        return

    prev_cwd = os.getcwd()
    try:
        mgr = InferCellManager(root)
        cell = mgr.get_cell(trace_id)
        if not cell:
            cell = mgr.ensure_cell_for_trace(str(trace_id), parent_trace_id=parent_trace_id, reason=f"strp:{req_id}")
        workspace = mgr.storage_dir / cell.cell_id / "workspace"
        workspace.mkdir(parents=True, exist_ok=True)

        emit_bus(
            bus_dir,
            topic="infercell.exec.bound",
            kind="metric",
            level="info",
            actor=actor,
            data={
                "req_id": req_id,
                "trace_id": trace_id,
                "parent_trace_id": parent_trace_id,
                "cell_id": cell.cell_id,
                "workspace": str(workspace),
                "prev_cwd": prev_cwd,
            },
            trace_id=trace_id,
        )
        os.chdir(str(workspace))
        yield str(workspace)
    except Exception as e:
        emit_bus(
            bus_dir,
            topic="infercell.exec.bound",
            kind="metric",
            level="warn",
            actor=actor,
            data={"req_id": req_id, "trace_id": trace_id, "error": str(e), "fallback": True},
            trace_id=trace_id,
        )
        os.chdir(prev_cwd)
        yield None
    finally:
        try:
            os.chdir(prev_cwd)
        except Exception:
            pass


def classify_failure_text(text: str) -> str | None:
    t = (text or "").lower()
    if not t.strip():
        return None
    if "please run /login" in t or "run /login" in t or "invalid api key" in t:
        return "blocked_auth"
    if "resource_exhausted" in t or "quota exceeded" in t or "http error: 429" in t:
        return "blocked_quota"
    if "no provider configured" in t or "missing api key" in t:
        return "blocked_config"
    if "unsupported provider" in t:
        return "blocked_config"
    return "error"


def load_progress(responses_path: Path) -> dict[str, dict]:
    progress: dict[str, dict] = {}
    for obj in iter_ndjson(responses_path):
        rid = obj.get("req_id")
        if isinstance(rid, str) and rid:
            st = progress.setdefault(rid, {"attempts": 0, "success": False, "last_ts": 0.0})
            st["attempts"] = int(st.get("attempts") or 0) + 1
            try:
                ts = float(obj.get("ts") or 0.0)
            except Exception:
                ts = 0.0
            if ts >= float(st.get("last_ts") or 0.0):
                st["last_ts"] = ts
                try:
                    st["last_exit_code"] = int(obj.get("exit_code"))
                except Exception:
                    st["last_exit_code"] = None
                stderr = obj.get("stderr")
                out = obj.get("output")
                msg = "\n".join([str(stderr or ""), str(out or "")]).strip()
                st["last_classification"] = obj.get("classification") if isinstance(obj.get("classification"), str) else classify_failure_text(msg)
            if int(obj.get("exit_code") or 1) == 0:
                st["success"] = True
    return progress


def cmd_run(args: argparse.Namespace) -> int:
    root = Path(args.root).expanduser().resolve() if args.root else (find_rhizome_root(Path.cwd()) or Path.cwd().resolve())
    idx = root / ".pluribus" / "index"
    ensure_dir(idx)

    requests_path = idx / "requests.ndjson"
    responses_path = idx / "responses.ndjson"
    actor = default_actor()
    bus_dir = args.bus_dir or os.environ.get("PLURIBUS_BUS_DIR")

    emit_bus(bus_dir, topic="strp.worker.start", kind="log", level="info", actor=actor, data={"root": str(root)})
    progress = load_progress(responses_path)
    retry_after_s = float(os.environ.get("PLURIBUS_RETRY_AFTER_S") or "300")
    max_attempts = int(os.environ.get("PLURIBUS_MAX_ATTEMPTS") or "3")
    if isinstance(getattr(args, "retry_after_s", None), (float, int)) and args.retry_after_s is not None:
        retry_after_s = float(args.retry_after_s)
    if isinstance(getattr(args, "max_attempts", None), int) and args.max_attempts is not None:
        max_attempts = int(args.max_attempts)
    allow_fallback = (os.environ.get("PLURIBUS_ALLOW_FALLBACK") or "1").strip().lower() not in {"0", "false", "no", "off"}
    allow_mock = (os.environ.get("PLURIBUS_ALLOW_MOCK") or "").strip().lower() in {"1", "true", "yes", "on"}
    def load_vps_session() -> dict:
        try:
            sess_path = root / ".pluribus" / "vps_session.json"
            if sess_path.exists():
                return json.loads(sess_path.read_text(encoding="utf-8"))
        except Exception:
            pass
        return {}

    # Initial load
    vps_session = load_vps_session()
    
    # Priority: Env -> Session -> Default
    fallback_chain_env = (os.environ.get("PLURIBUS_FALLBACK_PROVIDERS") or "").strip()
    if fallback_chain_env:
        fallback_chain = [p.strip() for p in fallback_chain_env.split(",") if p.strip()]
    else:
        # Session key is snake_case in vps_session.json; keep camelCase for backward-compat.
        fb = vps_session.get("fallback_order")
        if not isinstance(fb, list):
            fb = vps_session.get("fallbackOrder")
        if isinstance(fb, list) and fb:
            fallback_chain = [str(p).strip() for p in fb if isinstance(p, str) and p.strip()]
        else:
            # Default chain (web-only): browser-backed sessions only (no CLI/API/mock fallbacks).
            fallback_chain = ["chatgpt-web", "claude-web", "gemini-web"]
    if not allow_mock:
        fallback_chain = [p for p in fallback_chain if p != "mock"]

    def tick() -> int:
        # Refresh session live
        nonlocal fallback_chain
        current_session = load_vps_session()
        fb2 = current_session.get("fallback_order")
        if not isinstance(fb2, list):
            fb2 = current_session.get("fallbackOrder")
        if isinstance(fb2, list) and fb2:
            fallback_chain = [str(p).strip() for p in fb2 if isinstance(p, str) and p.strip()]
            if not allow_mock:
                fallback_chain = [p for p in fallback_chain if p != "mock"]
            
        if not requests_path.exists():
            return 0
        new_done = 0
        for req in iter_ndjson(requests_path):
            req_id = req.get("req_id")
            trace_id = req.get("trace_id")
            parent_trace_id = req.get("parent_trace_id")
            if not isinstance(req_id, str) or not req_id:
                continue
            if args.only_req_id and req_id != args.only_req_id:
                continue
            st = progress.get(req_id)
            if st and st.get("success"):
                continue
            if st and max_attempts > 0 and int(st.get("attempts") or 0) >= max_attempts:
                continue
            if st and retry_after_s > 0 and float(st.get("last_ts") or 0.0) > 0 and (time.time() - float(st.get("last_ts") or 0.0)) < retry_after_s:
                continue
            
            # Provider resolution: Request Hint -> CLI Arg -> Active Fallback -> Auto
            provider = req.get("provider_hint")
            if not provider or provider == "auto":
                provider = args.provider
            if not provider or provider == "auto":
                active_fb = current_session.get("active_fallback")
                if not isinstance(active_fb, str) or not active_fb.strip():
                    active_fb = current_session.get("activeFallback")
                provider = (active_fb or "auto")
            
            if not provider: provider = "auto"

            with infercell_execution_context(root=root, bus_dir=bus_dir, actor=actor, req_id=req_id, trace_id=trace_id, parent_trace_id=parent_trace_id):
                decision = choose_topology(req if isinstance(req, dict) else {})
                topology = decision.get("topology") or "single"
                fanout = int(decision.get("fanout") or 1)
                isolation = decision.get("isolation") or "thread"  # Iso-STRp: thread or process
                reason = str(decision.get("reason") or "")
                lineage = req.get("lineage") or "core.strp"  # Pillar 1: Lineage tracking

                coord_budget_tokens = req.get("coord_budget_tokens")
                if isinstance(coord_budget_tokens, int) and coord_budget_tokens > 0:
                    base_prompt_tokens = approx_tokens_from_text(build_prompt(req))
                    if base_prompt_tokens > 0:
                        max_fanout = max(1, int(coord_budget_tokens // base_prompt_tokens))
                        if fanout > max_fanout:
                            topology = "single"
                            fanout = 1
                            reason = "coord_budget_enforced"

                emit_bus(
                    bus_dir,
                    topic="agent.topology.chosen",
                    kind="metric",
                    level="info",
                    actor=actor,
                    data={
                        "req_id": req_id,
                        "topology": topology,
                        "fanout": fanout,
                        "isolation": isolation,
                        "reason": reason,
                        "provider": provider,
                        "decision": decision,
                        "lineage": lineage,  # Pillar 1
                    },
                    trace_id=trace_id,
                )

                # Liveness check (omega-gate)
                monitor = TimeBoundMonitor(max_seconds=60.0)

                t0 = time.perf_counter()
                prompt_tokens = 0
                output_tokens = 0
                runs: list[dict] = []

                def run_with_fallback(prompt_text: str) -> tuple[int, str, str, str]:
                    code0, out0, err0 = 1, "", "init"
                    used = provider
                    for attempt in range(3):
                        code0, out0, err0 = run_router(prompt=prompt_text, provider=provider, model=args.model, timeout=120.0)
                        if code0 == 0:
                            return code0, out0, err0, provider
                        classification = classify_failure_text("\n".join([err0 or "", out0 or ""])) or "error"
                        if classification == "blocked_quota" and attempt < 2:
                            time.sleep(2.0 * (attempt + 1))
                            continue
                        break

                    if code0 == 0 or not allow_fallback:
                        return code0, out0, err0, used

                    classification = classify_failure_text("\n".join([err0 or "", out0 or ""])) or "error"
                    for fb in fallback_chain:
                        if fb == used:
                            continue
                        code1, out1, err1 = run_router(prompt=prompt_text, provider=fb, model=args.model, timeout=60.0)
                        runs.append({"variant": f"fallback:{fb}", "provider": fb, "exit_code": code1, "stderr": err1.strip(), "output": out1.strip()})
                        if code1 == 0:
                            emit_bus(
                                bus_dir,
                                topic="agent.coord.fallback",
                                kind="metric",
                                level="warn",
                                actor=actor,
                                data={"req_id": req_id, "from": used, "to": fb, "classification": classification},
                                trace_id=trace_id,
                            )
                            return code1, out1, err1, fb
                    return code0, out0, err0, used

                if topology == "single" or fanout <= 1:
                    prompt = build_prompt(req)
                    prompt_tokens = approx_tokens_from_text(prompt)

                    # Iso-STRp: Use process isolation for risky/network tasks
                    if isolation == "process":
                        iso_exec = IsoExecutor(bus_dir=bus_dir)
                        sub_trace_id = str(uuid.uuid4())
                        emit_bus(
                            bus_dir,
                            topic="strp.subagent.spawn",
                            kind="metric",
                            level="info",
                            actor=actor,
                            data={
                                "req_id": req_id,
                                "sub_id": 1,
                                "total": 1,
                                "topology": "single",
                                "parent_trace_id": trace_id,
                                "isolation": "process",
                                "reason": "iso_strp_enforced",
                            },
                            trace_id=sub_trace_id,
                        )
                        res = iso_exec.spawn(
                            agent=provider if provider != "auto" else "ring0.architect",
                            goal=prompt,
                            trace_id=sub_trace_id,
                            parent_id=req_id,
                            timeout=120
                        )
                        if res.exit_code == 0:
                            try:
                                data = json.loads(res.stdout)
                                code, out, err = 0, data.get("text", res.stdout), ""
                                used_provider = data.get("provider", "iso-process")
                            except json.JSONDecodeError:
                                code, out, err = 0, res.stdout, ""
                                used_provider = "iso-process"
                        else:
                            code, out, err = res.exit_code, res.stdout, res.stderr
                            used_provider = "iso-process-fail"
                        runs.append({"variant": "single-isolated", "provider": used_provider, "exit_code": code, "stderr": err.strip(), "output": out.strip(), "isolation": "process"})
                    else:
                        # Thread-level execution (fast, shared state)
                        code, out, err, used_provider = run_with_fallback(prompt)
                        runs.append({"variant": "single", "provider": used_provider, "exit_code": code, "stderr": err.strip(), "output": out.strip(), "isolation": "thread"})

                    final_code, final_out, final_err = code, out, err
                    output_tokens = approx_tokens_from_text(final_out)
                elif topology in {"star", "peer_debate"}:
                    iso_exec = IsoExecutor(bus_dir=bus_dir)
                    futures = {}

                    def run_isolated_subtask(prompt_text, sub_trace_id):
                        # Process-isolated execution via IsoExecutor -> PluriChat
                        # This ensures environment/memory isolation for each sub-agent
                        res = iso_exec.spawn(
                            agent="ring0.architect" if args.provider == "auto" else args.provider,  # default persona
                            goal=prompt_text,
                            trace_id=sub_trace_id,
                            parent_id=req_id,
                            timeout=120
                        )

                        if res.exit_code == 0:
                            try:
                                # Parse structured JSON from plurichat --json-output
                                data = json.loads(res.stdout)
                                return 0, data.get("text", ""), "", data.get("provider", "iso-process")
                            except json.JSONDecodeError:
                                # Fallback if non-JSON output (shouldn't happen with --json-output but safe)
                                return 0, res.stdout, "", "iso-process"

                        return res.exit_code, res.stdout, res.stderr, "iso-process-fail"

                    with concurrent.futures.ThreadPoolExecutor(max_workers=fanout) as executor:
                        for i in range(fanout):
                            # Context Forking: Generate sub-trace-id
                            sub_trace_id = str(uuid.uuid4())
                            # Signal intent to spawn (Dashboard visibility)
                            emit_bus(
                                bus_dir,
                                topic="strp.subagent.spawn",
                                kind="metric",
                                level="info",
                                actor=actor,
                                data={
                                    "req_id": req_id,
                                    "sub_id": i + 1,
                                    "total": fanout,
                                    "topology": topology,
                                    "parent_trace_id": trace_id,
                                    "isolation": "process",
                                },
                                trace_id=sub_trace_id
                            )

                            prompt = build_variant_prompt(req, variant=f"independent_attempt_{i+1}_of_{fanout}")
                            prompt_tokens += approx_tokens_from_text(prompt)
                            # Submit the task to IsoExecutor
                            futures[executor.submit(run_isolated_subtask, prompt, sub_trace_id)] = i

                        for future in concurrent.futures.as_completed(futures):
                            i = futures[future]
                            try:
                                code, out, err, used_provider = future.result()
                                runs.append({"variant": f"attempt_{i+1}", "provider": used_provider, "exit_code": code, "stderr": err.strip(), "output": out.strip()})
                            except Exception as e:
                                runs.append({"variant": f"attempt_{i+1}", "provider": "error", "exit_code": 1, "stderr": str(e), "output": ""})

                    candidates = []
                    for r in runs:
                        if r.get("exit_code") == 0 and r.get("output"):
                            candidates.append({"variant": r.get("variant"), "output": r.get("output")})

                    if not candidates:
                        final_code, final_out, final_err = 1, "", "all subruns failed"
                    else:
                        agg_prompt = build_aggregate_prompt(req, candidates=candidates)
                        prompt_tokens += approx_tokens_from_text(agg_prompt)
                        final_code, final_out, final_err, _used = run_with_fallback(agg_prompt)
                        output_tokens = approx_tokens_from_text(final_out)
                else:
                    prompt = build_prompt(req)
                    prompt_tokens = approx_tokens_from_text(prompt)
                    final_code, final_out, final_err, used_provider = run_with_fallback(prompt)
                    runs.append({"variant": "fallback_single", "provider": used_provider, "exit_code": final_code, "stderr": final_err.strip(), "output": final_out.strip()})
                    output_tokens = approx_tokens_from_text(final_out)

                latency_s = max(0.0, time.perf_counter() - t0)

                monitor.heartbeat({})
                is_healthy = monitor.is_healthy()

                emit_bus(
                    bus_dir,
                    topic="agent.coord.latency_s",
                    kind="metric",
                    level="info",
                    actor=actor,
                    data={"req_id": req_id, "topology": topology, "fanout": fanout, "latency_s": latency_s, "provider": provider, "liveness": is_healthy},
                )
                emit_bus(
                    bus_dir,
                    topic="agent.coord.tokens",
                    kind="metric",
                    level="info",
                    actor=actor,
                    data={
                        "req_id": req_id,
                        "topology": topology,
                        "fanout": fanout,
                        "prompt_tokens_est": int(prompt_tokens),
                        "output_tokens_est": int(output_tokens),
                        "total_tokens_est": int(prompt_tokens + output_tokens),
                        "coord_budget_tokens": coord_budget_tokens if isinstance(coord_budget_tokens, int) else None,
                        "provider": provider,
                    },
                )

                resp = {
                    "id": str(uuid.uuid4()),
                    "ts": time.time(),
                    "iso": now_iso_utc(),
                    "kind": "strp_response",
                    "req_id": req_id,
                    "provider": provider,
                    "classification": classify_failure_text("\n".join([str(final_err or ""), str(final_out or "")]).strip()) if int(final_code) != 0 else None,
                    "model": args.model,
                    "topology": topology,
                    "fanout": fanout,
                    "runs": runs,
                    "exit_code": int(final_code),
                    "stderr": final_err.strip() if isinstance(final_err, str) else None,
                    "output": final_out.strip() if isinstance(final_out, str) else None,
                    "provenance": {"added_by": actor},
                    "lineage": lineage,  # Pillar 1
                }

                # Optional grounding verifier: only enforce when explicitly requested by constraints.
                require_citations = False
                validate_refs = False
                constraints = req.get("constraints") if isinstance(req, dict) else None
                if isinstance(constraints, dict):
                    if bool(constraints.get("require_citations")):
                        require_citations = True
                    grounding_cfg = constraints.get("grounding")
                    if isinstance(grounding_cfg, dict):
                        if bool(grounding_cfg.get("require_citations")):
                            require_citations = True
                        if bool(grounding_cfg.get("validate_refs")):
                            validate_refs = True
                if int(final_code) == 0 and isinstance(resp.get("output"), str) and require_citations:
                    parsed = extract_json_object(str(resp.get("output") or ""))
                    ok, issues = verify_grounded_output(
                        parsed or {},
                        require_citations=True,
                        validate_refs=validate_refs,
                        root=str(req.get("rhizome_root") or "") if isinstance(req, dict) else None,
                    )
                    resp["grounding"] = {"ok": ok, "issues": issues}
                    emit_bus(
                        bus_dir,
                        topic="strp.output.grounding",
                        kind="metric",
                        level="info" if ok else "warn",
                        actor=actor,
                        data={"req_id": req_id, "ok": ok, "issues": issues},
                    )
                    if not ok:
                        resp["exit_code"] = 1
                        resp["stderr"] = "grounding_verifier_failed: " + ",".join(issues)
                        resp["classification"] = "grounding_failed"

                if int(resp.get("exit_code") or 0) != 0:
                    classification = resp.get("classification") or "error"
                    topic = "strp.worker.blocked" if str(classification).startswith("blocked_") else "strp.worker.error"
                    emit_bus(
                        bus_dir,
                        topic=topic,
                        kind="request",
                        level="warn" if str(classification).startswith("blocked_") else "error",
                        actor=actor,
                        data={"req_id": req_id, "goal": req.get("goal"), "provider": provider, "classification": classification, "retry_after_s": retry_after_s},
                    )
                append_ndjson(responses_path, resp)
                st2 = progress.setdefault(req_id, {"attempts": 0, "success": False, "last_ts": 0.0})
                st2["attempts"] = int(st2.get("attempts") or 0) + 1
                st2["last_ts"] = float(resp.get("ts") or time.time())
                st2["last_exit_code"] = int(resp.get("exit_code") or 0)
                st2["last_classification"] = resp.get("classification")
                if int(resp.get("exit_code") or 0) == 0:
                    st2["success"] = True
                new_done += 1
                emit_bus(
                    bus_dir,
                    topic="strp.worker.item",
                    kind="response",
                    level="info" if int(resp.get("exit_code") or 0) == 0 else "error",
                    actor=actor,
                    data=resp,
                )
        return new_done

    if args.once:
        done = tick()
        sys.stdout.write(f"processed {done}\n")
        return 0

    poll = max(0.2, float(args.poll))
    while True:
        _ = tick()
        time.sleep(poll)


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="strp_worker.py", description="STRp worker: consume .pluribus/index/requests.ndjson and produce responses via provider router.")
    p.add_argument("--root", default=None, help="Rhizome root (defaults: search upward for .pluribus/rhizome.json).")
    p.add_argument("--bus-dir", default=None, help="Bus dir (or set PLURIBUS_BUS_DIR).")
    p.add_argument("--provider", default="auto", help="auto|chatgpt-web|claude-web|gemini-web")
    p.add_argument("--model", default=None, help="Optional model override (provider-specific).")
    p.add_argument("--poll", default="1.0", help="Poll interval seconds (default: 1.0).")
    p.add_argument("--once", action="store_true", help="Process current backlog once, then exit.")
    p.add_argument("--only-req-id", default=None, help="If set, only process a single request id (useful for interactive callers).")
    p.add_argument("--max-attempts", type=int, default=None, help="Override PLURIBUS_MAX_ATTEMPTS (default: 3).")
    p.add_argument("--retry-after-s", type=float, default=None, help="Override PLURIBUS_RETRY_AFTER_S (default: 300).")
    p.set_defaults(func=cmd_run)
    return p


def main(argv: list[str]) -> int:
    args = build_parser().parse_args(argv)
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
