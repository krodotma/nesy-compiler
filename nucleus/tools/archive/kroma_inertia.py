#!/usr/bin/env python3
"""
kroma_inertia.py — Computational Inertia Tensor Management

Implements the "I AM" principle: bounded policy/memory/goal updates
that preserve agent identity across state transitions.

Domains:
- Policy inertia (TRPO/PPO-style trust regions)
- Memory inertia (EWC/SI-style importance weights)
- Goal inertia (ACT-R/SOAR-style stack persistence)
- Control inertia (jerk penalties, smooth trajectories)
- Equilibrium inertia (ESS, replicator dynamics stability)

References:
- TRPO: Schulman et al., 2015
- PPO: Schulman et al., 2017
- EWC: Kirkpatrick et al., 2017
- Synaptic Intelligence: Zenke et al., 2017
"""
from __future__ import annotations

import argparse
import json
import math
import os
import subprocess
import sys
import time
import uuid
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Optional

sys.dont_write_bytecode = True


# --- Constants ---

DEFAULT_POLICY_BOUND = 0.01  # KL divergence bound (TRPO-style)
DEFAULT_MEMORY_LAMBDA = 0.5  # EWC regularization strength
DEFAULT_GOAL_POP_COST = 1.0  # Cost to pop goal from stack
DEFAULT_CONTROL_JERK_PENALTY = 0.1  # Penalty for control discontinuities
DEFAULT_EQUILIBRIUM_STABILITY = 0.95  # Replicator dynamics stability threshold


# --- Data Structures ---

@dataclass
class InertiaConfig:
    """Configuration for inertia tensor."""
    policy_step_bound: float = DEFAULT_POLICY_BOUND
    memory_consolidation_lambda: float = DEFAULT_MEMORY_LAMBDA
    goal_stack_pop_cost: float = DEFAULT_GOAL_POP_COST
    control_smoothness_penalty: float = DEFAULT_CONTROL_JERK_PENALTY
    equilibrium_stability_threshold: float = DEFAULT_EQUILIBRIUM_STABILITY


@dataclass
class InertiaTensor:
    """
    The "I AM" tensor: tracks agent identity across transitions.

    Invariant: ||I_AM(t+δ) - I_AM(t)|| ≤ ε·δ (Lipschitz continuity)
    """
    config: InertiaConfig
    policy_hash: str = ""
    memory_importance: dict = field(default_factory=dict)
    goal_stack: list = field(default_factory=list)
    control_history: list = field(default_factory=list)
    equilibrium_state: dict = field(default_factory=dict)
    last_update_ts: float = 0.0

    def to_dict(self) -> dict:
        return {
            "config": asdict(self.config),
            "policy_hash": self.policy_hash,
            "memory_importance": self.memory_importance,
            "goal_stack": self.goal_stack,
            "control_history": self.control_history[-10:],  # Keep last 10
            "equilibrium_state": self.equilibrium_state,
            "last_update_ts": self.last_update_ts,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "InertiaTensor":
        config = InertiaConfig(**d.get("config", {}))
        return cls(
            config=config,
            policy_hash=d.get("policy_hash", ""),
            memory_importance=d.get("memory_importance", {}),
            goal_stack=d.get("goal_stack", []),
            control_history=d.get("control_history", []),
            equilibrium_state=d.get("equilibrium_state", {}),
            last_update_ts=d.get("last_update_ts", 0.0),
        )


@dataclass
class InertiaUpdate:
    """Proposed update to the inertia tensor."""
    update_id: str
    domain: str  # policy | memory | goal | control | equilibrium
    delta: dict
    magnitude: float
    timestamp: float
    accepted: bool = False
    rejection_reason: Optional[str] = None


# --- Inertia Checking ---

def check_policy_inertia(tensor: InertiaTensor, new_policy_hash: str, kl_divergence: float) -> tuple[bool, str]:
    """
    Check if policy update violates trust region (TRPO/PPO style).

    Returns (accepted, reason)
    """
    if kl_divergence > tensor.config.policy_step_bound:
        return False, f"KL divergence {kl_divergence:.4f} exceeds bound {tensor.config.policy_step_bound}"
    return True, "Policy update within trust region"


def check_memory_inertia(tensor: InertiaTensor, param_id: str, proposed_delta: float) -> tuple[bool, str]:
    """
    Check if memory update violates importance-weighted constraint (EWC style).

    Returns (accepted, reason)
    """
    importance = tensor.memory_importance.get(param_id, 0.0)
    effective_penalty = importance * tensor.config.memory_consolidation_lambda

    if abs(proposed_delta) * effective_penalty > 1.0:
        return False, f"Memory update to {param_id} violates consolidation (importance={importance:.3f})"
    return True, "Memory update within consolidation bounds"


def check_goal_inertia(tensor: InertiaTensor, action: str, goal_id: Optional[str] = None) -> tuple[bool, str]:
    """
    Check if goal stack manipulation is valid (ACT-R/SOAR style).

    Actions: push, pop, peek
    Returns (accepted, reason)
    """
    if action == "push":
        return True, "Goal push always allowed"

    if action == "pop":
        if not tensor.goal_stack:
            return False, "Cannot pop from empty goal stack"
        # Pop cost check would go here if we had a budget
        return True, f"Goal pop allowed (cost={tensor.config.goal_stack_pop_cost})"

    if action == "peek":
        return True, "Goal peek always allowed"

    return False, f"Unknown goal action: {action}"


def check_control_inertia(tensor: InertiaTensor, new_control: dict) -> tuple[bool, float, str]:
    """
    Check if control update causes excessive jerk (control theory style).

    Returns (accepted, jerk_magnitude, reason)
    """
    if not tensor.control_history:
        return True, 0.0, "First control action"

    last_control = tensor.control_history[-1]

    # Compute jerk as rate of change of acceleration (third derivative)
    # Simplified: just measure change magnitude
    jerk = 0.0
    for key in set(new_control.keys()) | set(last_control.get("values", {}).keys()):
        old_val = last_control.get("values", {}).get(key, 0.0)
        new_val = new_control.get(key, 0.0)
        jerk += (new_val - old_val) ** 2
    jerk = math.sqrt(jerk)

    if jerk > 1.0 / tensor.config.control_smoothness_penalty:
        return False, jerk, f"Control jerk {jerk:.3f} exceeds smoothness bound"

    return True, jerk, "Control within smoothness bounds"


def check_equilibrium_inertia(tensor: InertiaTensor, perturbation: dict) -> tuple[bool, str]:
    """
    Check if perturbation destabilizes equilibrium (ESS/replicator style).

    Returns (accepted, reason)
    """
    # Simplified: check if any state variable would exceed stability threshold
    for key, delta in perturbation.items():
        current = tensor.equilibrium_state.get(key, 0.5)
        new_val = current + delta
        if new_val < (1 - tensor.config.equilibrium_stability_threshold) or new_val > tensor.config.equilibrium_stability_threshold:
            return False, f"Equilibrium perturbation to {key} would destabilize (new={new_val:.3f})"

    return True, "Equilibrium perturbation within stability bounds"


# --- Bus Integration ---

def emit_bus(bus_dir: str | None, *, topic: str, kind: str, level: str, actor: str, data) -> None:
    """Emit event to agent bus."""
    if not bus_dir:
        return
    tool = Path(__file__).with_name("agent_bus.py")
    if not tool.exists():
        return
    subprocess.run(
        [
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
        ],
        check=False,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )


# --- File Operations ---

def find_kroma_root(start: Path) -> Path | None:
    """Find Kroma/Pluribus root directory."""
    cur = start.resolve()
    for cand in [cur, *cur.parents]:
        if (cand / ".pluribus" / "rhizome.json").exists():
            return cand
    return None


def load_tensor(root: Path) -> InertiaTensor:
    """Load inertia tensor from disk."""
    tensor_path = root / ".pluribus" / "kroma" / "inertia_tensor.json"
    if tensor_path.exists():
        data = json.loads(tensor_path.read_text(encoding="utf-8"))
        return InertiaTensor.from_dict(data)
    return InertiaTensor(config=InertiaConfig())


def save_tensor(root: Path, tensor: InertiaTensor) -> None:
    """Save inertia tensor to disk."""
    kroma_dir = root / ".pluribus" / "kroma"
    kroma_dir.mkdir(parents=True, exist_ok=True)
    tensor_path = kroma_dir / "inertia_tensor.json"
    tensor_path.write_text(json.dumps(tensor.to_dict(), indent=2, ensure_ascii=False), encoding="utf-8")


def append_update_log(root: Path, update: InertiaUpdate) -> None:
    """Append update to log."""
    kroma_dir = root / ".pluribus" / "kroma"
    kroma_dir.mkdir(parents=True, exist_ok=True)
    log_path = kroma_dir / "inertia_updates.ndjson"
    line = json.dumps(asdict(update), ensure_ascii=False) + "\n"
    with log_path.open("a", encoding="utf-8") as f:
        f.write(line)


# --- CLI Commands ---

def cmd_init(args: argparse.Namespace) -> int:
    """Initialize inertia tensor."""
    root = Path(args.root).expanduser().resolve() if args.root else find_kroma_root(Path.cwd())
    if not root:
        sys.stderr.write("Error: No Pluribus root found. Run from a pluribus workspace.\n")
        return 1

    config = InertiaConfig(
        policy_step_bound=args.policy_bound,
        memory_consolidation_lambda=args.memory_lambda,
        goal_stack_pop_cost=args.goal_pop_cost,
        control_smoothness_penalty=args.control_jerk_penalty,
        equilibrium_stability_threshold=args.equilibrium_stability,
    )
    tensor = InertiaTensor(config=config, last_update_ts=time.time())
    save_tensor(root, tensor)

    print(f"Initialized inertia tensor at {root / '.pluribus' / 'kroma' / 'inertia_tensor.json'}")

    bus_dir = args.bus_dir or os.environ.get("PLURIBUS_BUS_DIR")
    if bus_dir:
        emit_bus(bus_dir, topic="kroma.inertia.init", kind="event", level="info",
                 actor=os.environ.get("PLURIBUS_ACTOR", "kroma"), data=tensor.to_dict())

    return 0


def cmd_check(args: argparse.Namespace) -> int:
    """Check if proposed update violates inertia bounds."""
    root = Path(args.root).expanduser().resolve() if args.root else find_kroma_root(Path.cwd())
    if not root:
        sys.stderr.write("Error: No Pluribus root found.\n")
        return 1

    tensor = load_tensor(root)

    domain = args.domain
    delta = json.loads(args.delta)

    update = InertiaUpdate(
        update_id=str(uuid.uuid4()),
        domain=domain,
        delta=delta,
        magnitude=0.0,
        timestamp=time.time(),
    )

    if domain == "policy":
        kl = delta.get("kl_divergence", 0.0)
        new_hash = delta.get("policy_hash", "")
        accepted, reason = check_policy_inertia(tensor, new_hash, kl)
        update.magnitude = kl
    elif domain == "memory":
        param_id = delta.get("param_id", "")
        proposed_delta = delta.get("delta", 0.0)
        accepted, reason = check_memory_inertia(tensor, param_id, proposed_delta)
        update.magnitude = abs(proposed_delta)
    elif domain == "goal":
        action = delta.get("action", "")
        goal_id = delta.get("goal_id")
        accepted, reason = check_goal_inertia(tensor, action, goal_id)
    elif domain == "control":
        new_control = delta.get("control", {})
        accepted, jerk, reason = check_control_inertia(tensor, new_control)
        update.magnitude = jerk
    elif domain == "equilibrium":
        perturbation = delta.get("perturbation", {})
        accepted, reason = check_equilibrium_inertia(tensor, perturbation)
    else:
        sys.stderr.write(f"Unknown domain: {domain}\n")
        return 2

    update.accepted = accepted
    if not accepted:
        update.rejection_reason = reason

    append_update_log(root, update)

    result = {
        "update_id": update.update_id,
        "domain": domain,
        "accepted": accepted,
        "reason": reason,
        "magnitude": update.magnitude,
    }

    print(json.dumps(result, indent=2))

    bus_dir = args.bus_dir or os.environ.get("PLURIBUS_BUS_DIR")
    if bus_dir:
        emit_bus(bus_dir, topic=f"kroma.inertia.check.{domain}", kind="check",
                 level="info" if accepted else "warn",
                 actor=os.environ.get("PLURIBUS_ACTOR", "kroma"), data=result)

    return 0 if accepted else 1


def cmd_status(args: argparse.Namespace) -> int:
    """Show current inertia tensor status."""
    root = Path(args.root).expanduser().resolve() if args.root else find_kroma_root(Path.cwd())
    if not root:
        sys.stderr.write("Error: No Pluribus root found.\n")
        return 1

    tensor = load_tensor(root)
    print(json.dumps(tensor.to_dict(), indent=2))
    return 0


def cmd_apply(args: argparse.Namespace) -> int:
    """Apply an update to the inertia tensor (after check passed)."""
    root = Path(args.root).expanduser().resolve() if args.root else find_kroma_root(Path.cwd())
    if not root:
        sys.stderr.write("Error: No Pluribus root found.\n")
        return 1

    tensor = load_tensor(root)
    domain = args.domain
    delta = json.loads(args.delta)

    if domain == "policy":
        tensor.policy_hash = delta.get("policy_hash", tensor.policy_hash)
    elif domain == "memory":
        param_id = delta.get("param_id", "")
        importance = delta.get("importance", 0.0)
        tensor.memory_importance[param_id] = importance
    elif domain == "goal":
        action = delta.get("action", "")
        if action == "push":
            tensor.goal_stack.append(delta.get("goal", {}))
        elif action == "pop" and tensor.goal_stack:
            tensor.goal_stack.pop()
    elif domain == "control":
        new_control = delta.get("control", {})
        tensor.control_history.append({
            "ts": time.time(),
            "values": new_control,
        })
    elif domain == "equilibrium":
        for key, val in delta.get("perturbation", {}).items():
            current = tensor.equilibrium_state.get(key, 0.5)
            tensor.equilibrium_state[key] = current + val

    tensor.last_update_ts = time.time()
    save_tensor(root, tensor)

    print(f"Applied {domain} update")

    bus_dir = args.bus_dir or os.environ.get("PLURIBUS_BUS_DIR")
    if bus_dir:
        emit_bus(bus_dir, topic=f"kroma.inertia.apply.{domain}", kind="apply", level="info",
                 actor=os.environ.get("PLURIBUS_ACTOR", "kroma"), data={"domain": domain, "delta": delta})

    return 0


# --- Main ---

def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="kroma_inertia.py",
        description="Computational Inertia Tensor Management for Kroma.live",
    )
    p.add_argument("--root", default=None, help="Pluribus root directory")
    p.add_argument("--bus-dir", default=None, help="Agent bus directory")

    sub = p.add_subparsers(dest="cmd", required=True)

    # init
    init_p = sub.add_parser("init", help="Initialize inertia tensor")
    init_p.add_argument("--policy-bound", type=float, default=DEFAULT_POLICY_BOUND)
    init_p.add_argument("--memory-lambda", type=float, default=DEFAULT_MEMORY_LAMBDA)
    init_p.add_argument("--goal-pop-cost", type=float, default=DEFAULT_GOAL_POP_COST)
    init_p.add_argument("--control-jerk-penalty", type=float, default=DEFAULT_CONTROL_JERK_PENALTY)
    init_p.add_argument("--equilibrium-stability", type=float, default=DEFAULT_EQUILIBRIUM_STABILITY)
    init_p.set_defaults(func=cmd_init)

    # check
    check_p = sub.add_parser("check", help="Check if update violates inertia bounds")
    check_p.add_argument("domain", choices=["policy", "memory", "goal", "control", "equilibrium"])
    check_p.add_argument("delta", help="JSON delta to check")
    check_p.set_defaults(func=cmd_check)

    # status
    status_p = sub.add_parser("status", help="Show current inertia tensor")
    status_p.set_defaults(func=cmd_status)

    # apply
    apply_p = sub.add_parser("apply", help="Apply update to tensor")
    apply_p.add_argument("domain", choices=["policy", "memory", "goal", "control", "equilibrium"])
    apply_p.add_argument("delta", help="JSON delta to apply")
    apply_p.set_defaults(func=cmd_apply)

    return p


def main(argv: list[str]) -> int:
    args = build_parser().parse_args(argv)
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
