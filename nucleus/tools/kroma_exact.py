#!/usr/bin/env python3
"""
kroma_exact.py — ExACT: Reflective MCTS + Exploratory Learning

Implements the ExACT framework for o1-like agentic applications:
- R-MCTS: Reflective Monte Carlo Tree Search with contrastive reflection
- Exploratory Learning: Tree traversal flattening for training
- VOR/MCTS integration: Physical-world causal navigation patterns

References:
- ExACT: https://agent-e3.github.io/ExACT/
- VOR navigation: Gauss-Schläfli-Hertz lineage
- MCTS: Coulom 2006, Silver et al. 2016

Axiom: ∀n ∃m [¬(m ≤ n) ∧ m ∈ X] — Infinity axiom for ω-automata
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import random
import subprocess
import sys
import time
import uuid
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Optional, Callable, List, Dict, Any

sys.dont_write_bytecode = True


# --- Constants ---

DEFAULT_UCT_C = 1.414  # UCT exploration constant (sqrt(2))
DEFAULT_MAX_DEPTH = 50
DEFAULT_SIMULATIONS = 100
DEFAULT_REFLECTION_THRESHOLD = 0.3


# --- Data Structures ---

@dataclass
class Node:
    """MCTS tree node."""
    id: str
    state: dict
    parent_id: Optional[str] = None
    action_from_parent: Optional[str] = None
    children: List[str] = field(default_factory=list)
    visits: int = 0
    value: float = 0.0
    is_terminal: bool = False
    is_accepting: bool = False  # ω-automata acceptance
    reflection: Optional[dict] = None


@dataclass
class Reflection:
    """Contrastive reflection result."""
    successes: List[dict]
    failures: List[dict]
    insights: List[str]
    policy_adjustments: dict


@dataclass
class MCTSConfig:
    """Configuration for R-MCTS."""
    uct_c: float = DEFAULT_UCT_C
    max_depth: int = DEFAULT_MAX_DEPTH
    simulations: int = DEFAULT_SIMULATIONS
    reflection_threshold: float = DEFAULT_REFLECTION_THRESHOLD
    inertia_bonus_weight: float = 0.1  # Kroma inertia integration
    multi_agent_debate: bool = False
    num_debate_agents: int = 3


@dataclass
class SearchResult:
    """Result of R-MCTS search."""
    best_action: str
    best_value: float
    visits: int
    tree_size: int
    reflections_used: int
    trajectory: List[dict]
    exploratory_trajectory: List[dict]  # For training


# --- R-MCTS Core ---

class RMCTS:
    """
    Reflective Monte Carlo Tree Search.

    Extends classical MCTS with:
    1. Contrastive reflection: Learn from past successes/failures
    2. Multi-agent debate: Collaborative state estimation
    3. Inertia bonus: Prefer actions consistent with policy history
    4. ω-acceptance: Track Büchi/Rabin acceptance during search
    """

    def __init__(self, config: MCTSConfig):
        self.config = config
        self.nodes: Dict[str, Node] = {}
        self.root_id: Optional[str] = None
        self.reflections: List[Reflection] = []
        self.policy_history: List[str] = []

    def create_node(self, state: dict, parent_id: Optional[str] = None,
                    action: Optional[str] = None) -> Node:
        """Create a new tree node."""
        node_id = hashlib.sha256(json.dumps(state, sort_keys=True).encode()).hexdigest()[:16]
        node = Node(
            id=node_id,
            state=state,
            parent_id=parent_id,
            action_from_parent=action,
        )
        self.nodes[node_id] = node
        if parent_id and parent_id in self.nodes:
            self.nodes[parent_id].children.append(node_id)
        return node

    def uct_score(self, node: Node, parent_visits: int) -> float:
        """
        UCT selection score with inertia bonus.

        UCT = Q/N + C * sqrt(ln(parent_N) / N) + inertia_bonus
        """
        if node.visits == 0:
            return float('inf')

        exploitation = node.value / node.visits
        exploration = self.config.uct_c * math.sqrt(math.log(parent_visits) / node.visits)

        # Inertia bonus: prefer actions similar to recent policy
        inertia_bonus = 0.0
        if node.action_from_parent and self.policy_history:
            # Simple heuristic: bonus if action matches recent actions
            recent_actions = self.policy_history[-5:]
            if node.action_from_parent in recent_actions:
                inertia_bonus = self.config.inertia_bonus_weight

        return exploitation + exploration + inertia_bonus

    def select(self, node_id: str) -> str:
        """Select best child according to UCT."""
        node = self.nodes[node_id]
        if not node.children:
            return node_id

        best_score = float('-inf')
        best_child_id = node.children[0]

        for child_id in node.children:
            child = self.nodes[child_id]
            score = self.uct_score(child, node.visits)
            if score > best_score:
                best_score = score
                best_child_id = child_id

        return best_child_id

    def expand(self, node_id: str, actions: List[str], next_state_fn: Callable) -> List[str]:
        """Expand node with all possible actions."""
        node = self.nodes[node_id]
        new_children = []

        for action in actions:
            next_state = next_state_fn(node.state, action)
            child = self.create_node(next_state, parent_id=node_id, action=action)
            new_children.append(child.id)

        return new_children

    def simulate(self, state: dict, rollout_policy: Callable, value_fn: Callable,
                 max_steps: int = 50) -> float:
        """
        Run simulation from state.

        With multi-agent debate if enabled.
        """
        if self.config.multi_agent_debate:
            # Multi-agent debate for value estimation
            estimates = []
            for _ in range(self.config.num_debate_agents):
                # Each "agent" runs independent rollout with noise
                current_state = dict(state)
                total_reward = 0.0
                for step in range(max_steps):
                    action = rollout_policy(current_state)
                    reward, next_state, done = value_fn(current_state, action)
                    total_reward += reward * (0.99 ** step)  # Discount
                    if done:
                        break
                    current_state = next_state
                estimates.append(total_reward)

            # Consensus via weighted median (robust to outliers)
            estimates.sort()
            return estimates[len(estimates) // 2]
        else:
            # Single-agent rollout
            current_state = dict(state)
            total_reward = 0.0
            for step in range(max_steps):
                action = rollout_policy(current_state)
                reward, next_state, done = value_fn(current_state, action)
                total_reward += reward * (0.99 ** step)
                if done:
                    break
                current_state = next_state
            return total_reward

    def backpropagate(self, node_id: str, value: float) -> None:
        """Backpropagate value up the tree."""
        current_id = node_id
        while current_id is not None:
            node = self.nodes[current_id]
            node.visits += 1
            node.value += value
            current_id = node.parent_id

    def reflect(self, trajectory: List[dict]) -> Reflection:
        """
        Contrastive reflection: analyze trajectory for insights.

        Identifies what worked vs what failed.
        """
        successes = [s for s in trajectory if s.get("reward", 0) > self.config.reflection_threshold]
        failures = [s for s in trajectory if s.get("reward", 0) <= self.config.reflection_threshold]

        insights = []
        policy_adjustments = {}

        # Generate insights
        if successes:
            success_actions = [s.get("action") for s in successes if s.get("action")]
            insights.append(f"Successful actions: {success_actions[:3]}")
            for action in success_actions[:3]:
                policy_adjustments[action] = policy_adjustments.get(action, 0) + 0.1

        if failures:
            failure_actions = [s.get("action") for s in failures if s.get("action")]
            insights.append(f"Failed actions to avoid: {failure_actions[:3]}")
            for action in failure_actions[:3]:
                policy_adjustments[action] = policy_adjustments.get(action, 0) - 0.1

        reflection = Reflection(
            successes=successes,
            failures=failures,
            insights=insights,
            policy_adjustments=policy_adjustments,
        )
        self.reflections.append(reflection)
        return reflection

    def search(self, root_state: dict, actions_fn: Callable, next_state_fn: Callable,
               rollout_policy: Callable, value_fn: Callable,
               terminal_fn: Callable, accepting_fn: Callable) -> SearchResult:
        """
        Run R-MCTS search.

        Args:
            root_state: Initial state
            actions_fn: Returns available actions for a state
            next_state_fn: Returns next state given state and action
            rollout_policy: Policy for simulation rollouts
            value_fn: Returns (reward, next_state, done) for state-action
            terminal_fn: Returns True if state is terminal
            accepting_fn: Returns True if state satisfies ω-acceptance
        """
        # Initialize root
        root = self.create_node(root_state)
        self.root_id = root.id

        trajectory = []
        exploratory_trajectory = []  # For training
        reflections_used = 0

        for sim in range(self.config.simulations):
            # Selection: traverse tree using UCT
            current_id = self.root_id
            path = [current_id]

            while self.nodes[current_id].children:
                current_id = self.select(current_id)
                path.append(current_id)

            current_node = self.nodes[current_id]

            # Check terminal/accepting
            if terminal_fn(current_node.state):
                current_node.is_terminal = True
                if accepting_fn(current_node.state):
                    current_node.is_accepting = True

            if not current_node.is_terminal:
                # Expansion
                actions = actions_fn(current_node.state)
                if actions:
                    new_children = self.expand(current_id, actions, next_state_fn)
                    if new_children:
                        # Select one child for simulation
                        current_id = random.choice(new_children)
                        path.append(current_id)

            # Simulation
            current_node = self.nodes[current_id]
            value = self.simulate(current_node.state, rollout_policy, value_fn)

            # Backpropagation
            self.backpropagate(current_id, value)

            # Record for trajectory
            step_record = {
                "sim": sim,
                "state": current_node.state,
                "action": current_node.action_from_parent,
                "value": value,
                "reward": value,  # For reflection
                "path_length": len(path),
            }
            trajectory.append(step_record)

            # Exploratory trajectory includes backtracking
            for node_id in path:
                node = self.nodes[node_id]
                exploratory_trajectory.append({
                    "state": node.state,
                    "action": node.action_from_parent,
                    "visits": node.visits,
                    "value": node.value,
                })

            # Periodic reflection
            if sim > 0 and sim % 10 == 0:
                recent_trajectory = trajectory[-10:]
                reflection = self.reflect(recent_trajectory)
                reflections_used += 1

                # Apply reflection insights to policy history
                for action, adjustment in reflection.policy_adjustments.items():
                    if adjustment > 0:
                        self.policy_history.append(action)

        # Select best action from root
        root_node = self.nodes[self.root_id]
        best_child_id = None
        best_visits = -1

        for child_id in root_node.children:
            child = self.nodes[child_id]
            if child.visits > best_visits:
                best_visits = child.visits
                best_child_id = child_id

        best_action = ""
        best_value = 0.0
        if best_child_id:
            best_node = self.nodes[best_child_id]
            best_action = best_node.action_from_parent or ""
            best_value = best_node.value / best_node.visits if best_node.visits > 0 else 0.0

        return SearchResult(
            best_action=best_action,
            best_value=best_value,
            visits=root_node.visits,
            tree_size=len(self.nodes),
            reflections_used=reflections_used,
            trajectory=trajectory,
            exploratory_trajectory=exploratory_trajectory,
        )


# --- VOR/MCTS Integration ---

@dataclass
class VORState:
    """
    VOR navigation state (Gauss-Schläfli-Hertz lineage).

    Maps physical navigation to MCTS:
    - bearing: current heading (radians)
    - cdi: course deviation indicator (-1 to 1)
    - radial: target radial (degrees)
    - wind_correction: applied correction
    """
    bearing: float
    cdi: float
    radial: float
    wind_correction: float
    position: tuple  # (x, y) in abstract space


def vor_to_mcts_state(vor: VORState) -> dict:
    """Convert VOR navigation state to MCTS state dict."""
    return {
        "bearing": vor.bearing,
        "cdi": vor.cdi,
        "radial": vor.radial,
        "wind_correction": vor.wind_correction,
        "position": list(vor.position),
    }


def vor_actions() -> List[str]:
    """Available VOR navigation actions."""
    return [
        "maintain",      # Hold current heading
        "correct_left",  # Apply left correction
        "correct_right", # Apply right correction
        "increase_correction",  # Increase wind correction
        "decrease_correction",  # Decrease wind correction
    ]


def vor_transition(state: dict, action: str) -> dict:
    """Simulate VOR state transition."""
    new_state = dict(state)
    cdi = state["cdi"]
    wind_correction = state["wind_correction"]

    if action == "correct_left":
        new_state["bearing"] = (state["bearing"] - 0.1) % (2 * math.pi)
        new_state["cdi"] = max(-1, cdi - 0.1)
    elif action == "correct_right":
        new_state["bearing"] = (state["bearing"] + 0.1) % (2 * math.pi)
        new_state["cdi"] = min(1, cdi + 0.1)
    elif action == "increase_correction":
        new_state["wind_correction"] = min(0.5, wind_correction + 0.05)
    elif action == "decrease_correction":
        new_state["wind_correction"] = max(-0.5, wind_correction - 0.05)

    # Simulate CDI drift based on wind
    new_state["cdi"] += random.gauss(0, 0.05) - new_state["wind_correction"] * 0.1

    return new_state


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


def save_search_result(root: Path, result: SearchResult) -> str:
    """Save search result to disk."""
    kroma_dir = root / ".pluribus" / "kroma" / "searches"
    kroma_dir.mkdir(parents=True, exist_ok=True)

    result_id = str(uuid.uuid4())[:8]
    result_path = kroma_dir / f"search_{result_id}.json"

    data = {
        "result_id": result_id,
        "timestamp": time.time(),
        "best_action": result.best_action,
        "best_value": result.best_value,
        "visits": result.visits,
        "tree_size": result.tree_size,
        "reflections_used": result.reflections_used,
        "trajectory_length": len(result.trajectory),
        "exploratory_trajectory_length": len(result.exploratory_trajectory),
    }

    result_path.write_text(json.dumps(data, indent=2), encoding="utf-8")

    # Save full exploratory trajectory for training
    trajectory_path = kroma_dir / f"exploratory_{result_id}.ndjson"
    with trajectory_path.open("w", encoding="utf-8") as f:
        for step in result.exploratory_trajectory:
            f.write(json.dumps(step) + "\n")

    return result_id


# --- CLI Commands ---

def cmd_demo(args: argparse.Namespace) -> int:
    """Run VOR/MCTS demo."""
    print("Running VOR/MCTS navigation demo...")

    config = MCTSConfig(
        simulations=args.simulations,
        multi_agent_debate=args.debate,
    )
    mcts = RMCTS(config)

    # Initial VOR state
    initial_state = {
        "bearing": 0.0,
        "cdi": 0.3,  # Slightly off course
        "radial": 90,
        "wind_correction": 0.0,
        "position": [0, 0],
    }

    # Define MCTS functions
    def actions_fn(state):
        return vor_actions()

    def next_state_fn(state, action):
        return vor_transition(state, action)

    def rollout_policy(state):
        return random.choice(vor_actions())

    def value_fn(state, action):
        next_state = vor_transition(state, action)
        # Reward: negative of CDI deviation (want CDI = 0)
        reward = -abs(next_state["cdi"])
        done = abs(next_state["cdi"]) < 0.05
        return reward, next_state, done

    def terminal_fn(state):
        return abs(state["cdi"]) < 0.05

    def accepting_fn(state):
        # ω-acceptance: on course (Büchi condition)
        return abs(state["cdi"]) < 0.1

    result = mcts.search(
        initial_state,
        actions_fn,
        next_state_fn,
        rollout_policy,
        value_fn,
        terminal_fn,
        accepting_fn,
    )

    print(f"\nSearch complete:")
    print(f"  Best action: {result.best_action}")
    print(f"  Best value: {result.best_value:.4f}")
    print(f"  Total visits: {result.visits}")
    print(f"  Tree size: {result.tree_size}")
    print(f"  Reflections used: {result.reflections_used}")
    print(f"  Exploratory trajectory: {len(result.exploratory_trajectory)} steps")

    root = find_kroma_root(Path.cwd())
    if root:
        result_id = save_search_result(root, result)
        print(f"  Saved as: search_{result_id}")

    bus_dir = args.bus_dir or os.environ.get("PLURIBUS_BUS_DIR")
    if bus_dir:
        emit_bus(bus_dir, topic="kroma.exact.demo", kind="result", level="info",
                 actor=os.environ.get("PLURIBUS_ACTOR", "kroma"),
                 data={"best_action": result.best_action, "best_value": result.best_value})

    return 0


def cmd_search(args: argparse.Namespace) -> int:
    """Run R-MCTS search on provided state."""
    root = find_kroma_root(Path.cwd())
    if not root:
        sys.stderr.write("Error: No Pluribus root found.\n")
        return 1

    state = json.loads(args.state)
    actions = args.actions.split(",") if args.actions else ["action1", "action2", "action3"]

    config = MCTSConfig(
        simulations=args.simulations,
        multi_agent_debate=args.debate,
    )
    mcts = RMCTS(config)

    # Simple default functions (override for real use)
    def actions_fn(s):
        return actions

    def next_state_fn(s, a):
        new_s = dict(s)
        new_s["last_action"] = a
        new_s["step"] = s.get("step", 0) + 1
        return new_s

    def rollout_policy(s):
        return random.choice(actions)

    def value_fn(s, a):
        return random.random(), next_state_fn(s, a), s.get("step", 0) > 10

    def terminal_fn(s):
        return s.get("step", 0) > 10

    def accepting_fn(s):
        return s.get("step", 0) > 5

    result = mcts.search(
        state,
        actions_fn,
        next_state_fn,
        rollout_policy,
        value_fn,
        terminal_fn,
        accepting_fn,
    )

    output = {
        "best_action": result.best_action,
        "best_value": result.best_value,
        "visits": result.visits,
        "tree_size": result.tree_size,
        "reflections_used": result.reflections_used,
    }

    print(json.dumps(output, indent=2))

    result_id = save_search_result(root, result)
    print(f"\nSaved as: search_{result_id}")

    return 0


# --- Main ---

def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="kroma_exact.py",
        description="ExACT: Reflective MCTS + Exploratory Learning for Kroma.live",
    )
    p.add_argument("--bus-dir", default=None, help="Agent bus directory")

    sub = p.add_subparsers(dest="cmd", required=True)

    # demo
    demo_p = sub.add_parser("demo", help="Run VOR/MCTS navigation demo")
    demo_p.add_argument("--simulations", type=int, default=50)
    demo_p.add_argument("--debate", action="store_true", help="Enable multi-agent debate")
    demo_p.set_defaults(func=cmd_demo)

    # search
    search_p = sub.add_parser("search", help="Run R-MCTS search")
    search_p.add_argument("state", help="Initial state as JSON")
    search_p.add_argument("--actions", default=None, help="Comma-separated actions")
    search_p.add_argument("--simulations", type=int, default=100)
    search_p.add_argument("--debate", action="store_true", help="Enable multi-agent debate")
    search_p.set_defaults(func=cmd_search)

    return p


def main(argv: list[str]) -> int:
    args = build_parser().parse_args(argv)
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
