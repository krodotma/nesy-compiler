#!/usr/bin/env python3
"""
icrl_context.py - In-Context Reinforcement Learning

DUALITY-BIND E12: Agents learn within context without weight updates.

Ring: 1 (Operator)
Protocol: DKIN v29
"""

import argparse
import json
import math
import random
import sys
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
import os

ICRL_DIR = Path(os.environ.get("PLURIBUS_ICRL_DIR", ".pluribus/icrl"))


@dataclass
class ContextEntry:
    """An entry in the ICRL context."""
    step: int
    action: str
    observation: str
    reward: float
    done: bool = False
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ICRLContext:
    """Maintains action-observation history for in-context learning."""
    context_id: str
    entries: List[ContextEntry] = field(default_factory=list)
    total_reward: float = 0.0
    step_count: int = 0
    
    def add(self, action: str, observation: str, reward: float, done: bool = False):
        self.step_count += 1
        self.total_reward += reward
        self.entries.append(ContextEntry(
            step=self.step_count,
            action=action,
            observation=observation,
            reward=reward,
            done=done,
        ))
    
    def to_prompt_context(self, max_entries: int = 10) -> str:
        """Format as context for LLM prompt."""
        recent = self.entries[-max_entries:]
        lines = ["Previous actions and outcomes:"]
        for e in recent:
            reward_str = f"+{e.reward:.1f}" if e.reward >= 0 else f"{e.reward:.1f}"
            lines.append(f"  Step {e.step}: {e.action} -> {e.observation} ({reward_str})")
        lines.append(f"Total reward so far: {self.total_reward:.2f}")
        return "\n".join(lines)


class ICRLAgent:
    """In-Context Reinforcement Learning agent."""
    
    def __init__(self, icrl_dir: Path = None):
        self.icrl_dir = icrl_dir or ICRL_DIR
        self.icrl_dir.mkdir(parents=True, exist_ok=True)
        self.contexts: Dict[str, ICRLContext] = {}
    
    def get_or_create_context(self, context_id: str) -> ICRLContext:
        """Get or create a context."""
        if context_id not in self.contexts:
            self.contexts[context_id] = ICRLContext(context_id=context_id)
        return self.contexts[context_id]
    
    def thompson_sample(self, context: ICRLContext, actions: List[str]) -> str:
        """
        Thompson sampling for action selection based on context history.
        
        Uses Beta distribution where:
        - alpha = successes + 1
        - beta = failures + 1
        """
        if not actions:
            return ""
        
        # Count successes/failures per action
        action_stats = {a: {"success": 1, "fail": 1} for a in actions}
        
        for entry in context.entries:
            if entry.action in action_stats:
                if entry.reward > 0:
                    action_stats[entry.action]["success"] += 1
                else:
                    action_stats[entry.action]["fail"] += 1
        
        # Sample from Beta distribution
        samples = {}
        for action, stats in action_stats.items():
            samples[action] = random.betavariate(stats["success"], stats["fail"])
        
        # Return action with highest sample
        return max(samples, key=samples.get)
    
    def epsilon_greedy(self, context: ICRLContext, actions: List[str], epsilon: float = 0.1) -> str:
        """Epsilon-greedy action selection."""
        if not actions:
            return ""
        
        if random.random() < epsilon:
            return random.choice(actions)
        
        # Compute average reward per action
        action_rewards = {a: [] for a in actions}
        for entry in context.entries:
            if entry.action in action_rewards:
                action_rewards[entry.action].append(entry.reward)
        
        # Return action with highest average (or random if unexplored)
        best_action = None
        best_avg = float("-inf")
        for action, rewards in action_rewards.items():
            avg = sum(rewards) / len(rewards) if rewards else 0
            if avg > best_avg:
                best_avg = avg
                best_action = action
        
        return best_action or random.choice(actions)
    
    def ucb_select(self, context: ICRLContext, actions: List[str], c: float = 1.0) -> str:
        """Upper Confidence Bound action selection."""
        if not actions:
            return ""
        
        # Count and reward per action
        action_stats = {a: {"count": 0, "reward": 0.0} for a in actions}
        total_count = 0
        
        for entry in context.entries:
            if entry.action in action_stats:
                action_stats[entry.action]["count"] += 1
                action_stats[entry.action]["reward"] += entry.reward
                total_count += 1
        
        if total_count == 0:
            return random.choice(actions)
        
        # Compute UCB for each action
        ucb_scores = {}
        for action, stats in action_stats.items():
            if stats["count"] == 0:
                ucb_scores[action] = float("inf")  # Unexplored
            else:
                avg = stats["reward"] / stats["count"]
                exploration = c * math.sqrt(math.log(total_count) / stats["count"])
                ucb_scores[action] = avg + exploration
        
        return max(ucb_scores, key=ucb_scores.get)
    
    def save_context(self, context: ICRLContext):
        """Save context to disk."""
        path = self.icrl_dir / f"{context.context_id}.json"
        with open(path, "w") as f:
            data = asdict(context)
            json.dump(data, f)
    
    def load_context(self, context_id: str) -> Optional[ICRLContext]:
        """Load context from disk."""
        path = self.icrl_dir / f"{context_id}.json"
        if path.exists():
            data = json.loads(path.read_text())
            entries = [ContextEntry(**e) for e in data.pop("entries", [])]
            ctx = ICRLContext(**data)
            ctx.entries = entries
            return ctx
        return None


def main():
    parser = argparse.ArgumentParser(description="In-Context RL")
    subparsers = parser.add_subparsers(dest="command", required=True)
    
    p_add = subparsers.add_parser("add", help="Add entry")
    p_add.add_argument("--context", required=True)
    p_add.add_argument("--action", required=True)
    p_add.add_argument("--observation", required=True)
    p_add.add_argument("--reward", type=float, required=True)
    
    p_select = subparsers.add_parser("select", help="Select action")
    p_select.add_argument("--context", required=True)
    p_select.add_argument("--actions", nargs="+", required=True)
    p_select.add_argument("--method", default="thompson", choices=["thompson", "epsilon", "ucb"])
    
    p_show = subparsers.add_parser("show", help="Show context")
    p_show.add_argument("context_id")
    
    args = parser.parse_args()
    agent = ICRLAgent()
    
    if args.command == "add":
        ctx = agent.get_or_create_context(args.context)
        ctx.add(args.action, args.observation, args.reward)
        agent.save_context(ctx)
        print(f"✅ Added entry to {args.context}")
        return 0
    elif args.command == "select":
        ctx = agent.load_context(args.context) or agent.get_or_create_context(args.context)
        
        if args.method == "thompson":
            action = agent.thompson_sample(ctx, args.actions)
        elif args.method == "epsilon":
            action = agent.epsilon_greedy(ctx, args.actions)
        else:
            action = agent.ucb_select(ctx, args.actions)
        
        print(f"Selected action: {action}")
        return 0
    elif args.command == "show":
        ctx = agent.load_context(args.context_id)
        if ctx:
            print(ctx.to_prompt_context())
        else:
            print(f"No context: {args.context_id}")
        return 0


if __name__ == "__main__":
    sys.exit(main())
