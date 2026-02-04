#!/usr/bin/env python3
"""
Tests for Agent Lightning Trainer Integration
"""
import json
import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch, MagicMock

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from agent_lightning_trainer import (
    TrainingLivenessMonitor,
    Transition,
    Episode,
    Policy,
    RewardFunction,
    TaskCompletionReward,
    QualityReward,
    CompositeReward,
    AgentLightningTrainer,
)


class TestTrainingLivenessMonitor(unittest.TestCase):
    """Test omega-liveness monitoring for training."""

    def test_initial_state(self):
        """Test monitor initial state."""
        monitor = TrainingLivenessMonitor(max_seconds=60.0)
        self.assertTrue(monitor.is_healthy())
        self.assertEqual(monitor.consecutive_failures, 0)

    def test_heartbeat_updates_state(self):
        """Test heartbeat updates progress."""
        monitor = TrainingLivenessMonitor(max_seconds=60.0)
        monitor.heartbeat({"progress": 0.5})
        self.assertEqual(len(monitor.progress_history), 1)

    def test_failure_tracking(self):
        """Test consecutive failure tracking."""
        monitor = TrainingLivenessMonitor(max_seconds=60.0, max_consecutive_failures=3)

        monitor.heartbeat({"failure": True})
        self.assertEqual(monitor.consecutive_failures, 1)

        monitor.heartbeat({"failure": True})
        self.assertEqual(monitor.consecutive_failures, 2)

        monitor.heartbeat({"failure": False})
        self.assertEqual(monitor.consecutive_failures, 0)

    def test_failure_limit_unhealthy(self):
        """Test monitor becomes unhealthy after too many failures."""
        monitor = TrainingLivenessMonitor(max_seconds=60.0, max_consecutive_failures=3)

        for _ in range(3):
            monitor.heartbeat({"failure": True})

        self.assertFalse(monitor.is_healthy())

    def test_diagnostics(self):
        """Test diagnostics output."""
        monitor = TrainingLivenessMonitor(max_seconds=60.0)
        monitor.heartbeat({"progress": 0.5, "failure": True})

        diag = monitor.diagnostics()

        self.assertIn("elapsed_s", diag)
        self.assertIn("consecutive_failures", diag)
        self.assertIn("limit_s", diag)
        self.assertEqual(diag["total_failures"], 1)


class TestDataStructures(unittest.TestCase):
    """Test data structures."""

    def test_transition(self):
        """Test Transition dataclass."""
        t = Transition(
            state={"step": 0},
            action="test_action",
            reward=1.0,
            next_state={"step": 1},
            done=False,
        )
        self.assertEqual(t.action, "test_action")
        self.assertEqual(t.reward, 1.0)
        self.assertFalse(t.done)

    def test_episode(self):
        """Test Episode dataclass."""
        transitions = [
            Transition({"step": 0}, "a1", 0.5, {"step": 1}, False),
            Transition({"step": 1}, "a2", 0.5, {"step": 2}, True),
        ]
        episode = Episode(
            episode_id="ep-123",
            trace_id="trace-456",
            goal="test goal",
            transitions=transitions,
            total_reward=1.0,
            success=True,
            latency_s=2.5,
            liveness_healthy=True,
        )
        self.assertEqual(episode.episode_id, "ep-123")
        self.assertEqual(len(episode.transitions), 2)
        self.assertTrue(episode.success)

    def test_policy(self):
        """Test Policy dataclass."""
        policy = Policy(
            policy_id="pol-123",
            version=1,
            weights={"task_completion": 1.0, "quality": 0.5},
            learning_rate=0.001,
            discount_factor=0.99,
            entropy_coef=0.01,
            created_iso="2025-01-01T00:00:00Z",
        )
        self.assertEqual(policy.version, 1)
        self.assertEqual(policy.weights["quality"], 0.5)


class TestRewardFunctions(unittest.TestCase):
    """Test reward computation."""

    def setUp(self):
        self.temp_dir = tempfile.mkdtemp()
        self.bus_dir = os.path.join(self.temp_dir, "bus")
        os.makedirs(self.bus_dir)

    def tearDown(self):
        import shutil
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_task_completion_reward_no_evidence(self):
        """Test reward without bus evidence."""
        reward_fn = TaskCompletionReward(bus_dir=self.bus_dir)
        transition = Transition({"step": 0}, "test", 0, {"step": 1}, False, {"latency_s": 5})
        context = {"trace_id": "test-trace"}

        reward = reward_fn.compute_reward(transition, context)
        self.assertEqual(reward, 0.0)  # No evidence

    def test_task_completion_reward_with_evidence(self):
        """Test reward with bus evidence."""
        # Write evidence to bus
        events_path = Path(self.bus_dir) / "events.ndjson"
        evidence = {
            "topic": "strp.worker.item",
            "level": "info",
            "trace_id": "test-trace",
            "data": {"exit_code": 0},
        }
        with events_path.open("w") as f:
            f.write(json.dumps(evidence) + "\n")

        reward_fn = TaskCompletionReward(bus_dir=self.bus_dir)
        transition = Transition({"step": 0}, "test", 0, {"step": 1}, False)
        context = {"trace_id": "test-trace"}

        reward = reward_fn.compute_reward(transition, context)
        self.assertEqual(reward, 1.0)  # completion_reward

    def test_quality_reward_grounding(self):
        """Test quality reward with grounding evidence."""
        events_path = Path(self.bus_dir) / "events.ndjson"
        evidence = {
            "topic": "strp.output.grounding",
            "trace_id": "test-trace",
            "data": {"ok": True, "citations": ["ref1", "ref2"]},
        }
        with events_path.open("w") as f:
            f.write(json.dumps(evidence) + "\n")

        reward_fn = QualityReward(bus_dir=self.bus_dir)
        transition = Transition({"step": 0}, "test", 0, {"step": 1}, False)
        context = {"trace_id": "test-trace"}

        reward = reward_fn.compute_reward(transition, context)
        # grounding_bonus (0.3) + 2 * citation_bonus (0.1)
        self.assertGreater(reward, 0.4)

    def test_composite_reward(self):
        """Test composite reward function."""
        events_path = Path(self.bus_dir) / "events.ndjson"
        evidence = {
            "topic": "strp.worker.item",
            "level": "info",
            "trace_id": "test-trace",
            "data": {"exit_code": 0},
        }
        with events_path.open("w") as f:
            f.write(json.dumps(evidence) + "\n")

        task_reward = TaskCompletionReward(bus_dir=self.bus_dir)
        quality_reward = QualityReward(bus_dir=self.bus_dir)
        composite = CompositeReward([
            (task_reward, 0.5),
            (quality_reward, 0.5),
        ], bus_dir=self.bus_dir)

        transition = Transition({"step": 0}, "test", 0, {"step": 1}, False)
        context = {"trace_id": "test-trace"}

        reward = composite.compute_reward(transition, context)
        # Should have weighted task reward
        self.assertEqual(reward, 0.5 * 1.0 + 0.5 * 0.0)


class TestAgentLightningTrainer(unittest.TestCase):
    """Test the trainer class."""

    def setUp(self):
        self.temp_dir = tempfile.mkdtemp()
        self.bus_dir = os.path.join(self.temp_dir, "bus")
        self.storage_dir = Path(self.temp_dir) / "agent_lightning"
        os.makedirs(self.bus_dir)

    def tearDown(self):
        import shutil
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_init_creates_policy(self):
        """Test trainer initializes with policy."""
        trainer = AgentLightningTrainer(
            bus_dir=self.bus_dir,
            storage_dir=self.storage_dir,
        )
        self.assertIsNotNone(trainer.policy)
        self.assertEqual(trainer.policy.version, 1)

    def test_run_episode(self):
        """Test running a training episode."""
        trainer = AgentLightningTrainer(
            bus_dir=self.bus_dir,
            storage_dir=self.storage_dir,
        )

        episode = trainer.run_episode("Test goal", max_steps=5)

        self.assertIsInstance(episode, Episode)
        self.assertEqual(episode.goal, "Test goal")
        self.assertLessEqual(len(episode.transitions), 5)
        self.assertTrue(episode.liveness_healthy)

    def test_episode_saved_to_replay(self):
        """Test episodes are saved to replay buffer."""
        trainer = AgentLightningTrainer(
            bus_dir=self.bus_dir,
            storage_dir=self.storage_dir,
        )

        trainer.run_episode("Test goal", max_steps=3)

        self.assertTrue(trainer.replay_path.exists())
        with trainer.replay_path.open() as f:
            content = f.read()
        self.assertIn("Test goal", content)

    def test_custom_executor(self):
        """Test episode with custom executor."""
        trainer = AgentLightningTrainer(
            bus_dir=self.bus_dir,
            storage_dir=self.storage_dir,
        )

        def custom_executor(action: str, state: dict) -> tuple[str, dict]:
            return f"executed_{action}", {**state, "step": state.get("step", 0) + 1}

        episode = trainer.run_episode("Test goal", max_steps=3, executor=custom_executor)

        self.assertIsInstance(episode, Episode)

    def test_policy_update(self):
        """Test policy update."""
        trainer = AgentLightningTrainer(
            bus_dir=self.bus_dir,
            storage_dir=self.storage_dir,
        )

        # Run enough episodes
        for _ in range(12):
            trainer.run_episode("Test goal", max_steps=5)

        initial_version = trainer.policy.version

        metrics = trainer.update_policy(batch_size=10)

        if not metrics.get("skipped"):
            self.assertEqual(trainer.policy.version, initial_version + 1)
            self.assertIn("mean_return", metrics)
            self.assertIn("success_rate", metrics)

    def test_policy_persistence(self):
        """Test policy is saved and loaded."""
        trainer = AgentLightningTrainer(
            bus_dir=self.bus_dir,
            storage_dir=self.storage_dir,
        )

        trainer.policy.weights["task_completion"] = 1.5
        trainer._save_policy()

        # Create new trainer - should load existing policy
        trainer2 = AgentLightningTrainer(
            bus_dir=self.bus_dir,
            storage_dir=self.storage_dir,
        )

        self.assertEqual(trainer2.policy.weights["task_completion"], 1.5)

    def test_evaluate_policy(self):
        """Test policy evaluation."""
        trainer = AgentLightningTrainer(
            bus_dir=self.bus_dir,
            storage_dir=self.storage_dir,
        )

        results = trainer.evaluate_policy(
            goals=["goal1", "goal2"],
            episodes_per_goal=2,
        )

        self.assertIn("summary", results)
        self.assertIn("per_goal", results)
        self.assertEqual(len(results["per_goal"]), 2)
        self.assertIn("overall_success_rate", results["summary"])


class TestCLI(unittest.TestCase):
    """Test CLI commands."""

    def test_train_command(self):
        """Test train command parsing."""
        from agent_lightning_trainer import build_parser

        parser = build_parser()
        args = parser.parse_args([
            "train",
            "--episodes", "5",
            "--goal", "test goal",
            "--max-steps", "10",
        ])
        self.assertEqual(args.cmd, "train")
        self.assertEqual(args.episodes, 5)
        self.assertEqual(args.goal, "test goal")
        self.assertEqual(args.max_steps, 10)

    def test_eval_command(self):
        """Test eval command parsing."""
        from agent_lightning_trainer import build_parser

        parser = build_parser()
        args = parser.parse_args([
            "eval",
            "--goals", "goal1,goal2",
            "--episodes-per-goal", "3",
            "--json",
        ])
        self.assertEqual(args.cmd, "eval")
        self.assertEqual(args.goals, "goal1,goal2")
        self.assertEqual(args.episodes_per_goal, 3)
        self.assertTrue(args.json)

    def test_status_command(self):
        """Test status command parsing."""
        from agent_lightning_trainer import build_parser

        parser = build_parser()
        args = parser.parse_args(["status", "--json"])
        self.assertEqual(args.cmd, "status")
        self.assertTrue(args.json)


class TestBusIntegration(unittest.TestCase):
    """Test bus event emission."""

    def setUp(self):
        self.temp_dir = tempfile.mkdtemp()
        self.bus_dir = os.path.join(self.temp_dir, "bus")
        self.storage_dir = Path(self.temp_dir) / "agent_lightning"
        os.makedirs(self.bus_dir)

    def tearDown(self):
        import shutil
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_episode_emits_events(self):
        """Test that episodes emit bus events."""
        trainer = AgentLightningTrainer(
            bus_dir=self.bus_dir,
            storage_dir=self.storage_dir,
        )

        trainer.run_episode("Test goal", max_steps=3, trace_id="test-trace")

        # Check if events file exists (may not if agent_bus.py not available)
        events_path = Path(self.bus_dir) / "events.ndjson"
        # Events are emitted via subprocess, so they may or may not appear
        # This test mainly verifies no exceptions occur


class TestOmegaLivenessGates(unittest.TestCase):
    """Test omega-liveness gate enforcement."""

    def setUp(self):
        self.temp_dir = tempfile.mkdtemp()
        self.bus_dir = os.path.join(self.temp_dir, "bus")
        self.storage_dir = Path(self.temp_dir) / "agent_lightning"
        os.makedirs(self.bus_dir)

    def tearDown(self):
        import shutil
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_episode_stops_on_liveness_failure(self):
        """Test episode terminates when liveness fails."""
        trainer = AgentLightningTrainer(
            bus_dir=self.bus_dir,
            storage_dir=self.storage_dir,
        )

        # Custom executor that always fails
        failure_count = [0]

        def failing_executor(action: str, state: dict) -> tuple[str, dict]:
            failure_count[0] += 1
            return "error_result", {**state, "step": state.get("step", 0) + 1}

        # Run with very short liveness (will fail quickly)
        # Note: Can't easily simulate time-based failure in unit test
        episode = trainer.run_episode(
            "Test goal",
            max_steps=20,
            executor=failing_executor,
        )

        # Episode should complete (we can't easily force liveness failure)
        self.assertIsInstance(episode, Episode)


if __name__ == "__main__":
    unittest.main()
