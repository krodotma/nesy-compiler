import unittest
import shutil
import tempfile
import os
import time
from nucleus.ark.cl.replay_buffer import PrioritizedReplayBuffer, Experience
from nucleus.ark.cl.curriculum import CurriculumLearning
from nucleus.ark.cl.checkpoint import CheckpointManager

class TestCLEngine(unittest.TestCase):
    def setUp(self):
        self.test_dir = tempfile.mkdtemp()
        os.environ["ARK_HOME"] = self.test_dir
        # Mock home for the test environment
        self.db_path = os.path.join(self.test_dir, "test_replay.db")

    def tearDown(self):
        shutil.rmtree(self.test_dir)

    def test_replay_buffer(self):
        """Test replay buffer operations."""
        buffer = PrioritizedReplayBuffer(capacity=10, db_path=self.db_path)
        
        # Add experiences
        for i in range(5):
            exp = Experience(
                commit_sha=f"sha_{i}",
                etymology="test",
                cmp_before=0.5,
                cmp_after=0.5 + (0.1 * i),
                entropy_before={},
                entropy_after={},
                gate_results={},
                success=True,
                timestamp=time.time()
            )
            buffer.add(exp)
            
        self.assertEqual(len(buffer), 5)
        
        # Sampling
        batch = buffer.sample_batch(3)
        self.assertEqual(len(batch), 3)

    def test_curriculum_progression(self):
        """Test curriculum level advancement."""
        # Use a temporary file for curriculum storage
        storage = os.path.join(self.test_dir, "curriculum.json")
        cl = CurriculumLearning(storage_path=storage)
        cl.reset()
        
        stats = cl.get_statistics()
        self.assertEqual(stats["current_level"], 0)
        
        # Simulate successes needed to advance (default beginner needs 5)
        for _ in range(10):
            cl.record_result(success=True, cmp=0.5, entropy_total=0.5)
            
        stats = cl.get_statistics()
        # Should have advanced from level 0 to 1
        self.assertEqual(stats["current_level"], 1)

    def test_checkpoint_manager(self):
        """Test checkpoint saving and loading."""
        manager = CheckpointManager()
        # Override storage path to test dir
        manager.storage_path = os.path.join(self.test_dir, "checkpoints.json")
        
        version = manager.save(
            parameters={"w": 1},
            fisher={},
            cmp_mean=0.8,
            success_rate=0.9,
            curriculum_level=2
        )
        
        self.assertIsNotNone(version)
        
        loaded = manager.load(version)
        self.assertIsNotNone(loaded)
        self.assertEqual(loaded.curriculum_level, 2)

if __name__ == '__main__':
    unittest.main()
