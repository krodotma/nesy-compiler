"""
ARK Continual Learning Engine

Phase 2.2 Implementation (P2-021 to P2-040)

Provides:
- Experience replay buffer with prioritization
- Pattern memory (successful commits)
- Anti-pattern memory (rejected commits)
- EWC-style forgetting prevention
- Online learning from commits
- Curriculum learning for gates
"""

from .replay_buffer import ReplayBuffer, Experience, PrioritizedReplayBuffer
from .pattern_memory import PatternMemory, Pattern, AntiPattern
from .ewc import ElasticWeightConsolidation, EWCConfig
from .curriculum import CurriculumLearning, DifficultyScheduler
from .checkpoint import CheckpointManager, ModelCheckpoint
