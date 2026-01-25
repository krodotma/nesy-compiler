import json
from .learner import MetaLearner

def handle_feedback(event: dict):
    """Entry point for OHM to forward feedback events to MetaLearner."""
    learner = MetaLearner()
    learner.record_experience(event)
