#!/usr/bin/env python3
"""
Neural subpackage for Code Agent.

Provides neural-guided code generation and proposal mechanisms.
"""

from .proposal_generator import NeuralCodeProposalGenerator, CodeProposal

__all__ = [
    "NeuralCodeProposalGenerator",
    "CodeProposal",
]
