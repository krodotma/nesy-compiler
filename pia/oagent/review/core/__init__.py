#!/usr/bin/env python3
"""
Review Agent Core Systems (Steps 161-170)

This module provides the core review systems:

- Step 161: OmegaVetoIntegration - Omega veto request handling
- Step 162: PRReviewAutomator - Pull request automation
- Step 163: CodeQualityScorer - Quality scoring
- Step 164: ReviewHistoryTracker - Review history tracking
- Step 165: DiffAnalyzer - Intelligent diff analysis
- Step 166: SuggestionEngine - Improvement suggestions
- Step 167: PriorityCalculator - Review priority calculation
- Step 168: AssignmentRouter - Review assignment routing
- Step 169: MergeBlocker - Merge blocking logic
- Step 170: ApprovalManager - Approval workflow management

Bus Topics:
- omega.veto.request
- review.pr.automate
- review.quality.score
- review.history.track
- review.diff.analyze
- review.suggestion.generate
- review.priority.calculate
- review.assignment.route
- review.merge.block
- review.approval.manage

Protocol: DKIN v30, CITIZEN v2, PAIP v16
"""

from .omega_veto import OmegaVetoIntegration, VetoRequest, VetoDecision, VetoResult
from .pr_automator import PRReviewAutomator, PRInfo, AutomatedReview
from .quality_scorer import CodeQualityScorer, QualityScore, QualityDimension
from .history_tracker import ReviewHistoryTracker, ReviewRecord, ReviewHistory
from .diff_analyzer import DiffAnalyzer, DiffHunk, DiffAnalysis
from .suggestion_engine import SuggestionEngine, Suggestion, SuggestionType
from .priority_calculator import PriorityCalculator, ReviewPriority, PriorityFactors
from .assignment_router import AssignmentRouter, AssignmentRule, ReviewAssignment
from .merge_blocker import MergeBlocker, BlockReason, MergeDecision
from .approval_manager import ApprovalManager, ApprovalState, ApprovalWorkflow

__all__ = [
    # Step 161
    "OmegaVetoIntegration",
    "VetoRequest",
    "VetoDecision",
    "VetoResult",
    # Step 162
    "PRReviewAutomator",
    "PRInfo",
    "AutomatedReview",
    # Step 163
    "CodeQualityScorer",
    "QualityScore",
    "QualityDimension",
    # Step 164
    "ReviewHistoryTracker",
    "ReviewRecord",
    "ReviewHistory",
    # Step 165
    "DiffAnalyzer",
    "DiffHunk",
    "DiffAnalysis",
    # Step 166
    "SuggestionEngine",
    "Suggestion",
    "SuggestionType",
    # Step 167
    "PriorityCalculator",
    "ReviewPriority",
    "PriorityFactors",
    # Step 168
    "AssignmentRouter",
    "AssignmentRule",
    "ReviewAssignment",
    # Step 169
    "MergeBlocker",
    "BlockReason",
    "MergeDecision",
    # Step 170
    "ApprovalManager",
    "ApprovalState",
    "ApprovalWorkflow",
]

__version__ = "0.1.0"
__step_range__ = "161-170"
