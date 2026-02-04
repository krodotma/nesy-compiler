#!/usr/bin/env python3
"""
proposal_generator.py - Neural Code Proposal Generator (Step 52)

PBTSO Phase: PLAN

Provides:
- LLM-guided code proposal generation
- Context retrieval from Research Agent
- Multi-edit proposal bundling
- Confidence scoring for proposals

Bus Topics:
- code.proposal.generate
- a2a.research.query
- code.proposal.ready

Protocol: DKIN v30, CITIZEN v2
"""

from __future__ import annotations

import asyncio
import json
import os
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Protocol


# =============================================================================
# Types
# =============================================================================

@dataclass
class CodeEdit:
    """Represents a single code edit operation."""
    path: str
    operation: str  # insert, replace, delete, append
    location: str  # line number, function name, or "end"
    old_content: Optional[str] = None
    new_content: Optional[str] = None
    description: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "path": self.path,
            "operation": self.operation,
            "location": self.location,
            "old_content": self.old_content,
            "new_content": self.new_content,
            "description": self.description,
        }


@dataclass
class CodeProposal:
    """
    A bundled code modification proposal.

    Contains all edits needed to implement a task,
    with confidence scoring and affected file tracking.
    """
    id: str
    description: str
    files_affected: List[str]
    estimated_lines: int
    confidence: float
    edits: List[CodeEdit]
    created_at: float = field(default_factory=time.time)
    research_context: Optional[Dict[str, Any]] = None
    rationale: str = ""
    risks: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "description": self.description,
            "files_affected": self.files_affected,
            "estimated_lines": self.estimated_lines,
            "confidence": self.confidence,
            "edits": [e.to_dict() for e in self.edits],
            "created_at": self.created_at,
            "research_context": self.research_context,
            "rationale": self.rationale,
            "risks": self.risks,
        }


# =============================================================================
# Research Agent Client Protocol
# =============================================================================

class ResearchAgentClient(Protocol):
    """Protocol for Research Agent communication."""

    async def get_context(self, query: str) -> Dict[str, Any]:
        """Get relevant context for a query."""
        ...

    async def search_codebase(self, pattern: str) -> List[Dict[str, Any]]:
        """Search codebase for pattern matches."""
        ...


class DefaultResearchClient:
    """
    Default Research Agent client using bus communication.

    Falls back to local file analysis if Research Agent unavailable.
    """

    def __init__(self, bus: Optional[Any] = None):
        self.bus = bus
        self._cache: Dict[str, Dict[str, Any]] = {}

    async def get_context(self, query: str) -> Dict[str, Any]:
        """Get context via Research Agent or local analysis."""
        # Check cache first
        cache_key = query[:100]
        if cache_key in self._cache:
            return self._cache[cache_key]

        # Emit query to bus (Research Agent will respond async)
        if self.bus:
            self.bus.emit({
                "topic": "a2a.research.query",
                "kind": "query",
                "actor": "code-agent",
                "data": {
                    "query": query,
                    "context_type": "code_generation",
                    "max_results": 10,
                }
            })

        # Return minimal context (real implementation would wait for response)
        context = {
            "query": query,
            "relevant_files": [],
            "similar_code": [],
            "documentation": [],
            "timestamp": time.time(),
        }

        self._cache[cache_key] = context
        return context

    async def search_codebase(self, pattern: str) -> List[Dict[str, Any]]:
        """Search codebase for pattern matches."""
        # Placeholder - real implementation would use grep/ripgrep
        return []


# =============================================================================
# Neural Code Proposal Generator
# =============================================================================

class NeuralCodeProposalGenerator:
    """
    Generate code proposals using neural models.

    PBTSO Phase: PLAN

    Responsibilities:
    - Generate structured code modification proposals
    - Coordinate with Research Agent for context
    - Score proposals by confidence
    - Bundle related edits into atomic proposals

    The generator uses a template-based approach for now,
    with hooks for LLM integration when available.
    """

    BUS_TOPICS = {
        "generate": "code.proposal.generate",
        "ready": "code.proposal.ready",
        "research_query": "a2a.research.query",
    }

    def __init__(
        self,
        research_client: Optional[ResearchAgentClient] = None,
        bus: Optional[Any] = None,
        confidence_threshold: float = 0.6,
    ):
        self.research = research_client or DefaultResearchClient(bus)
        self.bus = bus
        self.confidence_threshold = confidence_threshold
        self._proposals: Dict[str, CodeProposal] = {}

    async def generate_proposal(
        self,
        task: str,
        context: Optional[Dict[str, Any]] = None,
    ) -> CodeProposal:
        """
        Generate a code modification proposal for a task.

        Args:
            task: Natural language description of the task
            context: Additional context (files, constraints, etc.)

        Returns:
            CodeProposal with edits to implement the task
        """
        context = context or {}

        # Emit generation started event
        if self.bus:
            self.bus.emit({
                "topic": self.BUS_TOPICS["generate"],
                "kind": "proposal",
                "actor": "code-agent",
                "data": {"task": task, "context_keys": list(context.keys())},
            })

        # Get research context
        research_context = await self.research.get_context(task)

        # Generate edits (template-based for now)
        edits = await self._generate_edits(task, context, research_context)

        # Calculate metrics
        files_affected = list(set(e.path for e in edits))
        estimated_lines = sum(
            len((e.new_content or "").split("\n"))
            for e in edits if e.new_content
        )
        confidence = self._calculate_confidence(edits, research_context)

        # Create proposal
        proposal = CodeProposal(
            id=f"proposal-{uuid.uuid4().hex[:8]}",
            description=task,
            files_affected=files_affected,
            estimated_lines=estimated_lines,
            confidence=confidence,
            edits=edits,
            research_context=research_context,
            rationale=self._generate_rationale(task, edits),
            risks=self._identify_risks(edits),
        )

        self._proposals[proposal.id] = proposal

        # Emit proposal ready
        if self.bus:
            self.bus.emit({
                "topic": self.BUS_TOPICS["ready"],
                "kind": "proposal",
                "actor": "code-agent",
                "data": {
                    "proposal_id": proposal.id,
                    "files_affected": files_affected,
                    "confidence": confidence,
                    "edit_count": len(edits),
                },
            })

        return proposal

    async def _generate_edits(
        self,
        task: str,
        context: Dict[str, Any],
        research_context: Dict[str, Any],
    ) -> List[CodeEdit]:
        """
        Generate code edits for a task.

        This is the core neural generation step. Currently uses
        template matching; would integrate LLM in production.
        """
        edits: List[CodeEdit] = []

        # Extract target files from context
        target_files = context.get("files", [])

        # Pattern matching for common task types
        task_lower = task.lower()

        if "add" in task_lower and "import" in task_lower:
            # Import addition task
            for file_path in target_files:
                edits.append(CodeEdit(
                    path=file_path,
                    operation="insert",
                    location="1",  # Top of file
                    new_content="# TODO: Add required imports\n",
                    description="Add import statements",
                ))

        elif "add" in task_lower and ("method" in task_lower or "function" in task_lower):
            # Method/function addition
            for file_path in target_files:
                edits.append(CodeEdit(
                    path=file_path,
                    operation="append",
                    location="end",
                    new_content=self._generate_function_template(task),
                    description="Add new function/method",
                ))

        elif "fix" in task_lower or "bug" in task_lower:
            # Bug fix task
            for file_path in target_files:
                edits.append(CodeEdit(
                    path=file_path,
                    operation="replace",
                    location="unknown",  # Would be determined by analysis
                    old_content="# TODO: Identify buggy code",
                    new_content="# TODO: Fixed code",
                    description="Fix identified issue",
                ))

        elif "refactor" in task_lower:
            # Refactoring task
            for file_path in target_files:
                edits.append(CodeEdit(
                    path=file_path,
                    operation="replace",
                    location="unknown",
                    old_content="# Original code",
                    new_content="# Refactored code",
                    description="Refactor for improved structure",
                ))

        else:
            # Generic modification
            for file_path in target_files:
                edits.append(CodeEdit(
                    path=file_path,
                    operation="append",
                    location="end",
                    new_content=f"# TODO: Implement - {task}\n",
                    description=task,
                ))

        return edits

    def _generate_function_template(self, task: str) -> str:
        """Generate a function template based on task description."""
        # Extract potential function name from task
        words = task.lower().split()
        func_name = "new_function"

        for i, word in enumerate(words):
            if word in ("method", "function") and i + 1 < len(words):
                func_name = words[i + 1].replace("()", "").replace(",", "")
                break

        return f'''
def {func_name}():
    """
    TODO: Implement - {task}

    Generated by Code Agent Neural Proposal Generator.
    """
    raise NotImplementedError("{func_name} not yet implemented")
'''

    def _calculate_confidence(
        self,
        edits: List[CodeEdit],
        research_context: Dict[str, Any],
    ) -> float:
        """Calculate confidence score for proposal."""
        # Base confidence
        confidence = 0.7

        # Adjust based on research context quality
        if research_context.get("relevant_files"):
            confidence += 0.1

        if research_context.get("similar_code"):
            confidence += 0.1

        # Reduce confidence for complex edits
        if len(edits) > 5:
            confidence -= 0.1

        # Reduce confidence for unknown locations
        unknown_count = sum(1 for e in edits if e.location == "unknown")
        if unknown_count > 0:
            confidence -= 0.1 * min(unknown_count, 3)

        return max(0.1, min(1.0, confidence))

    def _generate_rationale(self, task: str, edits: List[CodeEdit]) -> str:
        """Generate rationale for the proposal."""
        file_count = len(set(e.path for e in edits))
        operations = set(e.operation for e in edits)

        return (
            f"Proposal addresses: {task}. "
            f"Affects {file_count} file(s) with operations: {', '.join(operations)}. "
            f"Generated using template matching with research context."
        )

    def _identify_risks(self, edits: List[CodeEdit]) -> List[str]:
        """Identify potential risks in the proposal."""
        risks: List[str] = []

        # Check for risky operations
        for edit in edits:
            if edit.operation == "delete":
                risks.append(f"Deletion in {edit.path} may remove needed code")

            if edit.operation == "replace" and edit.location == "unknown":
                risks.append(f"Replace location in {edit.path} not determined")

            if "__init__" in edit.path or "main" in edit.path:
                risks.append(f"Modification to critical file: {edit.path}")

        return risks

    def get_proposal(self, proposal_id: str) -> Optional[CodeProposal]:
        """Get a proposal by ID."""
        return self._proposals.get(proposal_id)

    def list_proposals(self) -> List[CodeProposal]:
        """List all proposals."""
        return list(self._proposals.values())

    async def refine_proposal(
        self,
        proposal_id: str,
        feedback: str,
    ) -> Optional[CodeProposal]:
        """
        Refine an existing proposal based on feedback.

        Args:
            proposal_id: ID of proposal to refine
            feedback: Feedback to incorporate

        Returns:
            Refined proposal or None if not found
        """
        original = self._proposals.get(proposal_id)
        if not original:
            return None

        # Generate refined proposal
        refined_task = f"{original.description} (Refined: {feedback})"
        context = {
            "files": original.files_affected,
            "original_proposal": proposal_id,
        }

        return await self.generate_proposal(refined_task, context)


# =============================================================================
# CLI
# =============================================================================

def main() -> int:
    """CLI entry point for Neural Proposal Generator."""
    import argparse

    parser = argparse.ArgumentParser(description="Neural Code Proposal Generator (Step 52)")
    subparsers = parser.add_subparsers(dest="command", required=True)

    # generate command
    gen_parser = subparsers.add_parser("generate", help="Generate code proposal")
    gen_parser.add_argument("task", help="Task description")
    gen_parser.add_argument("--files", nargs="+", help="Target files")
    gen_parser.add_argument("--json", action="store_true", help="JSON output")

    # list command
    subparsers.add_parser("list", help="List proposals")

    args = parser.parse_args()

    generator = NeuralCodeProposalGenerator()

    if args.command == "generate":
        async def run():
            context = {}
            if args.files:
                context["files"] = args.files

            proposal = await generator.generate_proposal(args.task, context)

            if args.json:
                print(json.dumps(proposal.to_dict(), indent=2))
            else:
                print(f"Proposal: {proposal.id}")
                print(f"  Description: {proposal.description}")
                print(f"  Files: {proposal.files_affected}")
                print(f"  Confidence: {proposal.confidence:.2f}")
                print(f"  Edits: {len(proposal.edits)}")
                for i, edit in enumerate(proposal.edits, 1):
                    print(f"    {i}. {edit.operation} @ {edit.path}:{edit.location}")

        asyncio.run(run())
        return 0

    elif args.command == "list":
        proposals = generator.list_proposals()
        if not proposals:
            print("No proposals generated yet")
        else:
            for p in proposals:
                print(f"  {p.id}: {p.description[:50]}...")
        return 0

    return 1


if __name__ == "__main__":
    import sys
    sys.exit(main())
