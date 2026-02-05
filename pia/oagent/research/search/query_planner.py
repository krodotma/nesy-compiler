#!/usr/bin/env python3
"""
query_planner.py - Query Planner (Step 18)

Optimal search strategy planning for research queries.
Determines the best combination of search methods and sources.

PBTSO Phase: PLAN

Bus Topics:
- a2a.research.query.plan
- research.query.execute
- research.query.complete

Protocol: DKIN v30, PAIP v16, CITIZEN v2
"""
from __future__ import annotations

import json
import os
import re
import time
from dataclasses import dataclass, field, asdict
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Set, Tuple

from ..bootstrap import AgentBus


# ============================================================================
# Data Models
# ============================================================================


class QueryIntent(Enum):
    """Intent of a research query."""
    FIND_SYMBOL = "find_symbol"           # Find where a symbol is defined
    FIND_USAGE = "find_usage"             # Find where a symbol is used
    EXPLAIN_CODE = "explain_code"         # Explain how code works
    FIND_SIMILAR = "find_similar"         # Find similar code patterns
    TRACE_DEPENDENCY = "trace_dependency" # Trace import chain
    FIND_DOCS = "find_docs"               # Find documentation
    SEARCH_CONCEPT = "search_concept"     # Search by concept/feature
    DEBUG_ISSUE = "debug_issue"           # Help debug an issue
    REFACTOR = "refactor"                 # Plan refactoring


class SearchMethod(Enum):
    """Available search methods."""
    SYMBOL_INDEX = "symbol_index"         # Query symbol store
    SEMANTIC_SEARCH = "semantic_search"   # Vector similarity
    FULLTEXT_SEARCH = "fulltext_search"   # Keyword search
    REFERENCE_RESOLVE = "reference_resolve" # Cross-file references
    DEPENDENCY_GRAPH = "dependency_graph" # Dependency analysis
    PATTERN_MATCH = "pattern_match"       # AST pattern matching
    KNOWLEDGE_GRAPH = "knowledge_graph"   # Domain knowledge


class StepType(Enum):
    """Type of query plan step."""
    SEARCH = "search"       # Perform a search
    FILTER = "filter"       # Filter results
    RANK = "rank"           # Rank/sort results
    AGGREGATE = "aggregate" # Combine results
    EXPAND = "expand"       # Expand with context


@dataclass
class QueryStep:
    """A step in the query execution plan."""

    step_id: int
    step_type: StepType
    method: Optional[SearchMethod] = None
    params: Dict[str, Any] = field(default_factory=dict)
    depends_on: List[int] = field(default_factory=list)
    estimated_time_ms: int = 100
    description: str = ""

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "step_id": self.step_id,
            "type": self.step_type.value,
            "method": self.method.value if self.method else None,
            "params": self.params,
            "depends_on": self.depends_on,
            "estimated_time_ms": self.estimated_time_ms,
            "description": self.description,
        }


@dataclass
class QueryPlan:
    """Complete query execution plan."""

    query: str
    intent: QueryIntent
    steps: List[QueryStep]
    estimated_total_time_ms: int = 0
    confidence: float = 0.8
    explanation: str = ""

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "query": self.query,
            "intent": self.intent.value,
            "steps": [s.to_dict() for s in self.steps],
            "estimated_total_time_ms": self.estimated_total_time_ms,
            "confidence": self.confidence,
            "explanation": self.explanation,
        }


@dataclass
class PlanResult:
    """Result of executing a query plan."""

    plan: QueryPlan
    results: List[Dict[str, Any]]
    execution_time_ms: float
    step_timings: Dict[int, float] = field(default_factory=dict)
    success: bool = True
    error: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "query": self.plan.query,
            "intent": self.plan.intent.value,
            "result_count": len(self.results),
            "execution_time_ms": self.execution_time_ms,
            "step_timings": self.step_timings,
            "success": self.success,
            "error": self.error,
        }


# ============================================================================
# Intent Detection
# ============================================================================


# Patterns for intent detection
INTENT_PATTERNS = {
    QueryIntent.FIND_SYMBOL: [
        r"where\s+is\s+(\w+)",
        r"find\s+(class|function|method)\s+(\w+)",
        r"locate\s+(\w+)",
        r"definition\s+of\s+(\w+)",
        r"what\s+file\s+.*\s+(\w+)",
    ],
    QueryIntent.FIND_USAGE: [
        r"who\s+uses\s+(\w+)",
        r"usages?\s+of\s+(\w+)",
        r"references?\s+to\s+(\w+)",
        r"callers?\s+of\s+(\w+)",
        r"where\s+.*\s+called",
    ],
    QueryIntent.EXPLAIN_CODE: [
        r"explain\s+",
        r"how\s+does\s+(\w+)\s+work",
        r"what\s+does\s+(\w+)\s+do",
        r"understand\s+",
    ],
    QueryIntent.FIND_SIMILAR: [
        r"similar\s+to",
        r"like\s+(\w+)",
        r"examples?\s+of",
        r"pattern",
    ],
    QueryIntent.TRACE_DEPENDENCY: [
        r"depends?\s+on",
        r"imports?\s+",
        r"requires?\s+",
        r"dependency",
    ],
    QueryIntent.FIND_DOCS: [
        r"documentation",
        r"docs?\s+for",
        r"readme",
        r"api\s+reference",
    ],
    QueryIntent.SEARCH_CONCEPT: [
        r"how\s+to\s+",
        r"implement",
        r"feature",
        r"functionality",
    ],
    QueryIntent.DEBUG_ISSUE: [
        r"error",
        r"bug",
        r"issue",
        r"fix",
        r"problem",
        r"fail",
    ],
    QueryIntent.REFACTOR: [
        r"refactor",
        r"rename",
        r"move",
        r"restructure",
        r"clean\s*up",
    ],
}


# ============================================================================
# Query Planner
# ============================================================================


class QueryPlanner:
    """
    Plan optimal search strategies for research queries.

    Analyzes query intent and generates execution plans that
    combine multiple search methods for best results.

    PBTSO Phase: PLAN

    Example:
        planner = QueryPlanner()
        plan = planner.plan_query("where is the User class defined?")
        print(plan.explanation)
    """

    # Method capabilities for different intents
    METHOD_CAPABILITIES = {
        QueryIntent.FIND_SYMBOL: [
            SearchMethod.SYMBOL_INDEX,
            SearchMethod.FULLTEXT_SEARCH,
        ],
        QueryIntent.FIND_USAGE: [
            SearchMethod.REFERENCE_RESOLVE,
            SearchMethod.SYMBOL_INDEX,
            SearchMethod.FULLTEXT_SEARCH,
        ],
        QueryIntent.EXPLAIN_CODE: [
            SearchMethod.SYMBOL_INDEX,
            SearchMethod.SEMANTIC_SEARCH,
            SearchMethod.KNOWLEDGE_GRAPH,
        ],
        QueryIntent.FIND_SIMILAR: [
            SearchMethod.SEMANTIC_SEARCH,
            SearchMethod.PATTERN_MATCH,
        ],
        QueryIntent.TRACE_DEPENDENCY: [
            SearchMethod.DEPENDENCY_GRAPH,
            SearchMethod.REFERENCE_RESOLVE,
        ],
        QueryIntent.FIND_DOCS: [
            SearchMethod.FULLTEXT_SEARCH,
            SearchMethod.SEMANTIC_SEARCH,
        ],
        QueryIntent.SEARCH_CONCEPT: [
            SearchMethod.SEMANTIC_SEARCH,
            SearchMethod.KNOWLEDGE_GRAPH,
            SearchMethod.FULLTEXT_SEARCH,
        ],
        QueryIntent.DEBUG_ISSUE: [
            SearchMethod.SEMANTIC_SEARCH,
            SearchMethod.SYMBOL_INDEX,
            SearchMethod.REFERENCE_RESOLVE,
        ],
        QueryIntent.REFACTOR: [
            SearchMethod.REFERENCE_RESOLVE,
            SearchMethod.DEPENDENCY_GRAPH,
            SearchMethod.SYMBOL_INDEX,
        ],
    }

    # Estimated time per method (ms)
    METHOD_TIMES = {
        SearchMethod.SYMBOL_INDEX: 50,
        SearchMethod.SEMANTIC_SEARCH: 200,
        SearchMethod.FULLTEXT_SEARCH: 100,
        SearchMethod.REFERENCE_RESOLVE: 300,
        SearchMethod.DEPENDENCY_GRAPH: 250,
        SearchMethod.PATTERN_MATCH: 400,
        SearchMethod.KNOWLEDGE_GRAPH: 150,
    }

    def __init__(
        self,
        bus: Optional[AgentBus] = None,
        available_methods: Optional[Set[SearchMethod]] = None,
    ):
        """
        Initialize the query planner.

        Args:
            bus: AgentBus for event emission
            available_methods: Methods available for use (default: all)
        """
        self.bus = bus or AgentBus()
        self.available_methods = available_methods or set(SearchMethod)

    def plan_query(
        self,
        query: str,
        context: Optional[Dict[str, Any]] = None,
        max_time_ms: int = 5000,
    ) -> QueryPlan:
        """
        Create an execution plan for a query.

        Args:
            query: Natural language query
            context: Additional context (current file, selection, etc.)
            max_time_ms: Maximum allowed execution time

        Returns:
            QueryPlan with steps to execute
        """
        # Detect intent
        intent = self._detect_intent(query)

        # Extract entities from query
        entities = self._extract_entities(query)

        # Select methods based on intent and availability
        methods = self._select_methods(intent, max_time_ms)

        # Build execution plan
        steps = self._build_steps(intent, methods, entities, context)

        # Calculate total time
        total_time = sum(s.estimated_time_ms for s in steps)

        # Generate explanation
        explanation = self._generate_explanation(intent, methods, entities)

        plan = QueryPlan(
            query=query,
            intent=intent,
            steps=steps,
            estimated_total_time_ms=total_time,
            confidence=self._calculate_confidence(intent, methods),
            explanation=explanation,
        )

        # Emit event
        self.bus.emit({
            "topic": "a2a.research.query.plan",
            "kind": "plan",
            "data": {
                "query": query,
                "intent": intent.value,
                "steps": len(steps),
                "estimated_time_ms": total_time,
            }
        })

        return plan

    def optimize_plan(
        self,
        plan: QueryPlan,
        feedback: Dict[str, Any],
    ) -> QueryPlan:
        """
        Optimize a plan based on execution feedback.

        Args:
            plan: Original plan
            feedback: Feedback from execution (timing, result quality)

        Returns:
            Optimized QueryPlan
        """
        # Adjust method times based on actual performance
        actual_times = feedback.get("step_timings", {})
        for step in plan.steps:
            if step.step_id in actual_times:
                # Adjust estimated time toward actual
                actual = actual_times[step.step_id]
                step.estimated_time_ms = int(0.7 * step.estimated_time_ms + 0.3 * actual)

        # Remove steps that produced no results
        empty_steps = feedback.get("empty_steps", [])
        plan.steps = [s for s in plan.steps if s.step_id not in empty_steps]

        # Recalculate total time
        plan.estimated_total_time_ms = sum(s.estimated_time_ms for s in plan.steps)

        return plan

    def explain_plan(self, plan: QueryPlan) -> str:
        """
        Generate human-readable explanation of a plan.

        Args:
            plan: Query plan

        Returns:
            Explanation string
        """
        lines = [
            f"Query Plan for: {plan.query}",
            f"Detected Intent: {plan.intent.value}",
            f"Estimated Time: {plan.estimated_total_time_ms}ms",
            f"Confidence: {plan.confidence:.0%}",
            "",
            "Execution Steps:",
        ]

        for step in plan.steps:
            deps = f" (after step {step.depends_on})" if step.depends_on else ""
            lines.append(f"  {step.step_id}. [{step.step_type.value}] {step.description}{deps}")

        lines.append("")
        lines.append(f"Explanation: {plan.explanation}")

        return "\n".join(lines)

    def _detect_intent(self, query: str) -> QueryIntent:
        """Detect query intent from natural language."""
        query_lower = query.lower()

        best_intent = QueryIntent.SEARCH_CONCEPT
        best_score = 0

        for intent, patterns in INTENT_PATTERNS.items():
            score = 0
            for pattern in patterns:
                if re.search(pattern, query_lower):
                    score += 1

            if score > best_score:
                best_score = score
                best_intent = intent

        return best_intent

    def _extract_entities(self, query: str) -> Dict[str, List[str]]:
        """Extract entities (symbols, paths, etc.) from query."""
        entities = {
            "symbols": [],
            "paths": [],
            "keywords": [],
        }

        # Find CamelCase words (likely class/type names)
        camel_case = re.findall(r"\b[A-Z][a-z]+(?:[A-Z][a-z]+)+\b", query)
        entities["symbols"].extend(camel_case)

        # Find snake_case words (likely function names)
        snake_case = re.findall(r"\b[a-z]+(?:_[a-z]+)+\b", query)
        entities["symbols"].extend(snake_case)

        # Find paths
        paths = re.findall(r"(?:\.?/)?[\w/]+\.\w+", query)
        entities["paths"].extend(paths)

        # Extract keywords (words in quotes or emphasized)
        quoted = re.findall(r"['\"](\w+)['\"]", query)
        entities["keywords"].extend(quoted)

        # Also add standalone words that look like identifiers
        words = re.findall(r"\b[a-zA-Z_]\w*\b", query)
        stop_words = {"the", "a", "an", "is", "are", "in", "of", "to", "for", "where", "how", "what"}
        for word in words:
            if word.lower() not in stop_words and word not in entities["symbols"]:
                if len(word) > 2:
                    entities["keywords"].append(word)

        return entities

    def _select_methods(
        self,
        intent: QueryIntent,
        max_time_ms: int,
    ) -> List[SearchMethod]:
        """Select search methods for intent within time budget."""
        candidates = self.METHOD_CAPABILITIES.get(intent, [])

        # Filter by availability
        candidates = [m for m in candidates if m in self.available_methods]

        # Sort by estimated time
        candidates.sort(key=lambda m: self.METHOD_TIMES.get(m, 500))

        # Select methods within time budget
        selected = []
        total_time = 0

        for method in candidates:
            method_time = self.METHOD_TIMES.get(method, 500)
            if total_time + method_time <= max_time_ms:
                selected.append(method)
                total_time += method_time
            elif not selected:
                # Always include at least one method
                selected.append(method)
                break

        return selected

    def _build_steps(
        self,
        intent: QueryIntent,
        methods: List[SearchMethod],
        entities: Dict[str, List[str]],
        context: Optional[Dict[str, Any]],
    ) -> List[QueryStep]:
        """Build execution steps for the plan."""
        steps = []
        step_id = 0

        # Create search steps for each method
        search_step_ids = []
        for method in methods:
            step_id += 1
            params = self._build_method_params(method, entities, context)

            steps.append(QueryStep(
                step_id=step_id,
                step_type=StepType.SEARCH,
                method=method,
                params=params,
                estimated_time_ms=self.METHOD_TIMES.get(method, 100),
                description=f"Search using {method.value}",
            ))
            search_step_ids.append(step_id)

        # Add aggregation step if multiple searches
        if len(search_step_ids) > 1:
            step_id += 1
            steps.append(QueryStep(
                step_id=step_id,
                step_type=StepType.AGGREGATE,
                params={"strategy": "union_rank"},
                depends_on=search_step_ids,
                estimated_time_ms=50,
                description="Combine and deduplicate results",
            ))

        # Add ranking step
        step_id += 1
        steps.append(QueryStep(
            step_id=step_id,
            step_type=StepType.RANK,
            params={
                "factors": ["relevance", "recency", "authority"],
                "limit": 20,
            },
            depends_on=[step_id - 1] if len(search_step_ids) > 1 else search_step_ids,
            estimated_time_ms=30,
            description="Rank results by relevance",
        ))

        # Add context expansion for certain intents
        if intent in (QueryIntent.EXPLAIN_CODE, QueryIntent.DEBUG_ISSUE):
            step_id += 1
            steps.append(QueryStep(
                step_id=step_id,
                step_type=StepType.EXPAND,
                params={"context_lines": 10, "include_imports": True},
                depends_on=[step_id - 1],
                estimated_time_ms=100,
                description="Expand results with surrounding context",
            ))

        return steps

    def _build_method_params(
        self,
        method: SearchMethod,
        entities: Dict[str, List[str]],
        context: Optional[Dict[str, Any]],
    ) -> Dict[str, Any]:
        """Build parameters for a search method."""
        params: Dict[str, Any] = {}

        if method == SearchMethod.SYMBOL_INDEX:
            if entities["symbols"]:
                params["name"] = entities["symbols"][0]
            params["limit"] = 50

        elif method == SearchMethod.SEMANTIC_SEARCH:
            # Build query from entities
            query_parts = entities["symbols"] + entities["keywords"]
            params["query"] = " ".join(query_parts[:5])
            params["top_k"] = 30

        elif method == SearchMethod.FULLTEXT_SEARCH:
            params["terms"] = entities["symbols"] + entities["keywords"]
            params["limit"] = 50

        elif method == SearchMethod.REFERENCE_RESOLVE:
            if entities["symbols"]:
                params["symbol_name"] = entities["symbols"][0]
            if entities["paths"]:
                params["file_path"] = entities["paths"][0]

        elif method == SearchMethod.DEPENDENCY_GRAPH:
            if entities["paths"]:
                params["start_path"] = entities["paths"][0]
            params["max_depth"] = 3

        elif method == SearchMethod.PATTERN_MATCH:
            params["patterns"] = entities["symbols"]

        elif method == SearchMethod.KNOWLEDGE_GRAPH:
            params["concepts"] = entities["keywords"]

        # Add context if available
        if context:
            if "current_file" in context:
                params["context_file"] = context["current_file"]
            if "selection" in context:
                params["selection"] = context["selection"]

        return params

    def _calculate_confidence(
        self,
        intent: QueryIntent,
        methods: List[SearchMethod],
    ) -> float:
        """Calculate confidence in the plan."""
        base_confidence = 0.5

        # Higher confidence if intent was clearly detected
        if intent != QueryIntent.SEARCH_CONCEPT:  # Default intent
            base_confidence += 0.2

        # Higher confidence with more methods
        base_confidence += 0.1 * min(len(methods), 3)

        return min(base_confidence, 0.95)

    def _generate_explanation(
        self,
        intent: QueryIntent,
        methods: List[SearchMethod],
        entities: Dict[str, List[str]],
    ) -> str:
        """Generate explanation of the plan."""
        intent_descriptions = {
            QueryIntent.FIND_SYMBOL: "find where a symbol is defined",
            QueryIntent.FIND_USAGE: "find where a symbol is used",
            QueryIntent.EXPLAIN_CODE: "explain how code works",
            QueryIntent.FIND_SIMILAR: "find similar code patterns",
            QueryIntent.TRACE_DEPENDENCY: "trace dependencies",
            QueryIntent.FIND_DOCS: "find documentation",
            QueryIntent.SEARCH_CONCEPT: "search for a concept",
            QueryIntent.DEBUG_ISSUE: "help debug an issue",
            QueryIntent.REFACTOR: "plan refactoring",
        }

        method_descriptions = {
            SearchMethod.SYMBOL_INDEX: "symbol index",
            SearchMethod.SEMANTIC_SEARCH: "semantic similarity",
            SearchMethod.FULLTEXT_SEARCH: "text search",
            SearchMethod.REFERENCE_RESOLVE: "reference analysis",
            SearchMethod.DEPENDENCY_GRAPH: "dependency graph",
            SearchMethod.PATTERN_MATCH: "pattern matching",
            SearchMethod.KNOWLEDGE_GRAPH: "knowledge graph",
        }

        parts = [f"Plan to {intent_descriptions.get(intent, 'search')}"]

        if entities["symbols"]:
            parts.append(f"for '{entities['symbols'][0]}'")

        method_names = [method_descriptions.get(m, m.value) for m in methods]
        parts.append(f"using {', '.join(method_names)}")

        return " ".join(parts) + "."


# ============================================================================
# CLI Entry Point
# ============================================================================


def main() -> int:
    """CLI entry point for Query Planner."""
    import argparse

    parser = argparse.ArgumentParser(
        description="Query Planner (Step 18)"
    )
    parser.add_argument(
        "query",
        nargs="?",
        help="Query to plan"
    )
    parser.add_argument(
        "--max-time",
        type=int,
        default=5000,
        help="Maximum execution time in ms"
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Output as JSON"
    )
    parser.add_argument(
        "--explain",
        action="store_true",
        help="Show detailed explanation"
    )

    args = parser.parse_args()

    planner = QueryPlanner()

    if args.query:
        plan = planner.plan_query(args.query, max_time_ms=args.max_time)

        if args.json:
            print(json.dumps(plan.to_dict(), indent=2))
        elif args.explain:
            print(planner.explain_plan(plan))
        else:
            print(f"Query: {plan.query}")
            print(f"Intent: {plan.intent.value}")
            print(f"Steps: {len(plan.steps)}")
            print(f"Estimated Time: {plan.estimated_total_time_ms}ms")
            print(f"\n{plan.explanation}")
    else:
        # Interactive mode
        print("Query Planner - Interactive Mode")
        print("Enter queries to see execution plans (Ctrl+C to exit)\n")

        while True:
            try:
                query = input("Query: ").strip()
                if not query:
                    continue

                plan = planner.plan_query(query)
                print(planner.explain_plan(plan))
                print()

            except KeyboardInterrupt:
                print("\nExiting.")
                break

    return 0


if __name__ == "__main__":
    import sys
    sys.exit(main())
