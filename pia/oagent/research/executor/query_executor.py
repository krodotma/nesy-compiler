#!/usr/bin/env python3
"""
query_executor.py - Query Executor (Step 21)

Execute planned research queries across multiple search backends.
Coordinates parallel execution and result aggregation.

PBTSO Phase: RESEARCH, ITERATE

Bus Topics:
- a2a.research.execute.start
- a2a.research.execute.complete
- a2a.research.execute.step
- research.execute.error

Protocol: DKIN v30, PAIP v16, CITIZEN v2
"""
from __future__ import annotations

import asyncio
import fcntl
import json
import os
import time
from dataclasses import dataclass, field, asdict
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Coroutine, Dict, List, Optional, Set, Tuple, Union

from ..bootstrap import AgentBus
from ..search.query_planner import (
    QueryPlan, QueryStep, StepType, SearchMethod, QueryIntent
)
from ..index.symbol_store import SymbolIndexStore, Symbol
from ..search.semantic_engine import SemanticSearchEngine
from ..analysis.reference_resolver import ReferenceResolver
from ..graph.dependency_builder import DependencyGraphBuilder


# ============================================================================
# Configuration
# ============================================================================


class ExecutionStrategy(Enum):
    """Query execution strategy."""
    SEQUENTIAL = "sequential"  # Execute steps one by one
    PARALLEL = "parallel"      # Execute independent steps in parallel
    ADAPTIVE = "adaptive"      # Choose based on plan structure


@dataclass
class ExecutionConfig:
    """Configuration for query executor."""

    strategy: ExecutionStrategy = ExecutionStrategy.ADAPTIVE
    timeout_seconds: int = 30
    max_parallel: int = 4
    retry_count: int = 2
    retry_delay_ms: int = 100
    enable_caching: bool = True
    bus_path: Optional[str] = None

    def __post_init__(self):
        if self.bus_path is None:
            pluribus_root = os.environ.get("PLURIBUS_ROOT", "/pluribus")
            self.bus_path = f"{pluribus_root}/.pluribus/bus/events.ndjson"


# ============================================================================
# Data Models
# ============================================================================


class StepStatus(Enum):
    """Status of a query step."""
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    SKIPPED = "skipped"


@dataclass
class StepResult:
    """Result of executing a single step."""

    step_id: int
    status: StepStatus
    results: List[Dict[str, Any]] = field(default_factory=list)
    execution_time_ms: float = 0
    error: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "step_id": self.step_id,
            "status": self.status.value,
            "result_count": len(self.results),
            "execution_time_ms": self.execution_time_ms,
            "error": self.error,
        }


@dataclass
class ExecutedQuery:
    """Result of executing a query plan."""

    plan: QueryPlan
    step_results: Dict[int, StepResult]
    aggregated_results: List[Dict[str, Any]]
    total_time_ms: float
    success: bool
    errors: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "query": self.plan.query,
            "intent": self.plan.intent.value,
            "steps_executed": len(self.step_results),
            "steps_successful": sum(
                1 for r in self.step_results.values()
                if r.status == StepStatus.COMPLETED
            ),
            "total_results": len(self.aggregated_results),
            "total_time_ms": self.total_time_ms,
            "success": self.success,
            "errors": self.errors,
        }


# ============================================================================
# Query Executor
# ============================================================================


class QueryExecutor:
    """
    Execute research query plans across multiple backends.

    Coordinates:
    - Parallel and sequential step execution
    - Backend-specific search methods
    - Result aggregation and deduplication
    - Error handling and retries

    PBTSO Phase: RESEARCH, ITERATE

    Example:
        executor = QueryExecutor()
        await executor.initialize()

        plan = query_planner.plan_query("find UserService")
        result = await executor.execute(plan)
        print(result.aggregated_results)
    """

    def __init__(
        self,
        config: Optional[ExecutionConfig] = None,
        bus: Optional[AgentBus] = None,
        symbol_store: Optional[SymbolIndexStore] = None,
        semantic_engine: Optional[SemanticSearchEngine] = None,
        reference_resolver: Optional[ReferenceResolver] = None,
        dependency_builder: Optional[DependencyGraphBuilder] = None,
    ):
        """
        Initialize the query executor.

        Args:
            config: Execution configuration
            bus: AgentBus for event emission
            symbol_store: Symbol index for symbol searches
            semantic_engine: Semantic search engine
            reference_resolver: Reference resolver for usage searches
            dependency_builder: Dependency graph builder
        """
        self.config = config or ExecutionConfig()
        self.bus = bus or AgentBus()

        # Search backends
        self._symbol_store = symbol_store
        self._semantic_engine = semantic_engine
        self._reference_resolver = reference_resolver
        self._dependency_builder = dependency_builder

        # Method handlers
        self._method_handlers: Dict[SearchMethod, Callable] = {}
        self._step_handlers: Dict[StepType, Callable] = {}

        # State
        self._initialized = False
        self._execution_cache: Dict[str, StepResult] = {}

    async def initialize(self) -> bool:
        """
        Initialize the executor and its backends.

        Returns:
            True if initialization successful
        """
        try:
            # Initialize backends if not provided
            if self._symbol_store is None:
                self._symbol_store = SymbolIndexStore(bus=self.bus)

            # Register method handlers
            self._register_method_handlers()
            self._register_step_handlers()

            self._initialized = True
            return True

        except Exception as e:
            self._emit_error("initialize", str(e))
            return False

    async def execute(
        self,
        plan: QueryPlan,
        context: Optional[Dict[str, Any]] = None,
    ) -> ExecutedQuery:
        """
        Execute a query plan.

        Args:
            plan: Query plan to execute
            context: Additional context for execution

        Returns:
            ExecutedQuery with results
        """
        start_time = time.time()

        # Emit start event
        self._emit_with_lock({
            "topic": "a2a.research.execute.start",
            "kind": "execute",
            "data": {
                "query": plan.query,
                "intent": plan.intent.value,
                "steps": len(plan.steps),
            }
        })

        step_results: Dict[int, StepResult] = {}
        errors: List[str] = []

        try:
            # Determine execution order
            execution_order = self._build_execution_order(plan)

            # Execute based on strategy
            if self.config.strategy == ExecutionStrategy.SEQUENTIAL:
                step_results = await self._execute_sequential(plan, execution_order, context)
            elif self.config.strategy == ExecutionStrategy.PARALLEL:
                step_results = await self._execute_parallel(plan, execution_order, context)
            else:  # ADAPTIVE
                step_results = await self._execute_adaptive(plan, execution_order, context)

            # Collect errors
            for result in step_results.values():
                if result.error:
                    errors.append(f"Step {result.step_id}: {result.error}")

            # Aggregate results
            aggregated = self._aggregate_results(step_results, plan)

        except asyncio.TimeoutError:
            errors.append("Execution timed out")
            aggregated = []
        except Exception as e:
            errors.append(str(e))
            aggregated = []

        total_time = (time.time() - start_time) * 1000

        result = ExecutedQuery(
            plan=plan,
            step_results=step_results,
            aggregated_results=aggregated,
            total_time_ms=total_time,
            success=len(errors) == 0,
            errors=errors,
        )

        # Emit completion event
        self._emit_with_lock({
            "topic": "a2a.research.execute.complete",
            "kind": "execute",
            "data": result.to_dict()
        })

        return result

    async def execute_step(
        self,
        step: QueryStep,
        prior_results: Dict[int, StepResult],
        context: Optional[Dict[str, Any]] = None,
    ) -> StepResult:
        """
        Execute a single query step.

        Args:
            step: Step to execute
            prior_results: Results from previous steps
            context: Additional context

        Returns:
            StepResult
        """
        start_time = time.time()

        # Check cache
        cache_key = self._make_cache_key(step)
        if self.config.enable_caching and cache_key in self._execution_cache:
            cached = self._execution_cache[cache_key]
            cached.metadata["cache_hit"] = True
            return cached

        # Get handler for step type
        handler = self._step_handlers.get(step.step_type)
        if not handler:
            return StepResult(
                step_id=step.step_id,
                status=StepStatus.FAILED,
                error=f"No handler for step type: {step.step_type.value}",
            )

        # Execute with retry
        last_error = None
        for attempt in range(self.config.retry_count + 1):
            try:
                results = await handler(step, prior_results, context)

                execution_time = (time.time() - start_time) * 1000

                result = StepResult(
                    step_id=step.step_id,
                    status=StepStatus.COMPLETED,
                    results=results,
                    execution_time_ms=execution_time,
                )

                # Emit step event
                self._emit_with_lock({
                    "topic": "a2a.research.execute.step",
                    "kind": "step",
                    "data": result.to_dict()
                })

                # Cache result
                if self.config.enable_caching:
                    self._execution_cache[cache_key] = result

                return result

            except Exception as e:
                last_error = str(e)
                if attempt < self.config.retry_count:
                    await asyncio.sleep(self.config.retry_delay_ms / 1000)

        # All retries failed
        return StepResult(
            step_id=step.step_id,
            status=StepStatus.FAILED,
            execution_time_ms=(time.time() - start_time) * 1000,
            error=last_error,
        )

    def register_method_handler(
        self,
        method: SearchMethod,
        handler: Callable[..., Coroutine],
    ) -> None:
        """Register a custom handler for a search method."""
        self._method_handlers[method] = handler

    def clear_cache(self) -> int:
        """Clear execution cache. Returns number of entries cleared."""
        count = len(self._execution_cache)
        self._execution_cache.clear()
        return count

    def get_stats(self) -> Dict[str, Any]:
        """Get executor statistics."""
        return {
            "initialized": self._initialized,
            "strategy": self.config.strategy.value,
            "cache_size": len(self._execution_cache),
            "method_handlers": list(m.value for m in self._method_handlers.keys()),
        }

    # ========================================================================
    # Execution Strategies
    # ========================================================================

    async def _execute_sequential(
        self,
        plan: QueryPlan,
        order: List[List[int]],
        context: Optional[Dict[str, Any]],
    ) -> Dict[int, StepResult]:
        """Execute steps sequentially."""
        results: Dict[int, StepResult] = {}
        step_map = {s.step_id: s for s in plan.steps}

        for level in order:
            for step_id in level:
                step = step_map.get(step_id)
                if step:
                    result = await self.execute_step(step, results, context)
                    results[step_id] = result

        return results

    async def _execute_parallel(
        self,
        plan: QueryPlan,
        order: List[List[int]],
        context: Optional[Dict[str, Any]],
    ) -> Dict[int, StepResult]:
        """Execute independent steps in parallel."""
        results: Dict[int, StepResult] = {}
        step_map = {s.step_id: s for s in plan.steps}

        for level in order:
            # Execute all steps at this level in parallel
            tasks = []
            for step_id in level:
                step = step_map.get(step_id)
                if step:
                    tasks.append(self.execute_step(step, results, context))

            if tasks:
                level_results = await asyncio.gather(*tasks, return_exceptions=True)

                for step_id, result in zip(level, level_results):
                    if isinstance(result, Exception):
                        results[step_id] = StepResult(
                            step_id=step_id,
                            status=StepStatus.FAILED,
                            error=str(result),
                        )
                    else:
                        results[step_id] = result

        return results

    async def _execute_adaptive(
        self,
        plan: QueryPlan,
        order: List[List[int]],
        context: Optional[Dict[str, Any]],
    ) -> Dict[int, StepResult]:
        """Adaptively choose execution strategy."""
        # Use parallel if we have multiple independent steps
        has_parallel_opportunity = any(len(level) > 1 for level in order)

        if has_parallel_opportunity:
            return await self._execute_parallel(plan, order, context)
        else:
            return await self._execute_sequential(plan, order, context)

    # ========================================================================
    # Step Handlers
    # ========================================================================

    def _register_step_handlers(self) -> None:
        """Register default step type handlers."""
        self._step_handlers[StepType.SEARCH] = self._handle_search_step
        self._step_handlers[StepType.FILTER] = self._handle_filter_step
        self._step_handlers[StepType.RANK] = self._handle_rank_step
        self._step_handlers[StepType.AGGREGATE] = self._handle_aggregate_step
        self._step_handlers[StepType.EXPAND] = self._handle_expand_step

    async def _handle_search_step(
        self,
        step: QueryStep,
        prior_results: Dict[int, StepResult],
        context: Optional[Dict[str, Any]],
    ) -> List[Dict[str, Any]]:
        """Handle a search step."""
        method = step.method
        if not method:
            return []

        handler = self._method_handlers.get(method)
        if handler:
            return await handler(step.params, context)

        return []

    async def _handle_filter_step(
        self,
        step: QueryStep,
        prior_results: Dict[int, StepResult],
        context: Optional[Dict[str, Any]],
    ) -> List[Dict[str, Any]]:
        """Handle a filter step."""
        # Get results from dependent steps
        results = []
        for dep_id in step.depends_on:
            if dep_id in prior_results:
                results.extend(prior_results[dep_id].results)

        # Apply filters from params
        filters = step.params.get("filters", {})

        filtered = []
        for result in results:
            include = True
            for key, value in filters.items():
                if key in result and result[key] != value:
                    include = False
                    break
            if include:
                filtered.append(result)

        return filtered

    async def _handle_rank_step(
        self,
        step: QueryStep,
        prior_results: Dict[int, StepResult],
        context: Optional[Dict[str, Any]],
    ) -> List[Dict[str, Any]]:
        """Handle a ranking step."""
        # Get results from dependent steps
        results = []
        for dep_id in step.depends_on:
            if dep_id in prior_results:
                results.extend(prior_results[dep_id].results)

        # Sort by score
        results.sort(key=lambda r: r.get("score", 0), reverse=True)

        # Apply limit
        limit = step.params.get("limit", 50)
        return results[:limit]

    async def _handle_aggregate_step(
        self,
        step: QueryStep,
        prior_results: Dict[int, StepResult],
        context: Optional[Dict[str, Any]],
    ) -> List[Dict[str, Any]]:
        """Handle an aggregation step."""
        strategy = step.params.get("strategy", "union")

        # Collect all results
        all_results = []
        for dep_id in step.depends_on:
            if dep_id in prior_results:
                all_results.extend(prior_results[dep_id].results)

        if strategy == "union":
            # Simple union with deduplication
            return self._deduplicate_results(all_results)

        elif strategy == "union_rank":
            # Union with re-ranking
            deduped = self._deduplicate_results(all_results)
            deduped.sort(key=lambda r: r.get("score", 0), reverse=True)
            return deduped

        elif strategy == "intersection":
            # Only results appearing in all sources
            if len(step.depends_on) <= 1:
                return all_results

            # Find common results by key
            result_sets = []
            for dep_id in step.depends_on:
                if dep_id in prior_results:
                    keys = set()
                    for r in prior_results[dep_id].results:
                        key = self._result_key(r)
                        keys.add(key)
                    result_sets.append(keys)

            if not result_sets:
                return []

            common_keys = result_sets[0]
            for rs in result_sets[1:]:
                common_keys &= rs

            return [r for r in all_results if self._result_key(r) in common_keys]

        return all_results

    async def _handle_expand_step(
        self,
        step: QueryStep,
        prior_results: Dict[int, StepResult],
        context: Optional[Dict[str, Any]],
    ) -> List[Dict[str, Any]]:
        """Handle a context expansion step."""
        # Get results from dependent steps
        results = []
        for dep_id in step.depends_on:
            if dep_id in prior_results:
                results.extend(prior_results[dep_id].results)

        context_lines = step.params.get("context_lines", 5)

        expanded = []
        for result in results:
            expanded_result = result.copy()

            # Try to expand with surrounding context
            if "path" in result and "line" in result:
                try:
                    path = result["path"]
                    line = result["line"]
                    content = Path(path).read_text(errors="ignore")
                    lines = content.split("\n")

                    start = max(0, line - 1 - context_lines)
                    end = min(len(lines), line + context_lines)

                    expanded_result["context"] = "\n".join(lines[start:end])
                    expanded_result["context_start"] = start + 1
                    expanded_result["context_end"] = end

                except Exception:
                    pass

            expanded.append(expanded_result)

        return expanded

    # ========================================================================
    # Method Handlers
    # ========================================================================

    def _register_method_handlers(self) -> None:
        """Register default search method handlers."""
        self._method_handlers[SearchMethod.SYMBOL_INDEX] = self._search_symbol_index
        self._method_handlers[SearchMethod.SEMANTIC_SEARCH] = self._search_semantic
        self._method_handlers[SearchMethod.FULLTEXT_SEARCH] = self._search_fulltext
        self._method_handlers[SearchMethod.REFERENCE_RESOLVE] = self._search_references
        self._method_handlers[SearchMethod.DEPENDENCY_GRAPH] = self._search_dependencies

    async def _search_symbol_index(
        self,
        params: Dict[str, Any],
        context: Optional[Dict[str, Any]],
    ) -> List[Dict[str, Any]]:
        """Search symbol index."""
        if not self._symbol_store:
            return []

        name = params.get("name")
        limit = params.get("limit", 50)

        if name:
            symbols = self._symbol_store.query(name=name, limit=limit)
        else:
            # Search by pattern or fulltext
            query = params.get("query", "")
            symbols = self._symbol_store.search(query, limit=limit)

        return [
            {
                "type": "symbol",
                "name": s.name,
                "kind": s.kind,
                "path": s.path,
                "line": s.line,
                "signature": s.signature,
                "docstring": s.docstring,
                "score": 1.0 if s.name == name else 0.8,
            }
            for s in symbols
        ]

    async def _search_semantic(
        self,
        params: Dict[str, Any],
        context: Optional[Dict[str, Any]],
    ) -> List[Dict[str, Any]]:
        """Search using semantic similarity."""
        if not self._semantic_engine:
            return []

        query = params.get("query", "")
        top_k = params.get("top_k", 30)

        results = self._semantic_engine.search(query, top_k=top_k)

        return [
            {
                "type": "semantic",
                "path": r.path,
                "content": r.content,
                "score": r.score,
                "line_start": r.line_start,
                "line_end": r.line_end,
            }
            for r in results
        ]

    async def _search_fulltext(
        self,
        params: Dict[str, Any],
        context: Optional[Dict[str, Any]],
    ) -> List[Dict[str, Any]]:
        """Search using fulltext matching."""
        # Use symbol store's search for now
        if not self._symbol_store:
            return []

        terms = params.get("terms", [])
        limit = params.get("limit", 50)

        results = []
        for term in terms[:5]:  # Limit term count
            symbols = self._symbol_store.search(term, limit=limit // len(terms) if terms else limit)
            for s in symbols:
                results.append({
                    "type": "fulltext",
                    "name": s.name,
                    "kind": s.kind,
                    "path": s.path,
                    "line": s.line,
                    "score": 0.6,
                })

        return results

    async def _search_references(
        self,
        params: Dict[str, Any],
        context: Optional[Dict[str, Any]],
    ) -> List[Dict[str, Any]]:
        """Search for symbol references."""
        if not self._reference_resolver:
            return []

        symbol_name = params.get("symbol_name")
        if not symbol_name:
            return []

        usages = self._reference_resolver.find_usages(symbol_name)

        return [
            {
                "type": "reference",
                "symbol": u.reference.symbol_name,
                "path": u.reference.source_path,
                "line": u.reference.source_line,
                "context": u.reference.context,
                "target_path": u.target_path,
                "score": u.confidence,
            }
            for u in usages[:50]
        ]

    async def _search_dependencies(
        self,
        params: Dict[str, Any],
        context: Optional[Dict[str, Any]],
    ) -> List[Dict[str, Any]]:
        """Search dependency graph."""
        if not self._dependency_builder:
            return []

        start_path = params.get("start_path")
        if not start_path:
            return []

        max_depth = params.get("max_depth", 3)

        results = []

        # Get dependencies
        deps = self._dependency_builder.get_dependencies(start_path)
        for dep in deps[:20]:
            results.append({
                "type": "dependency",
                "path": dep,
                "direction": "imports",
                "score": 0.7,
            })

        # Get dependents
        dependents = self._dependency_builder.get_dependents(start_path)
        for dep in dependents[:20]:
            results.append({
                "type": "dependency",
                "path": dep,
                "direction": "imported_by",
                "score": 0.7,
            })

        return results

    # ========================================================================
    # Helpers
    # ========================================================================

    def _build_execution_order(self, plan: QueryPlan) -> List[List[int]]:
        """
        Build execution order respecting dependencies.

        Returns list of levels, where each level contains step IDs
        that can be executed in parallel.
        """
        # Build dependency graph
        step_deps: Dict[int, Set[int]] = {}
        for step in plan.steps:
            step_deps[step.step_id] = set(step.depends_on)

        levels: List[List[int]] = []
        remaining = set(step_deps.keys())
        completed: Set[int] = set()

        while remaining:
            # Find steps with all dependencies satisfied
            ready = []
            for step_id in remaining:
                if step_deps[step_id] <= completed:
                    ready.append(step_id)

            if not ready:
                # Circular dependency or error - add all remaining
                levels.append(list(remaining))
                break

            levels.append(ready)
            completed.update(ready)
            remaining -= set(ready)

        return levels

    def _aggregate_results(
        self,
        step_results: Dict[int, StepResult],
        plan: QueryPlan,
    ) -> List[Dict[str, Any]]:
        """Aggregate results from all steps."""
        # Find the final step (highest ID with no dependents)
        final_steps = set(step_results.keys())
        for step in plan.steps:
            for dep in step.depends_on:
                final_steps.discard(dep)

        # Collect results from final steps
        results = []
        for step_id in final_steps:
            if step_id in step_results:
                results.extend(step_results[step_id].results)

        return self._deduplicate_results(results)

    def _deduplicate_results(self, results: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Deduplicate results by key."""
        seen: Set[str] = set()
        deduped = []

        for result in results:
            key = self._result_key(result)
            if key not in seen:
                seen.add(key)
                deduped.append(result)

        return deduped

    def _result_key(self, result: Dict[str, Any]) -> str:
        """Generate unique key for a result."""
        path = result.get("path", "")
        line = result.get("line", "")
        name = result.get("name", "")
        return f"{path}:{line}:{name}"

    def _make_cache_key(self, step: QueryStep) -> str:
        """Generate cache key for a step."""
        params_str = json.dumps(step.params, sort_keys=True)
        return f"{step.step_type.value}:{step.method.value if step.method else 'none'}:{params_str}"

    def _emit_with_lock(self, event: Dict[str, Any]) -> str:
        """Emit event with file locking."""
        bus_path = Path(self.config.bus_path)
        bus_path.parent.mkdir(parents=True, exist_ok=True)

        import socket
        from datetime import datetime, timezone
        import uuid

        event_id = str(uuid.uuid4())
        full_event = {
            "id": event_id,
            "ts": time.time(),
            "iso": datetime.now(timezone.utc).isoformat() + "Z",
            "topic": event.get("topic", "unknown"),
            "kind": event.get("kind", "event"),
            "level": event.get("level", "info"),
            "actor": "research-agent",
            "host": socket.gethostname(),
            "pid": os.getpid(),
            "data": event.get("data", {}),
        }

        with open(bus_path, "a") as f:
            fcntl.flock(f.fileno(), fcntl.LOCK_EX)
            try:
                f.write(json.dumps(full_event) + "\n")
            finally:
                fcntl.flock(f.fileno(), fcntl.LOCK_UN)

        return event_id

    def _emit_error(self, operation: str, error: str) -> None:
        """Emit error event."""
        self._emit_with_lock({
            "topic": "research.execute.error",
            "kind": "error",
            "level": "error",
            "data": {"operation": operation, "error": error}
        })


# ============================================================================
# CLI Entry Point
# ============================================================================


def main() -> int:
    """CLI entry point for Query Executor."""
    import argparse

    parser = argparse.ArgumentParser(
        description="Query Executor (Step 21)"
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    # Execute command
    exec_parser = subparsers.add_parser("execute", help="Execute a query")
    exec_parser.add_argument("query", help="Query to execute")
    exec_parser.add_argument("--strategy", choices=["sequential", "parallel", "adaptive"],
                            default="adaptive", help="Execution strategy")
    exec_parser.add_argument("--timeout", type=int, default=30, help="Timeout in seconds")
    exec_parser.add_argument("--json", action="store_true", help="Output as JSON")

    # Stats command
    stats_parser = subparsers.add_parser("stats", help="Show executor statistics")
    stats_parser.add_argument("--json", action="store_true", help="Output as JSON")

    args = parser.parse_args()

    async def run():
        config = ExecutionConfig(
            strategy=ExecutionStrategy(args.strategy) if hasattr(args, "strategy") else ExecutionStrategy.ADAPTIVE,
            timeout_seconds=getattr(args, "timeout", 30),
        )
        executor = QueryExecutor(config)
        await executor.initialize()

        if args.command == "execute":
            from ..search.query_planner import QueryPlanner

            planner = QueryPlanner()
            plan = planner.plan_query(args.query)
            result = await executor.execute(plan)

            if args.json:
                output = result.to_dict()
                output["results"] = result.aggregated_results[:10]
                print(json.dumps(output, indent=2))
            else:
                print(f"Query: {result.plan.query}")
                print(f"Intent: {result.plan.intent.value}")
                print(f"Success: {result.success}")
                print(f"Time: {result.total_time_ms:.1f}ms")
                print(f"Results: {len(result.aggregated_results)}")
                for r in result.aggregated_results[:10]:
                    print(f"  [{r.get('type', 'unknown')}] {r.get('name', r.get('path', 'N/A'))}")

        elif args.command == "stats":
            stats = executor.get_stats()
            if args.json:
                print(json.dumps(stats, indent=2))
            else:
                print("Query Executor Statistics:")
                print(f"  Initialized: {stats['initialized']}")
                print(f"  Strategy: {stats['strategy']}")
                print(f"  Cache Size: {stats['cache_size']}")

        return 0

    return asyncio.run(run())


if __name__ == "__main__":
    import sys
    sys.exit(main())
