#!/usr/bin/env python3
"""
orchestrator.py - Research Orchestrator (Step 20)

Coordinates all research components for unified query execution.
Manages the research pipeline from query to results.

PBTSO Phase: PLAN, DISTRIBUTE

Bus Topics:
- a2a.research.orchestrate
- a2a.research.query
- research.pipeline.start
- research.pipeline.complete
- research.pipeline.error

Protocol: DKIN v30, PAIP v16, CITIZEN v2
"""
from __future__ import annotations

import asyncio
import json
import os
import time
from dataclasses import dataclass, field, asdict
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Coroutine, Dict, List, Optional, Set, TypeVar, Union

from .bootstrap import AgentBus, ResearchAgentConfig
from .scanner import CodebaseScanner
from .index.symbol_store import SymbolIndexStore, Symbol
from .parsers.base import ParserRegistry, ParseResult
from .graph.dependency_builder import DependencyGraphBuilder
from .search.semantic_engine import SemanticSearchEngine, SemanticSearchConfig
from .search.context_assembler import ContextAssembler, ContextPriority, AssembledContext
from .search.query_planner import QueryPlanner, QueryPlan, QueryIntent, SearchMethod
from .analysis.reference_resolver import ReferenceResolver
from .analysis.impact_analyzer import ImpactAnalyzer
from .analysis.pattern_detector import PatternDetector
from .analysis.architecture_mapper import ArchitectureMapper
from .analysis.knowledge_extractor import KnowledgeExtractor
from .cache.cache_manager import CacheManager, CacheConfig


# ============================================================================
# Configuration
# ============================================================================


class ResearchPhase(Enum):
    """Phases of research execution."""
    PLAN = "plan"           # Query planning
    SEARCH = "search"       # Execute searches
    ANALYZE = "analyze"     # Analyze results
    ASSEMBLE = "assemble"   # Assemble context
    SYNTHESIZE = "synthesize"  # Synthesize findings


@dataclass
class ResearchOrchestratorConfig:
    """Configuration for research orchestrator."""

    root: Optional[str] = None
    max_results: int = 50
    context_tokens: int = 50000
    enable_caching: bool = True
    enable_semantic: bool = True
    parallel_searches: bool = True
    timeout_seconds: int = 30

    def __post_init__(self):
        if self.root is None:
            self.root = os.environ.get("PLURIBUS_ROOT", "/pluribus")


# ============================================================================
# Data Models
# ============================================================================


@dataclass
class ResearchQuery:
    """A research query to execute."""

    query: str
    context_file: Optional[str] = None
    context_selection: Optional[str] = None
    max_results: int = 50
    include_context: bool = True
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return asdict(self)


@dataclass
class ResearchResult:
    """Result of a research query."""

    query: ResearchQuery
    intent: QueryIntent
    results: List[Dict[str, Any]]
    context: Optional[AssembledContext] = None
    execution_time_ms: float = 0
    phases_completed: List[str] = field(default_factory=list)
    errors: List[str] = field(default_factory=list)
    cache_hit: bool = False

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "query": self.query.query,
            "intent": self.intent.value,
            "result_count": len(self.results),
            "has_context": self.context is not None,
            "execution_time_ms": self.execution_time_ms,
            "phases_completed": self.phases_completed,
            "errors": self.errors,
            "cache_hit": self.cache_hit,
        }


# ============================================================================
# Research Orchestrator
# ============================================================================


class ResearchOrchestrator:
    """
    Orchestrate all research components for unified query execution.

    Coordinates:
    - Query planning and intent detection
    - Symbol index searches
    - Semantic code search
    - Reference resolution
    - Context assembly
    - Result caching

    PBTSO Phase: PLAN, DISTRIBUTE

    Example:
        orchestrator = ResearchOrchestrator(root="/project")
        await orchestrator.initialize()

        result = await orchestrator.research("where is UserService defined?")
        print(result.results)
    """

    def __init__(
        self,
        config: Optional[ResearchOrchestratorConfig] = None,
        bus: Optional[AgentBus] = None,
    ):
        """
        Initialize the research orchestrator.

        Args:
            config: Orchestrator configuration
            bus: AgentBus for event emission
        """
        self.config = config or ResearchOrchestratorConfig()
        self.bus = bus or AgentBus()

        self.root = Path(self.config.root)

        # Component instances (initialized lazily)
        self._scanner: Optional[CodebaseScanner] = None
        self._symbol_store: Optional[SymbolIndexStore] = None
        self._dependency_builder: Optional[DependencyGraphBuilder] = None
        self._semantic_engine: Optional[SemanticSearchEngine] = None
        self._query_planner: Optional[QueryPlanner] = None
        self._reference_resolver: Optional[ReferenceResolver] = None
        self._impact_analyzer: Optional[ImpactAnalyzer] = None
        self._pattern_detector: Optional[PatternDetector] = None
        self._architecture_mapper: Optional[ArchitectureMapper] = None
        self._knowledge_extractor: Optional[KnowledgeExtractor] = None
        self._cache: Optional[CacheManager] = None
        self._context_assembler: Optional[ContextAssembler] = None

        # State
        self._initialized = False
        self._phase_handlers: Dict[ResearchPhase, Callable] = {}

        # Register default phase handlers
        self._register_default_handlers()

    async def initialize(self) -> bool:
        """
        Initialize all research components.

        Returns:
            True if initialization successful
        """
        try:
            self.bus.emit({
                "topic": "research.pipeline.start",
                "kind": "lifecycle",
                "data": {"phase": "initialize", "root": str(self.root)}
            })

            # Initialize core components
            self._symbol_store = SymbolIndexStore(bus=self.bus)
            self._query_planner = QueryPlanner(bus=self.bus)

            # Initialize cache
            if self.config.enable_caching:
                cache_config = CacheConfig(namespace="research")
                self._cache = CacheManager(config=cache_config, bus=self.bus)

            # Initialize dependency graph
            self._dependency_builder = DependencyGraphBuilder(
                root=self.root, bus=self.bus
            )

            # Initialize reference resolver
            self._reference_resolver = ReferenceResolver(
                root=self.root,
                symbol_store=self._symbol_store,
                bus=self.bus,
            )

            # Initialize semantic search if enabled
            if self.config.enable_semantic:
                semantic_config = SemanticSearchConfig(embedding_model="mock")
                self._semantic_engine = SemanticSearchEngine(
                    config=semantic_config, bus=self.bus
                )

            # Initialize analysis components
            self._impact_analyzer = ImpactAnalyzer(
                root=self.root,
                symbol_store=self._symbol_store,
                bus=self.bus,
            )

            self._pattern_detector = PatternDetector(bus=self.bus)

            self._architecture_mapper = ArchitectureMapper(
                root=self.root,
                symbol_store=self._symbol_store,
                bus=self.bus,
            )

            self._knowledge_extractor = KnowledgeExtractor(
                root=self.root,
                symbol_store=self._symbol_store,
                bus=self.bus,
            )

            self._initialized = True

            self.bus.emit({
                "topic": "research.pipeline.complete",
                "kind": "lifecycle",
                "data": {"phase": "initialize", "success": True}
            })

            return True

        except Exception as e:
            self.bus.emit({
                "topic": "research.pipeline.error",
                "kind": "error",
                "level": "error",
                "data": {"phase": "initialize", "error": str(e)}
            })
            return False

    async def research(
        self,
        query: Union[str, ResearchQuery],
        **kwargs,
    ) -> ResearchResult:
        """
        Execute a research query.

        Args:
            query: Query string or ResearchQuery object
            **kwargs: Additional options passed to ResearchQuery

        Returns:
            ResearchResult with findings
        """
        start_time = time.time()

        # Normalize query
        if isinstance(query, str):
            query = ResearchQuery(query=query, **kwargs)

        # Check cache
        if self._cache and self.config.enable_caching:
            cache_key = self._cache.make_key("query", self._cache.hash_key(query.query))
            cached = self._cache.get(cache_key)
            if cached:
                cached["cache_hit"] = True
                return ResearchResult(**cached)

        # Emit start event
        self.bus.emit({
            "topic": "a2a.research.query",
            "kind": "query",
            "data": {"query": query.query}
        })

        result = ResearchResult(
            query=query,
            intent=QueryIntent.SEARCH_CONCEPT,
            results=[],
        )

        try:
            # Phase 1: Plan
            plan = await self._execute_phase(ResearchPhase.PLAN, query, result)
            result.phases_completed.append("plan")
            result.intent = plan.intent

            # Phase 2: Search
            search_results = await self._execute_phase(ResearchPhase.SEARCH, query, result, plan=plan)
            result.phases_completed.append("search")
            result.results = search_results

            # Phase 3: Analyze
            analyzed = await self._execute_phase(ResearchPhase.ANALYZE, query, result)
            result.phases_completed.append("analyze")

            # Phase 4: Assemble context
            if query.include_context:
                context = await self._execute_phase(ResearchPhase.ASSEMBLE, query, result)
                result.context = context
                result.phases_completed.append("assemble")

            # Phase 5: Synthesize
            await self._execute_phase(ResearchPhase.SYNTHESIZE, query, result)
            result.phases_completed.append("synthesize")

        except Exception as e:
            result.errors.append(str(e))
            self.bus.emit({
                "topic": "research.pipeline.error",
                "kind": "error",
                "data": {"query": query.query, "error": str(e)}
            })

        result.execution_time_ms = (time.time() - start_time) * 1000

        # Cache result
        if self._cache and self.config.enable_caching and not result.errors:
            cache_key = self._cache.make_key("query", self._cache.hash_key(query.query))
            self._cache.set(cache_key, {
                "query": query.to_dict(),
                "intent": result.intent.value,
                "results": result.results,
                "execution_time_ms": result.execution_time_ms,
                "phases_completed": result.phases_completed,
                "errors": result.errors,
            }, ttl=300)  # Cache for 5 minutes

        # Emit completion
        self.bus.emit({
            "topic": "a2a.research.orchestrate",
            "kind": "complete",
            "data": result.to_dict()
        })

        return result

    async def search_symbol(
        self,
        name: str,
        kind: Optional[str] = None,
    ) -> List[Symbol]:
        """
        Search for a symbol by name.

        Args:
            name: Symbol name
            kind: Symbol kind (class, function, etc.)

        Returns:
            List of matching symbols
        """
        if not self._symbol_store:
            return []

        return self._symbol_store.query(name=name, kind=kind, limit=self.config.max_results)

    async def find_usages(self, symbol_name: str, path: Optional[str] = None) -> List[Dict[str, Any]]:
        """
        Find all usages of a symbol.

        Args:
            symbol_name: Symbol name
            path: Optional path where symbol is defined

        Returns:
            List of usage locations
        """
        if not self._reference_resolver:
            return []

        usages = self._reference_resolver.find_usages(symbol_name, path)
        return [u.to_dict() for u in usages]

    async def analyze_impact(
        self,
        path: str,
        symbol_name: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Analyze impact of changing a file or symbol.

        Args:
            path: File path
            symbol_name: Optional symbol name

        Returns:
            Impact analysis report
        """
        if not self._impact_analyzer:
            return {}

        if symbol_name:
            report = self._impact_analyzer.analyze_symbol_change(path, symbol_name)
        else:
            report = self._impact_analyzer.analyze_file_change(path)

        return report.to_dict()

    async def get_architecture(self) -> Dict[str, Any]:
        """
        Get architecture map of the codebase.

        Returns:
            Architecture map
        """
        if not self._architecture_mapper:
            return {}

        arch = self._architecture_mapper.map_architecture()
        return arch.to_dict()

    async def detect_patterns(self, path: str) -> List[Dict[str, Any]]:
        """
        Detect code patterns in a file.

        Args:
            path: File path

        Returns:
            List of detected patterns
        """
        if not self._pattern_detector:
            return []

        matches = self._pattern_detector.detect_patterns(path)
        return [m.to_dict() for m in matches]

    async def extract_knowledge(self) -> Dict[str, Any]:
        """
        Extract domain knowledge from codebase.

        Returns:
            Knowledge graph
        """
        if not self._knowledge_extractor:
            return {}

        graph = self._knowledge_extractor.extract_knowledge()
        return graph.to_dict()

    def register_phase_handler(
        self,
        phase: ResearchPhase,
        handler: Callable[..., Coroutine],
    ) -> None:
        """
        Register a custom phase handler.

        Args:
            phase: Research phase
            handler: Async handler function
        """
        self._phase_handlers[phase] = handler

    async def _execute_phase(
        self,
        phase: ResearchPhase,
        query: ResearchQuery,
        result: ResearchResult,
        **kwargs,
    ) -> Any:
        """Execute a research phase."""
        handler = self._phase_handlers.get(phase)
        if handler:
            return await handler(query, result, **kwargs)
        return None

    def _register_default_handlers(self) -> None:
        """Register default phase handlers."""
        self._phase_handlers[ResearchPhase.PLAN] = self._phase_plan
        self._phase_handlers[ResearchPhase.SEARCH] = self._phase_search
        self._phase_handlers[ResearchPhase.ANALYZE] = self._phase_analyze
        self._phase_handlers[ResearchPhase.ASSEMBLE] = self._phase_assemble
        self._phase_handlers[ResearchPhase.SYNTHESIZE] = self._phase_synthesize

    async def _phase_plan(
        self,
        query: ResearchQuery,
        result: ResearchResult,
        **kwargs,
    ) -> QueryPlan:
        """Plan phase: Create query execution plan."""
        context = {}
        if query.context_file:
            context["current_file"] = query.context_file
        if query.context_selection:
            context["selection"] = query.context_selection

        plan = self._query_planner.plan_query(query.query, context=context)
        return plan

    async def _phase_search(
        self,
        query: ResearchQuery,
        result: ResearchResult,
        plan: Optional[QueryPlan] = None,
        **kwargs,
    ) -> List[Dict[str, Any]]:
        """Search phase: Execute searches based on plan."""
        all_results = []

        if plan is None:
            # Default search
            plan = self._query_planner.plan_query(query.query)

        # Execute searches based on plan intent
        if plan.intent == QueryIntent.FIND_SYMBOL:
            # Search symbol index
            symbols = self._symbol_store.search(query.query, limit=query.max_results)
            for s in symbols:
                all_results.append({
                    "type": "symbol",
                    "name": s.name,
                    "kind": s.kind,
                    "path": s.path,
                    "line": s.line,
                    "signature": s.signature,
                    "docstring": s.docstring,
                })

        elif plan.intent == QueryIntent.FIND_USAGE:
            # Find usages
            # Extract symbol name from query
            words = query.query.split()
            for word in words:
                if word[0].isupper() or "_" in word:
                    usages = self._reference_resolver.find_usages(word)
                    for u in usages:
                        all_results.append({
                            "type": "usage",
                            "symbol": u.reference.symbol_name,
                            "path": u.reference.source_path,
                            "line": u.reference.source_line,
                            "context": u.reference.context,
                        })
                    break

        elif plan.intent in (QueryIntent.SEARCH_CONCEPT, QueryIntent.EXPLAIN_CODE):
            # Semantic search
            if self._semantic_engine:
                results = self._semantic_engine.search(query.query, top_k=query.max_results)
                for r in results:
                    all_results.append({
                        "type": "semantic",
                        "path": r.path,
                        "content": r.content[:500],
                        "score": r.score,
                        "line_start": r.line_start,
                        "line_end": r.line_end,
                    })

            # Also do symbol search
            symbols = self._symbol_store.search(query.query, limit=query.max_results // 2)
            for s in symbols:
                all_results.append({
                    "type": "symbol",
                    "name": s.name,
                    "kind": s.kind,
                    "path": s.path,
                    "line": s.line,
                })

        elif plan.intent == QueryIntent.TRACE_DEPENDENCY:
            # Trace dependencies
            # Find relevant paths
            if query.context_file:
                deps = self._dependency_builder.get_dependencies(query.context_file)
                for dep in deps:
                    all_results.append({
                        "type": "dependency",
                        "path": dep,
                        "direction": "imports",
                    })

                dependents = self._dependency_builder.get_dependents(query.context_file)
                for dep in dependents:
                    all_results.append({
                        "type": "dependency",
                        "path": dep,
                        "direction": "imported_by",
                    })

        else:
            # Default: combined search
            symbols = self._symbol_store.search(query.query, limit=query.max_results)
            for s in symbols:
                all_results.append({
                    "type": "symbol",
                    "name": s.name,
                    "kind": s.kind,
                    "path": s.path,
                    "line": s.line,
                })

        return all_results[:query.max_results]

    async def _phase_analyze(
        self,
        query: ResearchQuery,
        result: ResearchResult,
        **kwargs,
    ) -> Dict[str, Any]:
        """Analyze phase: Enhance results with analysis."""
        analysis = {}

        # Get unique paths from results
        paths = set()
        for r in result.results:
            if "path" in r:
                paths.add(r["path"])

        # Detect patterns in result files
        if paths and self._pattern_detector:
            for path in list(paths)[:5]:  # Limit to 5 files
                patterns = self._pattern_detector.detect_patterns(path)
                if patterns:
                    analysis.setdefault("patterns", []).extend([p.to_dict() for p in patterns])

        return analysis

    async def _phase_assemble(
        self,
        query: ResearchQuery,
        result: ResearchResult,
        **kwargs,
    ) -> AssembledContext:
        """Assemble phase: Build context from results."""
        assembler = ContextAssembler()

        # Add results to context
        for r in result.results:
            if r.get("type") == "semantic" and "content" in r:
                assembler.add_chunk(
                    content=r["content"],
                    source=r.get("path", "unknown"),
                    priority=ContextPriority.HIGH,
                    relevance=r.get("score", 0.5),
                    line_start=r.get("line_start"),
                    line_end=r.get("line_end"),
                )

            elif r.get("type") == "symbol" and r.get("path"):
                # Read symbol context from file
                try:
                    content = Path(r["path"]).read_text(errors="ignore")
                    lines = content.split("\n")
                    line = r.get("line", 1) - 1
                    start = max(0, line - 5)
                    end = min(len(lines), line + 20)
                    chunk = "\n".join(lines[start:end])

                    assembler.add_chunk(
                        content=chunk,
                        source=r["path"],
                        priority=ContextPriority.HIGH,
                        relevance=0.8,
                        line_start=start + 1,
                        line_end=end,
                    )
                except Exception:
                    pass

        # Add query context file if provided
        if query.context_file:
            assembler.add_file(
                query.context_file,
                priority=ContextPriority.CRITICAL,
                relevance=1.0,
            )

        return assembler.assemble(max_tokens=self.config.context_tokens)

    async def _phase_synthesize(
        self,
        query: ResearchQuery,
        result: ResearchResult,
        **kwargs,
    ) -> None:
        """Synthesize phase: Final processing and ranking."""
        # Sort results by relevance/score
        result.results.sort(
            key=lambda r: r.get("score", 0.5),
            reverse=True
        )

        # Deduplicate by path+line
        seen = set()
        unique_results = []
        for r in result.results:
            key = f"{r.get('path', '')}:{r.get('line', '')}"
            if key not in seen:
                seen.add(key)
                unique_results.append(r)

        result.results = unique_results

    def get_stats(self) -> Dict[str, Any]:
        """Get orchestrator statistics."""
        stats = {
            "initialized": self._initialized,
            "root": str(self.root),
        }

        if self._symbol_store:
            stats["symbol_store"] = self._symbol_store.stats()

        if self._cache:
            stats["cache"] = self._cache.get_stats().to_dict()

        if self._semantic_engine:
            stats["semantic_engine"] = self._semantic_engine.stats()

        return stats


# ============================================================================
# CLI Entry Point
# ============================================================================


def main() -> int:
    """CLI entry point for Research Orchestrator."""
    import argparse

    parser = argparse.ArgumentParser(
        description="Research Orchestrator (Step 20)"
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    # Research command
    research_parser = subparsers.add_parser("research", help="Execute research query")
    research_parser.add_argument("query", help="Research query")
    research_parser.add_argument("--context-file", help="Context file path")
    research_parser.add_argument("--max-results", type=int, default=20)
    research_parser.add_argument("--json", action="store_true", help="Output as JSON")

    # Stats command
    stats_parser = subparsers.add_parser("stats", help="Show orchestrator statistics")
    stats_parser.add_argument("--json", action="store_true", help="Output as JSON")

    # Init command
    init_parser = subparsers.add_parser("init", help="Initialize orchestrator")
    init_parser.add_argument("--root", default=".", help="Project root")

    args = parser.parse_args()

    async def run():
        config = ResearchOrchestratorConfig(
            root=getattr(args, "root", "."),
        )
        orchestrator = ResearchOrchestrator(config)

        if args.command == "init":
            success = await orchestrator.initialize()
            if success:
                print("Research orchestrator initialized successfully.")
            else:
                print("Failed to initialize orchestrator.")
            return 0 if success else 1

        elif args.command == "research":
            await orchestrator.initialize()

            result = await orchestrator.research(
                args.query,
                context_file=args.context_file,
                max_results=args.max_results,
            )

            if args.json:
                output = {
                    "query": result.query.query,
                    "intent": result.intent.value,
                    "results": result.results,
                    "execution_time_ms": result.execution_time_ms,
                    "errors": result.errors,
                }
                print(json.dumps(output, indent=2))
            else:
                print(f"Research Results for: {result.query.query}")
                print(f"Intent: {result.intent.value}")
                print(f"Execution Time: {result.execution_time_ms:.1f}ms")
                print(f"\nResults ({len(result.results)}):")
                for r in result.results[:10]:
                    print(f"  [{r.get('type', 'unknown')}] {r.get('name', r.get('path', 'N/A'))}")
                    if r.get('line'):
                        print(f"    Line {r['line']}")

            return 0

        elif args.command == "stats":
            await orchestrator.initialize()
            stats = orchestrator.get_stats()

            if args.json:
                print(json.dumps(stats, indent=2))
            else:
                print("Research Orchestrator Statistics:")
                print(f"  Initialized: {stats['initialized']}")
                print(f"  Root: {stats['root']}")
                if "symbol_store" in stats:
                    print(f"  Symbols: {stats['symbol_store'].get('total', 0)}")
                if "cache" in stats:
                    print(f"  Cache Hit Rate: {stats['cache'].get('hit_rate', 0):.1%}")

            return 0

        return 1

    return asyncio.run(run())


if __name__ == "__main__":
    import sys
    sys.exit(main())
