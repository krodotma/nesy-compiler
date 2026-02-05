#!/usr/bin/env python3
"""
knowledge_extractor.py - Knowledge Extractor (Step 17)

Extract domain knowledge from codebase including concepts, relationships,
glossary terms, and business rules.

PBTSO Phase: RESEARCH, DISTILL

Bus Topics:
- a2a.research.knowledge.extract
- research.knowledge.concept
- research.knowledge.relation

Protocol: DKIN v30, PAIP v16, CITIZEN v2
"""
from __future__ import annotations

import ast
import json
import os
import re
from collections import Counter, defaultdict
from dataclasses import dataclass, field, asdict
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

from ..bootstrap import AgentBus
from ..index.symbol_store import SymbolIndexStore, Symbol
from ..parsers.base import ParseResult


# ============================================================================
# Data Models
# ============================================================================


class KnowledgeType(Enum):
    """Type of extracted knowledge."""
    CONCEPT = "concept"          # Domain concept/entity
    RELATIONSHIP = "relationship"  # Relationship between concepts
    RULE = "rule"                # Business rule
    GLOSSARY = "glossary"        # Term definition
    PATTERN = "pattern"          # Usage pattern
    CONSTRAINT = "constraint"    # Validation/constraint


@dataclass
class Knowledge:
    """A piece of extracted knowledge."""

    knowledge_type: KnowledgeType
    name: str
    description: str
    source_path: str
    source_line: Optional[int] = None
    confidence: float = 0.8
    related: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "type": self.knowledge_type.value,
            "name": self.name,
            "description": self.description,
            "source_path": self.source_path,
            "source_line": self.source_line,
            "confidence": self.confidence,
            "related": self.related,
            "metadata": self.metadata,
        }


@dataclass
class Concept:
    """A domain concept extracted from code."""

    name: str
    description: str
    attributes: List[str] = field(default_factory=list)
    operations: List[str] = field(default_factory=list)
    relationships: List[Tuple[str, str]] = field(default_factory=list)  # (type, target)
    source_paths: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "name": self.name,
            "description": self.description,
            "attributes": self.attributes,
            "operations": self.operations,
            "relationships": [{"type": r[0], "target": r[1]} for r in self.relationships],
            "source_count": len(self.source_paths),
        }


@dataclass
class KnowledgeGraph:
    """Graph of extracted knowledge."""

    concepts: Dict[str, Concept] = field(default_factory=dict)
    knowledge: List[Knowledge] = field(default_factory=list)
    glossary: Dict[str, str] = field(default_factory=dict)
    metrics: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "concepts": {k: v.to_dict() for k, v in self.concepts.items()},
            "knowledge_count": len(self.knowledge),
            "glossary": self.glossary,
            "metrics": self.metrics,
        }

    def to_cypher(self) -> List[str]:
        """Generate Cypher queries for graph database."""
        queries = []

        # Create concept nodes
        for name, concept in self.concepts.items():
            queries.append(
                f"MERGE (c:Concept {{name: '{name}'}}) "
                f"SET c.description = '{concept.description[:200]}'"
            )

            # Create relationships
            for rel_type, target in concept.relationships:
                queries.append(
                    f"MATCH (a:Concept {{name: '{name}'}}), (b:Concept {{name: '{target}'}}) "
                    f"MERGE (a)-[:{rel_type.upper()}]->(b)"
                )

        return queries


# ============================================================================
# Knowledge Extractor
# ============================================================================


class KnowledgeExtractor:
    """
    Extract domain knowledge from codebase.

    Extracts:
    - Domain concepts from class names and docstrings
    - Relationships from inheritance and composition
    - Glossary terms from documentation
    - Business rules from validation logic
    - Usage patterns from common code structures

    PBTSO Phase: RESEARCH, DISTILL

    Example:
        extractor = KnowledgeExtractor(root="/project")
        graph = extractor.extract_knowledge()
        print(graph.concepts)
    """

    # Common words to ignore in concept extraction
    STOP_WORDS = {
        "base", "abstract", "mixin", "helper", "utils", "util",
        "manager", "handler", "processor", "service", "factory",
        "builder", "impl", "implementation", "test", "mock",
    }

    # Relationship indicators in docstrings
    RELATIONSHIP_PATTERNS = {
        "is_a": [r"is\s+a\s+(\w+)", r"extends\s+(\w+)", r"inherits\s+from\s+(\w+)"],
        "has_a": [r"has\s+(?:a|an|many)\s+(\w+)", r"contains?\s+(\w+)"],
        "uses": [r"uses?\s+(\w+)", r"depends?\s+on\s+(\w+)"],
        "creates": [r"creates?\s+(\w+)", r"produces?\s+(\w+)"],
    }

    def __init__(
        self,
        root: Optional[Path] = None,
        symbol_store: Optional[SymbolIndexStore] = None,
        bus: Optional[AgentBus] = None,
    ):
        """
        Initialize the knowledge extractor.

        Args:
            root: Project root directory
            symbol_store: Symbol index store
            bus: AgentBus for event emission
        """
        self.root = Path(root) if root else Path.cwd()
        self.symbol_store = symbol_store or SymbolIndexStore()
        self.bus = bus or AgentBus()

    def extract_knowledge(self) -> KnowledgeGraph:
        """
        Extract all knowledge from the codebase.

        Returns:
            KnowledgeGraph with concepts, relationships, and glossary
        """
        graph = KnowledgeGraph()

        # Extract concepts from classes
        self._extract_concepts(graph)

        # Extract relationships
        self._extract_relationships(graph)

        # Extract glossary from docstrings
        self._extract_glossary(graph)

        # Extract business rules
        self._extract_rules(graph)

        # Calculate metrics
        graph.metrics = self._calculate_metrics(graph)

        # Emit event
        self.bus.emit({
            "topic": "a2a.research.knowledge.extract",
            "kind": "knowledge",
            "data": {
                "concepts": len(graph.concepts),
                "knowledge": len(graph.knowledge),
                "glossary_terms": len(graph.glossary),
            }
        })

        return graph

    def extract_concepts_from_file(self, file_path: str) -> List[Concept]:
        """
        Extract concepts from a single file.

        Args:
            file_path: Path to the file

        Returns:
            List of extracted concepts
        """
        concepts = []

        try:
            content = Path(file_path).read_text(errors="ignore")
            tree = ast.parse(content)
        except Exception:
            return concepts

        for node in ast.walk(tree):
            if isinstance(node, ast.ClassDef):
                concept = self._class_to_concept(node, file_path)
                if concept:
                    concepts.append(concept)

        return concepts

    def find_related_concepts(self, concept_name: str) -> List[str]:
        """
        Find concepts related to the given concept.

        Args:
            concept_name: Name of the concept

        Returns:
            List of related concept names
        """
        graph = self.extract_knowledge()

        related = set()

        if concept_name in graph.concepts:
            concept = graph.concepts[concept_name]
            for rel_type, target in concept.relationships:
                related.add(target)

        # Also find reverse relationships
        for name, concept in graph.concepts.items():
            for rel_type, target in concept.relationships:
                if target == concept_name:
                    related.add(name)

        return list(related)

    def search_glossary(self, term: str) -> Optional[str]:
        """
        Search glossary for a term.

        Args:
            term: Term to search for

        Returns:
            Definition or None
        """
        graph = self.extract_knowledge()

        # Exact match
        if term in graph.glossary:
            return graph.glossary[term]

        # Case-insensitive search
        term_lower = term.lower()
        for key, definition in graph.glossary.items():
            if key.lower() == term_lower:
                return definition

        return None

    def _extract_concepts(self, graph: KnowledgeGraph) -> None:
        """Extract concepts from classes."""
        classes = self.symbol_store.query(kind="class", limit=10000)

        for symbol in classes:
            name = self._normalize_concept_name(symbol.name)

            if not name or name.lower() in self.STOP_WORDS:
                continue

            if name not in graph.concepts:
                graph.concepts[name] = Concept(
                    name=name,
                    description=symbol.docstring or "",
                    source_paths=[symbol.path],
                )
            else:
                if symbol.path not in graph.concepts[name].source_paths:
                    graph.concepts[name].source_paths.append(symbol.path)

            # Add attributes from methods starting with get_/set_
            methods = self.symbol_store.query(kind="method", parent=symbol.name, limit=100)
            for method in methods:
                if method.name.startswith("get_"):
                    attr = method.name[4:]
                    if attr not in graph.concepts[name].attributes:
                        graph.concepts[name].attributes.append(attr)
                elif not method.name.startswith("_"):
                    if method.name not in graph.concepts[name].operations:
                        graph.concepts[name].operations.append(method.name)

    def _extract_relationships(self, graph: KnowledgeGraph) -> None:
        """Extract relationships between concepts."""
        classes = self.symbol_store.query(kind="class", limit=10000)

        for symbol in classes:
            concept_name = self._normalize_concept_name(symbol.name)
            if concept_name not in graph.concepts:
                continue

            # Try to parse the file to get inheritance info
            try:
                content = Path(symbol.path).read_text(errors="ignore")
                tree = ast.parse(content)

                for node in ast.walk(tree):
                    if isinstance(node, ast.ClassDef) and node.name == symbol.name:
                        # Extract inheritance relationships
                        for base in node.bases:
                            base_name = self._get_name(base)
                            if base_name:
                                normalized = self._normalize_concept_name(base_name)
                                if normalized and normalized != concept_name:
                                    graph.concepts[concept_name].relationships.append(
                                        ("is_a", normalized)
                                    )

                        # Extract composition from type hints
                        for item in node.body:
                            if isinstance(item, ast.AnnAssign):
                                if item.annotation:
                                    type_name = self._get_type_name(item.annotation)
                                    if type_name:
                                        normalized = self._normalize_concept_name(type_name)
                                        if normalized and normalized != concept_name:
                                            graph.concepts[concept_name].relationships.append(
                                                ("has_a", normalized)
                                            )

            except Exception:
                pass

            # Extract from docstring
            if symbol.docstring:
                for rel_type, patterns in self.RELATIONSHIP_PATTERNS.items():
                    for pattern in patterns:
                        matches = re.findall(pattern, symbol.docstring, re.IGNORECASE)
                        for match in matches:
                            normalized = self._normalize_concept_name(match)
                            if normalized and normalized != concept_name:
                                graph.concepts[concept_name].relationships.append(
                                    (rel_type, normalized)
                                )

    def _extract_glossary(self, graph: KnowledgeGraph) -> None:
        """Extract glossary terms from docstrings."""
        # Get all symbols with docstrings
        all_symbols = self.symbol_store.search("", limit=1000)

        for symbol in all_symbols:
            if not symbol.docstring:
                continue

            # Look for definition patterns in docstrings
            patterns = [
                r"(?:^|\n)\s*(\w+)\s*:\s*(.+?)(?:\n|$)",  # term: definition
                r"(?:^|\n)\s*-\s*(\w+)\s*[-:]\s*(.+?)(?:\n|$)",  # - term: definition
            ]

            for pattern in patterns:
                matches = re.findall(pattern, symbol.docstring)
                for term, definition in matches:
                    if len(term) > 2 and len(definition) > 10:
                        term_clean = term.strip()
                        if term_clean not in graph.glossary:
                            graph.glossary[term_clean] = definition.strip()

            # Also add class/function names with their docstrings
            if symbol.kind in ("class", "function"):
                name = self._normalize_concept_name(symbol.name)
                if name and name not in graph.glossary:
                    # Take first line of docstring as definition
                    first_line = symbol.docstring.split("\n")[0].strip()
                    if len(first_line) > 10:
                        graph.glossary[name] = first_line

    def _extract_rules(self, graph: KnowledgeGraph) -> None:
        """Extract business rules from validation logic."""
        # Look for validation patterns
        functions = self.symbol_store.query(kind="function", limit=10000)

        for symbol in functions:
            # Check for validation-related names
            if any(kw in symbol.name.lower() for kw in ["validate", "check", "verify", "assert"]):
                knowledge = Knowledge(
                    knowledge_type=KnowledgeType.RULE,
                    name=symbol.name,
                    description=symbol.docstring or f"Validation rule: {symbol.name}",
                    source_path=symbol.path,
                    source_line=symbol.line,
                    confidence=0.7,
                )
                graph.knowledge.append(knowledge)

                self.bus.emit({
                    "topic": "research.knowledge.concept",
                    "kind": "rule",
                    "data": knowledge.to_dict()
                })

        # Look for constraint patterns in code
        for py_file in self.root.rglob("*.py"):
            try:
                content = py_file.read_text(errors="ignore")

                # Find raise statements with ValueError/ValidationError
                pattern = r"raise\s+(ValueError|ValidationError)\s*\(([^)]+)\)"
                matches = re.findall(pattern, content)

                for exc_type, message in matches:
                    knowledge = Knowledge(
                        knowledge_type=KnowledgeType.CONSTRAINT,
                        name=f"Constraint: {message[:50]}",
                        description=message,
                        source_path=str(py_file),
                        confidence=0.6,
                    )
                    graph.knowledge.append(knowledge)

            except Exception:
                pass

    def _class_to_concept(self, node: ast.ClassDef, file_path: str) -> Optional[Concept]:
        """Convert a class AST node to a Concept."""
        name = self._normalize_concept_name(node.name)
        if not name or name.lower() in self.STOP_WORDS:
            return None

        docstring = ast.get_docstring(node) or ""

        # Extract attributes
        attributes = []
        operations = []

        for item in node.body:
            if isinstance(item, ast.FunctionDef):
                if not item.name.startswith("_"):
                    operations.append(item.name)
                elif item.name.startswith("get_"):
                    attributes.append(item.name[4:])
            elif isinstance(item, ast.AnnAssign):
                if isinstance(item.target, ast.Name):
                    attributes.append(item.target.id)

        # Extract relationships from bases
        relationships = []
        for base in node.bases:
            base_name = self._get_name(base)
            if base_name:
                normalized = self._normalize_concept_name(base_name)
                if normalized and normalized != name:
                    relationships.append(("is_a", normalized))

        return Concept(
            name=name,
            description=docstring.split("\n")[0] if docstring else "",
            attributes=attributes,
            operations=operations,
            relationships=relationships,
            source_paths=[file_path],
        )

    def _normalize_concept_name(self, name: str) -> Optional[str]:
        """Normalize a name for concept extraction."""
        if not name:
            return None

        # Remove common suffixes/prefixes
        for suffix in ["Base", "Abstract", "Mixin", "Impl", "Interface"]:
            if name.endswith(suffix) and len(name) > len(suffix):
                name = name[:-len(suffix)]
            if name.startswith(suffix) and len(name) > len(suffix):
                name = name[len(suffix):]

        # Convert CamelCase to spaces for readability (internal use keeps original)
        if name and name[0].isupper():
            return name

        return None

    def _get_name(self, node: ast.expr) -> Optional[str]:
        """Extract name from AST expression."""
        if isinstance(node, ast.Name):
            return node.id
        elif isinstance(node, ast.Attribute):
            return node.attr
        return None

    def _get_type_name(self, node: ast.expr) -> Optional[str]:
        """Extract type name from annotation."""
        if isinstance(node, ast.Name):
            return node.id
        elif isinstance(node, ast.Attribute):
            return node.attr
        elif isinstance(node, ast.Subscript):
            if isinstance(node.value, ast.Name):
                return node.value.id
        return None

    def _calculate_metrics(self, graph: KnowledgeGraph) -> Dict[str, Any]:
        """Calculate knowledge graph metrics."""
        # Relationship counts
        rel_counts = Counter()
        for concept in graph.concepts.values():
            for rel_type, _ in concept.relationships:
                rel_counts[rel_type] += 1

        # Most connected concepts
        connection_counts = {
            name: len(concept.relationships)
            for name, concept in graph.concepts.items()
        }
        top_connected = sorted(connection_counts.items(), key=lambda x: -x[1])[:10]

        return {
            "total_concepts": len(graph.concepts),
            "total_relationships": sum(rel_counts.values()),
            "relationship_types": dict(rel_counts),
            "glossary_terms": len(graph.glossary),
            "business_rules": sum(1 for k in graph.knowledge if k.knowledge_type == KnowledgeType.RULE),
            "top_connected": top_connected,
        }


# ============================================================================
# CLI Entry Point
# ============================================================================


def main() -> int:
    """CLI entry point for Knowledge Extractor."""
    import argparse

    parser = argparse.ArgumentParser(
        description="Knowledge Extractor (Step 17)"
    )
    parser.add_argument(
        "root",
        nargs="?",
        default=".",
        help="Project root directory"
    )
    parser.add_argument(
        "--concepts",
        action="store_true",
        help="Show extracted concepts"
    )
    parser.add_argument(
        "--glossary",
        action="store_true",
        help="Show glossary"
    )
    parser.add_argument(
        "--cypher",
        action="store_true",
        help="Output as Cypher queries"
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Output as JSON"
    )

    args = parser.parse_args()

    extractor = KnowledgeExtractor(root=Path(args.root))
    graph = extractor.extract_knowledge()

    if args.cypher:
        for query in graph.to_cypher():
            print(query)

    elif args.json:
        print(json.dumps(graph.to_dict(), indent=2))

    elif args.concepts:
        print(f"Extracted Concepts ({len(graph.concepts)}):\n")
        for name, concept in sorted(graph.concepts.items()):
            print(f"  {name}")
            if concept.description:
                print(f"    {concept.description[:80]}")
            if concept.attributes:
                print(f"    Attributes: {', '.join(concept.attributes[:5])}")
            if concept.operations:
                print(f"    Operations: {', '.join(concept.operations[:5])}")
            if concept.relationships:
                print(f"    Relationships:")
                for rel_type, target in concept.relationships[:5]:
                    print(f"      - {rel_type} -> {target}")
            print()

    elif args.glossary:
        print(f"Glossary ({len(graph.glossary)} terms):\n")
        for term, definition in sorted(graph.glossary.items()):
            print(f"  {term}: {definition[:100]}")

    else:
        print(f"Knowledge Graph for: {extractor.root}")
        print(f"\nMetrics:")
        for key, value in graph.metrics.items():
            print(f"  {key}: {value}")

    return 0


if __name__ == "__main__":
    import sys
    sys.exit(main())
