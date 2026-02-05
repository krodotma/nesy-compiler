#!/usr/bin/env python3
"""
impact_analyzer.py - Impact Analyzer (Step 14)

Change impact prediction and analysis.
Determines what files and symbols would be affected by a proposed change.

PBTSO Phase: PLAN, VERIFY

Bus Topics:
- a2a.research.impact.analyze
- research.impact.report
- research.impact.risk

Protocol: DKIN v30, PAIP v16, CITIZEN v2
"""
from __future__ import annotations

import json
import os
from dataclasses import dataclass, field, asdict
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

from ..bootstrap import AgentBus
from ..graph.dependency_builder import DependencyGraphBuilder
from ..index.symbol_store import SymbolIndexStore, Symbol
from .reference_resolver import ReferenceResolver, ReferenceType


# ============================================================================
# Data Models
# ============================================================================


class ImpactLevel(Enum):
    """Impact severity level."""
    CRITICAL = "critical"   # Breaking change, major refactoring needed
    HIGH = "high"           # Significant impact, careful review needed
    MEDIUM = "medium"       # Moderate impact, some changes needed
    LOW = "low"             # Minor impact, localized changes
    NONE = "none"           # No impact


class ChangeType(Enum):
    """Type of proposed change."""
    ADD = "add"
    MODIFY = "modify"
    REMOVE = "remove"
    RENAME = "rename"
    MOVE = "move"


@dataclass
class Change:
    """Represents a proposed code change."""

    change_type: ChangeType
    path: str
    symbol_name: Optional[str] = None
    old_value: Optional[str] = None
    new_value: Optional[str] = None
    line_start: Optional[int] = None
    line_end: Optional[int] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "change_type": self.change_type.value,
            "path": self.path,
            "symbol_name": self.symbol_name,
            "old_value": self.old_value,
            "new_value": self.new_value,
            "line_start": self.line_start,
            "line_end": self.line_end,
        }


@dataclass
class Impact:
    """Impact on a single file or symbol."""

    path: str
    symbol_name: Optional[str] = None
    level: ImpactLevel = ImpactLevel.LOW
    reason: str = ""
    line: Optional[int] = None
    required_changes: List[str] = field(default_factory=list)
    is_direct: bool = True  # Direct vs transitive impact

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "path": self.path,
            "symbol_name": self.symbol_name,
            "level": self.level.value,
            "reason": self.reason,
            "line": self.line,
            "required_changes": self.required_changes,
            "is_direct": self.is_direct,
        }


@dataclass
class ImpactReport:
    """Complete impact analysis report."""

    change: Change
    impacts: List[Impact]
    total_files_affected: int
    risk_score: float  # 0.0 to 1.0
    summary: str
    recommendations: List[str] = field(default_factory=list)
    analysis_time: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "change": self.change.to_dict(),
            "impacts": [i.to_dict() for i in self.impacts],
            "total_files_affected": self.total_files_affected,
            "risk_score": self.risk_score,
            "summary": self.summary,
            "recommendations": self.recommendations,
            "analysis_time": self.analysis_time,
        }

    @property
    def critical_impacts(self) -> List[Impact]:
        """Get critical level impacts."""
        return [i for i in self.impacts if i.level == ImpactLevel.CRITICAL]

    @property
    def high_impacts(self) -> List[Impact]:
        """Get high level impacts."""
        return [i for i in self.impacts if i.level == ImpactLevel.HIGH]


# ============================================================================
# Impact Analyzer
# ============================================================================


class ImpactAnalyzer:
    """
    Analyze the impact of proposed code changes.

    Determines:
    - Files that would need to be updated
    - Symbols that would be affected
    - Risk level of the change
    - Recommended precautions

    PBTSO Phase: PLAN, VERIFY

    Example:
        analyzer = ImpactAnalyzer(root="/project")
        change = Change(ChangeType.RENAME, "src/utils.py", "old_func", new_value="new_func")
        report = analyzer.analyze_impact(change)
    """

    # Weights for risk calculation
    RISK_WEIGHTS = {
        ImpactLevel.CRITICAL: 1.0,
        ImpactLevel.HIGH: 0.7,
        ImpactLevel.MEDIUM: 0.4,
        ImpactLevel.LOW: 0.1,
        ImpactLevel.NONE: 0.0,
    }

    def __init__(
        self,
        root: Optional[Path] = None,
        symbol_store: Optional[SymbolIndexStore] = None,
        dependency_builder: Optional[DependencyGraphBuilder] = None,
        reference_resolver: Optional[ReferenceResolver] = None,
        bus: Optional[AgentBus] = None,
    ):
        """
        Initialize the impact analyzer.

        Args:
            root: Project root directory
            symbol_store: Symbol index for lookups
            dependency_builder: Dependency graph builder
            reference_resolver: Reference resolver
            bus: AgentBus for event emission
        """
        self.root = Path(root) if root else Path.cwd()
        self.symbol_store = symbol_store or SymbolIndexStore()
        self.bus = bus or AgentBus()

        self.dependency_builder = dependency_builder or DependencyGraphBuilder(
            root=self.root, bus=self.bus
        )
        self.reference_resolver = reference_resolver or ReferenceResolver(
            root=self.root, symbol_store=self.symbol_store, bus=self.bus
        )

    def analyze_impact(self, change: Change) -> ImpactReport:
        """
        Analyze the impact of a proposed change.

        Args:
            change: The proposed change

        Returns:
            ImpactReport with analysis results
        """
        import time
        start_time = time.time()

        impacts: List[Impact] = []

        # Dispatch to specific analyzers based on change type
        if change.change_type == ChangeType.REMOVE:
            impacts = self._analyze_removal(change)
        elif change.change_type == ChangeType.RENAME:
            impacts = self._analyze_rename(change)
        elif change.change_type == ChangeType.MODIFY:
            impacts = self._analyze_modification(change)
        elif change.change_type == ChangeType.MOVE:
            impacts = self._analyze_move(change)
        elif change.change_type == ChangeType.ADD:
            impacts = self._analyze_addition(change)

        # Calculate risk score
        risk_score = self._calculate_risk_score(impacts)

        # Generate summary and recommendations
        summary = self._generate_summary(change, impacts)
        recommendations = self._generate_recommendations(change, impacts)

        # Get unique affected files
        affected_files = set(i.path for i in impacts)

        analysis_time = time.time() - start_time

        report = ImpactReport(
            change=change,
            impacts=impacts,
            total_files_affected=len(affected_files),
            risk_score=risk_score,
            summary=summary,
            recommendations=recommendations,
            analysis_time=analysis_time,
        )

        # Emit event
        self.bus.emit({
            "topic": "research.impact.report",
            "kind": "analysis",
            "data": {
                "change_type": change.change_type.value,
                "path": change.path,
                "files_affected": report.total_files_affected,
                "risk_score": risk_score,
            }
        })

        return report

    def analyze_file_change(self, path: str) -> ImpactReport:
        """
        Analyze impact of changing an entire file.

        Args:
            path: File path being modified

        Returns:
            ImpactReport
        """
        change = Change(
            change_type=ChangeType.MODIFY,
            path=path,
        )
        return self.analyze_impact(change)

    def analyze_symbol_change(
        self,
        path: str,
        symbol_name: str,
        change_type: ChangeType = ChangeType.MODIFY,
    ) -> ImpactReport:
        """
        Analyze impact of changing a specific symbol.

        Args:
            path: File containing the symbol
            symbol_name: Name of the symbol
            change_type: Type of change

        Returns:
            ImpactReport
        """
        change = Change(
            change_type=change_type,
            path=path,
            symbol_name=symbol_name,
        )
        return self.analyze_impact(change)

    def get_dependents(self, path: str) -> List[str]:
        """
        Get files that depend on the given file.

        Args:
            path: File path

        Returns:
            List of dependent file paths
        """
        node = self.dependency_builder.get_node(path)
        if node:
            return list(node.imported_by)
        return []

    def get_dependencies(self, path: str) -> List[str]:
        """
        Get files that the given file depends on.

        Args:
            path: File path

        Returns:
            List of dependency file paths
        """
        node = self.dependency_builder.get_node(path)
        if node:
            return list(node.imports)
        return []

    def _analyze_removal(self, change: Change) -> List[Impact]:
        """Analyze impact of removing a file or symbol."""
        impacts = []

        if change.symbol_name:
            # Removing a symbol
            usages = self.reference_resolver.find_usages(
                change.symbol_name,
                symbol_path=change.path,
            )

            for usage in usages:
                if usage.reference.source_path != change.path:
                    impacts.append(Impact(
                        path=usage.reference.source_path,
                        symbol_name=usage.reference.symbol_name,
                        level=ImpactLevel.CRITICAL,
                        reason=f"Uses {change.symbol_name} which will be removed",
                        line=usage.reference.source_line,
                        required_changes=[f"Remove or replace usage of {change.symbol_name}"],
                        is_direct=True,
                    ))

        else:
            # Removing entire file
            dependents = self.get_dependents(change.path)

            for dep_path in dependents:
                impacts.append(Impact(
                    path=dep_path,
                    level=ImpactLevel.CRITICAL,
                    reason=f"Imports from {change.path} which will be removed",
                    required_changes=["Update imports to use alternative module"],
                    is_direct=True,
                ))

            # Check for transitive impacts
            for dep_path in dependents:
                transitive_deps = self.get_dependents(dep_path)
                for trans_path in transitive_deps:
                    if trans_path != change.path and trans_path not in dependents:
                        impacts.append(Impact(
                            path=trans_path,
                            level=ImpactLevel.MEDIUM,
                            reason=f"May be affected if {dep_path} changes",
                            is_direct=False,
                        ))

        return impacts

    def _analyze_rename(self, change: Change) -> List[Impact]:
        """Analyze impact of renaming a file or symbol."""
        impacts = []

        if change.symbol_name:
            # Renaming a symbol
            usages = self.reference_resolver.find_usages(
                change.symbol_name,
                symbol_path=change.path,
            )

            for usage in usages:
                impacts.append(Impact(
                    path=usage.reference.source_path,
                    symbol_name=usage.reference.symbol_name,
                    level=ImpactLevel.HIGH if usage.reference.source_path != change.path else ImpactLevel.LOW,
                    reason=f"Uses {change.symbol_name} which will be renamed to {change.new_value}",
                    line=usage.reference.source_line,
                    required_changes=[f"Rename {change.symbol_name} to {change.new_value}"],
                    is_direct=True,
                ))

        else:
            # Renaming/moving file
            dependents = self.get_dependents(change.path)

            for dep_path in dependents:
                impacts.append(Impact(
                    path=dep_path,
                    level=ImpactLevel.HIGH,
                    reason=f"Import path will change from {change.path} to {change.new_value}",
                    required_changes=["Update import statement"],
                    is_direct=True,
                ))

        return impacts

    def _analyze_modification(self, change: Change) -> List[Impact]:
        """Analyze impact of modifying a file or symbol."""
        impacts = []

        # Get symbol info if modifying specific symbol
        if change.symbol_name:
            symbols = self.symbol_store.query(
                name=change.symbol_name,
                path=change.path,
                limit=1,
            )

            if symbols:
                symbol = symbols[0]

                # Find usages
                usages = self.reference_resolver.find_usages(
                    change.symbol_name,
                    symbol_path=change.path,
                )

                for usage in usages:
                    # Determine impact level based on reference type
                    ref_type = usage.reference.reference_type

                    if ref_type == ReferenceType.INHERITANCE:
                        level = ImpactLevel.HIGH  # Changes to base class are significant
                    elif ref_type in (ReferenceType.CALL, ReferenceType.INSTANTIATION):
                        level = ImpactLevel.MEDIUM
                    else:
                        level = ImpactLevel.LOW

                    if usage.reference.source_path != change.path:
                        impacts.append(Impact(
                            path=usage.reference.source_path,
                            symbol_name=usage.reference.symbol_name,
                            level=level,
                            reason=f"Uses {change.symbol_name} ({ref_type.value})",
                            line=usage.reference.source_line,
                            is_direct=True,
                        ))

        else:
            # Modifying entire file - check all exports
            symbols = self.symbol_store.get_by_path(change.path)
            all_usages = set()

            for symbol in symbols:
                usages = self.reference_resolver.find_usages(
                    symbol.name,
                    symbol_path=change.path,
                )
                for usage in usages:
                    if usage.reference.source_path != change.path:
                        all_usages.add(usage.reference.source_path)

            for usage_path in all_usages:
                impacts.append(Impact(
                    path=usage_path,
                    level=ImpactLevel.MEDIUM,
                    reason=f"Uses symbols from {change.path}",
                    is_direct=True,
                ))

        return impacts

    def _analyze_move(self, change: Change) -> List[Impact]:
        """Analyze impact of moving a file."""
        # Similar to rename but also considers directory structure
        impacts = self._analyze_rename(change)

        # Additional check for relative imports
        from_dir = str(Path(change.path).parent)
        to_dir = str(Path(change.new_value or change.path).parent)

        if from_dir != to_dir:
            # Relative imports within the moved file may break
            impacts.append(Impact(
                path=change.path,
                level=ImpactLevel.MEDIUM,
                reason="Relative imports may need updating after move",
                required_changes=["Review and update relative imports"],
                is_direct=True,
            ))

        return impacts

    def _analyze_addition(self, change: Change) -> List[Impact]:
        """Analyze impact of adding a file or symbol."""
        impacts = []

        # Adding generally has low impact unless it shadows existing names
        if change.symbol_name:
            # Check for name conflicts
            existing = self.symbol_store.query(name=change.symbol_name, limit=10)

            for symbol in existing:
                if symbol.path != change.path:
                    impacts.append(Impact(
                        path=symbol.path,
                        symbol_name=symbol.name,
                        level=ImpactLevel.LOW,
                        reason=f"Symbol {change.symbol_name} already exists (potential confusion)",
                        is_direct=False,
                    ))

        return impacts

    def _calculate_risk_score(self, impacts: List[Impact]) -> float:
        """Calculate overall risk score from impacts."""
        if not impacts:
            return 0.0

        total_weight = sum(self.RISK_WEIGHTS[i.level] for i in impacts)
        max_possible = len(impacts) * self.RISK_WEIGHTS[ImpactLevel.CRITICAL]

        if max_possible == 0:
            return 0.0

        # Normalize to 0-1 range
        score = total_weight / max_possible

        # Boost score if there are critical impacts
        critical_count = sum(1 for i in impacts if i.level == ImpactLevel.CRITICAL)
        if critical_count > 0:
            score = min(1.0, score + 0.2 * critical_count)

        return round(score, 3)

    def _generate_summary(self, change: Change, impacts: List[Impact]) -> str:
        """Generate human-readable summary."""
        if not impacts:
            return f"{change.change_type.value.capitalize()} of {change.path} has no detected impact."

        critical = sum(1 for i in impacts if i.level == ImpactLevel.CRITICAL)
        high = sum(1 for i in impacts if i.level == ImpactLevel.HIGH)
        affected_files = len(set(i.path for i in impacts))

        parts = [f"{change.change_type.value.capitalize()} of "]

        if change.symbol_name:
            parts.append(f"'{change.symbol_name}' in {change.path}")
        else:
            parts.append(change.path)

        parts.append(f" affects {affected_files} file(s)")

        if critical > 0:
            parts.append(f" with {critical} critical impact(s)")
        elif high > 0:
            parts.append(f" with {high} high-priority update(s) needed")

        return "".join(parts) + "."

    def _generate_recommendations(
        self,
        change: Change,
        impacts: List[Impact],
    ) -> List[str]:
        """Generate recommendations based on impact analysis."""
        recommendations = []

        critical = [i for i in impacts if i.level == ImpactLevel.CRITICAL]
        high = [i for i in impacts if i.level == ImpactLevel.HIGH]

        if critical:
            recommendations.append("CRITICAL: This change will cause breaking issues in dependent code.")
            recommendations.append(f"Update {len(critical)} file(s) before making this change.")

        if high:
            recommendations.append(f"Review {len(high)} file(s) that heavily depend on this code.")

        if change.change_type == ChangeType.REMOVE:
            recommendations.append("Consider deprecation warning before removal.")
            recommendations.append("Check for alternative implementations.")

        if change.change_type == ChangeType.RENAME:
            recommendations.append("Use IDE refactoring tools for safer renaming.")
            recommendations.append("Update all documentation references.")

        if not recommendations:
            recommendations.append("This change appears safe to proceed with.")
            recommendations.append("Run tests to verify no unexpected breakages.")

        return recommendations


# ============================================================================
# CLI Entry Point
# ============================================================================


def main() -> int:
    """CLI entry point for Impact Analyzer."""
    import argparse

    parser = argparse.ArgumentParser(
        description="Impact Analyzer (Step 14)"
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    # File impact command
    file_parser = subparsers.add_parser("file", help="Analyze file change impact")
    file_parser.add_argument("path", help="File path")
    file_parser.add_argument("--change", choices=["modify", "remove", "rename", "move"], default="modify")
    file_parser.add_argument("--new-path", help="New path for rename/move")
    file_parser.add_argument("--json", action="store_true", help="Output as JSON")

    # Symbol impact command
    symbol_parser = subparsers.add_parser("symbol", help="Analyze symbol change impact")
    symbol_parser.add_argument("path", help="File path")
    symbol_parser.add_argument("symbol", help="Symbol name")
    symbol_parser.add_argument("--change", choices=["modify", "remove", "rename"], default="modify")
    symbol_parser.add_argument("--new-name", help="New name for rename")
    symbol_parser.add_argument("--json", action="store_true", help="Output as JSON")

    args = parser.parse_args()

    analyzer = ImpactAnalyzer()

    if args.command == "file":
        change = Change(
            change_type=ChangeType(args.change),
            path=args.path,
            new_value=args.new_path,
        )
        report = analyzer.analyze_impact(change)

    elif args.command == "symbol":
        change = Change(
            change_type=ChangeType(args.change),
            path=args.path,
            symbol_name=args.symbol,
            new_value=args.new_name,
        )
        report = analyzer.analyze_impact(change)

    if args.json:
        print(json.dumps(report.to_dict(), indent=2))
    else:
        print(f"Impact Analysis Report")
        print(f"=" * 50)
        print(f"\nChange: {report.change.change_type.value} {report.change.path}")
        if report.change.symbol_name:
            print(f"Symbol: {report.change.symbol_name}")
        print(f"\nRisk Score: {report.risk_score:.2f}")
        print(f"Files Affected: {report.total_files_affected}")
        print(f"\nSummary: {report.summary}")

        if report.impacts:
            print(f"\nImpacts:")
            for impact in sorted(report.impacts, key=lambda i: i.level.value):
                print(f"  [{impact.level.value:8}] {impact.path}")
                if impact.reason:
                    print(f"            {impact.reason}")

        print(f"\nRecommendations:")
        for rec in report.recommendations:
            print(f"  - {rec}")

    return 0


if __name__ == "__main__":
    import sys
    sys.exit(main())
