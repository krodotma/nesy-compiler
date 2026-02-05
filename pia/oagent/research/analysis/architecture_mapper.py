#!/usr/bin/env python3
"""
architecture_mapper.py - Architecture Mapper (Step 16)

Codebase architecture visualization and analysis.
Maps high-level structure, layers, and component relationships.

PBTSO Phase: RESEARCH, DISTILL

Bus Topics:
- a2a.research.architecture.map
- research.architecture.component
- research.architecture.layer

Protocol: DKIN v30, PAIP v16, CITIZEN v2
"""
from __future__ import annotations

import json
import os
import re
from collections import defaultdict
from dataclasses import dataclass, field, asdict
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

from ..bootstrap import AgentBus
from ..graph.dependency_builder import DependencyGraphBuilder
from ..index.symbol_store import SymbolIndexStore


# ============================================================================
# Data Models
# ============================================================================


class ComponentType(Enum):
    """Type of architectural component."""
    MODULE = "module"
    PACKAGE = "package"
    LAYER = "layer"
    SERVICE = "service"
    LIBRARY = "library"
    APPLICATION = "application"
    API = "api"
    DATABASE = "database"
    EXTERNAL = "external"


class LayerType(Enum):
    """Common architectural layers."""
    PRESENTATION = "presentation"   # UI, CLI, API endpoints
    APPLICATION = "application"     # Use cases, services
    DOMAIN = "domain"               # Business logic, entities
    INFRASTRUCTURE = "infrastructure"  # DB, external services
    SHARED = "shared"               # Utilities, common code
    TEST = "test"                   # Test code


@dataclass
class Component:
    """An architectural component."""

    name: str
    path: str
    component_type: ComponentType
    layer: Optional[LayerType] = None
    description: str = ""
    dependencies: List[str] = field(default_factory=list)
    dependents: List[str] = field(default_factory=list)
    files: List[str] = field(default_factory=list)
    symbols: List[str] = field(default_factory=list)
    metrics: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "name": self.name,
            "path": self.path,
            "type": self.component_type.value,
            "layer": self.layer.value if self.layer else None,
            "description": self.description,
            "dependencies": self.dependencies,
            "dependents": self.dependents,
            "file_count": len(self.files),
            "symbol_count": len(self.symbols),
            "metrics": self.metrics,
        }


@dataclass
class Boundary:
    """Boundary between components/layers."""

    from_component: str
    to_component: str
    relationship: str  # "depends_on", "calls", "implements"
    strength: int = 1  # Number of connections
    violations: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "from": self.from_component,
            "to": self.to_component,
            "relationship": self.relationship,
            "strength": self.strength,
            "violations": self.violations,
        }


@dataclass
class ArchitectureMap:
    """Complete architecture map of a codebase."""

    root: str
    components: List[Component]
    boundaries: List[Boundary]
    layers: Dict[str, List[str]]  # layer -> components
    metrics: Dict[str, Any]

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "root": self.root,
            "components": [c.to_dict() for c in self.components],
            "boundaries": [b.to_dict() for b in self.boundaries],
            "layers": self.layers,
            "metrics": self.metrics,
        }

    def to_mermaid(self) -> str:
        """Generate Mermaid diagram."""
        lines = ["graph TD"]

        # Group by layer
        for layer, comps in self.layers.items():
            lines.append(f"    subgraph {layer}")
            for comp in comps:
                lines.append(f"        {comp}")
            lines.append("    end")

        # Add boundaries
        for boundary in self.boundaries:
            if boundary.violations:
                lines.append(f"    {boundary.from_component} -.->|violation| {boundary.to_component}")
            else:
                lines.append(f"    {boundary.from_component} --> {boundary.to_component}")

        return "\n".join(lines)


# ============================================================================
# Layer Detection Heuristics
# ============================================================================


# Path patterns for layer detection
LAYER_PATTERNS = {
    LayerType.PRESENTATION: [
        r"(^|/)ui/",
        r"(^|/)cli/",
        r"(^|/)api/",
        r"(^|/)views?/",
        r"(^|/)handlers?/",
        r"(^|/)routes?/",
        r"(^|/)controllers?/",
        r"(^|/)endpoints?/",
    ],
    LayerType.APPLICATION: [
        r"(^|/)services?/",
        r"(^|/)use_?cases?/",
        r"(^|/)application/",
        r"(^|/)commands?/",
        r"(^|/)queries?/",
    ],
    LayerType.DOMAIN: [
        r"(^|/)domain/",
        r"(^|/)models?/",
        r"(^|/)entities?/",
        r"(^|/)core/",
        r"(^|/)business/",
    ],
    LayerType.INFRASTRUCTURE: [
        r"(^|/)infra(structure)?/",
        r"(^|/)db/",
        r"(^|/)database/",
        r"(^|/)repositories?/",
        r"(^|/)adapters?/",
        r"(^|/)external/",
        r"(^|/)clients?/",
    ],
    LayerType.SHARED: [
        r"(^|/)utils?/",
        r"(^|/)common/",
        r"(^|/)shared/",
        r"(^|/)lib/",
        r"(^|/)helpers?/",
    ],
    LayerType.TEST: [
        r"(^|/)tests?/",
        r"(^|/)spec/",
        r"_test\.py$",
        r"test_.*\.py$",
    ],
}


# ============================================================================
# Architecture Mapper
# ============================================================================


class ArchitectureMapper:
    """
    Map and visualize codebase architecture.

    Analyzes:
    - Package/module structure
    - Architectural layers
    - Component dependencies
    - Boundary violations

    PBTSO Phase: RESEARCH, DISTILL

    Example:
        mapper = ArchitectureMapper(root="/project")
        arch = mapper.map_architecture()
        print(arch.to_mermaid())
    """

    def __init__(
        self,
        root: Optional[Path] = None,
        dependency_builder: Optional[DependencyGraphBuilder] = None,
        symbol_store: Optional[SymbolIndexStore] = None,
        bus: Optional[AgentBus] = None,
    ):
        """
        Initialize the architecture mapper.

        Args:
            root: Project root directory
            dependency_builder: Dependency graph builder
            symbol_store: Symbol index store
            bus: AgentBus for event emission
        """
        self.root = Path(root) if root else Path.cwd()
        self.bus = bus or AgentBus()

        self.dependency_builder = dependency_builder or DependencyGraphBuilder(
            root=self.root, bus=self.bus
        )
        self.symbol_store = symbol_store or SymbolIndexStore()

    def map_architecture(self) -> ArchitectureMap:
        """
        Generate complete architecture map.

        Returns:
            ArchitectureMap with components, boundaries, and metrics
        """
        # Discover components
        components = self._discover_components()

        # Build dependency graph
        self._build_dependency_graph(components)

        # Detect layers
        self._assign_layers(components)

        # Detect boundaries
        boundaries = self._detect_boundaries(components)

        # Calculate metrics
        metrics = self._calculate_metrics(components, boundaries)

        # Group by layer
        layers = defaultdict(list)
        for comp in components:
            layer_name = comp.layer.value if comp.layer else "unknown"
            layers[layer_name].append(comp.name)

        arch = ArchitectureMap(
            root=str(self.root),
            components=components,
            boundaries=boundaries,
            layers=dict(layers),
            metrics=metrics,
        )

        # Emit event
        self.bus.emit({
            "topic": "a2a.research.architecture.map",
            "kind": "architecture",
            "data": {
                "components": len(components),
                "layers": len(layers),
                "boundaries": len(boundaries),
            }
        })

        return arch

    def map_component(self, path: str) -> Optional[Component]:
        """
        Map a single component in detail.

        Args:
            path: Path to component directory or file

        Returns:
            Component with detailed information
        """
        comp_path = Path(path)
        if not comp_path.exists():
            return None

        if comp_path.is_file():
            # Single file component
            name = comp_path.stem
            files = [str(comp_path)]
        else:
            # Package/directory component
            name = comp_path.name
            files = [str(f) for f in comp_path.rglob("*.py")]

        # Determine component type
        if comp_path.is_file():
            comp_type = ComponentType.MODULE
        elif (comp_path / "__init__.py").exists():
            comp_type = ComponentType.PACKAGE
        else:
            comp_type = ComponentType.MODULE

        # Get symbols
        symbols = []
        for file in files:
            file_symbols = self.symbol_store.get_by_path(file)
            symbols.extend([s.name for s in file_symbols])

        # Get dependencies
        deps = set()
        for file in files:
            node = self.dependency_builder.get_node(file)
            if node:
                deps.update(node.imports)

        component = Component(
            name=name,
            path=str(comp_path),
            component_type=comp_type,
            files=files,
            symbols=symbols,
            dependencies=list(deps),
            metrics={
                "file_count": len(files),
                "symbol_count": len(symbols),
                "dependency_count": len(deps),
            }
        )

        # Detect layer
        layer = self._detect_layer(str(comp_path))
        if layer:
            component.layer = layer

        return component

    def get_layer_diagram(self) -> str:
        """Generate a layer dependency diagram."""
        arch = self.map_architecture()
        return arch.to_mermaid()

    def check_layer_violations(self) -> List[Boundary]:
        """
        Check for layer boundary violations.

        Returns:
            List of boundaries that violate layer rules
        """
        arch = self.map_architecture()
        violations = []

        # Define allowed dependencies (layer -> layers it can depend on)
        allowed = {
            LayerType.PRESENTATION: {LayerType.APPLICATION, LayerType.SHARED},
            LayerType.APPLICATION: {LayerType.DOMAIN, LayerType.INFRASTRUCTURE, LayerType.SHARED},
            LayerType.DOMAIN: {LayerType.SHARED},
            LayerType.INFRASTRUCTURE: {LayerType.DOMAIN, LayerType.SHARED},
            LayerType.SHARED: set(),
            LayerType.TEST: {LayerType.PRESENTATION, LayerType.APPLICATION, LayerType.DOMAIN,
                           LayerType.INFRASTRUCTURE, LayerType.SHARED},
        }

        comp_by_name = {c.name: c for c in arch.components}

        for boundary in arch.boundaries:
            from_comp = comp_by_name.get(boundary.from_component)
            to_comp = comp_by_name.get(boundary.to_component)

            if from_comp and to_comp and from_comp.layer and to_comp.layer:
                if to_comp.layer not in allowed.get(from_comp.layer, set()):
                    boundary.violations.append(
                        f"{from_comp.layer.value} should not depend on {to_comp.layer.value}"
                    )
                    violations.append(boundary)

        return violations

    def _discover_components(self) -> List[Component]:
        """Discover components in the codebase."""
        components = []
        seen_paths = set()

        # First pass: find packages (__init__.py directories)
        for init_file in self.root.rglob("__init__.py"):
            pkg_path = init_file.parent

            # Skip if already covered by parent
            rel_path = str(pkg_path.relative_to(self.root))
            if any(rel_path.startswith(p + "/") for p in seen_paths):
                continue

            # Skip deep packages, prefer top-level
            depth = len(rel_path.split("/"))
            if depth > 3:  # Limit depth for top-level components
                continue

            seen_paths.add(rel_path)

            component = Component(
                name=pkg_path.name,
                path=rel_path,
                component_type=ComponentType.PACKAGE,
                files=list(str(f.relative_to(self.root)) for f in pkg_path.rglob("*.py")),
            )
            components.append(component)

        # Second pass: standalone modules
        for py_file in self.root.glob("*.py"):
            rel_path = str(py_file.relative_to(self.root))
            if rel_path not in seen_paths:
                seen_paths.add(rel_path)
                components.append(Component(
                    name=py_file.stem,
                    path=rel_path,
                    component_type=ComponentType.MODULE,
                    files=[rel_path],
                ))

        return components

    def _build_dependency_graph(self, components: List[Component]) -> None:
        """Build dependencies for all components."""
        for comp in components:
            deps = set()
            dependents = set()

            for file in comp.files:
                node = self.dependency_builder.get_node(file)
                if node:
                    deps.update(node.imports)
                    dependents.update(node.imported_by)

            # Map to component names
            path_to_comp = {}
            for c in components:
                for f in c.files:
                    path_to_comp[f] = c.name

            comp.dependencies = list(set(
                path_to_comp.get(d, d) for d in deps
                if path_to_comp.get(d, d) != comp.name
            ))

            comp.dependents = list(set(
                path_to_comp.get(d, d) for d in dependents
                if path_to_comp.get(d, d) != comp.name
            ))

    def _assign_layers(self, components: List[Component]) -> None:
        """Assign architectural layers to components."""
        for comp in components:
            layer = self._detect_layer(comp.path)
            comp.layer = layer

    def _detect_layer(self, path: str) -> Optional[LayerType]:
        """Detect architectural layer from path."""
        for layer, patterns in LAYER_PATTERNS.items():
            for pattern in patterns:
                if re.search(pattern, path, re.IGNORECASE):
                    return layer
        return None

    def _detect_boundaries(self, components: List[Component]) -> List[Boundary]:
        """Detect boundaries between components."""
        boundaries = []
        comp_by_name = {c.name: c for c in components}

        for comp in components:
            for dep in comp.dependencies:
                if dep in comp_by_name:
                    # Count connections
                    strength = 1  # Could be enhanced with actual import counts

                    boundary = Boundary(
                        from_component=comp.name,
                        to_component=dep,
                        relationship="depends_on",
                        strength=strength,
                    )
                    boundaries.append(boundary)

        return boundaries

    def _calculate_metrics(
        self,
        components: List[Component],
        boundaries: List[Boundary],
    ) -> Dict[str, Any]:
        """Calculate architecture metrics."""
        total_files = sum(len(c.files) for c in components)
        total_deps = sum(len(c.dependencies) for c in components)

        # Calculate coupling
        if len(components) > 0:
            avg_coupling = total_deps / len(components)
        else:
            avg_coupling = 0

        # Find highly coupled components
        high_coupling = [c.name for c in components if len(c.dependencies) > 5]

        # Layer distribution
        layer_dist = defaultdict(int)
        for comp in components:
            layer_name = comp.layer.value if comp.layer else "unknown"
            layer_dist[layer_name] += 1

        return {
            "total_components": len(components),
            "total_files": total_files,
            "total_boundaries": len(boundaries),
            "average_coupling": round(avg_coupling, 2),
            "high_coupling_components": high_coupling,
            "layer_distribution": dict(layer_dist),
        }


# ============================================================================
# CLI Entry Point
# ============================================================================


def main() -> int:
    """CLI entry point for Architecture Mapper."""
    import argparse

    parser = argparse.ArgumentParser(
        description="Architecture Mapper (Step 16)"
    )
    parser.add_argument(
        "root",
        nargs="?",
        default=".",
        help="Project root directory"
    )
    parser.add_argument(
        "--mermaid",
        action="store_true",
        help="Output as Mermaid diagram"
    )
    parser.add_argument(
        "--violations",
        action="store_true",
        help="Check for layer violations"
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Output as JSON"
    )

    args = parser.parse_args()

    mapper = ArchitectureMapper(root=Path(args.root))

    if args.violations:
        violations = mapper.check_layer_violations()
        if args.json:
            print(json.dumps([v.to_dict() for v in violations], indent=2))
        else:
            if violations:
                print(f"Found {len(violations)} layer violation(s):")
                for v in violations:
                    print(f"  {v.from_component} -> {v.to_component}")
                    for issue in v.violations:
                        print(f"    ! {issue}")
            else:
                print("No layer violations found.")

    elif args.mermaid:
        arch = mapper.map_architecture()
        print(arch.to_mermaid())

    elif args.json:
        arch = mapper.map_architecture()
        print(json.dumps(arch.to_dict(), indent=2))

    else:
        arch = mapper.map_architecture()
        print(f"Architecture Map for: {arch.root}")
        print(f"\nMetrics:")
        for key, value in arch.metrics.items():
            print(f"  {key}: {value}")

        print(f"\nComponents ({len(arch.components)}):")
        for comp in arch.components:
            layer = comp.layer.value if comp.layer else "unknown"
            print(f"  [{layer:15}] {comp.name} ({len(comp.files)} files)")

        print(f"\nLayers:")
        for layer, comps in arch.layers.items():
            print(f"  {layer}: {', '.join(comps)}")

    return 0


if __name__ == "__main__":
    import sys
    sys.exit(main())
