"""
DiTS Specification Loader
=========================

Loads DiTS specifications from various sources:
- JSON files
- YAML files
- Python dictionaries
- Template presets

Provides validation, transformation, and caching of specifications.
"""

from __future__ import annotations

import os
import time
import json
import hashlib
from typing import (
    Dict,
    Any,
    List,
    Optional,
    Set,
    Callable,
    Tuple,
    Union,
    Type,
)
from dataclasses import dataclass, field
from pathlib import Path
from enum import Enum


# Try to import YAML support
try:
    import yaml
    YAML_AVAILABLE = True
except ImportError:
    YAML_AVAILABLE = False


# Import kernel types
from .kernel import DiTSSpec, TransitionType


# =============================================================================
# TYPE DEFINITIONS
# =============================================================================

class SpecFormat(str, Enum):
    """Supported specification formats."""
    JSON = "json"
    YAML = "yaml"
    DICT = "dict"
    PRESET = "preset"


class ValidationLevel(str, Enum):
    """Level of specification validation."""
    NONE = "none"
    BASIC = "basic"
    STRICT = "strict"
    PEDANTIC = "pedantic"


# =============================================================================
# VALIDATION
# =============================================================================

@dataclass
class ValidationError:
    """A validation error."""
    field: str
    message: str
    severity: str = "error"  # error, warning, info


@dataclass
class ValidationResult:
    """Result of specification validation."""
    valid: bool
    errors: List[ValidationError] = field(default_factory=list)
    warnings: List[ValidationError] = field(default_factory=list)
    info: List[ValidationError] = field(default_factory=list)

    def add_error(self, field: str, message: str) -> None:
        """Add an error."""
        self.errors.append(ValidationError(field, message, "error"))
        self.valid = False

    def add_warning(self, field: str, message: str) -> None:
        """Add a warning."""
        self.warnings.append(ValidationError(field, message, "warning"))

    def add_info(self, field: str, message: str) -> None:
        """Add info message."""
        self.info.append(ValidationError(field, message, "info"))

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "valid": self.valid,
            "errors": [
                {"field": e.field, "message": e.message}
                for e in self.errors
            ],
            "warnings": [
                {"field": w.field, "message": w.message}
                for w in self.warnings
            ],
            "info": [
                {"field": i.field, "message": i.message}
                for i in self.info
            ],
        }


def validate_spec(
    spec: DiTSSpec,
    level: ValidationLevel = ValidationLevel.BASIC,
) -> ValidationResult:
    """
    Validate a DiTS specification.

    Args:
        spec: The specification to validate
        level: Validation strictness level

    Returns:
        ValidationResult with any errors/warnings
    """
    result = ValidationResult(valid=True)

    if level == ValidationLevel.NONE:
        return result

    # Basic validation
    if not spec.name:
        result.add_error("name", "Specification name is required")

    if not spec.states:
        result.add_warning("states", "Specification has no states defined")

    if not spec.initial_states and spec.states:
        result.add_warning("initial_states", "No initial states defined")

    # Check initial states are valid
    invalid_initial = spec.initial_states - spec.states
    if invalid_initial:
        result.add_error(
            "initial_states",
            f"Initial states not in state set: {invalid_initial}"
        )

    # Check final states are valid
    invalid_final = spec.final_states - spec.states
    if invalid_final:
        result.add_error(
            "final_states",
            f"Final states not in state set: {invalid_final}"
        )

    # Check transitions reference valid states
    for (source, label), targets in spec.transitions.items():
        if source not in spec.states:
            result.add_error(
                "transitions",
                f"Transition source '{source}' not in state set"
            )
        for target in targets:
            if target not in spec.states:
                result.add_error(
                    "transitions",
                    f"Transition target '{target}' not in state set"
                )

    if level.value >= ValidationLevel.STRICT.value:
        # Strict validation

        # Check for unreachable states
        reachable = _compute_reachable(spec)
        unreachable = spec.states - reachable

        if unreachable:
            result.add_warning(
                "states",
                f"Unreachable states from initial: {unreachable}"
            )

        # Check for dead-end states (non-final with no outgoing)
        for state in spec.states:
            if state not in spec.final_states:
                has_outgoing = any(
                    s == state for (s, _) in spec.transitions.keys()
                )
                if not has_outgoing:
                    result.add_warning(
                        "states",
                        f"State '{state}' has no outgoing transitions"
                    )

        # Check alphabet consistency
        used_labels = {label for (_, label) in spec.transitions.keys()}
        unused_labels = spec.alphabet - used_labels
        if unused_labels:
            result.add_info(
                "alphabet",
                f"Declared but unused labels: {unused_labels}"
            )

    if level == ValidationLevel.PEDANTIC:
        # Pedantic validation

        # Check for duplicate transitions
        seen_transitions: Set[Tuple[str, str, str]] = set()
        for (source, label), targets in spec.transitions.items():
            for target in targets:
                key = (source, label, target)
                if key in seen_transitions:
                    result.add_warning(
                        "transitions",
                        f"Duplicate transition: {source} --{label}--> {target}"
                    )
                seen_transitions.add(key)

        # Check naming conventions
        for state in spec.states:
            if not state.replace("_", "").replace("-", "").isalnum():
                result.add_info(
                    "states",
                    f"State '{state}' contains non-standard characters"
                )

    return result


def _compute_reachable(spec: DiTSSpec) -> Set[str]:
    """Compute states reachable from initial states."""
    reachable = set()
    frontier = spec.initial_states.copy()

    while frontier:
        state = frontier.pop()
        if state in reachable:
            continue

        reachable.add(state)

        for (source, _), targets in spec.transitions.items():
            if source == state:
                frontier.update(targets - reachable)

    return reachable


# =============================================================================
# PRESET SPECIFICATIONS
# =============================================================================

PRESET_SPECS: Dict[str, Dict[str, Any]] = {
    "three_act": {
        "name": "Three Act Structure",
        "description": "Classic three-act narrative structure",
        "version": "1.0.0",
        "states": ["setup", "confrontation", "resolution"],
        "initial_states": ["setup"],
        "final_states": ["resolution"],
        "alphabet": ["develop", "escalate", "conclude"],
        "transitions": {
            "setup:develop": ["confrontation"],
            "confrontation:escalate": ["confrontation"],
            "confrontation:conclude": ["resolution"],
        },
    },
    "freytag": {
        "name": "Freytag's Pyramid",
        "description": "Five-part dramatic structure",
        "version": "1.0.0",
        "states": [
            "exposition",
            "rising_action",
            "climax",
            "falling_action",
            "denouement",
        ],
        "initial_states": ["exposition"],
        "final_states": ["denouement"],
        "alphabet": ["introduce", "complicate", "peak", "resolve", "conclude"],
        "transitions": {
            "exposition:introduce": ["rising_action"],
            "rising_action:complicate": ["rising_action", "climax"],
            "climax:peak": ["falling_action"],
            "falling_action:resolve": ["denouement"],
        },
    },
    "hero_journey": {
        "name": "Hero's Journey",
        "description": "Campbell's monomyth structure",
        "version": "1.0.0",
        "states": [
            "ordinary_world",
            "call_to_adventure",
            "refusal",
            "meeting_mentor",
            "crossing_threshold",
            "tests_allies",
            "approach",
            "ordeal",
            "reward",
            "road_back",
            "resurrection",
            "return",
        ],
        "initial_states": ["ordinary_world"],
        "final_states": ["return"],
        "alphabet": [
            "call", "refuse", "meet", "cross", "test",
            "approach", "face", "reward", "return",
            "transform", "complete",
        ],
        "transitions": {
            "ordinary_world:call": ["call_to_adventure"],
            "call_to_adventure:refuse": ["refusal"],
            "call_to_adventure:meet": ["meeting_mentor"],
            "refusal:meet": ["meeting_mentor"],
            "meeting_mentor:cross": ["crossing_threshold"],
            "crossing_threshold:test": ["tests_allies"],
            "tests_allies:test": ["tests_allies"],
            "tests_allies:approach": ["approach"],
            "approach:face": ["ordeal"],
            "ordeal:reward": ["reward"],
            "reward:return": ["road_back"],
            "road_back:transform": ["resurrection"],
            "resurrection:complete": ["return"],
        },
    },
    "simple_loop": {
        "name": "Simple Loop",
        "description": "Basic looping narrative structure",
        "version": "1.0.0",
        "states": ["start", "middle", "end"],
        "initial_states": ["start"],
        "final_states": ["end"],
        "alphabet": ["next", "loop", "finish"],
        "transitions": {
            "start:next": ["middle"],
            "middle:next": ["middle"],
            "middle:loop": ["start"],
            "middle:finish": ["end"],
        },
    },
    "branching": {
        "name": "Branching Narrative",
        "description": "Multiple-path narrative structure",
        "version": "1.0.0",
        "states": ["start", "branch", "path_a", "path_b", "merge", "end"],
        "initial_states": ["start"],
        "final_states": ["end"],
        "alphabet": ["begin", "choose_a", "choose_b", "progress", "merge", "conclude"],
        "transitions": {
            "start:begin": ["branch"],
            "branch:choose_a": ["path_a"],
            "branch:choose_b": ["path_b"],
            "path_a:progress": ["merge"],
            "path_b:progress": ["merge"],
            "merge:conclude": ["end"],
        },
    },
    "dialogue": {
        "name": "Dialogue Flow",
        "description": "Conversational exchange structure",
        "version": "1.0.0",
        "states": ["greeting", "exchange", "deepening", "resolution", "farewell"],
        "initial_states": ["greeting"],
        "final_states": ["farewell"],
        "alphabet": ["greet", "respond", "deepen", "resolve", "part"],
        "transitions": {
            "greeting:greet": ["exchange"],
            "exchange:respond": ["exchange"],
            "exchange:deepen": ["deepening"],
            "deepening:respond": ["deepening"],
            "deepening:resolve": ["resolution"],
            "resolution:part": ["farewell"],
        },
    },
}


# =============================================================================
# LOADERS
# =============================================================================

@dataclass
class LoadResult:
    """Result of a spec load operation."""
    spec: Optional[DiTSSpec]
    validation: ValidationResult
    source: str
    format: SpecFormat
    load_time: float
    metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def success(self) -> bool:
        """Check if load was successful."""
        return self.spec is not None and self.validation.valid


def load_from_dict(
    data: Dict[str, Any],
    validate: ValidationLevel = ValidationLevel.BASIC,
) -> LoadResult:
    """
    Load a DiTS specification from a dictionary.

    Args:
        data: Dictionary containing specification
        validate: Validation level

    Returns:
        LoadResult with spec and validation info
    """
    start_time = time.time()

    try:
        spec = DiTSSpec.from_dict(data)
        validation = validate_spec(spec, validate)

        return LoadResult(
            spec=spec,
            validation=validation,
            source="dict",
            format=SpecFormat.DICT,
            load_time=time.time() - start_time,
        )

    except Exception as e:
        validation = ValidationResult(valid=False)
        validation.add_error("parse", str(e))

        return LoadResult(
            spec=None,
            validation=validation,
            source="dict",
            format=SpecFormat.DICT,
            load_time=time.time() - start_time,
        )


def load_from_json(
    source: Union[str, Path],
    validate: ValidationLevel = ValidationLevel.BASIC,
) -> LoadResult:
    """
    Load a DiTS specification from JSON.

    Args:
        source: JSON file path or JSON string
        validate: Validation level

    Returns:
        LoadResult with spec and validation info
    """
    start_time = time.time()
    source_str = str(source)

    try:
        # Check if it's a file path
        if os.path.isfile(source_str):
            with open(source_str, "r") as f:
                data = json.load(f)
            source_str = f"file:{source_str}"
        else:
            # Assume it's a JSON string
            data = json.loads(source_str)
            source_str = "json_string"

        spec = DiTSSpec.from_dict(data)
        validation = validate_spec(spec, validate)

        return LoadResult(
            spec=spec,
            validation=validation,
            source=source_str,
            format=SpecFormat.JSON,
            load_time=time.time() - start_time,
        )

    except json.JSONDecodeError as e:
        validation = ValidationResult(valid=False)
        validation.add_error("parse", f"Invalid JSON: {e}")

        return LoadResult(
            spec=None,
            validation=validation,
            source=source_str,
            format=SpecFormat.JSON,
            load_time=time.time() - start_time,
        )

    except Exception as e:
        validation = ValidationResult(valid=False)
        validation.add_error("load", str(e))

        return LoadResult(
            spec=None,
            validation=validation,
            source=source_str,
            format=SpecFormat.JSON,
            load_time=time.time() - start_time,
        )


def load_from_yaml(
    source: Union[str, Path],
    validate: ValidationLevel = ValidationLevel.BASIC,
) -> LoadResult:
    """
    Load a DiTS specification from YAML.

    Args:
        source: YAML file path or YAML string
        validate: Validation level

    Returns:
        LoadResult with spec and validation info
    """
    if not YAML_AVAILABLE:
        validation = ValidationResult(valid=False)
        validation.add_error(
            "yaml",
            "YAML support not available. Install pyyaml package."
        )
        return LoadResult(
            spec=None,
            validation=validation,
            source=str(source),
            format=SpecFormat.YAML,
            load_time=0.0,
        )

    start_time = time.time()
    source_str = str(source)

    try:
        # Check if it's a file path
        if os.path.isfile(source_str):
            with open(source_str, "r") as f:
                data = yaml.safe_load(f)
            source_str = f"file:{source_str}"
        else:
            # Assume it's a YAML string
            data = yaml.safe_load(source_str)
            source_str = "yaml_string"

        spec = DiTSSpec.from_dict(data)
        validation = validate_spec(spec, validate)

        return LoadResult(
            spec=spec,
            validation=validation,
            source=source_str,
            format=SpecFormat.YAML,
            load_time=time.time() - start_time,
        )

    except Exception as e:
        validation = ValidationResult(valid=False)
        validation.add_error("load", str(e))

        return LoadResult(
            spec=None,
            validation=validation,
            source=source_str,
            format=SpecFormat.YAML,
            load_time=time.time() - start_time,
        )


def load_preset(
    preset_name: str,
    validate: ValidationLevel = ValidationLevel.BASIC,
) -> LoadResult:
    """
    Load a preset DiTS specification.

    Args:
        preset_name: Name of the preset
        validate: Validation level

    Returns:
        LoadResult with spec and validation info
    """
    start_time = time.time()

    if preset_name not in PRESET_SPECS:
        validation = ValidationResult(valid=False)
        validation.add_error(
            "preset",
            f"Unknown preset: {preset_name}. "
            f"Available: {list(PRESET_SPECS.keys())}"
        )
        return LoadResult(
            spec=None,
            validation=validation,
            source=f"preset:{preset_name}",
            format=SpecFormat.PRESET,
            load_time=time.time() - start_time,
        )

    data = PRESET_SPECS[preset_name].copy()
    result = load_from_dict(data, validate)
    result.source = f"preset:{preset_name}"
    result.format = SpecFormat.PRESET

    return result


def list_presets() -> List[Dict[str, str]]:
    """
    List available preset specifications.

    Returns:
        List of preset info dictionaries
    """
    return [
        {
            "name": name,
            "description": data.get("description", ""),
            "states": len(data.get("states", [])),
        }
        for name, data in PRESET_SPECS.items()
    ]


# =============================================================================
# SPEC LOADER CLASS
# =============================================================================

@dataclass
class LoaderConfig:
    """Configuration for the spec loader."""
    default_validation: ValidationLevel = ValidationLevel.BASIC
    enable_caching: bool = True
    cache_ttl: float = 3600.0  # seconds
    search_paths: List[str] = field(default_factory=list)


class SpecLoader:
    """
    Unified loader for DiTS specifications.

    Supports loading from multiple sources with caching,
    validation, and transformation.

    Example usage:
        >>> loader = SpecLoader()
        >>> result = loader.load("freytag")  # Load preset
        >>> result = loader.load("path/to/spec.json")  # Load from file
        >>> result = loader.load({"name": "custom", ...})  # Load from dict
    """

    def __init__(self, config: Optional[LoaderConfig] = None):
        """
        Initialize SpecLoader.

        Args:
            config: Optional configuration
        """
        self.config = config or LoaderConfig()
        self._cache: Dict[str, Tuple[DiTSSpec, float]] = {}
        self._stats = {
            "loads": 0,
            "cache_hits": 0,
            "validations": 0,
        }

    def load(
        self,
        source: Union[str, Path, Dict[str, Any]],
        validate: Optional[ValidationLevel] = None,
        use_cache: Optional[bool] = None,
    ) -> LoadResult:
        """
        Load a DiTS specification from any supported source.

        Args:
            source: Spec source (preset name, file path, JSON string, or dict)
            validate: Validation level (defaults to config default)
            use_cache: Whether to use caching (defaults to config setting)

        Returns:
            LoadResult with spec and validation info
        """
        validate = validate or self.config.default_validation
        use_cache = use_cache if use_cache is not None else self.config.enable_caching

        self._stats["loads"] += 1

        # Check cache
        cache_key = self._cache_key(source)
        if use_cache and cache_key in self._cache:
            spec, cached_time = self._cache[cache_key]
            if time.time() - cached_time < self.config.cache_ttl:
                self._stats["cache_hits"] += 1
                return LoadResult(
                    spec=spec,
                    validation=ValidationResult(valid=True),
                    source=f"cache:{cache_key}",
                    format=SpecFormat.DICT,
                    load_time=0.0,
                    metadata={"cached": True},
                )

        # Determine source type and load
        result = self._dispatch_load(source, validate)

        # Cache successful loads
        if use_cache and result.success and result.spec:
            self._cache[cache_key] = (result.spec, time.time())

        self._stats["validations"] += 1 if validate != ValidationLevel.NONE else 0

        return result

    def _dispatch_load(
        self,
        source: Union[str, Path, Dict[str, Any]],
        validate: ValidationLevel,
    ) -> LoadResult:
        """Dispatch to appropriate loader based on source type."""
        # Dictionary
        if isinstance(source, dict):
            return load_from_dict(source, validate)

        source_str = str(source)

        # Preset name
        if source_str in PRESET_SPECS:
            return load_preset(source_str, validate)

        # File path
        if os.path.isfile(source_str):
            ext = os.path.splitext(source_str)[1].lower()
            if ext in [".yaml", ".yml"]:
                return load_from_yaml(source_str, validate)
            else:
                return load_from_json(source_str, validate)

        # Search in search paths
        for search_path in self.config.search_paths:
            for ext in [".json", ".yaml", ".yml"]:
                full_path = os.path.join(search_path, f"{source_str}{ext}")
                if os.path.isfile(full_path):
                    if ext in [".yaml", ".yml"]:
                        return load_from_yaml(full_path, validate)
                    else:
                        return load_from_json(full_path, validate)

        # Try as JSON/YAML string
        if source_str.strip().startswith("{"):
            return load_from_json(source_str, validate)
        elif YAML_AVAILABLE and (":" in source_str or "\n" in source_str):
            return load_from_yaml(source_str, validate)

        # Unknown source
        validation = ValidationResult(valid=False)
        validation.add_error(
            "source",
            f"Unable to determine source type: {source_str[:100]}"
        )
        return LoadResult(
            spec=None,
            validation=validation,
            source=source_str,
            format=SpecFormat.DICT,
            load_time=0.0,
        )

    def _cache_key(self, source: Union[str, Path, Dict[str, Any]]) -> str:
        """Generate cache key for source."""
        if isinstance(source, dict):
            content = json.dumps(source, sort_keys=True)
        else:
            content = str(source)

        return hashlib.sha256(content.encode()).hexdigest()[:16]

    def clear_cache(self) -> int:
        """Clear the specification cache."""
        count = len(self._cache)
        self._cache.clear()
        return count

    def preload_presets(self) -> Dict[str, bool]:
        """Preload all presets into cache."""
        results = {}
        for preset_name in PRESET_SPECS:
            result = self.load(preset_name)
            results[preset_name] = result.success
        return results

    @property
    def statistics(self) -> Dict[str, int]:
        """Get loader statistics."""
        return {
            **self._stats,
            "cache_size": len(self._cache),
        }


# =============================================================================
# CONVENIENCE FUNCTIONS
# =============================================================================

def load_spec(
    source: Union[str, Path, Dict[str, Any]],
    validate: ValidationLevel = ValidationLevel.BASIC,
) -> DiTSSpec:
    """
    Convenience function to load a spec, raising on error.

    Args:
        source: Spec source
        validate: Validation level

    Returns:
        Loaded DiTSSpec

    Raises:
        ValueError: If load fails
    """
    loader = SpecLoader()
    result = loader.load(source, validate)

    if not result.success:
        errors = "; ".join(e.message for e in result.validation.errors)
        raise ValueError(f"Failed to load spec: {errors}")

    return result.spec


def create_spec(
    name: str,
    states: Optional[List[str]] = None,
    transitions: Optional[Dict[str, List[str]]] = None,
    initial: Optional[List[str]] = None,
    final: Optional[List[str]] = None,
) -> DiTSSpec:
    """
    Convenience function to create a spec programmatically.

    Args:
        name: Specification name
        states: List of state names
        transitions: Dict of "source:label" -> [targets]
        initial: Initial state names
        final: Final state names

    Returns:
        New DiTSSpec
    """
    spec = DiTSSpec(name=name)

    if states:
        spec.states = set(states)

    if initial:
        spec.initial_states = set(initial)
    elif states:
        spec.initial_states = {states[0]}

    if final:
        spec.final_states = set(final)
    elif states:
        spec.final_states = {states[-1]}

    if transitions:
        for key, targets in transitions.items():
            source, label = key.split(":", 1)
            spec.add_transition(source, label, targets[0])
            for target in targets[1:]:
                spec.transitions[(source, label)].add(target)

    return spec


# =============================================================================
# EXPORTS
# =============================================================================

__all__ = [
    # Types
    "SpecFormat",
    "ValidationLevel",
    # Validation
    "ValidationError",
    "ValidationResult",
    "validate_spec",
    # Loading
    "LoadResult",
    "load_from_dict",
    "load_from_json",
    "load_from_yaml",
    "load_preset",
    "list_presets",
    # Presets
    "PRESET_SPECS",
    # Loader class
    "LoaderConfig",
    "SpecLoader",
    # Convenience
    "load_spec",
    "create_spec",
]


if __name__ == "__main__":
    # Demo usage
    print("DiTS Spec Loader Demo")
    print("=" * 50)
    print()

    # List presets
    print("Available Presets:")
    for preset in list_presets():
        print(f"  - {preset['name']}: {preset['description']}")
    print()

    # Load a preset
    loader = SpecLoader()
    result = loader.load("freytag")

    print(f"Loaded: {result.spec.name}")
    print(f"Format: {result.format.value}")
    print(f"Load time: {result.load_time:.4f}s")
    print(f"Valid: {result.validation.valid}")
    print()

    print("States:", list(result.spec.states))
    print("Initial:", list(result.spec.initial_states))
    print("Final:", list(result.spec.final_states))
    print()

    print("Transitions:")
    for (source, label), targets in result.spec.transitions.items():
        print(f"  {source} --{label}--> {targets}")
    print()

    # Load same spec again (cached)
    result2 = loader.load("freytag")
    print(f"Second load (cached): {result2.source}")
    print()

    # Create custom spec
    custom = create_spec(
        name="custom_story",
        states=["begin", "develop", "end"],
        transitions={
            "begin:start": ["develop"],
            "develop:continue": ["develop"],
            "develop:finish": ["end"],
        },
        initial=["begin"],
        final=["end"],
    )
    print(f"Created custom spec: {custom.name}")
    print(f"States: {custom.states}")
    print()

    # Validate with strict level
    validation = validate_spec(custom, ValidationLevel.STRICT)
    print(f"Strict validation: valid={validation.valid}")
    for warning in validation.warnings:
        print(f"  Warning: {warning.message}")
    print()

    print(f"Loader statistics: {loader.statistics}")
