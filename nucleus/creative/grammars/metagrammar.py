"""
Metagrammar Transformations
============================

This module provides metagrammar capabilities - grammars that operate on
other grammars. This enables:

- Pattern-based grammar transformations
- Grammar optimization and simplification
- Rule composition and decomposition
- Grammar evolution and learning
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Any, Callable, Optional, Union


class VariableType(Enum):
    """Types of pattern variables in metagrammar rules."""
    SYMBOL = auto()       # Matches any single symbol
    SEQUENCE = auto()     # Matches a sequence of symbols
    OPTIONAL = auto()     # Matches zero or one symbol
    NONTERMINAL = auto()  # Matches only nonterminals
    TERMINAL = auto()     # Matches only terminals


@dataclass
class PatternVariable:
    """
    A variable in a metagrammar pattern.

    Pattern variables can match parts of grammar rules and be used
    in transformations. For example, ?X might match any symbol,
    while *X matches a sequence.

    Attributes:
        name: Variable name (e.g., "X", "expr", "term").
        var_type: Type of pattern matching.
        constraint: Optional predicate that must be satisfied.
        binding: Current bound value (set during matching).
    """
    name: str
    var_type: VariableType = VariableType.SYMBOL
    constraint: Optional[Callable[[str], bool]] = None
    binding: Optional[Any] = None

    def matches(self, value: Any) -> bool:
        """
        Check if this variable matches a value.

        Args:
            value: The value to match against.

        Returns:
            True if the value matches, False otherwise.
        """
        if self.constraint and not self.constraint(value):
            return False

        if self.var_type == VariableType.SYMBOL:
            return isinstance(value, str) and len(value) > 0

        elif self.var_type == VariableType.SEQUENCE:
            return isinstance(value, (list, tuple))

        elif self.var_type == VariableType.OPTIONAL:
            return value is None or isinstance(value, str)

        elif self.var_type == VariableType.NONTERMINAL:
            if isinstance(value, str):
                return value.startswith("<") and value.endswith(">")
            return False

        elif self.var_type == VariableType.TERMINAL:
            if isinstance(value, str):
                return not (value.startswith("<") and value.endswith(">"))
            return False

        return True

    def bind(self, value: Any) -> PatternVariable:
        """Bind a value to this variable and return self."""
        self.binding = value
        return self

    def unbind(self) -> None:
        """Clear the binding."""
        self.binding = None

    def copy(self) -> PatternVariable:
        """Create a copy of this variable."""
        return PatternVariable(
            name=self.name,
            var_type=self.var_type,
            constraint=self.constraint,
            binding=self.binding
        )

    def __repr__(self) -> str:
        prefix = {
            VariableType.SYMBOL: "?",
            VariableType.SEQUENCE: "*",
            VariableType.OPTIONAL: "~",
            VariableType.NONTERMINAL: "!",
            VariableType.TERMINAL: "$",
        }.get(self.var_type, "?")
        return f"{prefix}{self.name}"

    @staticmethod
    def parse(text: str) -> Optional[PatternVariable]:
        """
        Parse a pattern variable from text.

        Syntax:
        - ?name: Single symbol
        - *name: Sequence
        - ~name: Optional
        - !name: Nonterminal only
        - $name: Terminal only

        Args:
            text: Text to parse.

        Returns:
            PatternVariable or None if not a valid pattern.
        """
        if len(text) < 2:
            return None

        prefix = text[0]
        name = text[1:]

        type_map = {
            "?": VariableType.SYMBOL,
            "*": VariableType.SEQUENCE,
            "~": VariableType.OPTIONAL,
            "!": VariableType.NONTERMINAL,
            "$": VariableType.TERMINAL,
        }

        if prefix in type_map:
            return PatternVariable(name=name, var_type=type_map[prefix])

        return None


@dataclass
class Pattern:
    """
    A pattern that can match against grammar structures.

    Patterns are lists of elements that can be:
    - Literal strings
    - PatternVariable objects
    - Nested patterns (lists)
    """
    elements: list[Union[str, PatternVariable, "Pattern"]]

    def match(self, target: list[str], bindings: Optional[dict[str, Any]] = None
             ) -> Optional[dict[str, Any]]:
        """
        Attempt to match this pattern against a target.

        Args:
            target: List of strings to match.
            bindings: Existing variable bindings.

        Returns:
            Dictionary of bindings if match succeeds, None otherwise.
        """
        if bindings is None:
            bindings = {}

        return self._match_impl(self.elements, target, bindings)

    def _match_impl(self, pattern: list, target: list, bindings: dict
                   ) -> Optional[dict]:
        """Recursive pattern matching implementation."""
        pat_idx = 0
        tgt_idx = 0

        while pat_idx < len(pattern):
            elem = pattern[pat_idx]

            if isinstance(elem, str):
                # Literal match
                if tgt_idx >= len(target) or target[tgt_idx] != elem:
                    return None
                tgt_idx += 1
                pat_idx += 1

            elif isinstance(elem, PatternVariable):
                if elem.var_type == VariableType.SEQUENCE:
                    # Sequence variable - try to match rest
                    for seq_len in range(len(target) - tgt_idx + 1):
                        seq = target[tgt_idx:tgt_idx + seq_len]
                        new_bindings = bindings.copy()
                        new_bindings[elem.name] = seq

                        rest_result = self._match_impl(
                            pattern[pat_idx + 1:],
                            target[tgt_idx + seq_len:],
                            new_bindings
                        )
                        if rest_result is not None:
                            return rest_result
                    return None

                elif elem.var_type == VariableType.OPTIONAL:
                    # Try with and without
                    for consume in [1, 0]:
                        if consume and tgt_idx < len(target):
                            if not elem.matches(target[tgt_idx]):
                                continue
                            new_bindings = bindings.copy()
                            new_bindings[elem.name] = target[tgt_idx]
                            rest_result = self._match_impl(
                                pattern[pat_idx + 1:],
                                target[tgt_idx + 1:],
                                new_bindings
                            )
                        else:
                            new_bindings = bindings.copy()
                            new_bindings[elem.name] = None
                            rest_result = self._match_impl(
                                pattern[pat_idx + 1:],
                                target[tgt_idx:],
                                new_bindings
                            )
                        if rest_result is not None:
                            return rest_result
                    return None

                else:
                    # Single symbol variable
                    if tgt_idx >= len(target):
                        return None
                    if not elem.matches(target[tgt_idx]):
                        return None

                    # Check if already bound
                    if elem.name in bindings:
                        if bindings[elem.name] != target[tgt_idx]:
                            return None
                    else:
                        bindings = bindings.copy()
                        bindings[elem.name] = target[tgt_idx]

                    tgt_idx += 1
                    pat_idx += 1

            elif isinstance(elem, Pattern):
                # Nested pattern - not yet implemented
                pat_idx += 1

            else:
                return None

        # Check all target consumed
        if tgt_idx == len(target):
            return bindings
        return None

    @staticmethod
    def parse(text: str) -> Pattern:
        """
        Parse a pattern from text.

        Elements are space-separated. Pattern variables start with
        special prefixes (?*~!$).

        Args:
            text: Pattern text to parse.

        Returns:
            Parsed Pattern object.
        """
        tokens = text.split()
        elements: list[Union[str, PatternVariable, Pattern]] = []

        for token in tokens:
            var = PatternVariable.parse(token)
            if var:
                elements.append(var)
            else:
                elements.append(token)

        return Pattern(elements=elements)


@dataclass
class TransformRule:
    """
    A grammar transformation rule.

    Transformation rules specify how to rewrite parts of a grammar.
    They have a source pattern that matches against rules, and a
    target pattern that specifies the replacement.

    Attributes:
        name: Human-readable name for the rule.
        source: Pattern to match against.
        target: Pattern to produce (with variable substitution).
        condition: Optional predicate for when rule applies.
        priority: Higher priority rules are tried first.
        bidirectional: If True, rule can be applied in reverse.
    """
    name: str
    source: Pattern
    target: Pattern
    condition: Optional[Callable[[dict[str, Any]], bool]] = None
    priority: int = 0
    bidirectional: bool = False

    def apply(self, production: list[str]) -> Optional[list[str]]:
        """
        Attempt to apply this transformation rule.

        Args:
            production: The grammar production to transform.

        Returns:
            Transformed production, or None if rule doesn't match.
        """
        bindings = self.source.match(production)
        if bindings is None:
            return None

        if self.condition and not self.condition(bindings):
            return None

        # Substitute into target
        result: list[str] = []
        for elem in self.target.elements:
            if isinstance(elem, str):
                result.append(elem)
            elif isinstance(elem, PatternVariable):
                if elem.name in bindings:
                    val = bindings[elem.name]
                    if isinstance(val, list):
                        result.extend(val)
                    elif val is not None:
                        result.append(val)

        return result

    def apply_reverse(self, production: list[str]) -> Optional[list[str]]:
        """Apply rule in reverse direction (if bidirectional)."""
        if not self.bidirectional:
            return None

        # Swap source and target
        bindings = self.target.match(production)
        if bindings is None:
            return None

        result: list[str] = []
        for elem in self.source.elements:
            if isinstance(elem, str):
                result.append(elem)
            elif isinstance(elem, PatternVariable):
                if elem.name in bindings:
                    val = bindings[elem.name]
                    if isinstance(val, list):
                        result.extend(val)
                    elif val is not None:
                        result.append(val)

        return result

    @staticmethod
    def from_strings(name: str, source: str, target: str,
                     priority: int = 0, bidirectional: bool = False
                    ) -> TransformRule:
        """
        Create a transform rule from pattern strings.

        Args:
            name: Rule name.
            source: Source pattern string.
            target: Target pattern string.
            priority: Rule priority.
            bidirectional: Allow reverse application.

        Returns:
            TransformRule object.
        """
        return TransformRule(
            name=name,
            source=Pattern.parse(source),
            target=Pattern.parse(target),
            priority=priority,
            bidirectional=bidirectional
        )


class MetagrammarRegistry:
    """
    Registry of metagrammar transformation rules.

    The registry manages collections of rules and provides methods
    for applying transformations to grammars.
    """

    def __init__(self):
        """Initialize empty registry."""
        self._rules: dict[str, TransformRule] = {}
        self._categories: dict[str, list[str]] = {}

    def register(self, rule: TransformRule, category: str = "default") -> None:
        """
        Register a transformation rule.

        Args:
            rule: The rule to register.
            category: Category to organize rules.
        """
        self._rules[rule.name] = rule
        if category not in self._categories:
            self._categories[category] = []
        if rule.name not in self._categories[category]:
            self._categories[category].append(rule.name)

    def unregister(self, name: str) -> Optional[TransformRule]:
        """
        Remove a rule from the registry.

        Args:
            name: Name of rule to remove.

        Returns:
            The removed rule, or None if not found.
        """
        rule = self._rules.pop(name, None)
        if rule:
            for cat_rules in self._categories.values():
                if name in cat_rules:
                    cat_rules.remove(name)
        return rule

    def get(self, name: str) -> Optional[TransformRule]:
        """Get a rule by name."""
        return self._rules.get(name)

    def get_category(self, category: str) -> list[TransformRule]:
        """Get all rules in a category."""
        names = self._categories.get(category, [])
        return [self._rules[n] for n in names if n in self._rules]

    def all_rules(self) -> list[TransformRule]:
        """Get all registered rules sorted by priority."""
        rules = list(self._rules.values())
        rules.sort(key=lambda r: r.priority, reverse=True)
        return rules

    def categories(self) -> list[str]:
        """Get list of all categories."""
        return list(self._categories.keys())

    def transform(self, production: list[str],
                  max_iterations: int = 100,
                  categories: Optional[list[str]] = None
                 ) -> list[str]:
        """
        Apply transformations to a production until fixed point.

        Args:
            production: The production to transform.
            max_iterations: Maximum transformation iterations.
            categories: Limit to rules from these categories.

        Returns:
            Transformed production.
        """
        if categories:
            rules = []
            for cat in categories:
                rules.extend(self.get_category(cat))
        else:
            rules = self.all_rules()

        current = production
        for _ in range(max_iterations):
            changed = False
            for rule in rules:
                result = rule.apply(current)
                if result is not None:
                    current = result
                    changed = True
                    break

            if not changed:
                break

        return current

    def transform_all(self, production: list[str],
                      categories: Optional[list[str]] = None
                     ) -> list[list[str]]:
        """
        Find all possible single-step transformations.

        Args:
            production: The production to transform.
            categories: Limit to rules from these categories.

        Returns:
            List of all possible transformations.
        """
        if categories:
            rules = []
            for cat in categories:
                rules.extend(self.get_category(cat))
        else:
            rules = self.all_rules()

        results = []
        for rule in rules:
            result = rule.apply(production)
            if result is not None:
                results.append(result)

            # Try reverse if bidirectional
            if rule.bidirectional:
                reverse = rule.apply_reverse(production)
                if reverse is not None:
                    results.append(reverse)

        return results

    def add_standard_rules(self) -> None:
        """Add commonly used grammar transformation rules."""
        # Distributive law: A (B | C) -> A B | A C
        self.register(
            TransformRule.from_strings(
                "distribute_left",
                "?A ( ?B | ?C )",
                "?A ?B | ?A ?C",
                priority=10
            ),
            category="algebra"
        )

        # Factor common prefix: A B | A C -> A (B | C)
        self.register(
            TransformRule.from_strings(
                "factor_prefix",
                "?A ?B | ?A ?C",
                "?A ( ?B | ?C )",
                priority=10
            ),
            category="algebra"
        )

        # Eliminate empty: A | epsilon -> A?
        # (Simplified representation)
        self.register(
            TransformRule.from_strings(
                "optional_empty",
                "?A | epsilon",
                "?A ?",
                priority=5
            ),
            category="simplify"
        )

        # Remove redundant parens: ( A ) -> A
        self.register(
            TransformRule.from_strings(
                "remove_parens",
                "( ?A )",
                "?A",
                priority=15
            ),
            category="simplify"
        )

        # Left recursion elimination prep (basic)
        # A ::= A a | b  ->  A ::= b A'   A' ::= a A' | epsilon
        # This is a simplified pattern - full implementation would need
        # grammar-level transformations
        self.register(
            TransformRule.from_strings(
                "left_rec_identify",
                "?A ::= ?A ?alpha | ?beta",
                "?A ::= ?beta ?A' | ?A' ::= ?alpha ?A' | epsilon",
                priority=20
            ),
            category="normalize"
        )

    def __len__(self) -> int:
        return len(self._rules)

    def __contains__(self, name: str) -> bool:
        return name in self._rules

    def __iter__(self):
        return iter(self._rules.values())


# Singleton default registry
_default_registry: Optional[MetagrammarRegistry] = None


def get_default_registry() -> MetagrammarRegistry:
    """Get the default metagrammar registry."""
    global _default_registry
    if _default_registry is None:
        _default_registry = MetagrammarRegistry()
        _default_registry.add_standard_rules()
    return _default_registry
