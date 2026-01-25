#!/usr/bin/env python3
"""
grammar.py - Grammar-Guided Synthesis (SyGuS)

Constrains AST transformations to a defined grammar.
Only allowed patterns can be synthesized.
"""

import ast
from dataclasses import dataclass, field
from typing import Set, List, Dict, Optional, Tuple
from enum import Enum


class NodeCategory(Enum):
    """Categories of AST nodes."""
    STRUCTURAL = "structural"    # Module, Class, Function
    CONTROL = "control"          # If, For, While, Try
    EXPRESSION = "expression"    # Name, Call, BinOp
    LITERAL = "literal"          # Constant, List, Dict
    IMPORT = "import"            # Import, ImportFrom


@dataclass
class GrammarRule:
    """A rule in the synthesis grammar."""
    node_type: str
    allowed: bool = True
    max_depth: int = 10
    max_children: int = 50
    conditions: List[str] = field(default_factory=list)


@dataclass
class SynthesisGrammar:
    """
    Grammar definition for code synthesis.
    
    Defines what AST patterns are allowed/forbidden.
    """
    name: str = "pluribus_default"
    version: str = "1.0"
    rules: Dict[str, GrammarRule] = field(default_factory=dict)
    
    # Anti-patterns (always forbidden)
    forbidden_patterns: Set[str] = field(default_factory=lambda: {
        "AbstractFactoryFactory",
        "AbstractManagerFactory",
        "HelperHelper",
        "UtilUtil",
    })
    
    # Maximum complexity thresholds
    max_function_lines: int = 100
    max_class_bases: int = 3
    max_nesting_depth: int = 6
    
    def add_rule(self, node_type: str, allowed: bool = True, **kwargs) -> None:
        """Add a grammar rule."""
        self.rules[node_type] = GrammarRule(
            node_type=node_type,
            allowed=allowed,
            **kwargs
        )
    
    def is_allowed(self, node_type: str) -> bool:
        """Check if a node type is allowed."""
        if node_type in self.rules:
            return self.rules[node_type].allowed
        return True  # Allow by default
    
    @classmethod
    def strict(cls) -> "SynthesisGrammar":
        """Create a strict grammar for production code."""
        g = cls(name="strict", version="1.0")
        
        # Forbidden in strict mode
        g.add_rule("Global", allowed=False)
        g.add_rule("Exec", allowed=False)
        g.add_rule("Eval", allowed=False)
        
        # Stricter limits
        g.max_function_lines = 50
        g.max_nesting_depth = 4
        g.max_class_bases = 2
        
        return g
    
    @classmethod
    def permissive(cls) -> "SynthesisGrammar":
        """Create a permissive grammar for exploration."""
        g = cls(name="permissive", version="1.0")
        
        g.max_function_lines = 200
        g.max_nesting_depth = 8
        g.max_class_bases = 5
        
        return g


class GrammarFilter:
    """
    Filters AST transformations against a synthesis grammar.
    
    Ensures synthesized code adheres to the grammar constraints.
    """
    
    def __init__(self, grammar: Optional[SynthesisGrammar] = None):
        self.grammar = grammar or SynthesisGrammar()
        self.violations: List[str] = []
    
    def validate(self, code: str) -> Tuple[bool, List[str]]:
        """
        Validate code against the grammar.
        
        Returns:
            (valid, violations)
        """
        self.violations = []
        
        try:
            tree = ast.parse(code)
        except SyntaxError as e:
            self.violations.append(f"Syntax error: {e}")
            return False, self.violations
        
        # Walk AST and check all nodes
        self._check_tree(tree)
        
        return len(self.violations) == 0, self.violations
    
    def _check_tree(self, tree: ast.AST, depth: int = 0) -> None:
        """Recursively check AST tree."""
        for node in ast.walk(tree):
            self._check_node(node, depth)
        
        # Check nesting depth
        max_depth = self._calculate_max_depth(tree)
        if max_depth > self.grammar.max_nesting_depth:
            self.violations.append(
                f"Nesting depth {max_depth} exceeds max {self.grammar.max_nesting_depth}"
            )
    
    def _check_node(self, node: ast.AST, depth: int) -> None:
        """Check a single AST node."""
        node_type = type(node).__name__
        
        # Check if node type is allowed
        if not self.grammar.is_allowed(node_type):
            self.violations.append(f"Forbidden node type: {node_type}")
        
        # Check class-specific rules
        if isinstance(node, ast.ClassDef):
            self._check_class(node)
        elif isinstance(node, ast.FunctionDef):
            self._check_function(node)
        elif isinstance(node, ast.Name):
            self._check_name(node)
    
    def _check_class(self, node: ast.ClassDef) -> None:
        """Check class definition rules."""
        # Check inheritance depth
        if len(node.bases) > self.grammar.max_class_bases:
            self.violations.append(
                f"Class {node.name} has {len(node.bases)} bases, max is {self.grammar.max_class_bases}"
            )
        
        # Check for anti-pattern names
        for pattern in self.grammar.forbidden_patterns:
            if pattern in node.name:
                self.violations.append(f"Anti-pattern class name: {node.name}")
    
    def _check_function(self, node: ast.FunctionDef) -> None:
        """Check function definition rules."""
        # Check line count
        if hasattr(node, 'end_lineno') and hasattr(node, 'lineno'):
            lines = node.end_lineno - node.lineno
            if lines > self.grammar.max_function_lines:
                self.violations.append(
                    f"Function {node.name} has {lines} lines, max is {self.grammar.max_function_lines}"
                )
        
        # Check for anti-pattern names
        for pattern in self.grammar.forbidden_patterns:
            if pattern.lower() in node.name.lower():
                self.violations.append(f"Anti-pattern function name: {node.name}")
    
    def _check_name(self, node: ast.Name) -> None:
        """Check name references."""
        # Check for dangerous builtins
        dangerous = {"eval", "exec", "compile", "__import__"}
        if node.id in dangerous:
            self.violations.append(f"Dangerous builtin: {node.id}")
    
    def _calculate_max_depth(self, tree: ast.AST) -> int:
        """Calculate maximum nesting depth."""
        def depth(node: ast.AST, current: int = 0) -> int:
            max_child = current
            for child in ast.iter_child_nodes(node):
                if isinstance(child, (ast.If, ast.For, ast.While, ast.With, 
                                     ast.Try, ast.FunctionDef, ast.ClassDef)):
                    max_child = max(max_child, depth(child, current + 1))
                else:
                    max_child = max(max_child, depth(child, current))
            return max_child
        
        return depth(tree)
    
    def get_report(self) -> str:
        """Get human-readable validation report."""
        if not self.violations:
            return "✅ Code adheres to synthesis grammar"
        
        lines = ["❌ Grammar violations:"]
        for v in self.violations:
            lines.append(f"  - {v}")
        return "\n".join(lines)
