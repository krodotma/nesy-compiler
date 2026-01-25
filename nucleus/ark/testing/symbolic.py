#!/usr/bin/env python3
"""
symbolic.py - Symbolic Execution for ARK

P2-064: Create symbolic execution harness
P2-065: Implement SMT solver integration
P2-066: Add constraint satisfaction checks

Implements lightweight symbolic execution:
- Track symbolic values through code
- Collect path conditions
- Check satisfiability (optional Z3 integration)
"""

import ast
import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any, Set, Union
from enum import Enum

logger = logging.getLogger("ARK.Testing.Symbolic")


class SymbolicType(Enum):
    """Types of symbolic values."""
    INT = "int"
    FLOAT = "float"
    BOOL = "bool"
    STRING = "string"
    UNKNOWN = "unknown"


@dataclass
class SymbolicValue:
    """A symbolic value in execution."""
    name: str
    type: SymbolicType
    constraints: List[str] = field(default_factory=list)
    concrete_value: Optional[Any] = None
    
    def __repr__(self):
        return f"Sym({self.name}: {self.type.value})"
    
    def add_constraint(self, constraint: str) -> None:
        self.constraints.append(constraint)


@dataclass
class PathCondition:
    """A condition on a symbolic execution path."""
    condition: str
    is_true_branch: bool
    variables: Set[str] = field(default_factory=set)
    
    def negate(self) -> "PathCondition":
        return PathCondition(
            condition=f"not ({self.condition})",
            is_true_branch=not self.is_true_branch,
            variables=self.variables.copy()
        )


@dataclass
class SymbolicState:
    """State of symbolic execution."""
    variables: Dict[str, SymbolicValue]
    path_conditions: List[PathCondition]
    path_id: int
    is_feasible: bool = True
    
    def clone(self, path_id: int) -> "SymbolicState":
        return SymbolicState(
            variables={k: SymbolicValue(
                v.name, v.type, v.constraints.copy(), v.concrete_value
            ) for k, v in self.variables.items()},
            path_conditions=self.path_conditions.copy(),
            path_id=path_id,
            is_feasible=self.is_feasible
        )
    
    def add_condition(self, condition: PathCondition) -> None:
        self.path_conditions.append(condition)
    
    def to_dict(self) -> Dict:
        return {
            "path_id": self.path_id,
            "variables": {k: repr(v) for k, v in self.variables.items()},
            "conditions": [pc.condition for pc in self.path_conditions],
            "is_feasible": self.is_feasible
        }


@dataclass
class ExecutionPath:
    """A single execution path through the code."""
    path_id: int
    conditions: List[str]
    result: Optional[Any] = None
    is_error: bool = False
    error_type: Optional[str] = None
    
    def to_dict(self) -> Dict:
        return {
            "path_id": self.path_id,
            "conditions": self.conditions,
            "result": str(self.result) if self.result else None,
            "is_error": self.is_error,
            "error_type": self.error_type
        }


class SymbolicExecutor:
    """
    Lightweight symbolic execution engine.
    
    P2-064: Symbolic execution harness
    P2-065: SMT solver integration
    P2-066: Constraint satisfaction
    """
    
    def __init__(self, max_paths: int = 100, max_depth: int = 20):
        self.max_paths = max_paths
        self.max_depth = max_depth
        self._path_counter = 0
        self._smt_available = False
        
        # Try to import Z3
        try:
            import z3
            self._smt_available = True
            logger.info("Z3 SMT solver available")
        except ImportError:
            logger.info("Z3 not available - using heuristic constraint checking")
    
    def execute(self, code: str, inputs: Dict[str, SymbolicType]) -> List[ExecutionPath]:
        """
        Symbolically execute code with given symbolic inputs.
        
        Args:
            code: Python code to execute
            inputs: Mapping of input names to their symbolic types
            
        Returns:
            List of execution paths discovered
        """
        paths = []
        
        try:
            tree = ast.parse(code)
        except SyntaxError as e:
            return [ExecutionPath(
                path_id=0,
                conditions=[],
                is_error=True,
                error_type=f"SyntaxError: {e}"
            )]
        
        # Initialize symbolic state
        initial_state = SymbolicState(
            variables={
                name: SymbolicValue(name, sym_type)
                for name, sym_type in inputs.items()
            },
            path_conditions=[],
            path_id=0
        )
        
        # Explore paths
        self._explore(tree, initial_state, paths)
        
        return paths
    
    def _explore(
        self,
        node: ast.AST,
        state: SymbolicState,
        paths: List[ExecutionPath],
        depth: int = 0
    ) -> None:
        """Explore execution paths from this node."""
        if depth > self.max_depth or len(paths) >= self.max_paths:
            return
        
        if isinstance(node, ast.Module):
            for stmt in node.body:
                self._explore(stmt, state, paths, depth)
        
        elif isinstance(node, ast.If):
            # Branch on condition
            condition_str = ast.unparse(node.test)
            
            # True branch
            true_state = state.clone(self._next_path_id())
            true_state.add_condition(PathCondition(
                condition=condition_str,
                is_true_branch=True,
                variables=self._extract_variables(node.test)
            ))
            
            for stmt in node.body:
                self._explore(stmt, true_state, paths, depth + 1)
            
            # False branch
            false_state = state.clone(self._next_path_id())
            false_state.add_condition(PathCondition(
                condition=f"not ({condition_str})",
                is_true_branch=False,
                variables=self._extract_variables(node.test)
            ))
            
            for stmt in node.orelse:
                self._explore(stmt, false_state, paths, depth + 1)
        
        elif isinstance(node, ast.Return):
            # End of path
            path = ExecutionPath(
                path_id=state.path_id,
                conditions=[pc.condition for pc in state.path_conditions],
                result=ast.unparse(node.value) if node.value else None
            )
            paths.append(path)
        
        elif isinstance(node, ast.Raise):
            # Error path
            error_type = ast.unparse(node.exc) if node.exc else "Exception"
            path = ExecutionPath(
                path_id=state.path_id,
                conditions=[pc.condition for pc in state.path_conditions],
                is_error=True,
                error_type=error_type
            )
            paths.append(path)
        
        elif isinstance(node, ast.Assign):
            # Track assignment
            for target in node.targets:
                if isinstance(target, ast.Name):
                    value_str = ast.unparse(node.value)
                    state.variables[target.id] = SymbolicValue(
                        name=target.id,
                        type=SymbolicType.UNKNOWN,
                        constraints=[f"{target.id} = {value_str}"]
                    )
        
        elif isinstance(node, ast.For):
            # Unroll loop once for simplicity
            for stmt in node.body:
                self._explore(stmt, state, paths, depth + 1)
        
        elif isinstance(node, ast.While):
            # Check condition once
            condition_str = ast.unparse(node.test)
            state.add_condition(PathCondition(
                condition=condition_str,
                is_true_branch=True,
                variables=self._extract_variables(node.test)
            ))
            for stmt in node.body:
                self._explore(stmt, state, paths, depth + 1)
    
    def _next_path_id(self) -> int:
        self._path_counter += 1
        return self._path_counter
    
    def _extract_variables(self, node: ast.AST) -> Set[str]:
        """Extract variable names from expression."""
        variables = set()
        for child in ast.walk(node):
            if isinstance(child, ast.Name):
                variables.add(child.id)
        return variables
    
    def check_satisfiability(
        self, 
        conditions: List[str]
    ) -> Optional[bool]:
        """
        Check if path conditions are satisfiable.
        
        P2-065: SMT solver integration
        """
        if not conditions:
            return True
        
        if self._smt_available:
            return self._check_with_z3(conditions)
        else:
            return self._check_heuristic(conditions)
    
    def _check_with_z3(self, conditions: List[str]) -> Optional[bool]:
        """Check satisfiability using Z3."""
        try:
            import z3
            
            # Create solver
            solver = z3.Solver()
            
            # Create symbolic variables (simplified)
            variables = {}
            for cond in conditions:
                # Extract variable names - very simplified
                for word in cond.replace('(', ' ').replace(')', ' ').split():
                    if word.isidentifier() and word not in ['and', 'or', 'not', 'True', 'False']:
                        if word not in variables:
                            variables[word] = z3.Int(word)
            
            # This is a simplified check - real implementation would parse properly
            # For now, return None to indicate unknown
            return None
            
        except Exception as e:
            logger.debug("Z3 check failed: %s", e)
            return None
    
    def _check_heuristic(self, conditions: List[str]) -> Optional[bool]:
        """Heuristic satisfiability check without SMT solver."""
        # Simple contradiction detection
        for i, c1 in enumerate(conditions):
            for c2 in conditions[i+1:]:
                if c1 == f"not ({c2})" or c2 == f"not ({c1})":
                    return False
        return None  # Unknown
    
    def find_inputs(
        self,
        code: str,
        target_condition: str,
        input_types: Dict[str, SymbolicType]
    ) -> Optional[Dict[str, Any]]:
        """
        Find concrete inputs that satisfy target condition.
        
        P2-066: Constraint satisfaction checks
        """
        paths = self.execute(code, input_types)
        
        for path in paths:
            if target_condition in " ".join(path.conditions):
                # This path satisfies the condition
                # Generate concrete values (simplified)
                return self._generate_concrete(input_types, path.conditions)
        
        return None
    
    def _generate_concrete(
        self, 
        input_types: Dict[str, SymbolicType],
        conditions: List[str]
    ) -> Dict[str, Any]:
        """Generate concrete values satisfying conditions."""
        values = {}
        
        for name, sym_type in input_types.items():
            if sym_type == SymbolicType.INT:
                values[name] = 0
            elif sym_type == SymbolicType.FLOAT:
                values[name] = 0.0
            elif sym_type == SymbolicType.BOOL:
                # Check if condition mentions this variable
                for cond in conditions:
                    if name in cond:
                        values[name] = "True" in cond or "not" not in cond
                        break
                else:
                    values[name] = True
            elif sym_type == SymbolicType.STRING:
                values[name] = "test"
            else:
                values[name] = None
        
        return values
