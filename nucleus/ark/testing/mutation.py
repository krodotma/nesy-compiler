#!/usr/bin/env python3
"""
mutation.py - Mutation Testing for ARK Gates

P2-063: Add mutation testing for gates

Implements mutation testing:
- Generate mutants of gate logic
- Run tests against mutants
- Report mutation score (killed / total)
"""

import ast
import copy
import random
import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any, Callable, Set
from enum import Enum

logger = logging.getLogger("ARK.Testing.Mutation")


class MutantStatus(Enum):
    ALIVE = "alive"      # Mutant not detected (bad!)
    KILLED = "killed"    # Mutant detected (good!)
    TIMEOUT = "timeout"
    ERROR = "error"


@dataclass
class Mutant:
    """A mutation of the original code."""
    id: str
    operator: str       # Type of mutation
    location: str       # File:line
    original: str       # Original code
    mutated: str        # Mutated code
    status: MutantStatus = MutantStatus.ALIVE
    killed_by: Optional[str] = None  # Test that killed it
    
    def to_dict(self) -> Dict:
        return {
            "id": self.id,
            "operator": self.operator,
            "location": self.location,
            "original": self.original[:50],
            "mutated": self.mutated[:50],
            "status": self.status.value,
            "killed_by": self.killed_by
        }


@dataclass
class MutationResult:
    """Result of mutation testing."""
    total_mutants: int
    killed: int
    alive: int
    timeout: int
    errors: int
    mutation_score: float  # killed / total
    mutants: List[Mutant] = field(default_factory=list)
    
    def to_dict(self) -> Dict:
        return {
            "total_mutants": self.total_mutants,
            "killed": self.killed,
            "alive": self.alive,
            "timeout": self.timeout,
            "errors": self.errors,
            "mutation_score": self.mutation_score,
            "surviving_mutants": [
                m.to_dict() for m in self.mutants if m.status == MutantStatus.ALIVE
            ]
        }


class MutationOperator:
    """Base class for mutation operators."""
    name: str = "base"
    
    def apply(self, node: ast.AST) -> Optional[ast.AST]:
        """Apply mutation to AST node. Return None if not applicable."""
        raise NotImplementedError


class ArithmeticOperatorMutation(MutationOperator):
    """Mutate arithmetic operators: + -> -, * -> /, etc."""
    name = "arithmetic_op"
    
    replacements = {
        ast.Add: ast.Sub,
        ast.Sub: ast.Add,
        ast.Mult: ast.Div,
        ast.Div: ast.Mult,
    }
    
    def apply(self, node: ast.AST) -> Optional[ast.AST]:
        if isinstance(node, ast.BinOp):
            op_type = type(node.op)
            if op_type in self.replacements:
                new_node = copy.deepcopy(node)
                new_node.op = self.replacements[op_type]()
                return new_node
        return None


class ComparisonOperatorMutation(MutationOperator):
    """Mutate comparison operators: < -> <=, == -> !=, etc."""
    name = "comparison_op"
    
    replacements = {
        ast.Lt: ast.LtE,
        ast.LtE: ast.Lt,
        ast.Gt: ast.GtE,
        ast.GtE: ast.Gt,
        ast.Eq: ast.NotEq,
        ast.NotEq: ast.Eq,
    }
    
    def apply(self, node: ast.AST) -> Optional[ast.AST]:
        if isinstance(node, ast.Compare):
            for i, op in enumerate(node.ops):
                op_type = type(op)
                if op_type in self.replacements:
                    new_node = copy.deepcopy(node)
                    new_node.ops[i] = self.replacements[op_type]()
                    return new_node
        return None


class BooleanOperatorMutation(MutationOperator):
    """Mutate boolean operators: and -> or, True -> False."""
    name = "boolean_op"
    
    def apply(self, node: ast.AST) -> Optional[ast.AST]:
        if isinstance(node, ast.BoolOp):
            new_node = copy.deepcopy(node)
            if isinstance(node.op, ast.And):
                new_node.op = ast.Or()
            else:
                new_node.op = ast.And()
            return new_node
        
        if isinstance(node, ast.Constant):
            if node.value is True:
                return ast.Constant(value=False)
            elif node.value is False:
                return ast.Constant(value=True)
        
        return None


class ConstantMutation(MutationOperator):
    """Mutate constants: 0 -> 1, 1 -> 0, string -> empty."""
    name = "constant"
    
    def apply(self, node: ast.AST) -> Optional[ast.AST]:
        if isinstance(node, ast.Constant):
            if isinstance(node.value, int):
                if node.value == 0:
                    return ast.Constant(value=1)
                elif node.value == 1:
                    return ast.Constant(value=0)
                else:
                    return ast.Constant(value=node.value + 1)
            elif isinstance(node.value, str):
                return ast.Constant(value="")
            elif isinstance(node.value, float):
                return ast.Constant(value=node.value + 0.1)
        return None


class ReturnValueMutation(MutationOperator):
    """Mutate return values."""
    name = "return_value"
    
    def apply(self, node: ast.AST) -> Optional[ast.AST]:
        if isinstance(node, ast.Return):
            if node.value is not None:
                new_node = copy.deepcopy(node)
                new_node.value = ast.Constant(value=None)
                return new_node
        return None


class MutationTester:
    """
    Mutation testing engine for ARK.
    
    P2-063: Add mutation testing for gates
    """
    
    def __init__(self, timeout_seconds: float = 5.0):
        self.timeout = timeout_seconds
        self.operators = [
            ArithmeticOperatorMutation(),
            ComparisonOperatorMutation(),
            BooleanOperatorMutation(),
            ConstantMutation(),
            ReturnValueMutation()
        ]
    
    def generate_mutants(self, code: str, max_mutants: int = 50) -> List[Mutant]:
        """Generate mutants from source code."""
        mutants = []
        
        try:
            tree = ast.parse(code)
        except SyntaxError:
            return mutants
        
        mutant_id = 0
        
        for node in ast.walk(tree):
            for operator in self.operators:
                mutated_node = operator.apply(node)
                if mutated_node is not None and mutant_id < max_mutants:
                    # Create mutant
                    try:
                        original_str = ast.unparse(node)
                        mutated_str = ast.unparse(mutated_node)
                        
                        if original_str != mutated_str:
                            line = getattr(node, 'lineno', 0)
                            mutants.append(Mutant(
                                id=f"m{mutant_id}",
                                operator=operator.name,
                                location=f"line:{line}",
                                original=original_str,
                                mutated=mutated_str
                            ))
                            mutant_id += 1
                    except Exception:
                        pass
        
        return mutants
    
    def run_tests(
        self,
        code: str,
        test_runner: Callable[[str], bool],
        max_mutants: int = 50
    ) -> MutationResult:
        """
        Run mutation testing.
        
        Args:
            code: Source code to mutate
            test_runner: Function that takes mutated code and returns True if tests pass
            max_mutants: Maximum mutants to generate
        """
        mutants = self.generate_mutants(code, max_mutants)
        
        killed = 0
        alive = 0
        timeout_count = 0
        error_count = 0
        
        for mutant in mutants:
            try:
                # Create mutated code
                mutated_code = code.replace(mutant.original, mutant.mutated, 1)
                
                # Run tests on mutant
                tests_pass = test_runner(mutated_code)
                
                if tests_pass:
                    mutant.status = MutantStatus.ALIVE
                    alive += 1
                else:
                    mutant.status = MutantStatus.KILLED
                    killed += 1
                    
            except TimeoutError:
                mutant.status = MutantStatus.TIMEOUT
                timeout_count += 1
            except Exception as e:
                mutant.status = MutantStatus.ERROR
                error_count += 1
                logger.debug("Mutant error: %s", e)
        
        total = len(mutants)
        score = killed / total if total > 0 else 1.0
        
        return MutationResult(
            total_mutants=total,
            killed=killed,
            alive=alive,
            timeout=timeout_count,
            errors=error_count,
            mutation_score=score,
            mutants=mutants
        )
    
    def get_gate_mutants(self) -> List[str]:
        """Get common mutations for gate code."""
        return [
            # Threshold mutations
            "h_total > 0.7 -> h_total > 0.5",
            "cmp >= 0.5 -> cmp >= 0.3",
            # Boolean flips
            "passed = True -> passed = False",
            "and -> or",
            # Return mutations
            "return True -> return False"
        ]
