#!/usr/bin/env python3
"""
Step 111 (continued): Mutant Generator

Generates code mutants by applying systematic transformations to source code.

PBTSO Phase: SKILL
Bus Topics:
- test.mutant.generate (subscribes)
- test.mutant.generated (emits)

Dependencies: Step 111 (Mutation Engine)
"""
from __future__ import annotations

import ast
import copy
import uuid
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple


# ============================================================================
# Mutation Types
# ============================================================================

class MutationType(Enum):
    """Types of mutations that can be applied."""
    # Arithmetic operators
    ARITHMETIC = "arithmetic"  # + -> -, * -> /, etc.

    # Relational operators
    RELATIONAL = "relational"  # < -> <=, == -> !=, etc.

    # Logical operators
    LOGICAL = "logical"  # and -> or, not removal

    # Boundary mutations
    BOUNDARY = "boundary"  # x < y -> x <= y, off-by-one

    # Constant mutations
    CONSTANT = "constant"  # 0 -> 1, True -> False

    # Return value mutations
    RETURN = "return"  # return x -> return None, return 0

    # Statement deletion
    STATEMENT = "statement"  # Remove entire statement

    # Assignment mutations
    ASSIGNMENT = "assignment"  # a = b -> a = 0

    # Exception mutations
    EXCEPTION = "exception"  # Remove try/except, change exception type

    # Negate conditionals
    NEGATE_CONDITIONAL = "negate_conditional"  # if x -> if not x


# ============================================================================
# Data Types
# ============================================================================

@dataclass
class Mutant:
    """Represents a code mutation."""
    id: str
    file_path: str
    line_number: int
    mutation_type: MutationType
    original_code: str
    mutated_code: str
    original_ast_dump: str = ""
    mutated_ast_dump: str = ""
    description: str = ""

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "id": self.id,
            "file_path": self.file_path,
            "line_number": self.line_number,
            "mutation_type": self.mutation_type.value,
            "original_code": self.original_code,
            "mutated_code": self.mutated_code,
            "description": self.description,
        }


@dataclass
class MutationLocation:
    """Location where a mutation can be applied."""
    line_number: int
    column: int
    mutation_type: MutationType
    node: ast.AST
    original_snippet: str


# ============================================================================
# AST Mutation Transformers
# ============================================================================

class ArithmeticMutator(ast.NodeTransformer):
    """Mutate arithmetic operators."""

    MUTATIONS = {
        ast.Add: [ast.Sub, ast.Mult],
        ast.Sub: [ast.Add, ast.Div],
        ast.Mult: [ast.Div, ast.Add],
        ast.Div: [ast.Mult, ast.Sub],
        ast.Mod: [ast.Div, ast.Mult],
        ast.Pow: [ast.Mult, ast.Div],
        ast.FloorDiv: [ast.Div, ast.Mult],
    }

    def __init__(self, target_line: int, replacement_op: type):
        self.target_line = target_line
        self.replacement_op = replacement_op
        self.mutated = False

    def visit_BinOp(self, node: ast.BinOp) -> ast.BinOp:
        if node.lineno == self.target_line and not self.mutated:
            op_type = type(node.op)
            if op_type in self.MUTATIONS and self.replacement_op in self.MUTATIONS[op_type]:
                node.op = self.replacement_op()
                self.mutated = True
        return node


class RelationalMutator(ast.NodeTransformer):
    """Mutate relational/comparison operators."""

    MUTATIONS = {
        ast.Lt: [ast.LtE, ast.Gt, ast.Eq],
        ast.LtE: [ast.Lt, ast.GtE, ast.Eq],
        ast.Gt: [ast.GtE, ast.Lt, ast.Eq],
        ast.GtE: [ast.Gt, ast.LtE, ast.Eq],
        ast.Eq: [ast.NotEq, ast.Lt, ast.Gt],
        ast.NotEq: [ast.Eq],
        ast.Is: [ast.IsNot],
        ast.IsNot: [ast.Is],
        ast.In: [ast.NotIn],
        ast.NotIn: [ast.In],
    }

    def __init__(self, target_line: int, replacement_op: type):
        self.target_line = target_line
        self.replacement_op = replacement_op
        self.mutated = False

    def visit_Compare(self, node: ast.Compare) -> ast.Compare:
        if node.lineno == self.target_line and not self.mutated:
            if len(node.ops) > 0:
                op_type = type(node.ops[0])
                if op_type in self.MUTATIONS and self.replacement_op in self.MUTATIONS[op_type]:
                    node.ops[0] = self.replacement_op()
                    self.mutated = True
        return node


class LogicalMutator(ast.NodeTransformer):
    """Mutate logical operators."""

    def __init__(self, target_line: int, mutation_kind: str):
        self.target_line = target_line
        self.mutation_kind = mutation_kind
        self.mutated = False

    def visit_BoolOp(self, node: ast.BoolOp) -> ast.BoolOp:
        if node.lineno == self.target_line and not self.mutated:
            if self.mutation_kind == "and_to_or" and isinstance(node.op, ast.And):
                node.op = ast.Or()
                self.mutated = True
            elif self.mutation_kind == "or_to_and" and isinstance(node.op, ast.Or):
                node.op = ast.And()
                self.mutated = True
        return node

    def visit_UnaryOp(self, node: ast.UnaryOp) -> ast.AST:
        if node.lineno == self.target_line and not self.mutated:
            if self.mutation_kind == "remove_not" and isinstance(node.op, ast.Not):
                self.mutated = True
                return node.operand
        return node


class BoundaryMutator(ast.NodeTransformer):
    """Mutate boundary conditions (off-by-one errors)."""

    def __init__(self, target_line: int, mutation_kind: str):
        self.target_line = target_line
        self.mutation_kind = mutation_kind
        self.mutated = False

    def visit_Constant(self, node: ast.Constant) -> ast.Constant:
        if node.lineno == self.target_line and not self.mutated:
            if isinstance(node.value, int):
                if self.mutation_kind == "increment":
                    node.value += 1
                    self.mutated = True
                elif self.mutation_kind == "decrement":
                    node.value -= 1
                    self.mutated = True
        return node


class ReturnMutator(ast.NodeTransformer):
    """Mutate return statements."""

    def __init__(self, target_line: int, mutation_kind: str):
        self.target_line = target_line
        self.mutation_kind = mutation_kind
        self.mutated = False

    def visit_Return(self, node: ast.Return) -> ast.Return:
        if node.lineno == self.target_line and not self.mutated:
            if self.mutation_kind == "return_none":
                node.value = ast.Constant(value=None)
                self.mutated = True
            elif self.mutation_kind == "return_zero":
                node.value = ast.Constant(value=0)
                self.mutated = True
            elif self.mutation_kind == "return_empty_string":
                node.value = ast.Constant(value="")
                self.mutated = True
            elif self.mutation_kind == "return_true":
                node.value = ast.Constant(value=True)
                self.mutated = True
            elif self.mutation_kind == "return_false":
                node.value = ast.Constant(value=False)
                self.mutated = True
        return node


class NegateConditionalMutator(ast.NodeTransformer):
    """Negate conditional expressions."""

    def __init__(self, target_line: int):
        self.target_line = target_line
        self.mutated = False

    def visit_If(self, node: ast.If) -> ast.If:
        if node.lineno == self.target_line and not self.mutated:
            # Wrap the condition in a Not()
            node.test = ast.UnaryOp(op=ast.Not(), operand=node.test)
            self.mutated = True
        return node

    def visit_While(self, node: ast.While) -> ast.While:
        if node.lineno == self.target_line and not self.mutated:
            node.test = ast.UnaryOp(op=ast.Not(), operand=node.test)
            self.mutated = True
        return node


# ============================================================================
# Mutant Generator
# ============================================================================

class MutantGenerator:
    """
    Generates code mutants by applying systematic transformations.

    Supports multiple mutation types:
    - Arithmetic operator changes
    - Relational operator changes
    - Logical operator changes
    - Boundary value changes
    - Return value mutations
    - Conditional negations
    """

    def __init__(self):
        self._mutation_strategies: Dict[MutationType, callable] = {
            MutationType.ARITHMETIC: self._generate_arithmetic_mutants,
            MutationType.RELATIONAL: self._generate_relational_mutants,
            MutationType.LOGICAL: self._generate_logical_mutants,
            MutationType.BOUNDARY: self._generate_boundary_mutants,
            MutationType.RETURN: self._generate_return_mutants,
            MutationType.NEGATE_CONDITIONAL: self._generate_negate_mutants,
        }

    def generate_mutants(
        self,
        source_code: str,
        file_path: str,
        mutation_types: Optional[List[MutationType]] = None,
    ) -> List[Mutant]:
        """
        Generate mutants for the given source code.

        Args:
            source_code: Python source code
            file_path: Path to the source file
            mutation_types: Types of mutations to generate (all if None)

        Returns:
            List of Mutant objects
        """
        if mutation_types is None:
            mutation_types = list(MutationType)

        try:
            tree = ast.parse(source_code)
        except SyntaxError:
            return []

        all_mutants = []

        for mutation_type in mutation_types:
            strategy = self._mutation_strategies.get(mutation_type)
            if strategy:
                mutants = strategy(source_code, file_path, tree)
                all_mutants.extend(mutants)

        return all_mutants

    def _generate_arithmetic_mutants(
        self,
        source_code: str,
        file_path: str,
        tree: ast.Module,
    ) -> List[Mutant]:
        """Generate arithmetic operator mutations."""
        mutants = []

        # Find all binary operations
        for node in ast.walk(tree):
            if isinstance(node, ast.BinOp):
                op_type = type(node.op)
                if op_type in ArithmeticMutator.MUTATIONS:
                    for replacement in ArithmeticMutator.MUTATIONS[op_type]:
                        mutated_tree = copy.deepcopy(tree)
                        mutator = ArithmeticMutator(node.lineno, replacement)
                        mutated_tree = mutator.visit(mutated_tree)

                        if mutator.mutated:
                            try:
                                mutated_code = ast.unparse(mutated_tree)
                                original_line = source_code.split('\n')[node.lineno - 1]
                                mutated_line = mutated_code.split('\n')[node.lineno - 1] if node.lineno <= len(mutated_code.split('\n')) else ""

                                mutants.append(Mutant(
                                    id=str(uuid.uuid4()),
                                    file_path=file_path,
                                    line_number=node.lineno,
                                    mutation_type=MutationType.ARITHMETIC,
                                    original_code=original_line.strip(),
                                    mutated_code=mutated_code,
                                    description=f"{op_type.__name__} -> {replacement.__name__}",
                                ))
                            except Exception:
                                continue

        return mutants

    def _generate_relational_mutants(
        self,
        source_code: str,
        file_path: str,
        tree: ast.Module,
    ) -> List[Mutant]:
        """Generate relational operator mutations."""
        mutants = []

        for node in ast.walk(tree):
            if isinstance(node, ast.Compare) and len(node.ops) > 0:
                op_type = type(node.ops[0])
                if op_type in RelationalMutator.MUTATIONS:
                    for replacement in RelationalMutator.MUTATIONS[op_type]:
                        mutated_tree = copy.deepcopy(tree)
                        mutator = RelationalMutator(node.lineno, replacement)
                        mutated_tree = mutator.visit(mutated_tree)

                        if mutator.mutated:
                            try:
                                mutated_code = ast.unparse(mutated_tree)
                                original_line = source_code.split('\n')[node.lineno - 1]

                                mutants.append(Mutant(
                                    id=str(uuid.uuid4()),
                                    file_path=file_path,
                                    line_number=node.lineno,
                                    mutation_type=MutationType.RELATIONAL,
                                    original_code=original_line.strip(),
                                    mutated_code=mutated_code,
                                    description=f"{op_type.__name__} -> {replacement.__name__}",
                                ))
                            except Exception:
                                continue

        return mutants

    def _generate_logical_mutants(
        self,
        source_code: str,
        file_path: str,
        tree: ast.Module,
    ) -> List[Mutant]:
        """Generate logical operator mutations."""
        mutants = []

        for node in ast.walk(tree):
            if isinstance(node, ast.BoolOp):
                if isinstance(node.op, ast.And):
                    mutations = [("and_to_or", "and -> or")]
                else:
                    mutations = [("or_to_and", "or -> and")]

                for mutation_kind, desc in mutations:
                    mutated_tree = copy.deepcopy(tree)
                    mutator = LogicalMutator(node.lineno, mutation_kind)
                    mutated_tree = mutator.visit(mutated_tree)

                    if mutator.mutated:
                        try:
                            mutated_code = ast.unparse(mutated_tree)
                            original_line = source_code.split('\n')[node.lineno - 1]

                            mutants.append(Mutant(
                                id=str(uuid.uuid4()),
                                file_path=file_path,
                                line_number=node.lineno,
                                mutation_type=MutationType.LOGICAL,
                                original_code=original_line.strip(),
                                mutated_code=mutated_code,
                                description=desc,
                            ))
                        except Exception:
                            continue

            elif isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.Not):
                mutated_tree = copy.deepcopy(tree)
                mutator = LogicalMutator(node.lineno, "remove_not")
                mutated_tree = mutator.visit(mutated_tree)

                if mutator.mutated:
                    try:
                        mutated_code = ast.unparse(mutated_tree)
                        original_line = source_code.split('\n')[node.lineno - 1]

                        mutants.append(Mutant(
                            id=str(uuid.uuid4()),
                            file_path=file_path,
                            line_number=node.lineno,
                            mutation_type=MutationType.LOGICAL,
                            original_code=original_line.strip(),
                            mutated_code=mutated_code,
                            description="remove not",
                        ))
                    except Exception:
                        continue

        return mutants

    def _generate_boundary_mutants(
        self,
        source_code: str,
        file_path: str,
        tree: ast.Module,
    ) -> List[Mutant]:
        """Generate boundary value mutations (off-by-one)."""
        mutants = []

        for node in ast.walk(tree):
            if isinstance(node, ast.Constant) and isinstance(node.value, int):
                for mutation_kind in ["increment", "decrement"]:
                    mutated_tree = copy.deepcopy(tree)
                    mutator = BoundaryMutator(node.lineno, mutation_kind)
                    mutated_tree = mutator.visit(mutated_tree)

                    if mutator.mutated:
                        try:
                            mutated_code = ast.unparse(mutated_tree)
                            original_line = source_code.split('\n')[node.lineno - 1]

                            mutants.append(Mutant(
                                id=str(uuid.uuid4()),
                                file_path=file_path,
                                line_number=node.lineno,
                                mutation_type=MutationType.BOUNDARY,
                                original_code=original_line.strip(),
                                mutated_code=mutated_code,
                                description=f"{mutation_kind} integer constant",
                            ))
                        except Exception:
                            continue

        return mutants

    def _generate_return_mutants(
        self,
        source_code: str,
        file_path: str,
        tree: ast.Module,
    ) -> List[Mutant]:
        """Generate return statement mutations."""
        mutants = []

        for node in ast.walk(tree):
            if isinstance(node, ast.Return) and node.value is not None:
                mutations = [
                    ("return_none", "return None"),
                    ("return_zero", "return 0"),
                    ("return_false", "return False"),
                ]

                for mutation_kind, desc in mutations:
                    mutated_tree = copy.deepcopy(tree)
                    mutator = ReturnMutator(node.lineno, mutation_kind)
                    mutated_tree = mutator.visit(mutated_tree)

                    if mutator.mutated:
                        try:
                            mutated_code = ast.unparse(mutated_tree)
                            original_line = source_code.split('\n')[node.lineno - 1]

                            mutants.append(Mutant(
                                id=str(uuid.uuid4()),
                                file_path=file_path,
                                line_number=node.lineno,
                                mutation_type=MutationType.RETURN,
                                original_code=original_line.strip(),
                                mutated_code=mutated_code,
                                description=desc,
                            ))
                        except Exception:
                            continue

        return mutants

    def _generate_negate_mutants(
        self,
        source_code: str,
        file_path: str,
        tree: ast.Module,
    ) -> List[Mutant]:
        """Generate conditional negation mutations."""
        mutants = []

        for node in ast.walk(tree):
            if isinstance(node, (ast.If, ast.While)):
                mutated_tree = copy.deepcopy(tree)
                mutator = NegateConditionalMutator(node.lineno)
                mutated_tree = mutator.visit(mutated_tree)

                if mutator.mutated:
                    try:
                        mutated_code = ast.unparse(mutated_tree)
                        original_line = source_code.split('\n')[node.lineno - 1]

                        mutants.append(Mutant(
                            id=str(uuid.uuid4()),
                            file_path=file_path,
                            line_number=node.lineno,
                            mutation_type=MutationType.NEGATE_CONDITIONAL,
                            original_code=original_line.strip(),
                            mutated_code=mutated_code,
                            description="negate conditional",
                        ))
                    except Exception:
                        continue

        return mutants

    def count_mutation_opportunities(
        self,
        source_code: str,
    ) -> Dict[MutationType, int]:
        """Count potential mutation opportunities in source code."""
        try:
            tree = ast.parse(source_code)
        except SyntaxError:
            return {}

        counts = {mt: 0 for mt in MutationType}

        for node in ast.walk(tree):
            if isinstance(node, ast.BinOp):
                if type(node.op) in ArithmeticMutator.MUTATIONS:
                    counts[MutationType.ARITHMETIC] += len(
                        ArithmeticMutator.MUTATIONS[type(node.op)]
                    )

            if isinstance(node, ast.Compare):
                if type(node.ops[0]) in RelationalMutator.MUTATIONS:
                    counts[MutationType.RELATIONAL] += len(
                        RelationalMutator.MUTATIONS[type(node.ops[0])]
                    )

            if isinstance(node, ast.BoolOp):
                counts[MutationType.LOGICAL] += 1

            if isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.Not):
                counts[MutationType.LOGICAL] += 1

            if isinstance(node, ast.Constant) and isinstance(node.value, int):
                counts[MutationType.BOUNDARY] += 2

            if isinstance(node, ast.Return) and node.value is not None:
                counts[MutationType.RETURN] += 3

            if isinstance(node, (ast.If, ast.While)):
                counts[MutationType.NEGATE_CONDITIONAL] += 1

        return counts
