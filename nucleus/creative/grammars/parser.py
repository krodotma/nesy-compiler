"""
Grammar Parser
===============

This module provides parsing capabilities for grammar specifications,
including BNF, EBNF, and custom grammar formats. It also includes
AST (Abstract Syntax Tree) construction and manipulation.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Any, Callable, Iterator, Optional, Union


class TokenType(Enum):
    """Types of tokens in grammar specifications."""
    NONTERMINAL = auto()    # <name> or Name (capitalized)
    TERMINAL = auto()       # "literal" or 'literal'
    OR = auto()             # |
    ASSIGN = auto()         # ::= or := or ->
    LPAREN = auto()         # (
    RPAREN = auto()         # )
    LBRACKET = auto()       # [
    RBRACKET = auto()       # ]
    LBRACE = auto()         # {
    RBRACE = auto()         # }
    STAR = auto()           # *
    PLUS = auto()           # +
    QUESTION = auto()       # ?
    SEMICOLON = auto()      # ;
    NEWLINE = auto()        # End of rule
    COMMENT = auto()        # # or //
    EPSILON = auto()        # epsilon or empty
    IDENTIFIER = auto()     # Plain identifier
    EOF = auto()            # End of input


@dataclass
class Token:
    """A token from lexical analysis."""
    type: TokenType
    value: str
    line: int = 1
    column: int = 1


class GrammarLexer:
    """Lexer for grammar specifications."""

    # Token patterns
    PATTERNS = [
        (r'#[^\n]*', TokenType.COMMENT),
        (r'//[^\n]*', TokenType.COMMENT),
        (r'::=|:=|->|=', TokenType.ASSIGN),
        (r'\|', TokenType.OR),
        (r'\(', TokenType.LPAREN),
        (r'\)', TokenType.RPAREN),
        (r'\[', TokenType.LBRACKET),
        (r'\]', TokenType.RBRACKET),
        (r'\{', TokenType.LBRACE),
        (r'\}', TokenType.RBRACE),
        (r'\*', TokenType.STAR),
        (r'\+', TokenType.PLUS),
        (r'\?', TokenType.QUESTION),
        (r';', TokenType.SEMICOLON),
        (r'\n', TokenType.NEWLINE),
        (r'"[^"]*"', TokenType.TERMINAL),
        (r"'[^']*'", TokenType.TERMINAL),
        (r'<[a-zA-Z_][a-zA-Z0-9_-]*>', TokenType.NONTERMINAL),
        (r'epsilon|EPSILON|empty|EMPTY|\u03b5', TokenType.EPSILON),
        (r'[a-zA-Z_][a-zA-Z0-9_-]*', TokenType.IDENTIFIER),
        (r'[ \t]+', None),  # Skip whitespace
    ]

    def __init__(self, text: str):
        """Initialize lexer with input text."""
        self.text = text
        self.pos = 0
        self.line = 1
        self.column = 1
        self._compiled = [(re.compile(p), t) for p, t in self.PATTERNS]

    def tokenize(self) -> Iterator[Token]:
        """Generate tokens from input."""
        while self.pos < len(self.text):
            match = None
            token_type = None

            for pattern, ttype in self._compiled:
                match = pattern.match(self.text, self.pos)
                if match:
                    token_type = ttype
                    break

            if match:
                value = match.group(0)

                if token_type is not None:
                    if token_type == TokenType.TERMINAL:
                        # Strip quotes
                        value = value[1:-1]

                    yield Token(
                        type=token_type,
                        value=value,
                        line=self.line,
                        column=self.column
                    )

                # Update position
                self.pos = match.end()
                if '\n' in value:
                    self.line += value.count('\n')
                    self.column = len(value) - value.rfind('\n')
                else:
                    self.column += len(value)
            else:
                # Skip unknown character
                self.pos += 1
                self.column += 1

        yield Token(type=TokenType.EOF, value='', line=self.line, column=self.column)


class ASTNodeType(Enum):
    """Types of AST nodes."""
    GRAMMAR = auto()      # Root node containing rules
    RULE = auto()         # A production rule
    ALTERNATIVE = auto()  # Alternatives (|)
    SEQUENCE = auto()     # Sequence of elements
    NONTERMINAL = auto()  # Reference to a nonterminal
    TERMINAL = auto()     # A literal terminal
    OPTIONAL = auto()     # Optional element [X] or X?
    REPETITION = auto()   # Zero or more {X} or X*
    ONE_OR_MORE = auto()  # One or more X+
    GROUP = auto()        # Grouped elements (X Y Z)
    EPSILON = auto()      # Empty production


@dataclass
class ASTNode:
    """
    Abstract Syntax Tree node.

    Represents parsed grammar structures in a tree format suitable
    for transformation and analysis.

    Attributes:
        node_type: Type of this AST node.
        value: Associated value (e.g., terminal string, nonterminal name).
        children: Child nodes.
        attributes: Additional metadata.
        line: Source line number.
        column: Source column number.
    """
    node_type: ASTNodeType
    value: Optional[str] = None
    children: list["ASTNode"] = field(default_factory=list)
    attributes: dict[str, Any] = field(default_factory=dict)
    line: int = 0
    column: int = 0

    def add_child(self, child: "ASTNode") -> "ASTNode":
        """Add a child node and return self."""
        self.children.append(child)
        return self

    def find_all(self, node_type: ASTNodeType) -> list["ASTNode"]:
        """Find all descendant nodes of a given type."""
        results: list[ASTNode] = []
        if self.node_type == node_type:
            results.append(self)
        for child in self.children:
            results.extend(child.find_all(node_type))
        return results

    def transform(self, func: Callable[["ASTNode"], Optional["ASTNode"]]
                 ) -> Optional["ASTNode"]:
        """
        Transform this tree using a function.

        The function is applied bottom-up. If it returns None,
        the original node is kept.
        """
        # Transform children first
        new_children = []
        for child in self.children:
            transformed = child.transform(func)
            if transformed is not None:
                new_children.append(transformed)
            else:
                new_children.append(child)

        # Create new node with transformed children
        new_node = ASTNode(
            node_type=self.node_type,
            value=self.value,
            children=new_children,
            attributes=self.attributes.copy(),
            line=self.line,
            column=self.column
        )

        # Apply function
        result = func(new_node)
        return result if result is not None else new_node

    def copy(self) -> "ASTNode":
        """Create a deep copy of this node."""
        return ASTNode(
            node_type=self.node_type,
            value=self.value,
            children=[c.copy() for c in self.children],
            attributes=self.attributes.copy(),
            line=self.line,
            column=self.column
        )

    def to_string(self, indent: int = 0) -> str:
        """Convert to human-readable string representation."""
        prefix = "  " * indent
        result = f"{prefix}{self.node_type.name}"
        if self.value:
            result += f": {self.value!r}"
        result += "\n"
        for child in self.children:
            result += child.to_string(indent + 1)
        return result

    def to_grammar_string(self) -> str:
        """Convert back to grammar string format."""
        if self.node_type == ASTNodeType.GRAMMAR:
            return "\n".join(c.to_grammar_string() for c in self.children)

        elif self.node_type == ASTNodeType.RULE:
            lhs = self.children[0].to_grammar_string() if self.children else ""
            rhs = self.children[1].to_grammar_string() if len(self.children) > 1 else ""
            return f"{lhs} ::= {rhs}"

        elif self.node_type == ASTNodeType.ALTERNATIVE:
            return " | ".join(c.to_grammar_string() for c in self.children)

        elif self.node_type == ASTNodeType.SEQUENCE:
            return " ".join(c.to_grammar_string() for c in self.children)

        elif self.node_type == ASTNodeType.NONTERMINAL:
            return f"<{self.value}>"

        elif self.node_type == ASTNodeType.TERMINAL:
            return f'"{self.value}"'

        elif self.node_type == ASTNodeType.OPTIONAL:
            inner = self.children[0].to_grammar_string() if self.children else ""
            return f"[{inner}]"

        elif self.node_type == ASTNodeType.REPETITION:
            inner = self.children[0].to_grammar_string() if self.children else ""
            return f"{{{inner}}}"

        elif self.node_type == ASTNodeType.ONE_OR_MORE:
            inner = self.children[0].to_grammar_string() if self.children else ""
            return f"({inner})+"

        elif self.node_type == ASTNodeType.GROUP:
            inner = " ".join(c.to_grammar_string() for c in self.children)
            return f"({inner})"

        elif self.node_type == ASTNodeType.EPSILON:
            return "epsilon"

        return ""

    def __repr__(self) -> str:
        return f"ASTNode({self.node_type.name}, {self.value!r}, {len(self.children)} children)"


@dataclass
class GrammarRule:
    """
    A grammar production rule.

    Represents a single production rule in a grammar, consisting of
    a left-hand side (nonterminal) and right-hand side (alternatives).

    Attributes:
        name: Name of the nonterminal being defined.
        alternatives: List of alternative productions.
        is_start: Whether this is the start rule.
        attributes: Additional metadata.
    """
    name: str
    alternatives: list[list[str]] = field(default_factory=list)
    is_start: bool = False
    attributes: dict[str, Any] = field(default_factory=dict)

    def add_alternative(self, production: list[str]) -> "GrammarRule":
        """Add an alternative production."""
        self.alternatives.append(production)
        return self

    def to_string(self) -> str:
        """Convert to grammar string format."""
        alts = " | ".join(" ".join(alt) for alt in self.alternatives)
        return f"<{self.name}> ::= {alts}"

    def first_set(self, grammar: "Grammar") -> set[str]:
        """
        Compute FIRST set for this rule.

        The FIRST set contains all terminals that can appear first
        in any derivation from this rule.
        """
        first: set[str] = set()

        for alt in self.alternatives:
            if not alt or alt == ["epsilon"]:
                first.add("epsilon")
            else:
                for symbol in alt:
                    if not symbol.startswith("<"):
                        # Terminal
                        first.add(symbol)
                        break
                    else:
                        # Nonterminal - get its FIRST set
                        name = symbol[1:-1]
                        if name in grammar.rules:
                            sub_first = grammar.rules[name].first_set(grammar)
                            first.update(sub_first - {"epsilon"})
                            if "epsilon" not in sub_first:
                                break

        return first

    def __repr__(self) -> str:
        return f"GrammarRule({self.name!r}, {len(self.alternatives)} alternatives)"


@dataclass
class Grammar:
    """
    A complete grammar specification.

    Contains a collection of production rules and metadata.
    """
    rules: dict[str, GrammarRule] = field(default_factory=dict)
    start_symbol: Optional[str] = None
    terminals: set[str] = field(default_factory=set)
    attributes: dict[str, Any] = field(default_factory=dict)

    def add_rule(self, rule: GrammarRule) -> "Grammar":
        """Add a rule to the grammar."""
        self.rules[rule.name] = rule
        if rule.is_start:
            self.start_symbol = rule.name

        # Update terminals
        for alt in rule.alternatives:
            for symbol in alt:
                if not symbol.startswith("<") and symbol != "epsilon":
                    self.terminals.add(symbol)

        return self

    def get_rule(self, name: str) -> Optional[GrammarRule]:
        """Get a rule by name."""
        return self.rules.get(name)

    def nonterminals(self) -> set[str]:
        """Get set of all nonterminals."""
        return set(self.rules.keys())

    def is_left_recursive(self, name: str) -> bool:
        """Check if a rule is directly left-recursive."""
        if name not in self.rules:
            return False
        rule = self.rules[name]
        for alt in rule.alternatives:
            if alt and alt[0] == f"<{name}>":
                return True
        return False

    def find_left_recursive(self) -> list[str]:
        """Find all directly left-recursive rules."""
        return [name for name in self.rules if self.is_left_recursive(name)]

    def to_string(self) -> str:
        """Convert to grammar string format."""
        lines = []
        for rule in self.rules.values():
            lines.append(rule.to_string())
        return "\n".join(lines)


class GrammarParser:
    """
    Parser for grammar specifications.

    Parses BNF, EBNF, and similar grammar formats into
    structured representations.
    """

    def __init__(self):
        """Initialize the parser."""
        self._tokens: list[Token] = []
        self._pos = 0

    def parse(self, text: str) -> ASTNode:
        """
        Parse grammar text into an AST.

        Args:
            text: Grammar specification text.

        Returns:
            Root AST node representing the grammar.
        """
        lexer = GrammarLexer(text)
        self._tokens = [t for t in lexer.tokenize()
                       if t.type not in (TokenType.COMMENT, TokenType.NEWLINE)]
        self._pos = 0

        return self._parse_grammar()

    def parse_to_grammar(self, text: str) -> Grammar:
        """
        Parse grammar text into a Grammar object.

        Args:
            text: Grammar specification text.

        Returns:
            Grammar object with rules.
        """
        ast = self.parse(text)
        return self._ast_to_grammar(ast)

    def _current(self) -> Token:
        """Get current token."""
        if self._pos < len(self._tokens):
            return self._tokens[self._pos]
        return Token(type=TokenType.EOF, value='')

    def _peek(self, offset: int = 0) -> Token:
        """Peek at token with offset."""
        idx = self._pos + offset
        if idx < len(self._tokens):
            return self._tokens[idx]
        return Token(type=TokenType.EOF, value='')

    def _advance(self) -> Token:
        """Consume and return current token."""
        token = self._current()
        self._pos += 1
        return token

    def _expect(self, token_type: TokenType) -> Token:
        """Expect and consume a token of given type."""
        token = self._current()
        if token.type != token_type:
            raise SyntaxError(
                f"Expected {token_type.name}, got {token.type.name} "
                f"at line {token.line}, column {token.column}"
            )
        return self._advance()

    def _parse_grammar(self) -> ASTNode:
        """Parse a complete grammar."""
        root = ASTNode(node_type=ASTNodeType.GRAMMAR)

        while self._current().type != TokenType.EOF:
            rule = self._parse_rule()
            if rule:
                root.add_child(rule)

        return root

    def _parse_rule(self) -> Optional[ASTNode]:
        """Parse a single rule."""
        # Parse left-hand side
        lhs = self._parse_lhs()
        if not lhs:
            return None

        # Expect assignment
        if self._current().type != TokenType.ASSIGN:
            return None
        self._advance()

        # Parse right-hand side
        rhs = self._parse_alternatives()

        # Optional semicolon
        if self._current().type == TokenType.SEMICOLON:
            self._advance()

        rule = ASTNode(
            node_type=ASTNodeType.RULE,
            line=lhs.line,
            column=lhs.column
        )
        rule.add_child(lhs)
        rule.add_child(rhs)

        return rule

    def _parse_lhs(self) -> Optional[ASTNode]:
        """Parse left-hand side (nonterminal)."""
        token = self._current()

        if token.type == TokenType.NONTERMINAL:
            self._advance()
            name = token.value[1:-1]  # Strip < >
            return ASTNode(
                node_type=ASTNodeType.NONTERMINAL,
                value=name,
                line=token.line,
                column=token.column
            )

        elif token.type == TokenType.IDENTIFIER:
            self._advance()
            return ASTNode(
                node_type=ASTNodeType.NONTERMINAL,
                value=token.value,
                line=token.line,
                column=token.column
            )

        return None

    def _parse_alternatives(self) -> ASTNode:
        """Parse alternatives (separated by |)."""
        first = self._parse_sequence()

        if self._current().type != TokenType.OR:
            return first

        alt = ASTNode(node_type=ASTNodeType.ALTERNATIVE)
        alt.add_child(first)

        while self._current().type == TokenType.OR:
            self._advance()
            alt.add_child(self._parse_sequence())

        return alt

    def _parse_sequence(self) -> ASTNode:
        """Parse a sequence of elements."""
        elements: list[ASTNode] = []

        while True:
            elem = self._parse_element()
            if elem is None:
                break
            elements.append(elem)

        if not elements:
            return ASTNode(node_type=ASTNodeType.EPSILON)

        if len(elements) == 1:
            return elements[0]

        seq = ASTNode(node_type=ASTNodeType.SEQUENCE)
        for elem in elements:
            seq.add_child(elem)
        return seq

    def _parse_element(self) -> Optional[ASTNode]:
        """Parse a single element (possibly with suffix)."""
        token = self._current()
        node: Optional[ASTNode] = None

        # Check for various element types
        if token.type == TokenType.NONTERMINAL:
            self._advance()
            name = token.value[1:-1]
            node = ASTNode(
                node_type=ASTNodeType.NONTERMINAL,
                value=name,
                line=token.line,
                column=token.column
            )

        elif token.type == TokenType.TERMINAL:
            self._advance()
            node = ASTNode(
                node_type=ASTNodeType.TERMINAL,
                value=token.value,
                line=token.line,
                column=token.column
            )

        elif token.type == TokenType.IDENTIFIER:
            self._advance()
            # Treat as terminal or nonterminal based on case
            if token.value[0].isupper():
                node = ASTNode(
                    node_type=ASTNodeType.NONTERMINAL,
                    value=token.value,
                    line=token.line,
                    column=token.column
                )
            else:
                node = ASTNode(
                    node_type=ASTNodeType.TERMINAL,
                    value=token.value,
                    line=token.line,
                    column=token.column
                )

        elif token.type == TokenType.EPSILON:
            self._advance()
            node = ASTNode(
                node_type=ASTNodeType.EPSILON,
                line=token.line,
                column=token.column
            )

        elif token.type == TokenType.LPAREN:
            self._advance()
            inner = self._parse_alternatives()
            self._expect(TokenType.RPAREN)
            node = ASTNode(node_type=ASTNodeType.GROUP)
            node.add_child(inner)

        elif token.type == TokenType.LBRACKET:
            self._advance()
            inner = self._parse_alternatives()
            self._expect(TokenType.RBRACKET)
            node = ASTNode(node_type=ASTNodeType.OPTIONAL)
            node.add_child(inner)

        elif token.type == TokenType.LBRACE:
            self._advance()
            inner = self._parse_alternatives()
            self._expect(TokenType.RBRACE)
            node = ASTNode(node_type=ASTNodeType.REPETITION)
            node.add_child(inner)

        else:
            return None

        # Check for suffix operators
        if self._current().type == TokenType.STAR:
            self._advance()
            wrapper = ASTNode(node_type=ASTNodeType.REPETITION)
            wrapper.add_child(node)
            return wrapper

        elif self._current().type == TokenType.PLUS:
            self._advance()
            wrapper = ASTNode(node_type=ASTNodeType.ONE_OR_MORE)
            wrapper.add_child(node)
            return wrapper

        elif self._current().type == TokenType.QUESTION:
            self._advance()
            wrapper = ASTNode(node_type=ASTNodeType.OPTIONAL)
            wrapper.add_child(node)
            return wrapper

        return node

    def _ast_to_grammar(self, ast: ASTNode) -> Grammar:
        """Convert AST to Grammar object."""
        grammar = Grammar()

        for rule_node in ast.children:
            if rule_node.node_type != ASTNodeType.RULE:
                continue

            if len(rule_node.children) < 2:
                continue

            lhs = rule_node.children[0]
            rhs = rule_node.children[1]

            if lhs.node_type != ASTNodeType.NONTERMINAL:
                continue

            rule = GrammarRule(name=lhs.value or "")

            # Convert RHS to alternatives
            if rhs.node_type == ASTNodeType.ALTERNATIVE:
                for alt_node in rhs.children:
                    rule.alternatives.append(self._node_to_symbols(alt_node))
            else:
                rule.alternatives.append(self._node_to_symbols(rhs))

            grammar.add_rule(rule)

            # First rule is start symbol
            if grammar.start_symbol is None:
                grammar.start_symbol = rule.name
                rule.is_start = True

        return grammar

    def _node_to_symbols(self, node: ASTNode) -> list[str]:
        """Convert AST node to list of symbols."""
        if node.node_type == ASTNodeType.SEQUENCE:
            symbols = []
            for child in node.children:
                symbols.extend(self._node_to_symbols(child))
            return symbols

        elif node.node_type == ASTNodeType.NONTERMINAL:
            return [f"<{node.value}>"]

        elif node.node_type == ASTNodeType.TERMINAL:
            return [f'"{node.value}"']

        elif node.node_type == ASTNodeType.EPSILON:
            return ["epsilon"]

        elif node.node_type in (ASTNodeType.OPTIONAL, ASTNodeType.REPETITION,
                                ASTNodeType.ONE_OR_MORE, ASTNodeType.GROUP):
            # For simplicity, just unwrap
            if node.children:
                return self._node_to_symbols(node.children[0])
            return []

        elif node.node_type == ASTNodeType.ALTERNATIVE:
            # This shouldn't happen at top level, but handle it
            if node.children:
                return self._node_to_symbols(node.children[0])
            return []

        return []
