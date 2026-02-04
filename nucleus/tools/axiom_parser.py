#!/usr/bin/env python3
"""
Axiom DSL Parser - Skeleton Implementation

Parses Axiom DSL into AST nodes per nucleus/specs/schema/axioms.schema.json

Authors: codex, claude
Protocol: DKIN v28
Status: Skeleton (Iteration 3)

Usage:
    from nucleus.tools.axiom_parser import parse_axiom, parse_program
    ast = parse_axiom('AXIOM test: forall x. P(x);')
"""

import re
import json
from dataclasses import dataclass, field, asdict
from typing import List, Optional, Union, Any
from enum import Enum, auto

# =============================================================================
# Token Types
# =============================================================================

class TokenType(Enum):
    # Keywords
    AXIOM = auto()
    DEF = auto()
    RULE = auto()
    DITS = auto()
    BIND = auto()
    FORALL = auto()
    EXISTS = auto()
    AND = auto()
    OR = auto()
    NOT = auto()
    IMPLIES = auto()
    TRUE = auto()
    FALSE = auto()
    # Temporal
    G = auto()  # Globally
    F = auto()  # Finally/Eventually
    X = auto()  # Next
    U = auto()  # Until
    R = auto()  # Release
    W = auto()  # Weak-until
    # Fixpoint
    MU = auto()
    NU = auto()
    # Punctuation
    LPAREN = auto()
    RPAREN = auto()
    LBRACE = auto()
    RBRACE = auto()
    COLON = auto()
    SEMICOLON = auto()
    COMMA = auto()
    DOT = auto()
    EQ = auto()
    NEQ = auto()
    LT = auto()
    LE = auto()
    GT = auto()
    GE = auto()
    # Literals
    IDENT = auto()
    NUMBER = auto()
    STRING = auto()
    # Special
    EOF = auto()

@dataclass
class Token:
    type: TokenType
    value: Any
    line: int
    col: int

# =============================================================================
# Lexer
# =============================================================================

KEYWORDS = {
    'AXIOM': TokenType.AXIOM,
    'DEF': TokenType.DEF,
    'RULE': TokenType.RULE,
    'DITS': TokenType.DITS,
    'BIND': TokenType.BIND,
    'forall': TokenType.FORALL,
    'exists': TokenType.EXISTS,
    'and': TokenType.AND,
    'or': TokenType.OR,
    'not': TokenType.NOT,
    'true': TokenType.TRUE,
    'false': TokenType.FALSE,
    'G': TokenType.G,
    'F': TokenType.F,
    'X': TokenType.X,
    'U': TokenType.U,
    'R': TokenType.R,
    'W': TokenType.W,
    'mu': TokenType.MU,
    'nu': TokenType.NU,
}

class Lexer:
    def __init__(self, source: str):
        self.source = source
        self.pos = 0
        self.line = 1
        self.col = 1
        self.tokens: List[Token] = []

    def peek(self, offset: int = 0) -> str:
        pos = self.pos + offset
        return self.source[pos] if pos < len(self.source) else ''

    def advance(self) -> str:
        ch = self.peek()
        self.pos += 1
        if ch == '\n':
            self.line += 1
            self.col = 1
        else:
            self.col += 1
        return ch

    def skip_whitespace(self):
        while self.peek() and self.peek() in ' \t\n\r':
            self.advance()

    def skip_comment(self):
        if self.peek() == '/' and self.peek(1) == '/':
            while self.peek() and self.peek() != '\n':
                self.advance()
        elif self.peek() == '/' and self.peek(1) == '*':
            self.advance()
            self.advance()
            while self.peek():
                if self.peek() == '*' and self.peek(1) == '/':
                    self.advance()
                    self.advance()
                    break
                self.advance()

    def read_string(self) -> str:
        quote = self.advance()  # consume opening quote
        value = ''
        while self.peek() and self.peek() != quote:
            if self.peek() == '\\':
                self.advance()
                ch = self.advance()
                escapes = {'n': '\n', 't': '\t', 'r': '\r', '\\': '\\', '"': '"', "'": "'"}
                value += escapes.get(ch, ch)
            else:
                value += self.advance()
        self.advance()  # consume closing quote
        return value

    def read_number(self) -> Union[int, float]:
        start = self.pos
        while self.peek() and (self.peek().isdigit() or self.peek() == '.'):
            self.advance()
        text = self.source[start:self.pos]
        return float(text) if '.' in text else int(text)

    def read_ident(self) -> str:
        start = self.pos
        while self.peek() and (self.peek().isalnum() or self.peek() == '_'):
            self.advance()
        return self.source[start:self.pos]

    def tokenize(self) -> List[Token]:
        while self.pos < len(self.source):
            # Skip whitespace and comments in a loop
            while True:
                self.skip_whitespace()
                if self.pos >= len(self.source):
                    break
                # Check for comment
                if self.peek() == '/' and self.peek(1) in '/*':
                    self.skip_comment()
                else:
                    break

            if self.pos >= len(self.source):
                break

            line, col = self.line, self.col
            ch = self.peek()

            # Single/double char tokens
            if ch == '(':
                self.tokens.append(Token(TokenType.LPAREN, '(', line, col))
                self.advance()
            elif ch == ')':
                self.tokens.append(Token(TokenType.RPAREN, ')', line, col))
                self.advance()
            elif ch == '{':
                self.tokens.append(Token(TokenType.LBRACE, '{', line, col))
                self.advance()
            elif ch == '}':
                self.tokens.append(Token(TokenType.RBRACE, '}', line, col))
                self.advance()
            elif ch == ':':
                self.tokens.append(Token(TokenType.COLON, ':', line, col))
                self.advance()
            elif ch == ';':
                self.tokens.append(Token(TokenType.SEMICOLON, ';', line, col))
                self.advance()
            elif ch == ',':
                self.tokens.append(Token(TokenType.COMMA, ',', line, col))
                self.advance()
            elif ch == '.':
                self.tokens.append(Token(TokenType.DOT, '.', line, col))
                self.advance()
            elif ch == '=' and self.peek(1) == '>':
                self.tokens.append(Token(TokenType.IMPLIES, '=>', line, col))
                self.advance()
                self.advance()
            elif ch == '=' and self.peek(1) != '>':
                self.tokens.append(Token(TokenType.EQ, '=', line, col))
                self.advance()
            elif ch == '!' and self.peek(1) == '=':
                self.tokens.append(Token(TokenType.NEQ, '!=', line, col))
                self.advance()
                self.advance()
            elif ch == '<' and self.peek(1) == '=':
                self.tokens.append(Token(TokenType.LE, '<=', line, col))
                self.advance()
                self.advance()
            elif ch == '<':
                self.tokens.append(Token(TokenType.LT, '<', line, col))
                self.advance()
            elif ch == '>' and self.peek(1) == '=':
                self.tokens.append(Token(TokenType.GE, '>=', line, col))
                self.advance()
                self.advance()
            elif ch == '>':
                self.tokens.append(Token(TokenType.GT, '>', line, col))
                self.advance()
            elif ch in '"\'':
                value = self.read_string()
                self.tokens.append(Token(TokenType.STRING, value, line, col))
            elif ch.isdigit():
                value = self.read_number()
                self.tokens.append(Token(TokenType.NUMBER, value, line, col))
            elif ch.isalpha() or ch == '_':
                ident = self.read_ident()
                tt = KEYWORDS.get(ident, TokenType.IDENT)
                self.tokens.append(Token(tt, ident, line, col))
            else:
                # Skip unknown
                self.advance()

        self.tokens.append(Token(TokenType.EOF, None, self.line, self.col))
        return self.tokens

# =============================================================================
# AST Nodes (matching axioms.schema.json)
# =============================================================================

@dataclass
class Variable:
    type: str = "Variable"
    name: str = ""

@dataclass
class Literal:
    type: str = "Literal"
    value: Union[int, float, str] = 0

@dataclass
class FunctionApp:
    type: str = "FunctionApp"
    name: str = ""
    arguments: List[Any] = field(default_factory=list)

@dataclass
class Predicate:
    type: str = "Predicate"
    name: str = ""
    arguments: List[Any] = field(default_factory=list)

@dataclass
class Comparison:
    type: str = "Comparison"
    operator: str = "="
    left: Any = None
    right: Any = None

@dataclass
class Constant:
    type: str = "Constant"
    value: bool = True

@dataclass
class Negation:
    type: str = "Negation"
    body: Any = None

@dataclass
class Conjunction:
    type: str = "Conjunction"
    operands: List[Any] = field(default_factory=list)

@dataclass
class Disjunction:
    type: str = "Disjunction"
    operands: List[Any] = field(default_factory=list)

@dataclass
class Implication:
    type: str = "Implication"
    antecedent: Any = None
    consequent: Any = None

@dataclass
class Quantifier:
    type: str = "Quantifier"
    quantifier: str = "forall"  # forall | exists
    variables: List[str] = field(default_factory=list)
    body: Any = None

@dataclass
class Modal:
    type: str = "Modal"
    operator: str = "G"  # G | F | X
    body: Any = None

@dataclass
class TemporalBinary:
    type: str = "TemporalBinary"
    operator: str = "U"  # U | R | W
    left: Any = None
    right: Any = None

@dataclass
class Fixpoint:
    type: str = "Fixpoint"
    operator: str = "mu"  # mu | nu
    variable: str = ""
    body: Any = None

@dataclass
class Binding:
    enforce: Optional[str] = None
    topic: Optional[str] = None
    tool: Optional[str] = None
    motif: Optional[str] = None

@dataclass
class Axiom:
    type: str = "Axiom"
    name: str = ""
    formula: Any = None
    binding: Optional[Binding] = None
    sources: List[str] = field(default_factory=list)

@dataclass
class Definition:
    type: str = "Definition"
    name: str = ""
    formula: Any = None

@dataclass
class Rule:
    type: str = "Rule"
    name: str = ""
    antecedent: Any = None
    consequent: Any = None

@dataclass
class DiTS:
    type: str = "DiTS"
    name: str = ""
    mu_spec: Any = None
    nu_spec: Any = None
    omega_spec: Any = None
    rank: Any = None
    binding: Optional[Binding] = None

@dataclass
class Program:
    type: str = "Program"
    declarations: List[Any] = field(default_factory=list)
    source: str = ""

# =============================================================================
# Parser
# =============================================================================

class ParseError(Exception):
    def __init__(self, message: str, token: Token):
        self.message = message
        self.token = token
        super().__init__(f"{message} at line {token.line}, col {token.col}")

class Parser:
    def __init__(self, tokens: List[Token]):
        self.tokens = tokens
        self.pos = 0

    def peek(self, offset: int = 0) -> Token:
        pos = self.pos + offset
        return self.tokens[pos] if pos < len(self.tokens) else self.tokens[-1]

    def check(self, *types: TokenType) -> bool:
        return self.peek().type in types

    def advance(self) -> Token:
        tok = self.peek()
        if tok.type != TokenType.EOF:
            self.pos += 1
        return tok

    def expect(self, tt: TokenType, msg: str = "") -> Token:
        if not self.check(tt):
            raise ParseError(msg or f"Expected {tt.name}", self.peek())
        return self.advance()

    def parse_program(self) -> Program:
        decls = []
        while not self.check(TokenType.EOF):
            decls.append(self.parse_declaration())
        return Program(declarations=decls)

    def parse_declaration(self) -> Any:
        if self.check(TokenType.AXIOM):
            return self.parse_axiom()
        elif self.check(TokenType.DEF):
            return self.parse_definition()
        elif self.check(TokenType.RULE):
            return self.parse_rule()
        elif self.check(TokenType.DITS):
            return self.parse_dits()
        else:
            raise ParseError(f"Expected declaration keyword", self.peek())

    def parse_axiom(self) -> Axiom:
        self.expect(TokenType.AXIOM)
        name = self.expect(TokenType.IDENT).value
        self.expect(TokenType.COLON)
        formula = self.parse_formula()
        binding = self.parse_binding_opt()
        self.expect(TokenType.SEMICOLON)
        return Axiom(name=name, formula=formula, binding=binding)

    def parse_definition(self) -> Definition:
        self.expect(TokenType.DEF)
        name = self.expect(TokenType.IDENT).value
        self.expect(TokenType.COLON)
        formula = self.parse_formula()
        self.expect(TokenType.SEMICOLON)
        return Definition(name=name, formula=formula)

    def parse_rule(self) -> Rule:
        self.expect(TokenType.RULE)
        name = self.expect(TokenType.IDENT).value
        self.expect(TokenType.COLON)
        # Parse full formula - if it's an implication, extract antecedent/consequent
        formula = self.parse_formula()
        self.expect(TokenType.SEMICOLON)

        # If formula is an implication, use its parts
        if isinstance(formula, dict) and formula.get('type') == 'Implication':
            return Rule(name=name, antecedent=formula['antecedent'], consequent=formula['consequent'])
        elif hasattr(formula, 'type') and formula.type == 'Implication':
            return Rule(name=name, antecedent=formula.antecedent, consequent=formula.consequent)
        else:
            # Non-implication rule - wrap as self-implication (tautology)
            return Rule(name=name, antecedent=formula, consequent=formula)

    def parse_dits(self) -> DiTS:
        self.expect(TokenType.DITS)
        name = self.expect(TokenType.IDENT).value
        self.expect(TokenType.COLON)
        self.expect(TokenType.LBRACE)

        mu_spec = None
        nu_spec = None
        omega_spec = None
        rank = None

        while not self.check(TokenType.RBRACE):
            # Handle mu/nu/omega as keywords or idents
            if self.check(TokenType.MU):
                self.advance()
                self.expect(TokenType.COLON)
                mu_spec = self.parse_formula()
            elif self.check(TokenType.NU):
                self.advance()
                self.expect(TokenType.COLON)
                nu_spec = self.parse_formula()
            elif self.check(TokenType.IDENT):
                key = self.advance().value
                self.expect(TokenType.COLON)
                if key == 'omega':
                    omega_spec = self.parse_formula()
                elif key == 'rank':
                    rank = self.parse_term()
            else:
                raise ParseError("Expected mu, nu, omega, or rank", self.peek())

            if self.check(TokenType.COMMA):
                self.advance()

        self.expect(TokenType.RBRACE)
        binding = self.parse_binding_opt()
        self.expect(TokenType.SEMICOLON)

        return DiTS(name=name, mu_spec=mu_spec, nu_spec=nu_spec,
                   omega_spec=omega_spec, rank=rank, binding=binding)

    def parse_binding_opt(self) -> Optional[Binding]:
        if not self.check(TokenType.BIND):
            return None
        self.advance()
        self.expect(TokenType.LBRACE)

        binding = Binding()
        while not self.check(TokenType.RBRACE):
            key = self.expect(TokenType.IDENT).value
            self.expect(TokenType.EQ)
            value = self.expect(TokenType.STRING).value

            if key == 'enforce':
                binding.enforce = value
            elif key == 'topic':
                binding.topic = value
            elif key == 'tool':
                binding.tool = value
            elif key == 'motif':
                binding.motif = value

            if self.check(TokenType.COMMA):
                self.advance()

        self.expect(TokenType.RBRACE)
        return binding

    def parse_formula(self) -> Any:
        return self.parse_implication()

    def parse_implication(self) -> Any:
        left = self.parse_disjunction()
        if self.check(TokenType.IMPLIES):
            self.advance()
            right = self.parse_formula()
            return Implication(antecedent=left, consequent=right)
        return left

    def parse_disjunction(self) -> Any:
        left = self.parse_conjunction()
        operands = [left]
        while self.check(TokenType.OR):
            self.advance()
            operands.append(self.parse_conjunction())
        return operands[0] if len(operands) == 1 else Disjunction(operands=operands)

    def parse_conjunction(self) -> Any:
        left = self.parse_unary()
        operands = [left]
        while self.check(TokenType.AND):
            self.advance()
            operands.append(self.parse_unary())
        return operands[0] if len(operands) == 1 else Conjunction(operands=operands)

    def parse_unary(self) -> Any:
        # Quantifiers
        if self.check(TokenType.FORALL, TokenType.EXISTS):
            quant = "forall" if self.peek().type == TokenType.FORALL else "exists"
            self.advance()
            variables = [self.expect(TokenType.IDENT).value]
            while self.check(TokenType.COMMA):
                self.advance()
                variables.append(self.expect(TokenType.IDENT).value)
            self.expect(TokenType.DOT)
            body = self.parse_formula()
            return Quantifier(quantifier=quant, variables=variables, body=body)

        # Fixpoints
        if self.check(TokenType.MU, TokenType.NU):
            op = "mu" if self.peek().type == TokenType.MU else "nu"
            self.advance()
            var = self.expect(TokenType.IDENT).value
            self.expect(TokenType.DOT)
            body = self.parse_formula()
            return Fixpoint(operator=op, variable=var, body=body)

        # Unary temporal
        if self.check(TokenType.G, TokenType.F, TokenType.X):
            op = self.peek().value
            self.advance()
            self.expect(TokenType.LPAREN)
            body = self.parse_formula()
            self.expect(TokenType.RPAREN)
            return Modal(operator=op, body=body)

        # Negation
        if self.check(TokenType.NOT):
            self.advance()
            body = self.parse_unary()
            return Negation(body=body)

        return self.parse_temporal_binary()

    def parse_temporal_binary(self) -> Any:
        left = self.parse_atom()
        if self.check(TokenType.U, TokenType.R, TokenType.W):
            op = self.peek().value
            self.advance()
            right = self.parse_atom()
            return TemporalBinary(operator=op, left=left, right=right)
        return left

    def parse_atom(self) -> Any:
        if self.check(TokenType.TRUE):
            self.advance()
            return Constant(value=True)
        if self.check(TokenType.FALSE):
            self.advance()
            return Constant(value=False)
        if self.check(TokenType.LPAREN):
            self.advance()
            expr = self.parse_formula()
            self.expect(TokenType.RPAREN)
            return expr
        if self.check(TokenType.IDENT):
            name = self.advance().value
            if self.check(TokenType.LPAREN):
                # Predicate or function
                self.advance()
                args = []
                if not self.check(TokenType.RPAREN):
                    args.append(self.parse_term())
                    while self.check(TokenType.COMMA):
                        self.advance()
                        args.append(self.parse_term())
                self.expect(TokenType.RPAREN)
                return Predicate(name=name, arguments=args)
            else:
                # Check for comparison
                if self.check(TokenType.EQ, TokenType.NEQ, TokenType.LT,
                             TokenType.LE, TokenType.GT, TokenType.GE):
                    op = self.advance().value
                    right = self.parse_term()
                    return Comparison(operator=op, left=Variable(name=name), right=right)
                return Variable(name=name)

        raise ParseError("Expected atom", self.peek())

    def parse_term(self) -> Any:
        if self.check(TokenType.NUMBER):
            return Literal(value=self.advance().value)
        if self.check(TokenType.STRING):
            return Literal(value=self.advance().value)
        if self.check(TokenType.IDENT):
            name = self.advance().value
            if self.check(TokenType.LPAREN):
                self.advance()
                args = []
                if not self.check(TokenType.RPAREN):
                    args.append(self.parse_term())
                    while self.check(TokenType.COMMA):
                        self.advance()
                        args.append(self.parse_term())
                self.expect(TokenType.RPAREN)
                return FunctionApp(name=name, arguments=args)
            return Variable(name=name)
        raise ParseError("Expected term", self.peek())

# =============================================================================
# Public API
# =============================================================================

def parse_axiom(source: str) -> dict:
    """Parse a single axiom/definition/rule/dits declaration."""
    lexer = Lexer(source)
    tokens = lexer.tokenize()
    parser = Parser(tokens)
    decl = parser.parse_declaration()
    return asdict(decl)

def parse_program(source: str) -> dict:
    """Parse a full program with multiple declarations."""
    lexer = Lexer(source)
    tokens = lexer.tokenize()
    parser = Parser(tokens)
    program = parser.parse_program()
    program.source = source
    return asdict(program)

def serialize_ast(ast: dict) -> str:
    """Serialize AST to JSON."""
    return json.dumps(ast, indent=2)

# =============================================================================
# CLI
# =============================================================================

if __name__ == '__main__':
    import sys

    if len(sys.argv) > 1:
        with open(sys.argv[1]) as f:
            source = f.read()
    else:
        source = '''
        AXIOM test: forall x. P(x) => Q(x);
        AXIOM liveness: G(F(done(s)));
        DITS cognition: {
            mu: mu x. (explore(x) and X(x)),
            nu: nu y. (safe(y) => X(y)),
            omega: G(consistent(p, c))
        } BIND { topic = "dits.check" };
        '''

    try:
        ast = parse_program(source)
        print(serialize_ast(ast))
    except ParseError as e:
        print(f"Parse error: {e}", file=sys.stderr)
        sys.exit(1)
