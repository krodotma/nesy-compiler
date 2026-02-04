#!/usr/bin/env python3
"""
Semantic Operators Lexer for Pluribus TUI/WebUI
================================================
Provides grammatical token recognition for semantic operators, slash commands,
and user-definable vocabulary with predictive autocomplete.

Features:
- Token recognition: OPERATOR, COMMAND, ARGUMENT, PROMPT
- Predictive autocomplete for operators and commands
- User-definable runtime operators
- Context-aware token coloring for TUI/WebUI
- Integrates with semops.json canonical registry

Usage:
    from semops_lexer import SemopsLexer

    lexer = SemopsLexer()
    tokens = lexer.tokenize("iterate --agent claude")
    completions = lexer.complete("ite")  # -> ["iterate"]
    lexer.define_operator("myop", {...})
"""
from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass, field
from enum import Enum, auto
from pathlib import Path
from time import gmtime, strftime
from typing import Any, Callable


class TokenType(Enum):
    """Token types for semantic operator grammar."""
    OPERATOR = auto()      # Semantic operator (CKIN, ITERATE, etc.)
    COMMAND = auto()       # Slash command (/help, /status)
    ARGUMENT = auto()      # Command argument (--flag, value)
    FLAG = auto()          # Flag (--name, -n)
    VALUE = auto()         # Argument value
    PROMPT = auto()        # Free-form prompt text
    WHITESPACE = auto()    # Whitespace
    USER_OP = auto()       # User-defined operator
    UNKNOWN = auto()       # Unknown token


@dataclass
class Token:
    """Lexical token with metadata."""
    type: TokenType
    value: str
    start: int
    end: int
    metadata: dict = field(default_factory=dict)

    def __repr__(self) -> str:
        return f"Token({self.type.name}, {self.value!r}, {self.start}:{self.end})"


@dataclass
class OperatorDef:
    """Definition of a semantic operator."""
    id: str
    name: str
    domain: str
    category: str
    description: str
    aliases: list[str] = field(default_factory=list)
    tool: str | None = None
    bus_topic: str | None = None
    bus_kind: str | None = None
    secondary_topic: str | None = None
    options: dict[str, str] = field(default_factory=dict)
    invocation: dict[str, Any] = field(default_factory=dict)
    targets: list[dict[str, Any]] = field(default_factory=list)
    ui: dict[str, Any] = field(default_factory=dict)
    agents: list[str] = field(default_factory=list)
    apps: list[str] = field(default_factory=list)
    extra: dict[str, Any] = field(default_factory=dict)
    user_defined: bool = False

    @classmethod
    def from_dict(cls, data: dict, op_id: str) -> "OperatorDef":
        known = {
            "id",
            "name",
            "domain",
            "category",
            "description",
            "aliases",
            "tool",
            "bus_topic",
            "bus_kind",
            "secondary_topic",
            "options",
            "invocation",
            "targets",
            "ui",
            "agents",
            "apps",
            "user_defined",
        }
        extra = {k: v for k, v in data.items() if k not in known}
        return cls(
            id=data.get("id", op_id),
            name=data.get("name", op_id.upper()),
            domain=data.get("domain", "user"),
            category=data.get("category", "custom"),
            description=data.get("description", ""),
            aliases=data.get("aliases", []),
            tool=data.get("tool"),
            bus_topic=data.get("bus_topic"),
            bus_kind=data.get("bus_kind"),
            secondary_topic=data.get("secondary_topic"),
            options=data.get("options", {}),
            invocation=data.get("invocation", {}),
            targets=data.get("targets", []) or [],
            ui=data.get("ui", {}) or {},
            agents=data.get("agents", []) or [],
            apps=data.get("apps", []) or [],
            extra=extra,
            user_defined=data.get("user_defined", False),
        )


class SemopsLexer:
    """
    Grammatical token lexer for Pluribus semantic operators.

    Recognizes:
    - Built-in operators from semops.json
    - User-defined runtime operators
    - Slash commands
    - Arguments and flags
    - Free-form prompts
    """

    REPO_ROOT = Path(__file__).resolve().parents[2]
    SEMOPS_PATH = REPO_ROOT / "nucleus" / "specs" / "semops.json"

    @staticmethod
    def resolve_user_ops_path(user_ops_path: Path | None = None) -> Path:
        """Resolve a writable user_operators.json path (primary → fallback)."""
        if user_ops_path is not None:
            return user_ops_path

        primary_root = Path(os.environ.get("PLURIBUS_STATE_DIR") or "/pluribus/.pluribus")
        fallback_root = Path(os.environ.get("PLURIBUS_FALLBACK_STATE_DIR") or str(SemopsLexer.REPO_ROOT / ".pluribus_local"))

        primary = (primary_root / "user_operators.json").resolve()
        fallback = (fallback_root / "user_operators.json").resolve()

        active = primary
        try:
            active.parent.mkdir(parents=True, exist_ok=True)
            with open(active, "a", encoding="utf-8"):
                pass
        except (PermissionError, OSError):
            active = fallback
            active.parent.mkdir(parents=True, exist_ok=True)
        return active

    # ANSI color codes for TUI highlighting
    COLORS = {
        TokenType.OPERATOR: "\033[1;32m",     # Bold green
        TokenType.COMMAND: "\033[1;36m",      # Bold cyan
        TokenType.FLAG: "\033[0;33m",         # Yellow
        TokenType.ARGUMENT: "\033[0;34m",     # Blue
        TokenType.VALUE: "\033[0;37m",        # White
        TokenType.USER_OP: "\033[1;35m",      # Bold magenta
        TokenType.PROMPT: "\033[0m",          # Default
        TokenType.WHITESPACE: "",
        TokenType.UNKNOWN: "\033[0;31m",      # Red
    }
    RESET = "\033[0m"

    def __init__(self, semops_path: Path | None = None, user_ops_path: Path | None = None):
        self.semops_path = semops_path or self.SEMOPS_PATH
        self.user_ops_path = self.resolve_user_ops_path(user_ops_path)

        # Operator registry: id -> OperatorDef
        self.operators: dict[str, OperatorDef] = {}
        # Alias map: alias -> operator_id
        self.alias_map: dict[str, str] = {}
        # Slash commands
        self.commands: set[str] = set()
        # Custom token handlers
        self.token_handlers: list[Callable[[str], Token | None]] = []

        # Load built-in operators
        self._load_semops()
        # Load user-defined operators
        self._load_user_ops()

    def _load_semops(self) -> None:
        """Load semantic operators from semops.json."""
        if not self.semops_path.exists():
            return
        try:
            with open(self.semops_path) as f:
                data = json.load(f)

            # Load operators
            for op_id, op_data in data.get("operators", {}).items():
                op_def = OperatorDef.from_dict(op_data, op_id)
                self.operators[op_id] = op_def

                # Register aliases
                for alias in op_def.aliases:
                    self.alias_map[alias.lower()] = op_id

            # Extract slash commands from grammar
            grammar = data.get("grammar", {})
            slash_pattern = grammar.get("slash_command_pattern", "")
            if slash_pattern:
                # Extract command names from pattern
                match = re.search(r'\/([\w|]+)', slash_pattern)
                if match:
                    self.commands = set(match.group(1).split('|'))

        except Exception as e:
            print(f"Warning: Failed to load semops.json: {e}")

    def _load_user_ops(self) -> None:
        """Load user-defined operators from user_operators.json."""
        if not self.user_ops_path.exists():
            return
        try:
            with open(self.user_ops_path) as f:
                data = json.load(f)

            for op_id, op_data in data.get("operators", {}).items():
                op_data["user_defined"] = True
                op_def = OperatorDef.from_dict(op_data, op_id)
                self.operators[op_id] = op_def

                for alias in op_def.aliases:
                    self.alias_map[alias.lower()] = op_id

        except Exception:
            pass  # User ops file may not exist

    def define_operator(
        self,
        op_key: str,
        *,
        op_id: str | None = None,
        name: str | None = None,
        description: str = "",
        aliases: list[str] | None = None,
        domain: str = "user",
        category: str = "custom",
        tool: str | None = None,
        bus_topic: str | None = None,
        bus_kind: str | None = None,
        secondary_topic: str | None = None,
        options: dict[str, str] | None = None,
        invocation: dict[str, Any] | None = None,
        targets: list[dict[str, Any]] | None = None,
        ui: dict[str, Any] | None = None,
        agents: list[str] | None = None,
        apps: list[str] | None = None,
        extra: dict[str, Any] | None = None,
        persist: bool = True,
    ) -> OperatorDef:
        """
        Define a new semantic operator at runtime.

        Args:
            op_key: Unique registry key (e.g., "CKIN", "PBFLUSH", "MYOP")
            op_id: Human-friendly id (defaults to op_key.lower())
            name: Display name (defaults to op_id.upper())
            description: Human-readable description
            aliases: List of alternative invocations
            domain: Operator domain (e.g., "user", "coordination")
            category: Operator category
            tool: Path to tool implementation
            bus_topic: Bus topic for events
            options: Command-line options
            persist: Whether to save to user_operators.json

        Returns:
            The newly defined OperatorDef
        """
        op_key_norm = op_key.strip().upper()
        if not op_key_norm:
            raise ValueError("op_key must be non-empty")
        op_def = OperatorDef(
            id=(op_id or op_key_norm.lower()).strip(),
            name=(name or op_key_norm).strip(),
            domain=domain,
            category=category,
            description=description,
            aliases=aliases or [(op_id or op_key_norm.lower()).strip(), op_key_norm],
            tool=tool,
            bus_topic=bus_topic,
            bus_kind=bus_kind,
            secondary_topic=secondary_topic,
            options=options or {},
            invocation=invocation or {},
            targets=targets or [],
            ui=ui or {},
            agents=agents or [],
            apps=apps or [],
            extra=extra or {},
            user_defined=True,
        )

        self.operators[op_key_norm] = op_def
        for alias in op_def.aliases:
            self.alias_map[alias.lower()] = op_key_norm

        if persist:
            self._save_user_ops()

        return op_def

    def undefine_operator(self, op_id: str) -> bool:
        """Remove a user-defined operator."""
        if op_id not in self.operators:
            return False
        op_def = self.operators[op_id]
        if not op_def.user_defined:
            return False  # Cannot remove built-in operators

        del self.operators[op_id]
        for alias in op_def.aliases:
            if self.alias_map.get(alias.lower()) == op_id:
                del self.alias_map[alias.lower()]

        self._save_user_ops()
        return True

    def _save_user_ops(self) -> None:
        """Save user-defined operators to file."""
        user_ops: dict[str, Any] = {}
        for op_key, op in self.operators.items():
            if not op.user_defined:
                continue
            payload = {
                "key": op_key,
                "id": op.id,
                "name": op.name,
                "domain": op.domain,
                "category": op.category,
                "description": op.description,
                "aliases": op.aliases,
                "tool": op.tool,
                "bus_topic": op.bus_topic,
                "bus_kind": op.bus_kind,
                "secondary_topic": op.secondary_topic,
                "options": op.options,
                "invocation": op.invocation,
                "targets": op.targets,
                "ui": op.ui,
                "agents": op.agents,
                "apps": op.apps,
                **(op.extra or {}),
            }
            user_ops[op_key] = payload

        self.user_ops_path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.user_ops_path, "w") as f:
            json.dump(
                {
                    "schema_version": 1,
                    "updated_iso": strftime("%Y-%m-%dT%H:%M:%SZ", gmtime()),
                    "operators": user_ops,
                },
                f,
                indent=2,
                ensure_ascii=False,
            )

    def tokenize(self, text: str) -> list[Token]:
        """
        Tokenize input text into semantic tokens.

        Args:
            text: Input text to tokenize

        Returns:
            List of Token objects
        """
        tokens: list[Token] = []
        pos = 0
        text_len = len(text)

        while pos < text_len:
            # Skip whitespace
            if text[pos].isspace():
                start = pos
                while pos < text_len and text[pos].isspace():
                    pos += 1
                tokens.append(Token(TokenType.WHITESPACE, text[start:pos], start, pos))
                continue

            # Check for slash command
            if text[pos] == '/':
                token = self._match_command(text, pos)
                if token:
                    tokens.append(token)
                    pos = token.end
                    continue

            # Check for flag
            if text[pos] == '-':
                token = self._match_flag(text, pos)
                if token:
                    tokens.append(token)
                    pos = token.end
                    continue

            # Check for operator (at start or after whitespace)
            if pos == 0 or (tokens and tokens[-1].type == TokenType.WHITESPACE):
                token = self._match_operator(text, pos)
                if token:
                    tokens.append(token)
                    pos = token.end
                    continue

            # Try custom handlers
            handled = False
            for handler in self.token_handlers:
                token = handler(text[pos:])
                if token:
                    token.start += pos
                    token.end += pos
                    tokens.append(token)
                    pos = token.end
                    handled = True
                    break
            if handled:
                continue

            # Default: consume as value/prompt word
            start = pos
            while pos < text_len and not text[pos].isspace() and text[pos] != '-':
                pos += 1
            word = text[start:pos]

            # Determine if it's an argument value or part of prompt
            if tokens and tokens[-1].type in (TokenType.FLAG, TokenType.COMMAND, TokenType.OPERATOR):
                tokens.append(Token(TokenType.VALUE, word, start, pos))
            else:
                tokens.append(Token(TokenType.PROMPT, word, start, pos))

        return tokens

    def _match_command(self, text: str, pos: int) -> Token | None:
        """Match a slash command at position."""
        if text[pos] != '/':
            return None

        # Extract word after /
        end = pos + 1
        while end < len(text) and (text[end].isalnum() or text[end] in '_-'):
            end += 1

        cmd = text[pos+1:end].lower()
        if cmd in self.commands:
            return Token(TokenType.COMMAND, text[pos:end], pos, end, {"command": cmd})

        # Check if it's an operator invoked as command
        if cmd in self.alias_map:
            return Token(TokenType.COMMAND, text[pos:end], pos, end,
                        {"command": cmd, "operator_id": self.alias_map[cmd]})

        return Token(TokenType.COMMAND, text[pos:end], pos, end, {"command": cmd})

    def _match_flag(self, text: str, pos: int) -> Token | None:
        """Match a command flag at position."""
        if text[pos] != '-':
            return None

        end = pos + 1
        # Handle --flag or -f
        if end < len(text) and text[end] == '-':
            end += 1

        while end < len(text) and (text[end].isalnum() or text[end] in '_-'):
            end += 1

        if end > pos + 1:
            return Token(TokenType.FLAG, text[pos:end], pos, end)
        return None

    def _match_operator(self, text: str, pos: int) -> Token | None:
        """Match a semantic operator at position."""
        # Extract word
        end = pos
        while end < len(text) and (text[end].isalnum() or text[end] in '_- '):
            # Handle multi-word operators like "checking in"
            if text[end] == ' ':
                # Look ahead for continuation
                next_word_end = end + 1
                while next_word_end < len(text) and text[next_word_end].isalnum():
                    next_word_end += 1
                candidate = text[pos:next_word_end].lower()
                if candidate in self.alias_map:
                    end = next_word_end
                    continue
                break
            end += 1

        word = text[pos:end].lower().strip()

        if word in self.alias_map:
            op_id = self.alias_map[word]
            op_def = self.operators.get(op_id)
            token_type = TokenType.USER_OP if (op_def and op_def.user_defined) else TokenType.OPERATOR
            return Token(token_type, text[pos:end], pos, end,
                        {"operator_id": op_id, "operator": op_def})

        return None

    def complete(self, prefix: str, context: str | None = None) -> list[tuple[str, str]]:
        """
        Get completions for a prefix.

        Args:
            prefix: Partial input to complete
            context: Optional context (e.g., current command)

        Returns:
            List of (completion, description) tuples
        """
        prefix_lower = prefix.lower()
        completions: list[tuple[str, str]] = []

        # Check for command completion
        if prefix.startswith('/'):
            cmd_prefix = prefix[1:].lower()
            for cmd in self.commands:
                if cmd.startswith(cmd_prefix):
                    completions.append((f"/{cmd}", f"Command: {cmd}"))
            for alias, op_id in self.alias_map.items():
                if alias.startswith(cmd_prefix):
                    op_def = self.operators.get(op_id)
                    desc = op_def.description[:50] if op_def else ""
                    completions.append((f"/{alias}", desc))
        else:
            # Operator completion
            for alias, op_id in self.alias_map.items():
                if alias.startswith(prefix_lower):
                    op_def = self.operators.get(op_id)
                    desc = op_def.description[:50] if op_def else ""
                    completions.append((alias, desc))

        # Sort by relevance (exact match first, then alphabetically)
        completions.sort(key=lambda x: (not x[0].lower().startswith(prefix_lower), x[0]))
        return completions

    def highlight(self, text: str) -> str:
        """
        Return text with ANSI color highlighting for tokens.

        Args:
            text: Input text to highlight

        Returns:
            Text with ANSI escape codes for colors
        """
        tokens = self.tokenize(text)
        result = []
        for token in tokens:
            color = self.COLORS.get(token.type, "")
            result.append(f"{color}{token.value}{self.RESET}")
        return "".join(result)

    def get_operator(self, name_or_alias: str) -> OperatorDef | None:
        """Get operator definition by name or alias."""
        op_id = self.alias_map.get(name_or_alias.lower())
        if op_id:
            return self.operators.get(op_id)
        return self.operators.get(name_or_alias)

    def list_operators(self, include_user: bool = True, domain: str | None = None) -> list[OperatorDef]:
        """List all operators, optionally filtered."""
        ops = list(self.operators.values())
        if not include_user:
            ops = [op for op in ops if not op.user_defined]
        if domain:
            ops = [op for op in ops if op.domain == domain]
        return sorted(ops, key=lambda x: x.name)

    def register_token_handler(self, handler: Callable[[str], Token | None]) -> None:
        """Register a custom token handler for extensibility."""
        self.token_handlers.append(handler)

    def to_json_schema(self) -> dict:
        """Export operator grammar as JSON schema for WebUI."""
        return {
            "operators": {
                op_id: {
                    "id": op.id,
                    "name": op.name,
                    "domain": op.domain,
                    "category": op.category,
                    "description": op.description,
                    "aliases": op.aliases,
                    "options": op.options,
                    "user_defined": op.user_defined,
                }
                for op_id, op in self.operators.items()
            },
            "commands": list(self.commands),
            "alias_map": self.alias_map,
        }


# Global lexer instance
_lexer: SemopsLexer | None = None

def get_lexer() -> SemopsLexer:
    """Get or create the global lexer instance."""
    global _lexer
    if _lexer is None:
        _lexer = SemopsLexer()
    return _lexer


# CLI interface for testing
if __name__ == "__main__":
    import sys

    lexer = SemopsLexer()

    if len(sys.argv) > 1:
        text = " ".join(sys.argv[1:])
        print(f"Input: {text!r}")
        print(f"Highlighted: {lexer.highlight(text)}")
        print("\nTokens:")
        for token in lexer.tokenize(text):
            print(f"  {token}")
        print("\nCompletions for first word:")
        first_word = text.split()[0] if text.split() else ""
        for comp, desc in lexer.complete(first_word)[:10]:
            print(f"  {comp}: {desc}")
    else:
        print("Loaded operators:")
        for op in lexer.list_operators():
            marker = "[USER]" if op.user_defined else "[BUILT-IN]"
            print(f"  {marker} {op.name}: {op.description[:60]}...")

        print(f"\nCommands: {sorted(lexer.commands)}")
        print(f"Total aliases: {len(lexer.alias_map)}")

        print("\nTest tokenization:")
        test_inputs = [
            "iterate",
            "/ckin --agent claude",
            "checking in",
            "OITERATE --goals 10x10",
            "hello world this is a prompt",
        ]
        for inp in test_inputs:
            print(f"\n  Input: {inp!r}")
            print(f"  Highlighted: {lexer.highlight(inp)}")
