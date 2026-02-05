#!/usr/bin/env python3
"""
stack.py - Undo/Redo Stack (Step 67)

PBTSO Phase: ITERATE

Provides:
- Multi-level undo/redo for edits
- Action grouping for atomic operations
- Per-file and global undo stacks
- Memory-efficient delta storage
- Action history browsing

Bus Topics:
- code.undo.execute
- code.redo.execute
- code.action.pushed

Protocol: DKIN v30, PAIP v16
"""

from __future__ import annotations

import copy
import hashlib
import time
import uuid
from collections import deque
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Deque, Dict, List, Optional, Set, Tuple


# =============================================================================
# Types
# =============================================================================

class ActionType(Enum):
    """Type of edit action."""
    INSERT = "insert"       # Insert content
    DELETE = "delete"       # Delete content
    REPLACE = "replace"     # Replace content
    CREATE = "create"       # Create file
    REMOVE = "remove"       # Remove file
    RENAME = "rename"       # Rename file
    TRANSFORM = "transform" # AST transformation


@dataclass
class TextDelta:
    """
    Represents a text change delta.

    Stores minimal information needed to undo/redo.
    """
    start_line: int
    start_col: int
    end_line: int
    end_col: int
    old_text: str
    new_text: str

    @property
    def is_insert(self) -> bool:
        return len(self.old_text) == 0 and len(self.new_text) > 0

    @property
    def is_delete(self) -> bool:
        return len(self.old_text) > 0 and len(self.new_text) == 0

    @property
    def is_replace(self) -> bool:
        return len(self.old_text) > 0 and len(self.new_text) > 0

    def inverse(self) -> "TextDelta":
        """Create inverse delta for undo."""
        return TextDelta(
            start_line=self.start_line,
            start_col=self.start_col,
            end_line=self.end_line,
            end_col=self.end_col,
            old_text=self.new_text,
            new_text=self.old_text,
        )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "start_line": self.start_line,
            "start_col": self.start_col,
            "end_line": self.end_line,
            "end_col": self.end_col,
            "old_text_len": len(self.old_text),
            "new_text_len": len(self.new_text),
        }


@dataclass
class EditAction:
    """
    A single undoable edit action.

    Contains all information needed to undo or redo the action.
    """
    id: str
    action_type: ActionType
    path: str
    delta: Optional[TextDelta] = None
    old_path: Optional[str] = None  # For rename
    timestamp: float = field(default_factory=time.time)
    description: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)

    # For file-level operations
    old_content: Optional[str] = None
    new_content: Optional[str] = None

    @property
    def affects_file(self) -> bool:
        return self.action_type in (ActionType.CREATE, ActionType.REMOVE, ActionType.RENAME)

    def inverse(self) -> "EditAction":
        """Create inverse action for undo."""
        if self.action_type == ActionType.INSERT:
            return EditAction(
                id=f"inv-{self.id}",
                action_type=ActionType.DELETE,
                path=self.path,
                delta=self.delta.inverse() if self.delta else None,
                description=f"Undo: {self.description}",
            )

        elif self.action_type == ActionType.DELETE:
            return EditAction(
                id=f"inv-{self.id}",
                action_type=ActionType.INSERT,
                path=self.path,
                delta=self.delta.inverse() if self.delta else None,
                description=f"Undo: {self.description}",
            )

        elif self.action_type == ActionType.REPLACE:
            return EditAction(
                id=f"inv-{self.id}",
                action_type=ActionType.REPLACE,
                path=self.path,
                delta=self.delta.inverse() if self.delta else None,
                old_content=self.new_content,
                new_content=self.old_content,
                description=f"Undo: {self.description}",
            )

        elif self.action_type == ActionType.CREATE:
            return EditAction(
                id=f"inv-{self.id}",
                action_type=ActionType.REMOVE,
                path=self.path,
                old_content=self.new_content,
                description=f"Undo: {self.description}",
            )

        elif self.action_type == ActionType.REMOVE:
            return EditAction(
                id=f"inv-{self.id}",
                action_type=ActionType.CREATE,
                path=self.path,
                new_content=self.old_content,
                description=f"Undo: {self.description}",
            )

        elif self.action_type == ActionType.RENAME:
            return EditAction(
                id=f"inv-{self.id}",
                action_type=ActionType.RENAME,
                path=self.old_path,
                old_path=self.path,
                description=f"Undo: {self.description}",
            )

        # Default: return copy with swapped content
        return EditAction(
            id=f"inv-{self.id}",
            action_type=self.action_type,
            path=self.path,
            old_content=self.new_content,
            new_content=self.old_content,
            description=f"Undo: {self.description}",
        )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "action_type": self.action_type.value,
            "path": self.path,
            "delta": self.delta.to_dict() if self.delta else None,
            "timestamp": self.timestamp,
            "description": self.description,
        }


@dataclass
class ActionGroup:
    """
    A group of related actions that should be undone together.

    Used for atomic multi-file operations.
    """
    id: str
    actions: List[EditAction]
    description: str
    created_at: float = field(default_factory=time.time)
    tags: List[str] = field(default_factory=list)

    @property
    def action_count(self) -> int:
        return len(self.actions)

    @property
    def files_affected(self) -> Set[str]:
        return set(a.path for a in self.actions)

    def inverse(self) -> "ActionGroup":
        """Create inverse group for undo."""
        return ActionGroup(
            id=f"inv-{self.id}",
            actions=[a.inverse() for a in reversed(self.actions)],
            description=f"Undo: {self.description}",
        )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "actions": [a.to_dict() for a in self.actions],
            "action_count": self.action_count,
            "files_affected": list(self.files_affected),
            "description": self.description,
            "created_at": self.created_at,
            "tags": self.tags,
        }


# =============================================================================
# Undo/Redo Stack
# =============================================================================

class UndoRedoStack:
    """
    Multi-level undo/redo stack for code edits.

    PBTSO Phase: ITERATE

    Features:
    - Unlimited undo levels (configurable)
    - Action grouping for atomic operations
    - Per-file undo stacks
    - Memory-efficient delta storage
    - Action history browsing

    Usage:
        stack = UndoRedoStack(working_dir)
        stack.push(action)
        stack.undo()
        stack.redo()
    """

    BUS_TOPICS = {
        "undo": "code.undo.execute",
        "redo": "code.redo.execute",
        "pushed": "code.action.pushed",
    }

    def __init__(
        self,
        working_dir: Path,
        bus: Optional[Any] = None,
        max_undo_levels: int = 100,
        max_memory_mb: int = 50,
        group_timeout_ms: int = 1000,
    ):
        self.working_dir = Path(working_dir)
        self.bus = bus
        self.max_undo_levels = max_undo_levels
        self.max_memory_mb = max_memory_mb
        self.group_timeout_ms = group_timeout_ms

        # Global undo/redo stacks
        self._undo_stack: Deque[ActionGroup] = deque(maxlen=max_undo_levels)
        self._redo_stack: Deque[ActionGroup] = deque(maxlen=max_undo_levels)

        # Per-file stacks for file-specific undo
        self._file_undo_stacks: Dict[str, Deque[EditAction]] = {}
        self._file_redo_stacks: Dict[str, Deque[EditAction]] = {}

        # Current action group being built
        self._current_group: Optional[ActionGroup] = None
        self._last_action_time: float = 0

        # Memory tracking
        self._memory_usage: int = 0

        # Action executors
        self._executors: Dict[ActionType, Callable[[EditAction], bool]] = {}
        self._register_default_executors()

    def _register_default_executors(self) -> None:
        """Register default action executors."""
        self._executors[ActionType.INSERT] = self._execute_insert
        self._executors[ActionType.DELETE] = self._execute_delete
        self._executors[ActionType.REPLACE] = self._execute_replace
        self._executors[ActionType.CREATE] = self._execute_create
        self._executors[ActionType.REMOVE] = self._execute_remove
        self._executors[ActionType.RENAME] = self._execute_rename

    # =========================================================================
    # Push Actions
    # =========================================================================

    def push(
        self,
        action: EditAction,
        auto_group: bool = True,
    ) -> None:
        """
        Push an action onto the undo stack.

        Args:
            action: The action to push
            auto_group: Whether to auto-group with recent actions
        """
        now = time.time()

        # Clear redo stack on new action
        self._redo_stack.clear()

        # Check if we should group with current
        if auto_group and self._current_group:
            time_since_last = (now - self._last_action_time) * 1000
            if time_since_last <= self.group_timeout_ms:
                self._current_group.actions.append(action)
                self._last_action_time = now
                return
            else:
                # Close current group and push
                self._undo_stack.append(self._current_group)
                self._current_group = None

        # Start new group
        self._current_group = ActionGroup(
            id=f"grp-{uuid.uuid4().hex[:8]}",
            actions=[action],
            description=action.description,
        )
        self._last_action_time = now

        # Also track in per-file stack
        if action.path not in self._file_undo_stacks:
            self._file_undo_stacks[action.path] = deque(maxlen=self.max_undo_levels)
        self._file_undo_stacks[action.path].append(action)

        # Update memory tracking
        self._update_memory_usage(action)

        # Emit event
        if self.bus:
            self.bus.emit({
                "topic": self.BUS_TOPICS["pushed"],
                "kind": "action",
                "actor": "code-agent",
                "data": action.to_dict(),
            })

    def push_group(self, group: ActionGroup) -> None:
        """Push a complete action group."""
        # Finalize any current group
        self._finalize_current_group()

        # Clear redo stack
        self._redo_stack.clear()

        # Push the group
        self._undo_stack.append(group)

        # Track per-file
        for action in group.actions:
            if action.path not in self._file_undo_stacks:
                self._file_undo_stacks[action.path] = deque(maxlen=self.max_undo_levels)
            self._file_undo_stacks[action.path].append(action)

    def _finalize_current_group(self) -> None:
        """Finalize and push the current action group."""
        if self._current_group:
            self._undo_stack.append(self._current_group)
            self._current_group = None

    def _update_memory_usage(self, action: EditAction) -> None:
        """Update memory usage tracking."""
        size = 0
        if action.old_content:
            size += len(action.old_content)
        if action.new_content:
            size += len(action.new_content)
        if action.delta:
            size += len(action.delta.old_text) + len(action.delta.new_text)

        self._memory_usage += size

        # Trim if over limit
        while self._memory_usage > self.max_memory_mb * 1024 * 1024 and self._undo_stack:
            old_group = self._undo_stack.popleft()
            for a in old_group.actions:
                if a.old_content:
                    self._memory_usage -= len(a.old_content)
                if a.new_content:
                    self._memory_usage -= len(a.new_content)

    # =========================================================================
    # Undo/Redo
    # =========================================================================

    def undo(self, count: int = 1) -> List[ActionGroup]:
        """
        Undo the last N action groups.

        Args:
            count: Number of groups to undo

        Returns:
            List of undone action groups
        """
        self._finalize_current_group()

        undone: List[ActionGroup] = []

        for _ in range(count):
            if not self._undo_stack:
                break

            group = self._undo_stack.pop()
            inverse = group.inverse()

            # Execute inverse actions
            success = self._execute_group(inverse)

            if success:
                self._redo_stack.append(group)
                undone.append(group)

                # Update per-file stacks
                for action in group.actions:
                    if action.path in self._file_undo_stacks:
                        file_stack = self._file_undo_stacks[action.path]
                        if file_stack and file_stack[-1].id == action.id:
                            file_stack.pop()

                # Emit event
                if self.bus:
                    self.bus.emit({
                        "topic": self.BUS_TOPICS["undo"],
                        "kind": "undo",
                        "actor": "code-agent",
                        "data": {
                            "group_id": group.id,
                            "action_count": group.action_count,
                            "files": list(group.files_affected),
                        },
                    })

        return undone

    def redo(self, count: int = 1) -> List[ActionGroup]:
        """
        Redo the last N undone action groups.

        Args:
            count: Number of groups to redo

        Returns:
            List of redone action groups
        """
        redone: List[ActionGroup] = []

        for _ in range(count):
            if not self._redo_stack:
                break

            group = self._redo_stack.pop()

            # Re-execute original actions
            success = self._execute_group(group)

            if success:
                self._undo_stack.append(group)
                redone.append(group)

                # Update per-file stacks
                for action in group.actions:
                    if action.path not in self._file_undo_stacks:
                        self._file_undo_stacks[action.path] = deque(maxlen=self.max_undo_levels)
                    self._file_undo_stacks[action.path].append(action)

                # Emit event
                if self.bus:
                    self.bus.emit({
                        "topic": self.BUS_TOPICS["redo"],
                        "kind": "redo",
                        "actor": "code-agent",
                        "data": {
                            "group_id": group.id,
                            "action_count": group.action_count,
                            "files": list(group.files_affected),
                        },
                    })

        return redone

    def undo_file(self, path: str, count: int = 1) -> List[EditAction]:
        """Undo actions for a specific file."""
        if path not in self._file_undo_stacks:
            return []

        undone: List[EditAction] = []
        stack = self._file_undo_stacks[path]

        for _ in range(count):
            if not stack:
                break

            action = stack.pop()
            inverse = action.inverse()

            executor = self._executors.get(inverse.action_type)
            if executor and executor(inverse):
                if path not in self._file_redo_stacks:
                    self._file_redo_stacks[path] = deque(maxlen=self.max_undo_levels)
                self._file_redo_stacks[path].append(action)
                undone.append(action)

        return undone

    def redo_file(self, path: str, count: int = 1) -> List[EditAction]:
        """Redo actions for a specific file."""
        if path not in self._file_redo_stacks:
            return []

        redone: List[EditAction] = []
        redo_stack = self._file_redo_stacks[path]

        for _ in range(count):
            if not redo_stack:
                break

            action = redo_stack.pop()

            executor = self._executors.get(action.action_type)
            if executor and executor(action):
                if path not in self._file_undo_stacks:
                    self._file_undo_stacks[path] = deque(maxlen=self.max_undo_levels)
                self._file_undo_stacks[path].append(action)
                redone.append(action)

        return redone

    # =========================================================================
    # Action Execution
    # =========================================================================

    def _execute_group(self, group: ActionGroup) -> bool:
        """Execute all actions in a group."""
        executed: List[EditAction] = []

        for action in group.actions:
            executor = self._executors.get(action.action_type)
            if not executor:
                # Rollback executed actions
                for a in reversed(executed):
                    inv_executor = self._executors.get(a.inverse().action_type)
                    if inv_executor:
                        inv_executor(a.inverse())
                return False

            if executor(action):
                executed.append(action)
            else:
                # Rollback executed actions
                for a in reversed(executed):
                    inv_executor = self._executors.get(a.inverse().action_type)
                    if inv_executor:
                        inv_executor(a.inverse())
                return False

        return True

    def _execute_insert(self, action: EditAction) -> bool:
        """Execute an insert action."""
        if not action.delta:
            return False

        path = self.working_dir / action.path
        if not path.exists():
            return False

        try:
            content = path.read_text()
            lines = content.splitlines(keepends=True)

            # Insert new text at position
            line_idx = action.delta.start_line - 1
            if line_idx < len(lines):
                line = lines[line_idx]
                col = action.delta.start_col
                lines[line_idx] = line[:col] + action.delta.new_text + line[col:]
            else:
                lines.append(action.delta.new_text)

            path.write_text("".join(lines))
            return True

        except Exception:
            return False

    def _execute_delete(self, action: EditAction) -> bool:
        """Execute a delete action."""
        if not action.delta:
            return False

        path = self.working_dir / action.path
        if not path.exists():
            return False

        try:
            content = path.read_text()
            lines = content.splitlines(keepends=True)

            # Delete text at position
            start_line = action.delta.start_line - 1
            end_line = action.delta.end_line - 1
            start_col = action.delta.start_col
            end_col = action.delta.end_col

            if start_line == end_line and start_line < len(lines):
                line = lines[start_line]
                lines[start_line] = line[:start_col] + line[end_col:]
            else:
                # Multi-line delete
                if start_line < len(lines):
                    lines[start_line] = lines[start_line][:start_col]
                if end_line < len(lines):
                    lines[start_line] += lines[end_line][end_col:]
                del lines[start_line + 1:end_line + 1]

            path.write_text("".join(lines))
            return True

        except Exception:
            return False

    def _execute_replace(self, action: EditAction) -> bool:
        """Execute a replace action."""
        path = self.working_dir / action.path
        if not path.exists():
            return False

        try:
            if action.new_content is not None:
                # Full file replace
                path.write_text(action.new_content)
            elif action.delta:
                # Delta replace
                content = path.read_text()
                lines = content.splitlines(keepends=True)

                start_line = action.delta.start_line - 1
                end_line = action.delta.end_line - 1
                start_col = action.delta.start_col
                end_col = action.delta.end_col

                if start_line == end_line and start_line < len(lines):
                    line = lines[start_line]
                    lines[start_line] = line[:start_col] + action.delta.new_text + line[end_col:]
                else:
                    # Multi-line replace
                    if start_line < len(lines):
                        before = lines[start_line][:start_col]
                    else:
                        before = ""
                    if end_line < len(lines):
                        after = lines[end_line][end_col:]
                    else:
                        after = ""

                    lines[start_line] = before + action.delta.new_text + after
                    del lines[start_line + 1:end_line + 1]

                path.write_text("".join(lines))

            return True

        except Exception:
            return False

    def _execute_create(self, action: EditAction) -> bool:
        """Execute a file create action."""
        path = self.working_dir / action.path

        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(action.new_content or "")
            return True
        except Exception:
            return False

    def _execute_remove(self, action: EditAction) -> bool:
        """Execute a file remove action."""
        path = self.working_dir / action.path

        try:
            if path.exists():
                path.unlink()
            return True
        except Exception:
            return False

    def _execute_rename(self, action: EditAction) -> bool:
        """Execute a file rename action."""
        old_path = self.working_dir / (action.old_path or action.path)
        new_path = self.working_dir / action.path

        try:
            if old_path.exists():
                new_path.parent.mkdir(parents=True, exist_ok=True)
                old_path.rename(new_path)
                return True
            return False
        except Exception:
            return False

    # =========================================================================
    # Query Operations
    # =========================================================================

    def can_undo(self) -> bool:
        """Check if undo is available."""
        return bool(self._undo_stack or self._current_group)

    def can_redo(self) -> bool:
        """Check if redo is available."""
        return bool(self._redo_stack)

    def can_undo_file(self, path: str) -> bool:
        """Check if file-specific undo is available."""
        return path in self._file_undo_stacks and bool(self._file_undo_stacks[path])

    def can_redo_file(self, path: str) -> bool:
        """Check if file-specific redo is available."""
        return path in self._file_redo_stacks and bool(self._file_redo_stacks[path])

    def peek_undo(self) -> Optional[ActionGroup]:
        """Peek at the next action group to be undone."""
        self._finalize_current_group()
        return self._undo_stack[-1] if self._undo_stack else None

    def peek_redo(self) -> Optional[ActionGroup]:
        """Peek at the next action group to be redone."""
        return self._redo_stack[-1] if self._redo_stack else None

    def get_history(self, limit: int = 50) -> List[ActionGroup]:
        """Get undo history."""
        self._finalize_current_group()
        return list(self._undo_stack)[-limit:]

    def get_file_history(self, path: str, limit: int = 50) -> List[EditAction]:
        """Get undo history for a specific file."""
        if path not in self._file_undo_stacks:
            return []
        return list(self._file_undo_stacks[path])[-limit:]

    def clear(self) -> None:
        """Clear all undo/redo history."""
        self._undo_stack.clear()
        self._redo_stack.clear()
        self._file_undo_stacks.clear()
        self._file_redo_stacks.clear()
        self._current_group = None
        self._memory_usage = 0

    def clear_file(self, path: str) -> None:
        """Clear undo/redo history for a specific file."""
        if path in self._file_undo_stacks:
            del self._file_undo_stacks[path]
        if path in self._file_redo_stacks:
            del self._file_redo_stacks[path]

    def get_stats(self) -> Dict[str, Any]:
        """Get stack statistics."""
        return {
            "undo_levels": len(self._undo_stack) + (1 if self._current_group else 0),
            "redo_levels": len(self._redo_stack),
            "files_tracked": len(self._file_undo_stacks),
            "memory_usage_bytes": self._memory_usage,
            "max_levels": self.max_undo_levels,
        }


# =============================================================================
# Helper Functions
# =============================================================================

def create_action_from_edit(
    path: str,
    old_content: str,
    new_content: str,
    description: str = "",
) -> EditAction:
    """Create an EditAction from before/after content."""
    return EditAction(
        id=f"act-{uuid.uuid4().hex[:8]}",
        action_type=ActionType.REPLACE,
        path=path,
        old_content=old_content,
        new_content=new_content,
        description=description,
    )


def create_insert_action(
    path: str,
    line: int,
    col: int,
    text: str,
    description: str = "",
) -> EditAction:
    """Create an insert action."""
    return EditAction(
        id=f"act-{uuid.uuid4().hex[:8]}",
        action_type=ActionType.INSERT,
        path=path,
        delta=TextDelta(
            start_line=line,
            start_col=col,
            end_line=line,
            end_col=col,
            old_text="",
            new_text=text,
        ),
        description=description,
    )


def create_delete_action(
    path: str,
    start_line: int,
    start_col: int,
    end_line: int,
    end_col: int,
    deleted_text: str,
    description: str = "",
) -> EditAction:
    """Create a delete action."""
    return EditAction(
        id=f"act-{uuid.uuid4().hex[:8]}",
        action_type=ActionType.DELETE,
        path=path,
        delta=TextDelta(
            start_line=start_line,
            start_col=start_col,
            end_line=end_line,
            end_col=end_col,
            old_text=deleted_text,
            new_text="",
        ),
        description=description,
    )


# =============================================================================
# CLI
# =============================================================================

def main() -> int:
    """CLI entry point for Undo/Redo Stack."""
    import argparse
    import json

    parser = argparse.ArgumentParser(description="Undo/Redo Stack (Step 67)")
    parser.add_argument("--working-dir", default=".", help="Working directory")

    subparsers = parser.add_subparsers(dest="command", required=True)

    # undo command
    undo_parser = subparsers.add_parser("undo", help="Undo actions")
    undo_parser.add_argument("--count", "-n", type=int, default=1, help="Number to undo")
    undo_parser.add_argument("--file", help="Undo for specific file")

    # redo command
    redo_parser = subparsers.add_parser("redo", help="Redo actions")
    redo_parser.add_argument("--count", "-n", type=int, default=1, help="Number to redo")
    redo_parser.add_argument("--file", help="Redo for specific file")

    # history command
    history_parser = subparsers.add_parser("history", help="Show history")
    history_parser.add_argument("--file", help="History for specific file")
    history_parser.add_argument("--limit", type=int, default=20)
    history_parser.add_argument("--json", action="store_true")

    # status command
    subparsers.add_parser("status", help="Show stack status")

    # clear command
    clear_parser = subparsers.add_parser("clear", help="Clear history")
    clear_parser.add_argument("--file", help="Clear for specific file")

    args = parser.parse_args()

    stack = UndoRedoStack(Path(args.working_dir))

    if args.command == "undo":
        if args.file:
            undone = stack.undo_file(args.file, args.count)
            print(f"Undid {len(undone)} actions for {args.file}")
        else:
            undone = stack.undo(args.count)
            print(f"Undid {len(undone)} action groups")
        return 0

    elif args.command == "redo":
        if args.file:
            redone = stack.redo_file(args.file, args.count)
            print(f"Redid {len(redone)} actions for {args.file}")
        else:
            redone = stack.redo(args.count)
            print(f"Redid {len(redone)} action groups")
        return 0

    elif args.command == "history":
        if args.file:
            actions = stack.get_file_history(args.file, args.limit)
            if args.json:
                print(json.dumps([a.to_dict() for a in actions], indent=2))
            else:
                for a in actions:
                    print(f"{a.id} {a.action_type.value} {a.description or '(no description)'}")
        else:
            groups = stack.get_history(args.limit)
            if args.json:
                print(json.dumps([g.to_dict() for g in groups], indent=2))
            else:
                for g in groups:
                    print(f"{g.id} ({g.action_count} actions) {g.description or '(no description)'}")
        return 0

    elif args.command == "status":
        stats = stack.get_stats()
        print(f"Undo levels: {stats['undo_levels']}")
        print(f"Redo levels: {stats['redo_levels']}")
        print(f"Files tracked: {stats['files_tracked']}")
        print(f"Memory usage: {stats['memory_usage_bytes']} bytes")
        print(f"Can undo: {stack.can_undo()}")
        print(f"Can redo: {stack.can_redo()}")
        return 0

    elif args.command == "clear":
        if args.file:
            stack.clear_file(args.file)
            print(f"Cleared history for {args.file}")
        else:
            stack.clear()
            print("Cleared all history")
        return 0

    return 1


if __name__ == "__main__":
    import sys
    sys.exit(main())
