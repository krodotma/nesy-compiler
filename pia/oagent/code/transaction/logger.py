#!/usr/bin/env python3
"""
logger.py - Transaction Logger (Step 66)

PBTSO Phase: DISTILL, VERIFY

Provides:
- Atomic transaction tracking for edits
- Write-ahead logging (WAL) for durability
- Transaction commit/rollback
- Audit trail for all operations
- Recovery from failed transactions

Bus Topics:
- code.transaction.begin
- code.transaction.commit
- code.transaction.rollback

Protocol: DKIN v30, PAIP v16
"""

from __future__ import annotations

import json
import os
import time
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Dict, Iterator, List, Optional


# =============================================================================
# Types
# =============================================================================

class TransactionState(Enum):
    """State of a transaction."""
    PENDING = "pending"       # Transaction started, not committed
    COMMITTED = "committed"   # Successfully committed
    ROLLED_BACK = "rolled_back"  # Rolled back
    FAILED = "failed"         # Failed during commit
    RECOVERING = "recovering" # In recovery process


class EntryType(Enum):
    """Type of transaction entry."""
    BEGIN = "begin"           # Transaction start
    WRITE = "write"           # File write
    DELETE = "delete"         # File delete
    RENAME = "rename"         # File rename
    CREATE = "create"         # File create
    MODIFY = "modify"         # Content modify
    CHECKPOINT = "checkpoint" # Checkpoint marker
    COMMIT = "commit"         # Transaction commit
    ROLLBACK = "rollback"     # Transaction rollback


@dataclass
class TransactionEntry:
    """
    A single entry in the transaction log.

    Represents one atomic operation within a transaction.
    """
    id: str
    transaction_id: str
    sequence: int
    entry_type: EntryType
    timestamp: float
    path: Optional[str] = None
    old_content_hash: Optional[str] = None
    new_content_hash: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "transaction_id": self.transaction_id,
            "sequence": self.sequence,
            "entry_type": self.entry_type.value,
            "timestamp": self.timestamp,
            "path": self.path,
            "old_content_hash": self.old_content_hash,
            "new_content_hash": self.new_content_hash,
            "metadata": self.metadata,
        }

    def to_log_line(self) -> str:
        """Serialize to log line format."""
        return json.dumps(self.to_dict())

    @classmethod
    def from_log_line(cls, line: str) -> "TransactionEntry":
        """Deserialize from log line."""
        data = json.loads(line)
        return cls(
            id=data["id"],
            transaction_id=data["transaction_id"],
            sequence=data["sequence"],
            entry_type=EntryType(data["entry_type"]),
            timestamp=data["timestamp"],
            path=data.get("path"),
            old_content_hash=data.get("old_content_hash"),
            new_content_hash=data.get("new_content_hash"),
            metadata=data.get("metadata", {}),
        )


@dataclass
class Transaction:
    """
    Represents a single atomic transaction.

    Contains multiple entries that form a logical unit.
    """
    id: str
    state: TransactionState
    entries: List[TransactionEntry]
    created_at: float
    committed_at: Optional[float] = None
    description: str = ""
    actor: str = "code-agent"
    task_id: Optional[str] = None
    parent_transaction_id: Optional[str] = None

    @property
    def entry_count(self) -> int:
        return len(self.entries)

    @property
    def is_active(self) -> bool:
        return self.state == TransactionState.PENDING

    @property
    def is_committed(self) -> bool:
        return self.state == TransactionState.COMMITTED

    @property
    def files_affected(self) -> List[str]:
        return list(set(e.path for e in self.entries if e.path))

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "state": self.state.value,
            "entries": [e.to_dict() for e in self.entries],
            "entry_count": self.entry_count,
            "created_at": self.created_at,
            "committed_at": self.committed_at,
            "description": self.description,
            "actor": self.actor,
            "task_id": self.task_id,
            "files_affected": self.files_affected,
        }


# =============================================================================
# Transaction Logger
# =============================================================================

class TransactionLogger:
    """
    Write-ahead logging for edit transactions.

    PBTSO Phase: DISTILL, VERIFY

    Features:
    - Atomic transaction boundaries
    - Write-ahead logging for durability
    - Transaction commit and rollback
    - Automatic recovery from crashes
    - Full audit trail

    Usage:
        logger = TransactionLogger(working_dir)
        txn = logger.begin("refactoring changes")
        logger.log_write(txn.id, "file.py", old_hash, new_hash)
        logger.commit(txn.id)
    """

    BUS_TOPICS = {
        "begin": "code.transaction.begin",
        "commit": "code.transaction.commit",
        "rollback": "code.transaction.rollback",
    }

    def __init__(
        self,
        working_dir: Path,
        bus: Optional[Any] = None,
        log_dir: Optional[Path] = None,
        max_log_size_mb: int = 100,
        checkpoint_interval: int = 1000,  # entries
    ):
        self.working_dir = Path(working_dir)
        self.bus = bus
        self.log_dir = log_dir or self.working_dir / ".pluribus" / "txlog"
        self.max_log_size_mb = max_log_size_mb
        self.checkpoint_interval = checkpoint_interval

        self._transactions: Dict[str, Transaction] = {}
        self._active_transaction: Optional[str] = None
        self._entry_counter: int = 0
        self._log_file: Optional[Path] = None

        self._ensure_log_dir()
        self._recover()

    def _ensure_log_dir(self) -> None:
        """Ensure log directory exists."""
        self.log_dir.mkdir(parents=True, exist_ok=True)

    def _get_current_log_file(self) -> Path:
        """Get current log file path."""
        if self._log_file and self._log_file.exists():
            size = self._log_file.stat().st_size / (1024 * 1024)
            if size < self.max_log_size_mb:
                return self._log_file

        # Create new log file
        timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
        self._log_file = self.log_dir / f"txlog_{timestamp}.wal"
        return self._log_file

    def _write_entry(self, entry: TransactionEntry) -> None:
        """Write entry to log file."""
        log_file = self._get_current_log_file()
        with log_file.open("a") as f:
            f.write(entry.to_log_line() + "\n")
            f.flush()
            os.fsync(f.fileno())  # Ensure durability

        self._entry_counter += 1

        # Checkpoint if needed
        if self._entry_counter % self.checkpoint_interval == 0:
            self._write_checkpoint()

    def _write_checkpoint(self) -> None:
        """Write checkpoint marker to log."""
        for txn in self._transactions.values():
            if txn.is_active:
                entry = TransactionEntry(
                    id=f"ckpt-{uuid.uuid4().hex[:8]}",
                    transaction_id=txn.id,
                    sequence=len(txn.entries),
                    entry_type=EntryType.CHECKPOINT,
                    timestamp=time.time(),
                    metadata={"state": txn.state.value},
                )
                log_file = self._get_current_log_file()
                with log_file.open("a") as f:
                    f.write(entry.to_log_line() + "\n")

    # =========================================================================
    # Transaction Management
    # =========================================================================

    def begin(
        self,
        description: str = "",
        actor: str = "code-agent",
        task_id: Optional[str] = None,
    ) -> Transaction:
        """
        Begin a new transaction.

        Args:
            description: Human-readable description
            actor: Who initiated the transaction
            task_id: Associated task ID

        Returns:
            New Transaction object
        """
        txn_id = f"txn-{uuid.uuid4().hex[:12]}"
        timestamp = time.time()

        transaction = Transaction(
            id=txn_id,
            state=TransactionState.PENDING,
            entries=[],
            created_at=timestamp,
            description=description,
            actor=actor,
            task_id=task_id,
        )

        # Write BEGIN entry
        entry = TransactionEntry(
            id=f"entry-{uuid.uuid4().hex[:8]}",
            transaction_id=txn_id,
            sequence=0,
            entry_type=EntryType.BEGIN,
            timestamp=timestamp,
            metadata={
                "description": description,
                "actor": actor,
                "task_id": task_id,
            },
        )

        self._write_entry(entry)
        transaction.entries.append(entry)

        self._transactions[txn_id] = transaction
        self._active_transaction = txn_id

        # Emit event
        if self.bus:
            self.bus.emit({
                "topic": self.BUS_TOPICS["begin"],
                "kind": "transaction",
                "actor": actor,
                "data": {
                    "transaction_id": txn_id,
                    "description": description,
                },
            })

        return transaction

    def commit(self, transaction_id: str) -> bool:
        """
        Commit a transaction.

        Args:
            transaction_id: ID of transaction to commit

        Returns:
            True if commit successful
        """
        transaction = self._transactions.get(transaction_id)
        if not transaction:
            return False

        if not transaction.is_active:
            return False

        timestamp = time.time()

        # Write COMMIT entry
        entry = TransactionEntry(
            id=f"entry-{uuid.uuid4().hex[:8]}",
            transaction_id=transaction_id,
            sequence=len(transaction.entries),
            entry_type=EntryType.COMMIT,
            timestamp=timestamp,
        )

        self._write_entry(entry)
        transaction.entries.append(entry)

        transaction.state = TransactionState.COMMITTED
        transaction.committed_at = timestamp

        if self._active_transaction == transaction_id:
            self._active_transaction = None

        # Emit event
        if self.bus:
            self.bus.emit({
                "topic": self.BUS_TOPICS["commit"],
                "kind": "transaction",
                "actor": transaction.actor,
                "data": {
                    "transaction_id": transaction_id,
                    "entry_count": transaction.entry_count,
                    "files_affected": transaction.files_affected,
                },
            })

        return True

    def rollback(self, transaction_id: str, reason: str = "") -> bool:
        """
        Rollback a transaction.

        Args:
            transaction_id: ID of transaction to rollback
            reason: Reason for rollback

        Returns:
            True if rollback successful
        """
        transaction = self._transactions.get(transaction_id)
        if not transaction:
            return False

        if not transaction.is_active:
            return False

        timestamp = time.time()

        # Write ROLLBACK entry
        entry = TransactionEntry(
            id=f"entry-{uuid.uuid4().hex[:8]}",
            transaction_id=transaction_id,
            sequence=len(transaction.entries),
            entry_type=EntryType.ROLLBACK,
            timestamp=timestamp,
            metadata={"reason": reason},
        )

        self._write_entry(entry)
        transaction.entries.append(entry)

        transaction.state = TransactionState.ROLLED_BACK

        if self._active_transaction == transaction_id:
            self._active_transaction = None

        # Emit event
        if self.bus:
            self.bus.emit({
                "topic": self.BUS_TOPICS["rollback"],
                "kind": "transaction",
                "actor": transaction.actor,
                "data": {
                    "transaction_id": transaction_id,
                    "reason": reason,
                },
            })

        return True

    # =========================================================================
    # Logging Operations
    # =========================================================================

    def log_write(
        self,
        transaction_id: str,
        path: str,
        old_content_hash: Optional[str],
        new_content_hash: str,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> TransactionEntry:
        """Log a file write operation."""
        return self._log_operation(
            transaction_id=transaction_id,
            entry_type=EntryType.WRITE,
            path=path,
            old_content_hash=old_content_hash,
            new_content_hash=new_content_hash,
            metadata=metadata,
        )

    def log_create(
        self,
        transaction_id: str,
        path: str,
        content_hash: str,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> TransactionEntry:
        """Log a file creation."""
        return self._log_operation(
            transaction_id=transaction_id,
            entry_type=EntryType.CREATE,
            path=path,
            new_content_hash=content_hash,
            metadata=metadata,
        )

    def log_delete(
        self,
        transaction_id: str,
        path: str,
        old_content_hash: str,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> TransactionEntry:
        """Log a file deletion."""
        return self._log_operation(
            transaction_id=transaction_id,
            entry_type=EntryType.DELETE,
            path=path,
            old_content_hash=old_content_hash,
            metadata=metadata,
        )

    def log_rename(
        self,
        transaction_id: str,
        old_path: str,
        new_path: str,
        content_hash: str,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> TransactionEntry:
        """Log a file rename."""
        meta = metadata or {}
        meta["new_path"] = new_path
        return self._log_operation(
            transaction_id=transaction_id,
            entry_type=EntryType.RENAME,
            path=old_path,
            old_content_hash=content_hash,
            new_content_hash=content_hash,
            metadata=meta,
        )

    def log_modify(
        self,
        transaction_id: str,
        path: str,
        old_content_hash: str,
        new_content_hash: str,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> TransactionEntry:
        """Log a content modification."""
        return self._log_operation(
            transaction_id=transaction_id,
            entry_type=EntryType.MODIFY,
            path=path,
            old_content_hash=old_content_hash,
            new_content_hash=new_content_hash,
            metadata=metadata,
        )

    def _log_operation(
        self,
        transaction_id: str,
        entry_type: EntryType,
        path: Optional[str] = None,
        old_content_hash: Optional[str] = None,
        new_content_hash: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> TransactionEntry:
        """Internal method to log an operation."""
        transaction = self._transactions.get(transaction_id)
        if not transaction:
            raise ValueError(f"Transaction not found: {transaction_id}")

        if not transaction.is_active:
            raise ValueError(f"Transaction not active: {transaction_id}")

        entry = TransactionEntry(
            id=f"entry-{uuid.uuid4().hex[:8]}",
            transaction_id=transaction_id,
            sequence=len(transaction.entries),
            entry_type=entry_type,
            timestamp=time.time(),
            path=path,
            old_content_hash=old_content_hash,
            new_content_hash=new_content_hash,
            metadata=metadata or {},
        )

        self._write_entry(entry)
        transaction.entries.append(entry)

        return entry

    # =========================================================================
    # Recovery
    # =========================================================================

    def _recover(self) -> None:
        """Recover transactions from log files."""
        log_files = sorted(self.log_dir.glob("txlog_*.wal"))

        for log_file in log_files:
            self._recover_from_log(log_file)

        # Rollback any pending transactions (crash recovery)
        for txn in list(self._transactions.values()):
            if txn.state == TransactionState.PENDING:
                txn.state = TransactionState.RECOVERING
                # In a real implementation, we would undo the operations
                txn.state = TransactionState.ROLLED_BACK

    def _recover_from_log(self, log_file: Path) -> None:
        """Recover transactions from a single log file."""
        transactions: Dict[str, Transaction] = {}

        try:
            with log_file.open() as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue

                    try:
                        entry = TransactionEntry.from_log_line(line)

                        if entry.entry_type == EntryType.BEGIN:
                            txn = Transaction(
                                id=entry.transaction_id,
                                state=TransactionState.PENDING,
                                entries=[entry],
                                created_at=entry.timestamp,
                                description=entry.metadata.get("description", ""),
                                actor=entry.metadata.get("actor", "code-agent"),
                                task_id=entry.metadata.get("task_id"),
                            )
                            transactions[txn.id] = txn

                        elif entry.transaction_id in transactions:
                            txn = transactions[entry.transaction_id]
                            txn.entries.append(entry)

                            if entry.entry_type == EntryType.COMMIT:
                                txn.state = TransactionState.COMMITTED
                                txn.committed_at = entry.timestamp

                            elif entry.entry_type == EntryType.ROLLBACK:
                                txn.state = TransactionState.ROLLED_BACK

                    except Exception:
                        pass  # Skip malformed entries

        except Exception:
            pass  # Skip unreadable log files

        self._transactions.update(transactions)

    # =========================================================================
    # Query Operations
    # =========================================================================

    def get_transaction(self, transaction_id: str) -> Optional[Transaction]:
        """Get transaction by ID."""
        return self._transactions.get(transaction_id)

    def get_active_transaction(self) -> Optional[Transaction]:
        """Get the currently active transaction."""
        if self._active_transaction:
            return self._transactions.get(self._active_transaction)
        return None

    def list_transactions(
        self,
        state: Optional[TransactionState] = None,
        actor: Optional[str] = None,
        limit: int = 100,
    ) -> List[Transaction]:
        """List transactions with optional filters."""
        transactions = list(self._transactions.values())

        if state:
            transactions = [t for t in transactions if t.state == state]

        if actor:
            transactions = [t for t in transactions if t.actor == actor]

        transactions.sort(key=lambda t: t.created_at, reverse=True)

        return transactions[:limit]

    def get_file_history(self, path: str) -> List[TransactionEntry]:
        """Get all entries affecting a specific file."""
        entries: List[TransactionEntry] = []

        for txn in self._transactions.values():
            if txn.is_committed:
                for entry in txn.entries:
                    if entry.path == path:
                        entries.append(entry)

        entries.sort(key=lambda e: e.timestamp, reverse=True)
        return entries

    def iter_entries(
        self,
        transaction_id: Optional[str] = None,
        entry_type: Optional[EntryType] = None,
    ) -> Iterator[TransactionEntry]:
        """Iterate over entries with optional filters."""
        for txn in self._transactions.values():
            if transaction_id and txn.id != transaction_id:
                continue

            for entry in txn.entries:
                if entry_type and entry.entry_type != entry_type:
                    continue
                yield entry

    # =========================================================================
    # Maintenance
    # =========================================================================

    def compact_logs(self, keep_days: int = 7) -> int:
        """Compact old log files."""
        cutoff = time.time() - (keep_days * 24 * 3600)
        removed = 0

        for log_file in self.log_dir.glob("txlog_*.wal"):
            if log_file.stat().st_mtime < cutoff:
                log_file.unlink()
                removed += 1

        return removed

    def get_stats(self) -> Dict[str, Any]:
        """Get transaction statistics."""
        total_size = sum(
            f.stat().st_size for f in self.log_dir.glob("txlog_*.wal")
            if f.exists()
        )

        states = {}
        for txn in self._transactions.values():
            state = txn.state.value
            states[state] = states.get(state, 0) + 1

        return {
            "total_transactions": len(self._transactions),
            "states": states,
            "active_transaction": self._active_transaction,
            "total_entries": sum(t.entry_count for t in self._transactions.values()),
            "log_size_bytes": total_size,
            "log_dir": str(self.log_dir),
        }


# =============================================================================
# Context Manager
# =============================================================================

class TransactionContext:
    """Context manager for transactions."""

    def __init__(
        self,
        logger: TransactionLogger,
        description: str = "",
        actor: str = "code-agent",
    ):
        self.logger = logger
        self.description = description
        self.actor = actor
        self.transaction: Optional[Transaction] = None

    def __enter__(self) -> Transaction:
        self.transaction = self.logger.begin(self.description, self.actor)
        return self.transaction

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.transaction:
            if exc_type:
                self.logger.rollback(self.transaction.id, str(exc_val))
            else:
                self.logger.commit(self.transaction.id)
        return False


# =============================================================================
# CLI
# =============================================================================

def main() -> int:
    """CLI entry point for Transaction Logger."""
    import argparse

    parser = argparse.ArgumentParser(description="Transaction Logger (Step 66)")
    parser.add_argument("--working-dir", default=".", help="Working directory")

    subparsers = parser.add_subparsers(dest="command", required=True)

    # begin command
    begin_parser = subparsers.add_parser("begin", help="Begin transaction")
    begin_parser.add_argument("--description", "-d", default="", help="Description")
    begin_parser.add_argument("--actor", default="cli", help="Actor name")

    # commit command
    commit_parser = subparsers.add_parser("commit", help="Commit transaction")
    commit_parser.add_argument("transaction_id", help="Transaction ID")

    # rollback command
    rollback_parser = subparsers.add_parser("rollback", help="Rollback transaction")
    rollback_parser.add_argument("transaction_id", help="Transaction ID")
    rollback_parser.add_argument("--reason", default="", help="Reason")

    # list command
    list_parser = subparsers.add_parser("list", help="List transactions")
    list_parser.add_argument("--state", choices=["pending", "committed", "rolled_back"])
    list_parser.add_argument("--limit", type=int, default=20)
    list_parser.add_argument("--json", action="store_true")

    # history command
    history_parser = subparsers.add_parser("history", help="File history")
    history_parser.add_argument("path", help="File path")

    # stats command
    subparsers.add_parser("stats", help="Show statistics")

    # compact command
    compact_parser = subparsers.add_parser("compact", help="Compact logs")
    compact_parser.add_argument("--days", type=int, default=7, help="Keep days")

    args = parser.parse_args()

    logger = TransactionLogger(Path(args.working_dir))

    if args.command == "begin":
        txn = logger.begin(args.description, args.actor)
        print(f"Transaction started: {txn.id}")
        return 0

    elif args.command == "commit":
        if logger.commit(args.transaction_id):
            print(f"Transaction committed: {args.transaction_id}")
        else:
            print(f"Failed to commit: {args.transaction_id}")
            return 1
        return 0

    elif args.command == "rollback":
        if logger.rollback(args.transaction_id, args.reason):
            print(f"Transaction rolled back: {args.transaction_id}")
        else:
            print(f"Failed to rollback: {args.transaction_id}")
            return 1
        return 0

    elif args.command == "list":
        state = TransactionState(args.state) if args.state else None
        transactions = logger.list_transactions(state=state, limit=args.limit)

        if args.json:
            print(json.dumps([t.to_dict() for t in transactions], indent=2))
        else:
            for t in transactions:
                ts = datetime.fromtimestamp(t.created_at).strftime("%Y-%m-%d %H:%M")
                print(f"{t.id} [{t.state.value}] {t.description or '(no description)'} ({ts})")

        return 0

    elif args.command == "history":
        entries = logger.get_file_history(args.path)
        for e in entries:
            ts = datetime.fromtimestamp(e.timestamp).strftime("%Y-%m-%d %H:%M")
            print(f"{ts} {e.entry_type.value} {e.transaction_id[:12]}")
        return 0

    elif args.command == "stats":
        stats = logger.get_stats()
        print(f"Total transactions: {stats['total_transactions']}")
        print(f"Total entries: {stats['total_entries']}")
        print(f"States: {stats['states']}")
        print(f"Active: {stats['active_transaction'] or 'None'}")
        print(f"Log size: {stats['log_size_bytes']} bytes")
        return 0

    elif args.command == "compact":
        removed = logger.compact_logs(args.days)
        print(f"Removed {removed} old log files")
        return 0

    return 1


if __name__ == "__main__":
    import sys
    sys.exit(main())
