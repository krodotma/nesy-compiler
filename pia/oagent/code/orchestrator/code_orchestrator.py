#!/usr/bin/env python3
"""
code_orchestrator.py - Code Orchestrator (Step 70)

PBTSO Phase: All Phases

Provides:
- Coordination of all Code Agent components
- Edit pipeline management
- Component lifecycle orchestration
- Cross-component event routing
- Unified API for code operations

Bus Topics:
- a2a.code.orchestrator.*
- code.pipeline.*
- code.operation.*

Protocol: DKIN v30, PAIP v16, CITIZEN v2
"""

from __future__ import annotations

import asyncio
import time
import uuid
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Coroutine, Dict, List, Optional, Set


# =============================================================================
# Types
# =============================================================================

class PipelineStage(Enum):
    """Stages in the edit pipeline."""
    VALIDATE = "validate"       # Validate inputs
    LOCK = "lock"               # Acquire workspace locks
    CHECKPOINT = "checkpoint"   # Create checkpoint
    TRANSFORM = "transform"     # Apply AST transformations
    DIFF = "diff"               # Generate diffs
    MERGE = "merge"             # Merge changes
    COMPILE = "compile"         # Incremental compilation
    STYLE = "style"             # Style enforcement
    COMMIT = "commit"           # Commit transaction
    SYNC = "sync"               # Sync with workspace
    NOTIFY = "notify"           # Send notifications


class OperationType(Enum):
    """Type of code operation."""
    EDIT = "edit"               # Edit files
    REFACTOR = "refactor"       # Refactoring operation
    GENERATE = "generate"       # Code generation
    TRANSFORM = "transform"     # AST transformation
    MERGE = "merge"             # Merge operation
    UNDO = "undo"               # Undo operation
    REDO = "redo"               # Redo operation


class OperationStatus(Enum):
    """Status of an operation."""
    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    FAILED = "failed"
    ROLLED_BACK = "rolled_back"


@dataclass
class OrchestrationConfig:
    """Configuration for the orchestrator."""
    working_dir: str = "/pluribus"
    enable_checkpoints: bool = True
    enable_transactions: bool = True
    enable_workspace_sync: bool = True
    enable_file_watcher: bool = True
    auto_style_enforcement: bool = True
    auto_compilation: bool = True
    parallel_operations: int = 4
    operation_timeout_s: int = 300
    checkpoint_on_error: bool = True

    def to_dict(self) -> Dict[str, Any]:
        return {
            "working_dir": self.working_dir,
            "enable_checkpoints": self.enable_checkpoints,
            "enable_transactions": self.enable_transactions,
            "enable_workspace_sync": self.enable_workspace_sync,
            "enable_file_watcher": self.enable_file_watcher,
            "auto_style_enforcement": self.auto_style_enforcement,
            "auto_compilation": self.auto_compilation,
            "parallel_operations": self.parallel_operations,
            "operation_timeout_s": self.operation_timeout_s,
        }


@dataclass
class EditOperation:
    """Represents a code editing operation."""
    id: str
    operation_type: OperationType
    files: List[str]
    description: str
    status: OperationStatus = OperationStatus.PENDING
    created_at: float = field(default_factory=time.time)
    started_at: Optional[float] = None
    completed_at: Optional[float] = None
    checkpoint_id: Optional[str] = None
    transaction_id: Optional[str] = None
    lock_ids: List[str] = field(default_factory=list)
    result: Optional[Dict[str, Any]] = None
    error: Optional[str] = None
    stages_completed: List[PipelineStage] = field(default_factory=list)

    @property
    def duration_ms(self) -> Optional[float]:
        if self.started_at and self.completed_at:
            return (self.completed_at - self.started_at) * 1000
        return None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "operation_type": self.operation_type.value,
            "files": self.files,
            "description": self.description,
            "status": self.status.value,
            "created_at": self.created_at,
            "started_at": self.started_at,
            "completed_at": self.completed_at,
            "duration_ms": self.duration_ms,
            "checkpoint_id": self.checkpoint_id,
            "transaction_id": self.transaction_id,
            "stages_completed": [s.value for s in self.stages_completed],
            "error": self.error,
        }


@dataclass
class EditPipeline:
    """Configuration for an edit pipeline."""
    stages: List[PipelineStage]
    parallel_stages: Set[PipelineStage] = field(default_factory=set)
    skip_on_error: Set[PipelineStage] = field(default_factory=set)
    retry_stages: Set[PipelineStage] = field(default_factory=set)
    max_retries: int = 3


# =============================================================================
# Default Pipelines
# =============================================================================

DEFAULT_EDIT_PIPELINE = EditPipeline(
    stages=[
        PipelineStage.VALIDATE,
        PipelineStage.LOCK,
        PipelineStage.CHECKPOINT,
        PipelineStage.TRANSFORM,
        PipelineStage.DIFF,
        PipelineStage.STYLE,
        PipelineStage.COMPILE,
        PipelineStage.COMMIT,
        PipelineStage.SYNC,
        PipelineStage.NOTIFY,
    ],
    skip_on_error={PipelineStage.STYLE, PipelineStage.NOTIFY},
    retry_stages={PipelineStage.LOCK, PipelineStage.SYNC},
)

REFACTOR_PIPELINE = EditPipeline(
    stages=[
        PipelineStage.VALIDATE,
        PipelineStage.LOCK,
        PipelineStage.CHECKPOINT,
        PipelineStage.TRANSFORM,
        PipelineStage.MERGE,
        PipelineStage.DIFF,
        PipelineStage.STYLE,
        PipelineStage.COMPILE,
        PipelineStage.COMMIT,
        PipelineStage.SYNC,
        PipelineStage.NOTIFY,
    ],
)

UNDO_PIPELINE = EditPipeline(
    stages=[
        PipelineStage.VALIDATE,
        PipelineStage.LOCK,
        PipelineStage.CHECKPOINT,
        PipelineStage.COMMIT,
        PipelineStage.SYNC,
        PipelineStage.NOTIFY,
    ],
)


# =============================================================================
# Code Orchestrator
# =============================================================================

class CodeOrchestrator:
    """
    Central orchestrator for all Code Agent components.

    PBTSO Phase: All Phases

    Responsibilities:
    - Coordinate component lifecycle
    - Manage edit pipelines
    - Route events between components
    - Handle operation execution
    - Provide unified API

    Components Coordinated:
    - DiffOptimizer (Step 61)
    - ConflictResolver (Step 62)
    - SemanticMerger (Step 63)
    - RollbackManager (Step 64)
    - CheckpointSystem (Step 65)
    - TransactionLogger (Step 66)
    - UndoRedoStack (Step 67)
    - WorkspaceSync (Step 68)
    - FileWatcher (Step 69)

    Usage:
        orchestrator = CodeOrchestrator(config)
        await orchestrator.initialize()
        result = await orchestrator.execute_edit(operation)
        await orchestrator.shutdown()
    """

    BUS_TOPICS = {
        "initialized": "a2a.code.orchestrator.initialized",
        "shutdown": "a2a.code.orchestrator.shutdown",
        "operation_start": "code.operation.start",
        "operation_complete": "code.operation.complete",
        "operation_failed": "code.operation.failed",
        "stage_start": "code.pipeline.stage.start",
        "stage_complete": "code.pipeline.stage.complete",
    }

    def __init__(
        self,
        config: Optional[OrchestrationConfig] = None,
        bus: Optional[Any] = None,
    ):
        self.config = config or OrchestrationConfig()
        self.bus = bus
        self.working_dir = Path(self.config.working_dir)

        # Components (initialized lazily)
        self._diff_optimizer = None
        self._conflict_resolver = None
        self._semantic_merger = None
        self._rollback_manager = None
        self._checkpoint_system = None
        self._transaction_logger = None
        self._undo_redo_stack = None
        self._workspace_sync = None
        self._file_watcher = None

        # State
        self._initialized = False
        self._operations: Dict[str, EditOperation] = {}
        self._active_operations: Set[str] = set()
        self._operation_semaphore: Optional[asyncio.Semaphore] = None

        # Pipelines
        self._pipelines: Dict[OperationType, EditPipeline] = {
            OperationType.EDIT: DEFAULT_EDIT_PIPELINE,
            OperationType.REFACTOR: REFACTOR_PIPELINE,
            OperationType.UNDO: UNDO_PIPELINE,
            OperationType.REDO: UNDO_PIPELINE,
        }

        # Stage handlers
        self._stage_handlers: Dict[PipelineStage, Callable] = {}

    # =========================================================================
    # Lifecycle
    # =========================================================================

    async def initialize(self) -> bool:
        """
        Initialize the orchestrator and all components.

        Returns:
            True if initialization successful
        """
        if self._initialized:
            return True

        try:
            # Import components
            from ..diff import DiffOptimizer
            from ..conflict import ConflictResolver
            from ..merge import SemanticMerger
            from ..rollback import RollbackManager
            from ..checkpoint import CheckpointSystem
            from ..transaction import TransactionLogger
            from ..undo import UndoRedoStack
            from ..workspace import WorkspaceSync
            from ..watcher import FileWatcher

            # Initialize components
            self._diff_optimizer = DiffOptimizer(bus=self.bus)
            self._conflict_resolver = ConflictResolver(bus=self.bus)
            self._semantic_merger = SemanticMerger(bus=self.bus)

            if self.config.enable_checkpoints:
                self._rollback_manager = RollbackManager(self.working_dir, bus=self.bus)
                self._checkpoint_system = CheckpointSystem(self.working_dir, bus=self.bus)

            if self.config.enable_transactions:
                self._transaction_logger = TransactionLogger(self.working_dir, bus=self.bus)

            self._undo_redo_stack = UndoRedoStack(self.working_dir, bus=self.bus)

            if self.config.enable_workspace_sync:
                self._workspace_sync = WorkspaceSync(
                    self.working_dir,
                    agent_id="code-orchestrator",
                    bus=self.bus,
                )

            if self.config.enable_file_watcher:
                self._file_watcher = FileWatcher(self.working_dir, bus=self.bus)
                await self._file_watcher.start()

            # Initialize operation semaphore
            self._operation_semaphore = asyncio.Semaphore(self.config.parallel_operations)

            # Register stage handlers
            self._register_stage_handlers()

            self._initialized = True

            # Emit initialized event
            if self.bus:
                self.bus.emit({
                    "topic": self.BUS_TOPICS["initialized"],
                    "kind": "orchestrator",
                    "actor": "code-orchestrator",
                    "data": {
                        "components": self._get_component_status(),
                        "config": self.config.to_dict(),
                    },
                })

            return True

        except Exception as e:
            if self.bus:
                self.bus.emit({
                    "topic": self.BUS_TOPICS["operation_failed"],
                    "kind": "orchestrator",
                    "actor": "code-orchestrator",
                    "data": {"error": f"Initialization failed: {e}"},
                })
            return False

    async def shutdown(self) -> None:
        """Shutdown the orchestrator and all components."""
        if not self._initialized:
            return

        # Stop file watcher
        if self._file_watcher:
            await self._file_watcher.stop()

        # Clean up components
        self._diff_optimizer = None
        self._conflict_resolver = None
        self._semantic_merger = None
        self._rollback_manager = None
        self._checkpoint_system = None
        self._transaction_logger = None
        self._undo_redo_stack = None
        self._workspace_sync = None
        self._file_watcher = None

        self._initialized = False

        # Emit shutdown event
        if self.bus:
            self.bus.emit({
                "topic": self.BUS_TOPICS["shutdown"],
                "kind": "orchestrator",
                "actor": "code-orchestrator",
                "data": {
                    "operations_completed": len(self._operations),
                },
            })

    def _get_component_status(self) -> Dict[str, bool]:
        """Get status of all components."""
        return {
            "diff_optimizer": self._diff_optimizer is not None,
            "conflict_resolver": self._conflict_resolver is not None,
            "semantic_merger": self._semantic_merger is not None,
            "rollback_manager": self._rollback_manager is not None,
            "checkpoint_system": self._checkpoint_system is not None,
            "transaction_logger": self._transaction_logger is not None,
            "undo_redo_stack": self._undo_redo_stack is not None,
            "workspace_sync": self._workspace_sync is not None,
            "file_watcher": self._file_watcher is not None,
        }

    def _register_stage_handlers(self) -> None:
        """Register handlers for each pipeline stage."""
        self._stage_handlers = {
            PipelineStage.VALIDATE: self._handle_validate,
            PipelineStage.LOCK: self._handle_lock,
            PipelineStage.CHECKPOINT: self._handle_checkpoint,
            PipelineStage.TRANSFORM: self._handle_transform,
            PipelineStage.DIFF: self._handle_diff,
            PipelineStage.MERGE: self._handle_merge,
            PipelineStage.COMPILE: self._handle_compile,
            PipelineStage.STYLE: self._handle_style,
            PipelineStage.COMMIT: self._handle_commit,
            PipelineStage.SYNC: self._handle_sync,
            PipelineStage.NOTIFY: self._handle_notify,
        }

    # =========================================================================
    # Operation Execution
    # =========================================================================

    async def execute_edit(
        self,
        files: List[str],
        changes: Dict[str, str],
        description: str = "",
        operation_type: OperationType = OperationType.EDIT,
    ) -> EditOperation:
        """
        Execute an edit operation through the pipeline.

        Args:
            files: Files to edit
            changes: Dict of path -> new content
            description: Operation description
            operation_type: Type of operation

        Returns:
            EditOperation with results
        """
        if not self._initialized:
            await self.initialize()

        operation = EditOperation(
            id=f"op-{uuid.uuid4().hex[:12]}",
            operation_type=operation_type,
            files=files,
            description=description,
        )

        self._operations[operation.id] = operation

        # Execute with semaphore for parallelism control
        async with self._operation_semaphore:
            self._active_operations.add(operation.id)
            try:
                await self._execute_pipeline(operation, changes)
            finally:
                self._active_operations.discard(operation.id)

        return operation

    async def _execute_pipeline(
        self,
        operation: EditOperation,
        changes: Dict[str, str],
    ) -> None:
        """Execute an operation through its pipeline."""
        pipeline = self._pipelines.get(operation.operation_type, DEFAULT_EDIT_PIPELINE)

        operation.status = OperationStatus.IN_PROGRESS
        operation.started_at = time.time()

        # Emit operation start
        if self.bus:
            self.bus.emit({
                "topic": self.BUS_TOPICS["operation_start"],
                "kind": "operation",
                "actor": "code-orchestrator",
                "data": operation.to_dict(),
            })

        context = {
            "operation": operation,
            "changes": changes,
            "working_dir": self.working_dir,
        }

        try:
            for stage in pipeline.stages:
                # Skip stages on error if configured
                if operation.status == OperationStatus.FAILED:
                    if stage in pipeline.skip_on_error:
                        continue
                    break

                await self._execute_stage(operation, stage, context, pipeline)

            if operation.status != OperationStatus.FAILED:
                operation.status = OperationStatus.COMPLETED
                operation.completed_at = time.time()

                # Emit success
                if self.bus:
                    self.bus.emit({
                        "topic": self.BUS_TOPICS["operation_complete"],
                        "kind": "operation",
                        "actor": "code-orchestrator",
                        "data": operation.to_dict(),
                    })

        except Exception as e:
            operation.status = OperationStatus.FAILED
            operation.error = str(e)
            operation.completed_at = time.time()

            # Attempt rollback if configured
            if self.config.checkpoint_on_error and operation.checkpoint_id:
                await self._rollback_operation(operation)

            # Emit failure
            if self.bus:
                self.bus.emit({
                    "topic": self.BUS_TOPICS["operation_failed"],
                    "kind": "operation",
                    "actor": "code-orchestrator",
                    "data": {
                        **operation.to_dict(),
                        "error": str(e),
                    },
                })

        finally:
            # Release locks
            await self._release_locks(operation)

    async def _execute_stage(
        self,
        operation: EditOperation,
        stage: PipelineStage,
        context: Dict[str, Any],
        pipeline: EditPipeline,
    ) -> None:
        """Execute a single pipeline stage."""
        handler = self._stage_handlers.get(stage)
        if not handler:
            return

        # Emit stage start
        if self.bus:
            self.bus.emit({
                "topic": self.BUS_TOPICS["stage_start"],
                "kind": "stage",
                "actor": "code-orchestrator",
                "data": {
                    "operation_id": operation.id,
                    "stage": stage.value,
                },
            })

        retries = 0
        max_retries = pipeline.max_retries if stage in pipeline.retry_stages else 1

        while retries < max_retries:
            try:
                await handler(operation, context)
                operation.stages_completed.append(stage)

                # Emit stage complete
                if self.bus:
                    self.bus.emit({
                        "topic": self.BUS_TOPICS["stage_complete"],
                        "kind": "stage",
                        "actor": "code-orchestrator",
                        "data": {
                            "operation_id": operation.id,
                            "stage": stage.value,
                            "retries": retries,
                        },
                    })

                return

            except Exception as e:
                retries += 1
                if retries >= max_retries:
                    operation.status = OperationStatus.FAILED
                    operation.error = f"Stage {stage.value} failed: {e}"
                    raise
                await asyncio.sleep(0.1 * retries)  # Exponential backoff

    # =========================================================================
    # Stage Handlers
    # =========================================================================

    async def _handle_validate(
        self,
        operation: EditOperation,
        context: Dict[str, Any],
    ) -> None:
        """Validate operation inputs."""
        changes = context.get("changes", {})

        # Validate files exist
        for path in operation.files:
            full_path = self.working_dir / path
            if path not in changes and not full_path.exists():
                raise ValueError(f"File not found: {path}")

        # Validate content
        for path, content in changes.items():
            if not isinstance(content, str):
                raise ValueError(f"Invalid content for {path}")

    async def _handle_lock(
        self,
        operation: EditOperation,
        context: Dict[str, Any],
    ) -> None:
        """Acquire workspace locks."""
        if not self._workspace_sync:
            return

        from ..workspace import LockType

        lock = self._workspace_sync.acquire_lock(
            operation.files,
            LockType.EXCLUSIVE,
            reason=operation.description,
            wait=True,
            wait_timeout_s=30,
        )

        if not lock:
            raise RuntimeError("Could not acquire locks")

        operation.lock_ids.append(lock.id)

    async def _handle_checkpoint(
        self,
        operation: EditOperation,
        context: Dict[str, Any],
    ) -> None:
        """Create checkpoint before changes."""
        if not self._checkpoint_system:
            return

        checkpoint = self._checkpoint_system.create(
            name=f"pre_{operation.id}",
            message=operation.description,
        )

        operation.checkpoint_id = checkpoint.id

    async def _handle_transform(
        self,
        operation: EditOperation,
        context: Dict[str, Any],
    ) -> None:
        """Apply changes to files."""
        changes = context.get("changes", {})

        for path, content in changes.items():
            full_path = self.working_dir / path
            full_path.parent.mkdir(parents=True, exist_ok=True)
            full_path.write_text(content)

        # Track in undo stack
        if self._undo_redo_stack:
            from ..undo import create_action_from_edit, ActionType

            for path, new_content in changes.items():
                full_path = self.working_dir / path
                old_content = ""
                if full_path.exists():
                    old_content = full_path.read_text()

                action = create_action_from_edit(
                    path=path,
                    old_content=old_content,
                    new_content=new_content,
                    description=operation.description,
                )
                self._undo_redo_stack.push(action)

    async def _handle_diff(
        self,
        operation: EditOperation,
        context: Dict[str, Any],
    ) -> None:
        """Generate diffs for changes."""
        if not self._diff_optimizer:
            return

        changes = context.get("changes", {})
        diffs = []

        for path, new_content in changes.items():
            full_path = self.working_dir / path
            old_content = ""
            if full_path.exists():
                old_content = full_path.read_text()

            diff = self._diff_optimizer.generate_diff(
                old_content, new_content, path
            )
            diffs.append(diff)

        context["diffs"] = diffs

    async def _handle_merge(
        self,
        operation: EditOperation,
        context: Dict[str, Any],
    ) -> None:
        """Handle merge operations."""
        if not self._semantic_merger:
            return

        # Merge logic would go here for refactoring operations
        pass

    async def _handle_compile(
        self,
        operation: EditOperation,
        context: Dict[str, Any],
    ) -> None:
        """Run incremental compilation."""
        if not self.config.auto_compilation:
            return

        # Would integrate with incremental compiler here
        pass

    async def _handle_style(
        self,
        operation: EditOperation,
        context: Dict[str, Any],
    ) -> None:
        """Enforce code style."""
        if not self.config.auto_style_enforcement:
            return

        # Would integrate with style enforcer here
        pass

    async def _handle_commit(
        self,
        operation: EditOperation,
        context: Dict[str, Any],
    ) -> None:
        """Commit transaction."""
        if not self._transaction_logger:
            return

        if operation.transaction_id:
            self._transaction_logger.commit(operation.transaction_id)

    async def _handle_sync(
        self,
        operation: EditOperation,
        context: Dict[str, Any],
    ) -> None:
        """Sync changes to workspace."""
        if not self._workspace_sync:
            return

        from ..workspace import ChangeType

        changes = []
        for path in operation.files:
            change = self._workspace_sync.track_file_change(
                path,
                ChangeType.MODIFY,
            )
            changes.append(change)

        self._workspace_sync.push_changes(changes)

    async def _handle_notify(
        self,
        operation: EditOperation,
        context: Dict[str, Any],
    ) -> None:
        """Send notifications."""
        # Would send notifications to other agents here
        pass

    async def _release_locks(self, operation: EditOperation) -> None:
        """Release all locks held by operation."""
        if not self._workspace_sync:
            return

        for lock_id in operation.lock_ids:
            self._workspace_sync.release_lock(lock_id)

        operation.lock_ids = []

    async def _rollback_operation(self, operation: EditOperation) -> None:
        """Rollback an operation on failure."""
        if not self._rollback_manager or not operation.checkpoint_id:
            return

        try:
            self._rollback_manager.rollback(operation.checkpoint_id)
            operation.status = OperationStatus.ROLLED_BACK
        except Exception:
            pass

    # =========================================================================
    # Convenience Methods
    # =========================================================================

    async def undo(self, count: int = 1) -> List[Any]:
        """Undo recent operations."""
        if not self._undo_redo_stack:
            return []
        return self._undo_redo_stack.undo(count)

    async def redo(self, count: int = 1) -> List[Any]:
        """Redo undone operations."""
        if not self._undo_redo_stack:
            return []
        return self._undo_redo_stack.redo(count)

    def get_operation(self, operation_id: str) -> Optional[EditOperation]:
        """Get an operation by ID."""
        return self._operations.get(operation_id)

    def list_operations(self, limit: int = 50) -> List[EditOperation]:
        """List recent operations."""
        ops = list(self._operations.values())
        ops.sort(key=lambda o: o.created_at, reverse=True)
        return ops[:limit]

    def get_stats(self) -> Dict[str, Any]:
        """Get orchestrator statistics."""
        return {
            "initialized": self._initialized,
            "components": self._get_component_status(),
            "total_operations": len(self._operations),
            "active_operations": len(self._active_operations),
            "config": self.config.to_dict(),
        }


# =============================================================================
# CLI
# =============================================================================

def main() -> int:
    """CLI entry point for Code Orchestrator."""
    import argparse
    import json

    parser = argparse.ArgumentParser(description="Code Orchestrator (Step 70)")
    parser.add_argument("--working-dir", default=".", help="Working directory")

    subparsers = parser.add_subparsers(dest="command", required=True)

    # edit command
    edit_parser = subparsers.add_parser("edit", help="Execute edit operation")
    edit_parser.add_argument("file", help="File to edit")
    edit_parser.add_argument("--content", help="New content (or read from stdin)")
    edit_parser.add_argument("--description", "-d", default="", help="Description")

    # undo command
    undo_parser = subparsers.add_parser("undo", help="Undo operations")
    undo_parser.add_argument("--count", "-n", type=int, default=1)

    # redo command
    redo_parser = subparsers.add_parser("redo", help="Redo operations")
    redo_parser.add_argument("--count", "-n", type=int, default=1)

    # status command
    subparsers.add_parser("status", help="Show orchestrator status")

    # operations command
    ops_parser = subparsers.add_parser("operations", help="List operations")
    ops_parser.add_argument("--limit", type=int, default=20)
    ops_parser.add_argument("--json", action="store_true")

    args = parser.parse_args()

    config = OrchestrationConfig(working_dir=args.working_dir)
    orchestrator = CodeOrchestrator(config)

    async def run():
        await orchestrator.initialize()

        if args.command == "edit":
            content = args.content
            if content is None:
                import sys
                content = sys.stdin.read()

            operation = await orchestrator.execute_edit(
                files=[args.file],
                changes={args.file: content},
                description=args.description,
            )

            print(f"Operation {operation.id}: {operation.status.value}")
            if operation.error:
                print(f"  Error: {operation.error}")

        elif args.command == "undo":
            undone = await orchestrator.undo(args.count)
            print(f"Undid {len(undone)} operations")

        elif args.command == "redo":
            redone = await orchestrator.redo(args.count)
            print(f"Redid {len(redone)} operations")

        elif args.command == "status":
            stats = orchestrator.get_stats()
            print(f"Initialized: {stats['initialized']}")
            print(f"Operations: {stats['total_operations']}")
            print(f"Active: {stats['active_operations']}")
            print("Components:")
            for name, status in stats['components'].items():
                status_str = "OK" if status else "Not initialized"
                print(f"  {name}: {status_str}")

        elif args.command == "operations":
            operations = orchestrator.list_operations(args.limit)
            if args.json:
                print(json.dumps([o.to_dict() for o in operations], indent=2))
            else:
                for op in operations:
                    print(f"{op.id} [{op.status.value}] {op.operation_type.value}: {op.description or '(no description)'}")

        await orchestrator.shutdown()

    asyncio.run(run())
    return 0


if __name__ == "__main__":
    import sys
    sys.exit(main())
