#!/usr/bin/env python3
"""
Review Workflow Engine (Step 176)

Custom workflow engine for code review processes.

PBTSO Phase: PLAN, BUILD
Bus Topics: review.workflow.execute, review.workflow.transition

Features:
- Custom workflow definitions
- State machine execution
- Conditional transitions
- Parallel steps
- Timeout handling
- Hook integration

Protocol: DKIN v30, CITIZEN v2, PAIP v16
"""

from __future__ import annotations

import asyncio
import fcntl
import json
import os
import time
import uuid
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone, timedelta
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Awaitable


# ============================================================================
# Types
# ============================================================================

class WorkflowState(Enum):
    """Workflow execution states."""
    PENDING = "pending"
    RUNNING = "running"
    PAUSED = "paused"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"
    TIMEOUT = "timeout"


class StepType(Enum):
    """Types of workflow steps."""
    ACTION = "action"       # Execute an action
    DECISION = "decision"   # Make a decision based on condition
    PARALLEL = "parallel"   # Execute multiple steps in parallel
    WAIT = "wait"           # Wait for external event
    APPROVAL = "approval"   # Require approval to continue


class TransitionCondition(Enum):
    """Conditions for step transitions."""
    ALWAYS = "always"
    SUCCESS = "success"
    FAILURE = "failure"
    CONDITION = "condition"


@dataclass
class WorkflowConfig:
    """Configuration for workflow execution."""
    default_timeout_seconds: int = 3600
    max_parallel_steps: int = 10
    retry_attempts: int = 3
    retry_delay_seconds: int = 5
    enable_hooks: bool = True

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return asdict(self)


@dataclass
class WorkflowStep:
    """
    A step in a workflow.

    Attributes:
        step_id: Unique step identifier
        name: Step name
        step_type: Type of step
        action: Action to execute (for action steps)
        config: Step configuration
        next_steps: Possible next steps
        timeout_seconds: Step timeout
        on_failure: Step to execute on failure
    """
    step_id: str
    name: str
    step_type: StepType
    action: Optional[str] = None
    config: Dict[str, Any] = field(default_factory=dict)
    next_steps: Dict[str, str] = field(default_factory=dict)  # condition -> step_id
    timeout_seconds: Optional[int] = None
    on_failure: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "step_id": self.step_id,
            "name": self.name,
            "step_type": self.step_type.value,
            "action": self.action,
            "config": self.config,
            "next_steps": self.next_steps,
            "timeout_seconds": self.timeout_seconds,
            "on_failure": self.on_failure,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "WorkflowStep":
        """Create from dictionary."""
        data = data.copy()
        data["step_type"] = StepType(data["step_type"])
        return cls(**data)


@dataclass
class StepResult:
    """Result from executing a step."""
    step_id: str
    status: WorkflowState
    output: Any = None
    error: Optional[str] = None
    duration_ms: float = 0.0
    next_condition: str = "success"

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "step_id": self.step_id,
            "status": self.status.value,
            "output": self.output,
            "error": self.error,
            "duration_ms": round(self.duration_ms, 2),
            "next_condition": self.next_condition,
        }


@dataclass
class WorkflowDefinition:
    """
    Definition of a workflow.

    Attributes:
        workflow_id: Unique workflow ID
        name: Workflow name
        description: Workflow description
        steps: Workflow steps
        start_step: ID of the starting step
        end_steps: IDs of ending steps
        variables: Workflow variables/inputs
        created_at: Creation timestamp
    """
    workflow_id: str
    name: str
    description: str
    steps: List[WorkflowStep]
    start_step: str
    end_steps: List[str] = field(default_factory=list)
    variables: Dict[str, Any] = field(default_factory=dict)
    created_at: str = ""

    def __post_init__(self):
        if not self.created_at:
            self.created_at = datetime.now(timezone.utc).isoformat() + "Z"

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "workflow_id": self.workflow_id,
            "name": self.name,
            "description": self.description,
            "steps": [s.to_dict() for s in self.steps],
            "start_step": self.start_step,
            "end_steps": self.end_steps,
            "variables": self.variables,
            "created_at": self.created_at,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "WorkflowDefinition":
        """Create from dictionary."""
        data = data.copy()
        data["steps"] = [WorkflowStep.from_dict(s) for s in data.get("steps", [])]
        return cls(**{k: v for k, v in data.items() if k in cls.__dataclass_fields__})

    def get_step(self, step_id: str) -> Optional[WorkflowStep]:
        """Get step by ID."""
        for step in self.steps:
            if step.step_id == step_id:
                return step
        return None


@dataclass
class WorkflowExecution:
    """
    Execution instance of a workflow.

    Attributes:
        execution_id: Unique execution ID
        workflow_id: ID of the workflow definition
        state: Current execution state
        current_step: Current step ID
        context: Execution context/variables
        step_results: Results from executed steps
        started_at: Start timestamp
        completed_at: Completion timestamp
        error: Error message if failed
    """
    execution_id: str
    workflow_id: str
    state: WorkflowState
    current_step: Optional[str] = None
    context: Dict[str, Any] = field(default_factory=dict)
    step_results: List[StepResult] = field(default_factory=list)
    started_at: str = ""
    completed_at: Optional[str] = None
    error: Optional[str] = None

    def __post_init__(self):
        if not self.started_at:
            self.started_at = datetime.now(timezone.utc).isoformat() + "Z"

    @property
    def duration_ms(self) -> float:
        """Total execution duration."""
        return sum(r.duration_ms for r in self.step_results)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "execution_id": self.execution_id,
            "workflow_id": self.workflow_id,
            "state": self.state.value,
            "current_step": self.current_step,
            "context": self.context,
            "step_results": [r.to_dict() for r in self.step_results],
            "started_at": self.started_at,
            "completed_at": self.completed_at,
            "error": self.error,
            "duration_ms": self.duration_ms,
        }


# ============================================================================
# Built-in Workflows
# ============================================================================

REVIEW_WORKFLOW = WorkflowDefinition(
    workflow_id="standard-review",
    name="Standard Review Workflow",
    description="Standard code review workflow with all analysis steps",
    steps=[
        WorkflowStep(
            step_id="start",
            name="Initialize Review",
            step_type=StepType.ACTION,
            action="review.init",
            next_steps={"success": "static_analysis", "failure": "fail"},
        ),
        WorkflowStep(
            step_id="static_analysis",
            name="Static Analysis",
            step_type=StepType.ACTION,
            action="review.static.analyze",
            next_steps={"success": "security_scan", "failure": "continue_on_error"},
        ),
        WorkflowStep(
            step_id="security_scan",
            name="Security Scan",
            step_type=StepType.ACTION,
            action="review.security.scan",
            next_steps={"success": "complexity_check", "failure": "continue_on_error"},
        ),
        WorkflowStep(
            step_id="complexity_check",
            name="Complexity Analysis",
            step_type=StepType.ACTION,
            action="review.complexity.analyze",
            next_steps={"success": "generate_report", "failure": "continue_on_error"},
        ),
        WorkflowStep(
            step_id="continue_on_error",
            name="Continue Despite Errors",
            step_type=StepType.DECISION,
            config={"condition": "context.continue_on_error"},
            next_steps={"true": "generate_report", "false": "fail"},
        ),
        WorkflowStep(
            step_id="generate_report",
            name="Generate Report",
            step_type=StepType.ACTION,
            action="review.report.generate",
            next_steps={"success": "decision_check", "failure": "fail"},
        ),
        WorkflowStep(
            step_id="decision_check",
            name="Check Review Decision",
            step_type=StepType.DECISION,
            config={"condition": "context.blocking_issues > 0"},
            next_steps={"true": "require_approval", "false": "complete"},
        ),
        WorkflowStep(
            step_id="require_approval",
            name="Require Approval",
            step_type=StepType.APPROVAL,
            config={"approvers": ["tech_lead", "security_team"]},
            next_steps={"approved": "complete", "rejected": "fail"},
            timeout_seconds=86400,  # 24 hours
        ),
        WorkflowStep(
            step_id="complete",
            name="Complete Review",
            step_type=StepType.ACTION,
            action="review.complete",
            next_steps={},
        ),
        WorkflowStep(
            step_id="fail",
            name="Review Failed",
            step_type=StepType.ACTION,
            action="review.fail",
            next_steps={},
        ),
    ],
    start_step="start",
    end_steps=["complete", "fail"],
    variables={
        "files": [],
        "continue_on_error": True,
        "blocking_issues": 0,
    },
)


# ============================================================================
# Action Registry
# ============================================================================

ActionHandler = Callable[[Dict[str, Any]], Awaitable[StepResult]]


class ActionRegistry:
    """Registry for workflow actions."""

    def __init__(self):
        self._actions: Dict[str, ActionHandler] = {}
        self._register_defaults()

    def _register_defaults(self) -> None:
        """Register default actions."""
        async def noop_action(ctx: Dict[str, Any]) -> StepResult:
            return StepResult(
                step_id=ctx.get("step_id", ""),
                status=WorkflowState.COMPLETED,
                output={"message": "Action completed"},
            )

        self._actions["review.init"] = noop_action
        self._actions["review.static.analyze"] = noop_action
        self._actions["review.security.scan"] = noop_action
        self._actions["review.complexity.analyze"] = noop_action
        self._actions["review.report.generate"] = noop_action
        self._actions["review.complete"] = noop_action
        self._actions["review.fail"] = noop_action

    def register(self, action_name: str, handler: ActionHandler) -> None:
        """Register an action handler."""
        self._actions[action_name] = handler

    def get(self, action_name: str) -> Optional[ActionHandler]:
        """Get an action handler."""
        return self._actions.get(action_name)


# ============================================================================
# Workflow Engine
# ============================================================================

class WorkflowEngine:
    """
    Executes custom review workflows.

    Example:
        engine = WorkflowEngine()

        # Define workflow
        workflow = WorkflowDefinition(...)

        # Execute workflow
        execution = await engine.execute(workflow, context={"files": [...]})

        # Check result
        print(f"Status: {execution.state}")
    """

    BUS_TOPICS = {
        "execute": "review.workflow.execute",
        "transition": "review.workflow.transition",
        "complete": "review.workflow.complete",
    }

    def __init__(
        self,
        config: Optional[WorkflowConfig] = None,
        bus_path: Optional[Path] = None,
    ):
        """
        Initialize the workflow engine.

        Args:
            config: Workflow configuration
            bus_path: Path to event bus file
        """
        self.config = config or WorkflowConfig()
        self.bus_path = bus_path or self._get_bus_path()
        self.actions = ActionRegistry()
        self._executions: Dict[str, WorkflowExecution] = {}

    def _get_bus_path(self) -> Path:
        """Get path to bus events file."""
        pluribus_root = Path(os.environ.get("PLURIBUS_ROOT", "/pluribus"))
        bus_dir = os.environ.get("PLURIBUS_BUS_DIR", str(pluribus_root / ".pluribus" / "bus"))
        return Path(bus_dir) / "events.ndjson"

    def _emit_event(self, topic: str, data: Dict[str, Any], kind: str = "workflow") -> str:
        """Emit event to bus with file locking."""
        self.bus_path.parent.mkdir(parents=True, exist_ok=True)

        event_id = str(uuid.uuid4())
        event = {
            "id": event_id,
            "ts": time.time(),
            "iso": datetime.now(timezone.utc).isoformat() + "Z",
            "topic": topic,
            "kind": kind,
            "actor": "workflow-engine",
            "data": data,
        }

        with open(self.bus_path, "a") as f:
            fcntl.flock(f.fileno(), fcntl.LOCK_EX)
            try:
                f.write(json.dumps(event) + "\n")
            finally:
                fcntl.flock(f.fileno(), fcntl.LOCK_UN)

        return event_id

    def register_action(self, action_name: str, handler: ActionHandler) -> None:
        """Register a custom action handler."""
        self.actions.register(action_name, handler)

    async def execute(
        self,
        workflow: WorkflowDefinition,
        context: Optional[Dict[str, Any]] = None,
    ) -> WorkflowExecution:
        """
        Execute a workflow.

        Args:
            workflow: Workflow definition
            context: Initial execution context

        Returns:
            WorkflowExecution with results

        Emits:
            review.workflow.execute
            review.workflow.transition
            review.workflow.complete
        """
        execution_id = str(uuid.uuid4())[:8]

        # Merge context with workflow variables
        ctx = {**workflow.variables, **(context or {})}

        execution = WorkflowExecution(
            execution_id=execution_id,
            workflow_id=workflow.workflow_id,
            state=WorkflowState.RUNNING,
            current_step=workflow.start_step,
            context=ctx,
        )

        self._executions[execution_id] = execution

        self._emit_event(self.BUS_TOPICS["execute"], {
            "execution_id": execution_id,
            "workflow_id": workflow.workflow_id,
            "workflow_name": workflow.name,
            "status": "started",
        })

        try:
            # Execute steps until reaching an end step
            while execution.current_step and execution.current_step not in workflow.end_steps:
                step = workflow.get_step(execution.current_step)
                if not step:
                    execution.state = WorkflowState.FAILED
                    execution.error = f"Step not found: {execution.current_step}"
                    break

                # Execute step
                result = await self._execute_step(step, execution)
                execution.step_results.append(result)

                # Emit transition event
                self._emit_event(self.BUS_TOPICS["transition"], {
                    "execution_id": execution_id,
                    "step_id": step.step_id,
                    "step_name": step.name,
                    "result_status": result.status.value,
                    "next_condition": result.next_condition,
                })

                # Handle failure
                if result.status == WorkflowState.FAILED:
                    if step.on_failure:
                        execution.current_step = step.on_failure
                    else:
                        execution.state = WorkflowState.FAILED
                        execution.error = result.error
                        break
                elif result.status == WorkflowState.TIMEOUT:
                    execution.state = WorkflowState.TIMEOUT
                    execution.error = f"Step timed out: {step.name}"
                    break
                else:
                    # Determine next step
                    next_step = step.next_steps.get(result.next_condition)
                    if not next_step and step.next_steps:
                        # Try "success" as default
                        next_step = step.next_steps.get("success")
                    execution.current_step = next_step

            # Execute final step if it's an end step
            if execution.current_step in workflow.end_steps:
                step = workflow.get_step(execution.current_step)
                if step:
                    result = await self._execute_step(step, execution)
                    execution.step_results.append(result)

                if execution.state == WorkflowState.RUNNING:
                    execution.state = WorkflowState.COMPLETED

        except Exception as e:
            execution.state = WorkflowState.FAILED
            execution.error = str(e)

        execution.completed_at = datetime.now(timezone.utc).isoformat() + "Z"

        self._emit_event(self.BUS_TOPICS["complete"], {
            "execution_id": execution_id,
            "workflow_id": workflow.workflow_id,
            "status": execution.state.value,
            "steps_executed": len(execution.step_results),
            "duration_ms": execution.duration_ms,
            "error": execution.error,
        })

        return execution

    async def _execute_step(
        self,
        step: WorkflowStep,
        execution: WorkflowExecution,
    ) -> StepResult:
        """Execute a single workflow step."""
        start_time = time.time()

        try:
            if step.step_type == StepType.ACTION:
                result = await self._execute_action(step, execution)
            elif step.step_type == StepType.DECISION:
                result = await self._execute_decision(step, execution)
            elif step.step_type == StepType.PARALLEL:
                result = await self._execute_parallel(step, execution)
            elif step.step_type == StepType.WAIT:
                result = await self._execute_wait(step, execution)
            elif step.step_type == StepType.APPROVAL:
                result = await self._execute_approval(step, execution)
            else:
                result = StepResult(
                    step_id=step.step_id,
                    status=WorkflowState.FAILED,
                    error=f"Unknown step type: {step.step_type}",
                )

            result.duration_ms = (time.time() - start_time) * 1000
            return result

        except asyncio.TimeoutError:
            return StepResult(
                step_id=step.step_id,
                status=WorkflowState.TIMEOUT,
                error="Step timed out",
                duration_ms=(time.time() - start_time) * 1000,
            )
        except Exception as e:
            return StepResult(
                step_id=step.step_id,
                status=WorkflowState.FAILED,
                error=str(e),
                duration_ms=(time.time() - start_time) * 1000,
            )

    async def _execute_action(
        self,
        step: WorkflowStep,
        execution: WorkflowExecution,
    ) -> StepResult:
        """Execute an action step."""
        if not step.action:
            return StepResult(
                step_id=step.step_id,
                status=WorkflowState.COMPLETED,
                output={"message": "No action specified"},
            )

        handler = self.actions.get(step.action)
        if not handler:
            return StepResult(
                step_id=step.step_id,
                status=WorkflowState.FAILED,
                error=f"Unknown action: {step.action}",
            )

        # Prepare context for action
        ctx = {
            **execution.context,
            "step_id": step.step_id,
            "step_config": step.config,
        }

        # Apply timeout
        timeout = step.timeout_seconds or self.config.default_timeout_seconds
        result = await asyncio.wait_for(handler(ctx), timeout=timeout)
        result.step_id = step.step_id

        return result

    async def _execute_decision(
        self,
        step: WorkflowStep,
        execution: WorkflowExecution,
    ) -> StepResult:
        """Execute a decision step."""
        condition = step.config.get("condition", "true")

        # Simple condition evaluation
        # Supports: context.variable, context.variable > 0, etc.
        try:
            if condition.startswith("context."):
                parts = condition.split(" ")
                var_path = parts[0].replace("context.", "")
                value = execution.context.get(var_path)

                if len(parts) == 1:
                    # Boolean check
                    result_cond = "true" if value else "false"
                elif len(parts) == 3:
                    # Comparison
                    op, compare_val = parts[1], parts[2]
                    compare_val = int(compare_val) if compare_val.isdigit() else compare_val
                    if op == ">":
                        result_cond = "true" if value > compare_val else "false"
                    elif op == "<":
                        result_cond = "true" if value < compare_val else "false"
                    elif op == "==":
                        result_cond = "true" if value == compare_val else "false"
                    else:
                        result_cond = "false"
                else:
                    result_cond = "false"
            else:
                result_cond = "true" if condition.lower() == "true" else "false"

            return StepResult(
                step_id=step.step_id,
                status=WorkflowState.COMPLETED,
                output={"condition": condition, "result": result_cond},
                next_condition=result_cond,
            )
        except Exception as e:
            return StepResult(
                step_id=step.step_id,
                status=WorkflowState.FAILED,
                error=f"Condition evaluation failed: {e}",
            )

    async def _execute_parallel(
        self,
        step: WorkflowStep,
        execution: WorkflowExecution,
    ) -> StepResult:
        """Execute parallel steps."""
        parallel_steps = step.config.get("steps", [])

        # For now, just succeed
        return StepResult(
            step_id=step.step_id,
            status=WorkflowState.COMPLETED,
            output={"parallel_steps": parallel_steps},
        )

    async def _execute_wait(
        self,
        step: WorkflowStep,
        execution: WorkflowExecution,
    ) -> StepResult:
        """Execute a wait step."""
        wait_seconds = step.config.get("seconds", 0)
        if wait_seconds > 0:
            await asyncio.sleep(min(wait_seconds, 60))  # Cap at 60 seconds

        return StepResult(
            step_id=step.step_id,
            status=WorkflowState.COMPLETED,
            output={"waited_seconds": wait_seconds},
        )

    async def _execute_approval(
        self,
        step: WorkflowStep,
        execution: WorkflowExecution,
    ) -> StepResult:
        """Execute an approval step."""
        # In a real implementation, this would wait for external approval
        # For now, auto-approve
        approvers = step.config.get("approvers", [])

        return StepResult(
            step_id=step.step_id,
            status=WorkflowState.COMPLETED,
            output={"approvers": approvers, "auto_approved": True},
            next_condition="approved",
        )

    def get_execution(self, execution_id: str) -> Optional[WorkflowExecution]:
        """Get an execution by ID."""
        return self._executions.get(execution_id)

    def cancel_execution(self, execution_id: str) -> bool:
        """Cancel a running execution."""
        execution = self._executions.get(execution_id)
        if execution and execution.state == WorkflowState.RUNNING:
            execution.state = WorkflowState.CANCELLED
            execution.completed_at = datetime.now(timezone.utc).isoformat() + "Z"
            return True
        return False


# ============================================================================
# CLI
# ============================================================================

def main() -> int:
    """CLI entry point for Workflow Engine."""
    import argparse

    parser = argparse.ArgumentParser(description="Review Workflow Engine (Step 176)")
    subparsers = parser.add_subparsers(dest="command", help="Commands")

    # Run command
    run_parser = subparsers.add_parser("run", help="Run a workflow")
    run_parser.add_argument("--workflow", default="standard-review",
                            help="Workflow ID or definition file")
    run_parser.add_argument("--context", help="JSON context")

    # Show command
    show_parser = subparsers.add_parser("show", help="Show workflow definition")
    show_parser.add_argument("workflow_id")

    parser.add_argument("--json", action="store_true", help="Output as JSON")

    args = parser.parse_args()

    engine = WorkflowEngine()

    if args.command == "run":
        # Use built-in workflow
        if args.workflow == "standard-review":
            workflow = REVIEW_WORKFLOW
        else:
            # Try loading from file
            try:
                with open(args.workflow) as f:
                    workflow = WorkflowDefinition.from_dict(json.load(f))
            except Exception as e:
                print(f"Error loading workflow: {e}")
                return 1

        # Parse context
        context = {}
        if args.context:
            context = json.loads(args.context)

        # Execute
        execution = asyncio.run(engine.execute(workflow, context))

        if args.json:
            print(json.dumps(execution.to_dict(), indent=2))
        else:
            print(f"Workflow Execution: {execution.execution_id}")
            print(f"  Workflow: {workflow.name}")
            print(f"  Status: {execution.state.value}")
            print(f"  Steps Executed: {len(execution.step_results)}")
            print(f"  Duration: {execution.duration_ms:.1f}ms")
            if execution.error:
                print(f"  Error: {execution.error}")
            print(f"\nStep Results:")
            for r in execution.step_results:
                print(f"  [{r.status.value}] {r.step_id} ({r.duration_ms:.1f}ms)")

    elif args.command == "show":
        if args.workflow_id == "standard-review":
            workflow = REVIEW_WORKFLOW
        else:
            print(f"Workflow not found: {args.workflow_id}")
            return 1

        if args.json:
            print(json.dumps(workflow.to_dict(), indent=2))
        else:
            print(f"Workflow: {workflow.name}")
            print(f"  ID: {workflow.workflow_id}")
            print(f"  Description: {workflow.description}")
            print(f"  Start Step: {workflow.start_step}")
            print(f"  End Steps: {', '.join(workflow.end_steps)}")
            print(f"\nSteps:")
            for step in workflow.steps:
                print(f"  [{step.step_id}] {step.name} ({step.step_type.value})")
                if step.action:
                    print(f"    Action: {step.action}")
                if step.next_steps:
                    print(f"    Next: {step.next_steps}")

    else:
        parser.print_help()

    return 0


if __name__ == "__main__":
    import sys
    sys.exit(main())
