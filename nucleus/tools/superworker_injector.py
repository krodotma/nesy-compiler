"""
superworker_injector.py - 5-Layer SUPERWORKER Injection Stack

Implements the full SUPERWORKER architecture from SUPERWORKERS.md:
1. Constitutional - P/E/L/R/Q gates, PLURIBUSCHECK
2. Idiolect - Semantic operators, EBNF grammar  
3. Persona - Archetype, lanes, effects budget
4. Context - Gestalt summary, file priority list
5. Observable - Bus event emission, gate validation

Phase 3 - SUPERWORKER Integration
"""

from typing import Dict, Any, Optional, List, Callable
from dataclasses import dataclass, field
from enum import Enum
import json
import os


# ============================================================================
# Layer Enums and Types
# ============================================================================

class Gate(Enum):
    """P/E/L/R/Q Constitutional Gates"""
    P = "P"  # Proprietary - model-specific configs
    E = "E"  # Evidence - append-only bus emissions
    L = "L"  # Lanes - communication channels
    R = "R"  # Ring - capability tier
    Q = "Q"  # Query - external API access


class Lane(Enum):
    """Three-Lane Coordination Channels"""
    DIALOGOS = "dialogos"  # Streaming SSE, interactive
    PBPAIR = "pbpair"      # Proposal→Constraints→Impl→Verify
    STRP = "strp"          # Request/Response fanout


class DepthClass(Enum):
    """Task Depth Classification"""
    NARROW = "narrow"  # Quick tasks, minimal context
    DEEP = "deep"      # Complex tasks, full context


class ContextMode(Enum):
    """Context Allocation Mode"""
    MIN = "min"    # ~2k tokens
    LITE = "lite"  # ~8k tokens  
    FULL = "full"  # ~32k tokens


# ============================================================================
# Layer 1: Constitutional Grounding
# ============================================================================

@dataclass
class ConstitutionalLayer:
    """
    Layer 1: Constitutional constraints and gates.
    Enforces immutable principles from CITIZEN.md
    """
    # Gate permissions
    gates: Dict[Gate, bool] = field(default_factory=lambda: {
        Gate.P: True,   # Proprietary configs allowed
        Gate.E: True,   # Evidence emission required
        Gate.L: True,   # Lane access
        Gate.R: True,   # Ring-appropriate
        Gate.Q: False,  # External queries restricted by default
    })
    
    # Constitutional principles
    append_only: bool = True
    non_blocking: bool = True
    tests_first: bool = True
    no_secrets_emission: bool = True
    
    # PLURIBUSCHECK assertion
    pluribus_check: str = "PLURIBUSCHECK"
    
    def validate_gate(self, gate: Gate) -> bool:
        """Check if gate is open."""
        return self.gates.get(gate, False)
    
    def emit_check(self) -> str:
        """Return PLURIBUSCHECK assertion for response suffix."""
        return f"\n{self.pluribus_check}"
    
    def to_prompt(self) -> str:
        """Generate constitutional grounding prompt segment."""
        gates_str = ", ".join(f"{g.value}={'✓' if v else '✗'}" for g, v in self.gates.items())
        return f"""## Constitutional Grounding
You operate under the Pluribus Citizen Constitution:
- Gates: [{gates_str}]
- Append-only evidence: All actions emit to bus
- Non-blocking IPC: Publish requests, never block
- Tests-first: Untested code does not exist
- End responses with {self.pluribus_check}
"""


# ============================================================================
# Layer 2: Idiolect Compression
# ============================================================================

@dataclass
class IdiolectLayer:
    """
    Layer 2: Semantic operators and compressed vocabulary.
    Enables efficient communication via shared terminology.
    """
    # Semantic operators
    operators: Dict[str, str] = field(default_factory=lambda: {
        "GESTALT": "Compressed summary of codebase state",
        "ENTELECHY": "Intrinsic drive toward ideal form",
        "REMEDIATE": "Fix identified issue with bus emission",
        "COLLIMATE": "Focus scattered context into coherent beam",
        "TRANSMUTE": "Transform from current to target state",
    })
    
    # EBNF grammar shortcuts
    grammar: Dict[str, str] = field(default_factory=lambda: {
        "⟨task⟩": "::= ⟨verb⟩ ⟨target⟩ [⟨constraints⟩]",
        "⟨verb⟩": "::= ANALYZE | IMPLEMENT | VERIFY | REMEDIATE",
        "⟨target⟩": "::= file_path | module_name | test_suite",
    })
    
    def expand(self, term: str) -> Optional[str]:
        """Expand a semantic operator."""
        return self.operators.get(term.upper())
    
    def to_prompt(self) -> str:
        """Generate idiolect prompt segment."""
        ops = "\n".join(f"- {k}: {v}" for k, v in self.operators.items())
        return f"""## Idiolect
Use these semantic operators in your responses:
{ops}
"""


# ============================================================================
# Layer 3: Persona Adoption
# ============================================================================

@dataclass
class PersonaLayer:
    """
    Layer 3: Archetype identity and capability constraints.
    """
    archetype: str = "general"
    persona_id: str = "agent.default"
    ring: int = 2
    
    # Lane preferences  
    preferred_lanes: List[Lane] = field(default_factory=lambda: [Lane.DIALOGOS])
    
    # Effects budget (what side effects are allowed)
    effects_budget: List[str] = field(default_factory=lambda: ["none", "file"])
    
    # Scope allowlist
    scope_allowlist: List[str] = field(default_factory=list)
    
    def can_use_lane(self, lane: Lane) -> bool:
        """Check if lane is in preferred list."""
        return lane in self.preferred_lanes
    
    def can_effect(self, effect: str) -> bool:
        """Check if side effect is in budget."""
        return effect in self.effects_budget
    
    def to_prompt(self) -> str:
        """Generate persona prompt segment."""
        lanes = ", ".join(l.value for l in self.preferred_lanes)
        effects = ", ".join(self.effects_budget)
        scope = ", ".join(self.scope_allowlist) if self.scope_allowlist else "unrestricted"
        return f"""## Persona
- Archetype: {self.archetype}
- Identity: {self.persona_id}
- Ring: {self.ring}
- Lanes: [{lanes}]
- Effects Budget: [{effects}]
- Scope: [{scope}]
"""


# ============================================================================
# Layer 4: Context Allocation
# ============================================================================

@dataclass
class ContextLayer:
    """
    Layer 4: Dynamic context injection based on task depth.
    """
    depth: DepthClass = DepthClass.NARROW
    codebase_size_mb: float = 0.0
    mode: ContextMode = ContextMode.MIN
    token_budget: int = 2000
    
    # Gestalt summary
    gestalt: str = ""
    
    # Priority file list
    priority_files: List[str] = field(default_factory=list)
    
    # Recent changes
    recent_changes: List[str] = field(default_factory=list)
    
    def classify_depth(self, task_description: str) -> DepthClass:
        """
        Classify task depth based on complexity signals.
        """
        deep_signals = [
            "refactor", "architect", "design", "implement",
            "complex", "system", "integration", "migration"
        ]
        task_lower = task_description.lower()
        
        for signal in deep_signals:
            if signal in task_lower:
                return DepthClass.DEEP
        return DepthClass.NARROW
    
    def allocate_context(self) -> ContextMode:
        """
        Allocate context mode based on depth and codebase size.
        From SUPERWORKERS.md 3.3 Context Allocation table.
        """
        if self.depth == DepthClass.NARROW:
            return ContextMode.MIN
        
        # Deep tasks
        if self.codebase_size_mb < 10:
            return ContextMode.FULL
        elif self.codebase_size_mb > 100:
            return ContextMode.LITE
        else:
            return ContextMode.FULL
    
    def get_token_budget(self) -> int:
        """Get token budget for current mode."""
        budgets = {
            ContextMode.MIN: 2000,
            ContextMode.LITE: 8000,
            ContextMode.FULL: 32000,
        }
        return budgets.get(self.mode, 2000)
    
    def to_prompt(self) -> str:
        """Generate context prompt segment."""
        files = "\n".join(f"  - {f}" for f in self.priority_files[:10])
        changes = "\n".join(f"  - {c}" for c in self.recent_changes[:5])
        
        return f"""## Context
- Depth: {self.depth.value}
- Mode: {self.mode.value} (~{self.token_budget} tokens)
- Codebase: {self.codebase_size_mb:.1f}MB

### Gestalt
{self.gestalt or "(No gestalt summary available)"}

### Priority Files
{files or "(No priority files)"}

### Recent Changes
{changes or "(No recent changes)"}
"""


# ============================================================================
# Layer 5: Observable State Emission
# ============================================================================

@dataclass
class ObservableLayer:
    """
    Layer 5: Bus event emission and gate validation.
    Ensures all actions are observable via the agent bus.
    """
    bus_dir: str = "/pluribus/.pluribus/bus"
    actor: str = "superworker"
    trace_id: Optional[str] = None
    run_id: Optional[str] = None
    
    # Pending emissions
    pending_events: List[Dict[str, Any]] = field(default_factory=list)
    
    def emit(self, topic: str, kind: str, level: str, data: Dict[str, Any]):
        """Queue an event for emission."""
        import time
        event = {
            "topic": topic,
            "kind": kind,
            "level": level,
            "actor": self.actor,
            "ts": time.time(),
            "data": data,
        }
        if self.trace_id:
            event["trace_id"] = self.trace_id
        if self.run_id:
            event["run_id"] = self.run_id
        
        self.pending_events.append(event)
    
    def flush(self) -> List[Dict[str, Any]]:
        """Return and clear pending events."""
        events = self.pending_events.copy()
        self.pending_events.clear()
        return events
    
    def to_prompt(self) -> str:
        """Generate observable prompt segment."""
        return f"""## Observable State
- Bus: {self.bus_dir}
- Actor: {self.actor}
- Trace: {self.trace_id or '(none)'}

Emit events to bus for all significant actions:
- Topic format: {{domain}}.{{entity}}.{{action}}
- Kinds: request, response, metric, log
- Levels: debug, info, warn, error
"""


# ============================================================================
# SUPERWORKER Stack (All 5 Layers)
# ============================================================================

@dataclass
class SuperworkerStack:
    """
    Complete 5-layer SUPERWORKER injection stack.
    Combines all layers into a coherent prompt injection.
    """
    constitutional: ConstitutionalLayer = field(default_factory=ConstitutionalLayer)
    idiolect: IdiolectLayer = field(default_factory=IdiolectLayer)
    persona: PersonaLayer = field(default_factory=PersonaLayer)
    context: ContextLayer = field(default_factory=ContextLayer)
    observable: ObservableLayer = field(default_factory=ObservableLayer)
    
    def inject_system_prompt(self) -> str:
        """
        Generate the complete system prompt injection.
        Combines all 5 layers into a coherent prompt.
        """
        sections = [
            "# SUPERWORKER System Injection\n",
            self.constitutional.to_prompt(),
            self.idiolect.to_prompt(),
            self.persona.to_prompt(),
            self.context.to_prompt(),
            self.observable.to_prompt(),
        ]
        return "\n".join(sections)
    
    def inject_user_prefix(self, task: str) -> str:
        """
        Generate user message prefix with context.
        """
        # Classify depth
        self.context.depth = self.context.classify_depth(task)
        self.context.mode = self.context.allocate_context()
        self.context.token_budget = self.context.get_token_budget()
        
        return f"""[{self.persona.archetype.upper()}] Task:
{task}

Context Mode: {self.context.mode.value} ({self.context.token_budget} tokens)
"""
    
    def validate_response(self, response: str) -> Dict[str, bool]:
        """
        Validate response against constitutional requirements.
        """
        return {
            "has_pluribuscheck": self.constitutional.pluribus_check in response,
            "evidence_compliant": True,  # Would check bus emissions
            "non_blocking": True,  # Would check for blocking patterns
        }
    
    @classmethod
    def from_cagent_config(cls, config_path: str) -> "SuperworkerStack":
        """
        Create stack from CAGENT config file.
        """
        with open(config_path) as f:
            config = json.load(f)
        
        stack = cls()
        
        # Apply persona config
        stack.persona.archetype = config.get("archetype", "general")
        stack.persona.persona_id = config.get("persona_id", "agent.default")
        stack.persona.scope_allowlist = config.get("scope_allowlist", [])
        stack.persona.effects_budget = config.get("effects_budget", ["none", "file"])
        
        # Parse lanes
        lane_map = {"dialogos": Lane.DIALOGOS, "pbpair": Lane.PBPAIR, "strp": Lane.STRP}
        stack.persona.preferred_lanes = [
            lane_map[l] for l in config.get("preferred_lanes", ["dialogos"])
            if l in lane_map
        ]
        
        # Observable config
        stack.observable.actor = config.get("persona_id", "superworker").split(".")[-1]
        
        return stack


# ============================================================================
# Factory Functions
# ============================================================================

def create_verifier_stack(scope: Optional[List[str]] = None) -> SuperworkerStack:
    """Create a stack configured for verification tasks."""
    stack = SuperworkerStack()
    stack.persona.archetype = "Verifier"
    stack.persona.persona_id = "subagent.verifier"
    stack.persona.effects_budget = ["none", "file"]
    stack.persona.scope_allowlist = scope or []
    stack.context.depth = DepthClass.NARROW
    return stack


def create_implementer_stack(scope: Optional[List[str]] = None) -> SuperworkerStack:
    """Create a stack configured for implementation tasks."""
    stack = SuperworkerStack()
    stack.persona.archetype = "Implementer"
    stack.persona.persona_id = "subagent.narrow_coder"
    stack.persona.effects_budget = ["none", "file", "network"]
    stack.persona.scope_allowlist = scope or []
    stack.context.depth = DepthClass.DEEP
    return stack


def create_orchestrator_stack() -> SuperworkerStack:
    """Create a stack configured for orchestration (Ring 0)."""
    stack = SuperworkerStack()
    stack.persona.archetype = "Orchestrator"
    stack.persona.persona_id = "ring0.architect"
    stack.persona.ring = 0
    stack.persona.effects_budget = ["none", "file", "network", "spawn"]
    stack.persona.preferred_lanes = [Lane.DIALOGOS, Lane.PBPAIR, Lane.STRP]
    stack.constitutional.gates[Gate.Q] = True  # Orchestrator can query
    stack.context.depth = DepthClass.DEEP
    stack.context.mode = ContextMode.FULL
    return stack


# ============================================================================
# CLI Self-Test
# ============================================================================

if __name__ == "__main__":
    print("=== SUPERWORKER Injection Stack Test ===\n")
    
    # Create orchestrator stack
    stack = create_orchestrator_stack()
    
    # Set some context
    stack.context.gestalt = "Auralux voice pipeline with avatar lip-sync integration."
    stack.context.priority_files = [
        "nucleus/auralux/viseme_mapper.ts",
        "nucleus/auralux/avatar_controller.ts",
        "nucleus/dashboard/src/components/VoiceHUD.tsx",
    ]
    stack.context.recent_changes = [
        "Added useAvatarController hook",
        "Created viseme_mapper.test.ts (24 tests)",
    ]
    stack.context.codebase_size_mb = 45.0
    
    # Generate system prompt
    print("=== System Prompt Injection ===")
    print(stack.inject_system_prompt())
    
    # Generate user prefix
    print("\n=== User Message Prefix ===")
    print(stack.inject_user_prefix("Implement error boundaries for AuraluxProvider"))
    
    # Validate sample response
    sample_response = "Done implementing error boundaries.\nPLURIBUSCHECK"
    print("\n=== Response Validation ===")
    print(stack.validate_response(sample_response))
