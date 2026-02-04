#!/usr/bin/env python3
"""
A2A Negotiation Protocol (RFC-A2A-01)
=====================================

Implements the formal negotiation state machine for Agent-to-Agent interactions.
This is the "Deepening" of A2A beyond simple forwarding.

Protocol States:
1. PROPOSE: Sender offers a task/contract.
2. CONSIDER: Receiver evaluates capability/capacity.
3. NEGOTIATE: Receiver counter-offers (e.g. "I can do X, but need more tokens").
4. AGREE/REJECT: Final contract status.

Usage:
    from nucleus.tools.a2a.negotiation import Contract, Negotiator
"""
from __future__ import annotations

import json
import time
import uuid
from dataclasses import dataclass, field
from enum import Enum
from typing import List, Optional, Any

class A2AState(Enum):
    PROPOSE = "propose"
    CONSIDER = "consider"
    NEGOTIATE = "negotiate"
    AGREE = "agree"
    REJECT = "reject"
    FAILED = "failed"

@dataclass
class Contract:
    """An A2A Service Agreement."""
    contract_id: str
    initiator: str
    target: str
    task_description: str
    constraints: dict
    compensation: dict = field(default_factory=dict) # e.g. token budget
    state: A2AState = A2AState.PROPOSE
    history: List[dict] = field(default_factory=list)

    def transition(self, new_state: A2AState, reason: str, payload: dict = None):
        self.state = new_state
        self.history.append({
            "ts": time.time(),
            "state": new_state.value,
            "reason": reason,
            "payload": payload or {}
        })

class Negotiator:
    """Handles the negotiation logic for an agent."""
    def __init__(self, agent_name: str, capabilities: List[str]):
        self.agent_name = agent_name
        self.capabilities = set(capabilities)

    def evaluate_proposal(self, contract: Contract) -> Contract:
        """Decide whether to accept, reject, or counter a proposal."""
        
        # 1. Capability Check
        # (Naive implementation: keyword matching. Deepening would use embeddings)
        req_caps = contract.constraints.get("required_capabilities", [])
        missing = [c for c in req_caps if c not in self.capabilities]
        
        if missing:
            contract.transition(A2AState.REJECT, f"Missing capabilities: {missing}")
            return contract

        # 2. Capacity/Cost Check (Stub)
        budget = contract.compensation.get("max_tokens", 0)
        est_cost = 1000 # Stub
        if budget > 0 and est_cost > budget:
             contract.transition(A2AState.NEGOTIATE, "Insufficient budget", {"required_tokens": est_cost})
             return contract

        # 3. Accept
        contract.transition(A2AState.AGREE, "Terms acceptable")
        return contract

def serialize_contract(c: Contract) -> str:
    return json.dumps(c.__dict__, default=lambda o: o.value if isinstance(o, Enum) else o, indent=2)
