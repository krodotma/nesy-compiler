from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class SemopsUIAction:
    id: str
    label: str
    kind: str = "secondary"  # primary|secondary|danger
    payload: dict[str, Any] | None = None

    def as_dict(self) -> dict[str, Any]:
        out: dict[str, Any] = {"id": self.id, "label": self.label, "kind": self.kind}
        if self.payload:
            out["payload"] = self.payload
        return out


def _safe_str(v: Any) -> str:
    return str(v) if isinstance(v, (str, int, float, bool)) else ""


def infer_flow_hints(*, operator_key: str, op: dict[str, Any]) -> list[str]:
    hints: set[str] = set()

    tool = _safe_str(op.get("tool")).strip()
    bus_topic = _safe_str(op.get("bus_topic")).strip()
    ui = op.get("ui") if isinstance(op.get("ui"), dict) else {}
    ui_route = _safe_str(ui.get("route")).strip()
    ui_component = _safe_str(ui.get("component")).strip()
    agents = op.get("agents") if isinstance(op.get("agents"), list) else []
    apps = op.get("apps") if isinstance(op.get("apps"), list) else []
    targets = op.get("targets") if isinstance(op.get("targets"), list) else []

    if tool or any(isinstance(t, dict) and str(t.get("type") or "") == "tool" for t in targets):
        hints.add("tool")
    if bus_topic or any(isinstance(t, dict) and str(t.get("type") or "") == "bus" for t in targets):
        hints.add("bus")
    if ui_route or ui_component or any(isinstance(t, dict) and str(t.get("type") or "") == "ui" for t in targets):
        hints.add("ui")
    if agents or any(isinstance(t, dict) and str(t.get("type") or "") == "agent" for t in targets):
        hints.add("agent")
    if apps or any(isinstance(t, dict) and str(t.get("type") or "") == "app" for t in targets):
        hints.add("app")

    domain = _safe_str(op.get("domain")).lower()
    category = _safe_str(op.get("category")).lower()
    name = _safe_str(op.get("name")).lower()
    desc = _safe_str(op.get("description")).lower()
    corpus = " ".join([operator_key.lower(), domain, category, name, desc, bus_topic.lower()])

    if re.search(r"\b(policy|safety|security|auth|guard|compliance)\b", corpus):
        hints.add("policy")
    if re.search(r"\b(evo|evolution|hgt|git|rhizome|quine)\b", corpus):
        hints.add("evolution")

    return sorted(hints)


def derive_ui_actions(*, operator_key: str, op: dict[str, Any]) -> list[dict[str, Any]]:
    user_defined = bool(op.get("user_defined"))
    tool = _safe_str(op.get("tool")).strip()
    bus_topic = _safe_str(op.get("bus_topic")).strip()
    ui = op.get("ui") if isinstance(op.get("ui"), dict) else {}
    ui_route = _safe_str(ui.get("route")).strip()
    ui_component = _safe_str(ui.get("component")).strip()

    actions: list[SemopsUIAction] = []
    actions.append(SemopsUIAction(id="select", label="Select", kind="secondary", payload={"operator_key": operator_key}))
    actions.append(SemopsUIAction(id="copy_key", label="Copy Key", kind="secondary", payload={"operator_key": operator_key}))

    if user_defined:
        actions.append(SemopsUIAction(id="edit", label="Edit", kind="primary", payload={"operator_key": operator_key}))
        actions.append(SemopsUIAction(id="delete", label="Delete", kind="danger", payload={"operator_key": operator_key}))
    else:
        actions.append(SemopsUIAction(id="clone", label="Clone", kind="primary", payload={"operator_key": operator_key}))

    if tool:
        actions.append(SemopsUIAction(id="open_tool", label="Open Tool", kind="secondary", payload={"path": tool}))
    if bus_topic:
        actions.append(SemopsUIAction(id="open_bus", label="Open Bus Topic", kind="secondary", payload={"topic": bus_topic}))
    if ui_route or ui_component:
        actions.append(
            SemopsUIAction(
                id="open_ui",
                label="Open UI",
                kind="secondary",
                payload={"route": ui_route, "component": ui_component},
            )
        )

    # Non-blocking: always allow emitting an invoke request to the bus.
    actions.append(SemopsUIAction(id="invoke", label="Invoke (bus)", kind="secondary", payload={"operator_key": operator_key}))

    return [a.as_dict() for a in actions]

