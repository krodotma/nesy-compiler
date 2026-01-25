"""
VIL UI Control Center Module
Floating overlay UI for VIL control and monitoring.

Features:
1. VIL status monitoring panel
2. ICL buffer visualization
3. CMP fitness tracking
4. Geometric embedding visualization
5. Control knobs for VIL parameters

Version: 1.0
Date: 2026-01-25
"""

import time
from typing import Dict, Any, List, Optional, Callable
from dataclasses import dataclass, field
from enum import Enum
import json

from nucleus.vil.cmp.manager import CladeState, PHI
from nucleus.vil.coordinator import VILState, VILConfig


class UIPanelType(str, Enum):
    """Types of UI panels."""

    STATUS = "status"  # Overall VIL status
    ICL_BUFFER = "icl_buffer"  # ICL buffer visualization
    CMP_TRACKING = "cmp_tracking"  # CMP fitness tracking
    GEOMETRIC = "geometric"  # Geometric embedding visualization
    CONTROLS = "controls"  # Parameter control knobs
    EVENTS = "events"  # Event log viewer


@dataclass
class UIPanelConfig:
    """
    Configuration for UI panel.

    Contains:
    - Panel type
    - Position (x, y)
    - Size (width, height)
    - Visibility
    - Z-index
    """

    panel_type: UIPanelType
    x: int = 10
    y: int = 10
    width: int = 300
    height: int = 400
    visible: bool = True
    z_index: int = 1000
    collapsible: bool = True
    draggable: bool = True


@dataclass
class VILStatusData:
    """
    VIL status data for UI display.

    Contains:
    - Overall state
    - Vision pipeline status
    - Metalearning status
    - CMP fitness
    - ICL buffer stats
    """

    state: str = "idle"
    vision_frames_processed: int = 0
    vision_accuracy: float = 0.0
    metalearning_iterations: int = 0
    metalearning_accuracy: float = 0.0
    cmp_fitness: float = 0.0
    cmp_clade_count: int = 0
    icl_buffer_size: int = 0
    icl_buffer_utilization: float = 0.0
    synthesis_count: int = 0
    synthesis_success_rate: float = 0.0
    uptime_seconds: float = 0.0


@dataclass
class UIEvent:
    """
    UI event for logging.

    Contains:
    - Event type
    - Message
    - Timestamp
    - Severity
    """

    event_type: str
    message: str
    timestamp: float = field(default_factory=time.time)
    severity: str = "info"  # info, warning, error, success
    metadata: Dict[str, Any] = field(default_factory=dict)


class VILControlCenter:
    """
    VIL UI Control Center for floating overlay.

    Features:
    1. Multiple panels (status, ICL, CMP, geometric, controls, events)
    2. Real-time data updates
    3. Interactive parameter controls
    4. Event logging
    5. Collapsible/draggable panels
    """

    def __init__(
        self,
        config: Optional[VILConfig] = None,
        bus_emitter: Optional[Callable] = None,
    ):
        self.config = config or VILConfig()
        self.bus_emitter = bus_emitter

        # UI panels
        self.panels: Dict[UIPanelType, UIPanelConfig] = {
            UIPanelType.STATUS: UIPanelConfig(
                panel_type=UIPanelType.STATUS,
                x=10, y=10,
                width=280, height=200,
            ),
            UIPanelType.ICL_BUFFER: UIPanelConfig(
                panel_type=UIPanelType.ICL_BUFFER,
                x=310, y=10,
                width=280, height=250,
            ),
            UIPanelType.CMP_TRACKING: UIPanelConfig(
                panel_type=UIPanelType.CMP_TRACKING,
                x=610, y=10,
                width=280, height=250,
            ),
            UIPanelType.GEOMETRIC: UIPanelConfig(
                panel_type=UIPanelType.GEOMETRIC,
                x=10, y=220,
                width=280, height=250,
            ),
            UIPanelType.CONTROLS: UIPanelConfig(
                panel_type=UIPanelType.CONTROLS,
                x=310, y=270,
                width=280, height=300,
            ),
            UIPanelType.EVENTS: UIPanelConfig(
                panel_type=UIPanelType.EVENTS,
                x=610, y=270,
                width=280, height=300,
            ),
        }

        # Status data
        self.status_data = VILStatusData()
        self.start_time = time.time()

        # Event log
        self.events: List[UIEvent] = []
        self.max_events = 100

        # Control values
        self.control_values = {
            "icel_temperature": 0.5,
            "mutation_rate": 0.1,
            "learning_rate": 0.01,
            "entropy_threshold": 0.7,
            "novelty_bonus": 0.5,
            "phi_weighting": True,
            "auto_evolve": True,
            "icl_strategy": "diverse",
        }

    def update_status(
        self,
        vil_state: Optional[VILState] = None,
        vision_stats: Optional[Dict[str, Any]] = None,
        metalearning_stats: Optional[Dict[str, Any]] = None,
        cmp_stats: Optional[Dict[str, Any]] = None,
        icl_stats: Optional[Dict[str, Any]] = None,
        synthesis_stats: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Update VIL status data."""
        if vil_state:
            self.status_data.state = vil_state.phase

        if vision_stats:
            self.status_data.vision_frames_processed = vision_stats.get("frames_processed", 0)
            self.status_data.vision_accuracy = vision_stats.get("accuracy", 0.0)

        if metalearning_stats:
            self.status_data.metalearning_iterations = metalearning_stats.get("iterations", 0)
            self.status_data.metalearning_accuracy = metalearning_stats.get("accuracy", 0.0)

        if cmp_stats:
            self.status_data.cmp_fitness = cmp_stats.get("avg_fitness", 0.0)
            self.status_data.cmp_clade_count = cmp_stats.get("total_clades", 0)

        if icl_stats:
            self.status_data.icl_buffer_size = icl_stats.get("total_examples", 0)
            self.status_data.icl_buffer_utilization = icl_stats.get("buffer_utilization", 0.0)

        if synthesis_stats:
            self.status_data.synthesis_count = synthesis_stats.get("syntheses_completed", 0)
            self.status_data.synthesis_success_rate = (
                synthesis_stats.get("syntheses_completed", 0) /
                max(synthesis_stats.get("requests_received", 1), 1)
            )

        self.status_data.uptime_seconds = time.time() - self.start_time

    def log_event(
        self,
        event_type: str,
        message: str,
        severity: str = "info",
        metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Log event to UI."""
        event = UIEvent(
            event_type=event_type,
            message=message,
            severity=severity,
            metadata=metadata or {},
        )

        self.events.append(event)

        # Prune old events
        if len(self.events) > self.max_events:
            self.events = self.events[-self.max_events:]

        # Emit event
        self._emit_ui_event(event)

    def set_control_value(self, key: str, value: Any) -> None:
        """Set control value."""
        if key in self.control_values:
            old_value = self.control_values[key]
            self.control_values[key] = value
            self.log_event(
                event_type="control_change",
                message=f"Updated {key}: {old_value} -> {value}",
                severity="info",
                metadata={"key": key, "old_value": old_value, "new_value": value},
            )

    def get_control_value(self, key: str, default: Any = None) -> Any:
        """Get control value."""
        return self.control_values.get(key, default)

    def toggle_panel(self, panel_type: UIPanelType) -> bool:
        """Toggle panel visibility."""
        if panel_type in self.panels:
            self.panels[panel_type].visible = not self.panels[panel_type].visible
            return self.panels[panel_type].visible
        return False

    def move_panel(self, panel_type: UIPanelType, x: int, y: int) -> None:
        """Move panel to position."""
        if panel_type in self.panels:
            self.panels[panel_type].x = x
            self.panels[panel_type].y = y

    def get_ui_config(self) -> Dict[str, Any]:
        """Get UI configuration for rendering."""
        return {
            "panels": {
                pt.value: {
                    "x": p.x,
                    "y": p.y,
                    "width": p.width,
                    "height": p.height,
                    "visible": p.visible,
                    "z_index": p.z_index,
                    "collapsible": p.collapsible,
                    "draggable": p.draggable,
                }
                for pt, p in self.panels.items()
            },
            "status": self._get_status_dict(),
            "controls": self.control_values.copy(),
            "events": [self._event_to_dict(e) for e in self.events[-20:]],  # Last 20
        }

    def _get_status_dict(self) -> Dict[str, Any]:
        """Get status data as dictionary."""
        return {
            "state": self.status_data.state,
            "vision_frames_processed": self.status_data.vision_frames_processed,
            "vision_accuracy": f"{self.status_data.vision_accuracy:.1%}",
            "metalearning_iterations": self.status_data.metalearning_iterations,
            "metalearning_accuracy": f"{self.status_data.metalearning_accuracy:.1%}",
            "cmp_fitness": f"{self.status_data.cmp_fitness:.3f}",
            "cmp_clade_count": self.status_data.cmp_clade_count,
            "icl_buffer_size": self.status_data.icl_buffer_size,
            "icl_buffer_utilization": f"{self.status_data.icl_buffer_utilization:.1%}",
            "synthesis_count": self.status_data.synthesis_count,
            "synthesis_success_rate": f"{self.status_data.synthesis_success_rate:.1%}",
            "uptime_seconds": self.status_data.uptime_seconds,
            "uptime_formatted": self._format_uptime(self.status_data.uptime_seconds),
        }

    def _format_uptime(self, seconds: float) -> str:
        """Format uptime as readable string."""
        hours = int(seconds // 3600)
        minutes = int((seconds % 3600) // 60)
        secs = int(seconds % 60)
        return f"{hours:02d}:{minutes:02d}:{secs:02d}"

    def _event_to_dict(self, event: UIEvent) -> Dict[str, Any]:
        """Convert event to dictionary."""
        return {
            "type": event.event_type,
            "message": event.message,
            "timestamp": event.timestamp,
            "severity": event.severity,
            "metadata": event.metadata,
        }

    def _emit_ui_event(self, event: UIEvent) -> None:
        """Emit UI event to bus."""
        if not self.bus_emitter:
            return

        bus_event = {
            "topic": "vil.ui.event",
            "data": {
                "type": event.event_type,
                "message": event.message,
                "severity": event.severity,
                "metadata": event.metadata,
            },
        }

        try:
            self.bus_emitter(bus_event)
        except Exception as e:
            print(f"[VILControlCenter] Bus emission error: {e}")

    def export_config(self) -> str:
        """Export UI configuration as JSON."""
        return json.dumps(self.get_ui_config(), indent=2)

    def get_panel_html(self, panel_type: UIPanelType) -> str:
        """Generate HTML for panel (for integration with dashboard)."""
        if panel_type == UIPanelType.STATUS:
            return self._status_panel_html()
        elif panel_type == UIPanelType.ICL_BUFFER:
            return self._icl_buffer_panel_html()
        elif panel_type == UIPanelType.CMP_TRACKING:
            return self._cmp_tracking_panel_html()
        elif panel_type == UIPanelType.CONTROLS:
            return self._controls_panel_html()
        elif panel_type == UIPanelType.EVENTS:
            return self._events_panel_html()
        else:
            return "<div>Unknown panel</div>"

    def _status_panel_html(self) -> str:
        """Generate status panel HTML."""
        status = self._get_status_dict()
        fitness_color = "green" if float(status["cmp_fitness"]) > PHI else "orange"

        return f"""
        <div class="vil-status-panel">
            <h3>VIL Status</h3>
            <div class="status-row">
                <span>State:</span>
                <span class="status-value">{status['state']}</span>
            </div>
            <div class="status-row">
                <span>Uptime:</span>
                <span class="status-value">{status['uptime_formatted']}</span>
            </div>
            <div class="status-row">
                <span>Vision Frames:</span>
                <span class="status-value">{status['vision_frames_processed']}</span>
            </div>
            <div class="status-row">
                <span>Vision Accuracy:</span>
                <span class="status-value">{status['vision_accuracy']}</span>
            </div>
            <div class="status-row">
                <span>CMP Fitness:</span>
                <span class="status-value" style="color: {fitness_color}">{status['cmp_fitness']}</span>
            </div>
            <div class="status-row">
                <span>ICL Buffer:</span>
                <span class="status-value">{status['icl_buffer_size']} examples ({status['icl_buffer_utilization']})</span>
            </div>
        </div>
        """

    def _icl_buffer_panel_html(self) -> str:
        """Generate ICL buffer panel HTML."""
        return """
        <div class="vil-icl-panel">
            <h3>ICL Buffer</h3>
            <div class="buffer-visualization">
                <!-- Placeholder for ICL buffer visualization -->
                <div class="buffer-bar" style="width: 70%;"></div>
            </div>
            <div class="buffer-stats">
                <div>Vision: 30</div>
                <div>Meta: 25</div>
                <div>Synthesis: 15</div>
            </div>
        </div>
        """

    def _cmp_tracking_panel_html(self) -> str:
        """Generate CMP tracking panel HTML."""
        return """
        <div class="vil-cmp-panel">
            <h3>CMP Tracking</h3>
            <div class="cmp-fitness-chart">
                <!-- Placeholder for fitness chart -->
            </div>
            <div class="clade-stats">
                <div>Active: 5</div>
                <div>Converging: 2</div>
                <div>Extinct: 1</div>
            </div>
        </div>
        """

    def _controls_panel_html(self) -> str:
        """Generate controls panel HTML."""
        return """
        <div class="vil-controls-panel">
            <h3>Controls</h3>
            <div class="control-row">
                <label>ICL Temperature</label>
                <input type="range" min="0" max="1" step="0.1" value="0.5">
            </div>
            <div class="control-row">
                <label>Mutation Rate</label>
                <input type="range" min="0" max="1" step="0.05" value="0.1">
            </div>
            <div class="control-row">
                <label>Learning Rate</label>
                <input type="range" min="0.001" max="0.1" step="0.001" value="0.01">
            </div>
            <div class="control-row">
                <label>
                    <input type="checkbox" checked>
                    Auto Evolve
                </label>
            </div>
            <div class="control-row">
                <label>
                    <input type="checkbox" checked>
                    Phi Weighting
                </label>
            </div>
        </div>
        """

    def _events_panel_html(self) -> str:
        """Generate events panel HTML."""
        events_html = ""
        for event in reversed(self.events[-10:]):
            severity_class = f"severity-{event.severity}"
            events_html += f"""
            <div class="event-row {severity_class}">
                <span class="event-time">{time.strftime('%H:%M:%S', time.localtime(event.timestamp))}</span>
                <span class="event-message">{event.message}</span>
            </div>
            """

        return f"""
        <div class="vil-events-panel">
            <h3>Event Log</h3>
            <div class="events-list">
                {events_html}
            </div>
        </div>
        """


def create_vil_control_center(
    config: Optional[VILConfig] = None,
    bus_emitter: Optional[Callable] = None,
) -> VILControlCenter:
    """Create VIL control center with default config."""
    return VILControlCenter(
        config=config,
        bus_emitter=bus_emitter,
    )


__all__ = [
    "UIPanelType",
    "UIPanelConfig",
    "VILStatusData",
    "UIEvent",
    "VILControlCenter",
    "create_vil_control_center",
]
