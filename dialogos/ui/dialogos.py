"""
VIL Dialogos Widget Enhancements
Enhanced Dialogos holographic UI for VIL visualization.

Features:
1. VIL state visualization with holographic effects
2. CMP fitness progress bars
3. ICL buffer geometric visualization
4. Event stream with severity highlighting
5. Real-time parameter controls

Version: 1.0
Date: 2026-01-25
"""

import json
from typing import Dict, Any, List, Optional, Callable
from dataclasses import dataclass, field
from enum import Enum


class DialogosPanelType(str, Enum):
    """Dialogos panel types for VIL."""

    VIL_OVERVIEW = "vil_overview"
    CMP_TRACKER = "cmp_tracker"
    ICL_VISUALIZER = "icl_visualizer"
    GEOMETRIC_VIEW = "geometric_view"
    EVENT_STREAM = "event_stream"
    CONTROL_PANEL = "control_panel"


@dataclass
class DialogosConfig:
    """
    Configuration for Dialogos widget.

    Contains:
    - Theme settings
    - Animation settings
    - Layout preferences
    - Color schemes
    """

    theme: str = "cyberpunk"  # cyberpunk, minimal, holographic
    animation_enabled: bool = True
    animation_speed: float = 1.0
    layout: str = "grid"  # grid, stack, tabs
    color_scheme: Dict[str, str] = field(default_factory=lambda: {
        "primary": "#00ffff",
        "secondary": "#ff00ff",
        "success": "#00ff00",
        "warning": "#ffff00",
        "error": "#ff0000",
        "background": "#0a0a1a",
        "panel_bg": "#111122",
        "text": "#ffffff",
    })
    show_timestamps: bool = True
    show_severity_icons: bool = True
    auto_refresh: bool = True
    refresh_interval_ms: int = 1000


@dataclass
class DialogosEvent:
    """
    Event for Dialogos display.

    Contains:
    - Event data
    - Display settings
    - Styling
    """

    event_type: str
    title: str
    message: str
    severity: str = "info"
    timestamp: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)
    icon: Optional[str] = None
    color: Optional[str] = None
    progress: Optional[float] = None


class DialogosVILEnhancements:
    """
    Dialogos widget enhancements for VIL.

    Features:
    1. Holographic VIL overview panel
    2. CMP fitness tracker with progress bars
    3. ICL buffer geometric visualization
    4. Real-time event stream
    5. Parameter control panel
    """

    def __init__(
        self,
        config: Optional[DialogosConfig] = None,
        bus_emitter: Optional[Callable] = None,
    ):
        self.config = config or DialogosConfig()
        self.bus_emitter = bus_emitter

        # Panel data
        self.panels: Dict[DialogosPanelType, Dict[str, Any]] = {
            pt: {} for pt in DialogosPanelType
        }

        # Event stream
        self.events: List[DialogosEvent] = []
        self.max_events = 50

        # Visualization data
        self.cmp_data: List[Dict[str, float]] = []
        self.icl_embeddings: List[Dict[str, Any]] = []

    def update_vil_overview(
        self,
        state: str,
        uptime: float,
        vision_frames: int,
        metalearning_iterations: int,
        synthesis_count: int,
    ) -> None:
        """Update VIL overview panel."""
        self.panels[DialogosPanelType.VIL_OVERVIEW] = {
            "state": state,
            "uptime": uptime,
            "vision_frames": vision_frames,
            "metalearning_iterations": metalearning_iterations,
            "synthesis_count": synthesis_count,
        }

    def update_cmp_tracker(
        self,
        fitness: float,
        clade_count: int,
        active_clades: int,
        converging_clades: int,
        fitness_history: Optional[List[float]] = None,
    ) -> None:
        """Update CMP tracker panel."""
        self.panels[DialogosPanelType.CMP_TRACKER] = {
            "fitness": fitness,
            "clade_count": clade_count,
            "active_clades": active_clades,
            "converging_clades": converging_clades,
            "fitness_history": fitness_history or self.cmp_data,
        }

        # Store history
        self.cmp_data.append({
            "timestamp": len(self.cmp_data),
            "fitness": fitness,
        })
        if len(self.cmp_data) > 100:
            self.cmp_data = self.cmp_data[-100:]

    def update_icl_visualizer(
        self,
        buffer_size: int,
        buffer_utilization: float,
        avg_novelty: float,
        avg_entropy: float,
        embeddings: Optional[List[Dict[str, Any]]] = None,
    ) -> None:
        """Update ICL visualizer panel."""
        self.panels[DialogosPanelType.ICL_VISUALIZER] = {
            "buffer_size": buffer_size,
            "buffer_utilization": buffer_utilization,
            "avg_novelty": avg_novelty,
            "avg_entropy": avg_entropy,
            "embeddings": embeddings or self.icl_embeddings,
        }

        # Store embeddings
        if embeddings:
            self.icl_embeddings = embeddings[-50:]  # Keep last 50

    def add_event(
        self,
        event_type: str,
        title: str,
        message: str,
        severity: str = "info",
        metadata: Optional[Dict[str, Any]] = None,
        progress: Optional[float] = None,
    ) -> None:
        """Add event to stream."""
        # Get icon and color based on severity
        icon, color = self._get_severity_style(severity)

        event = DialogosEvent(
            event_type=event_type,
            title=title,
            message=message,
            severity=severity,
            timestamp=0.0,  # Will be set on render
            metadata=metadata or {},
            icon=icon,
            color=color,
            progress=progress,
        )

        self.events.append(event)

        # Prune old events
        if len(self.events) > self.max_events:
            self.events = self.events[-self.max_events:]

    def _get_severity_style(self, severity: str) -> tuple:
        """Get icon and color for severity level."""
        styles = {
            "info": ("ℹ️", self.config.color_scheme["primary"]),
            "success": ("✅", self.config.color_scheme["success"]),
            "warning": ("⚠️", self.config.color_scheme["warning"]),
            "error": ("❌", self.config.color_scheme["error"]),
            "debug": ("🔍", "#888888"),
        }
        return styles.get(severity, ("•", self.config.color_scheme["text"]))

    def generate_html(self, panel: DialogosPanelType) -> str:
        """Generate HTML for panel."""
        if panel == DialogosPanelType.VIL_OVERVIEW:
            return self._vil_overview_html()
        elif panel == DialogosPanelType.CMP_TRACKER:
            return self._cmp_tracker_html()
        elif panel == DialogosPanelType.ICL_VISUALIZER:
            return self._icl_visualizer_html()
        elif panel == DialogosPanelType.EVENT_STREAM:
            return self._event_stream_html()
        elif panel == DialogosPanelType.CONTROL_PANEL:
            return self._control_panel_html()
        else:
            return "<div>Unknown panel</div>"

    def _vil_overview_html(self) -> str:
        """Generate VIL overview panel HTML."""
        data = self.panels[DialogosPanelType.VIL_OVERVIEW]
        if not data:
            return "<div>No data</div>"

        return f"""
        <div class="dialogos-vil-overview" data-theme="{self.config.theme}">
            <div class="panel-header holographic">
                <h2>VIL Overview</h2>
                <div class="status-indicator status-{data['state']}"></div>
            </div>
            <div class="panel-content">
                <div class="metric-row">
                    <div class="metric">
                        <span class="metric-label">State</span>
                        <span class="metric-value">{data['state']}</span>
                    </div>
                    <div class="metric">
                        <span class="metric-label">Uptime</span>
                        <span class="metric-value">{data.get('uptime_formatted', '00:00:00')}</span>
                    </div>
                </div>
                <div class="metric-row">
                    <div class="metric">
                        <span class="metric-label">Vision Frames</span>
                        <span class="metric-value">{data['vision_frames']}</span>
                    </div>
                    <div class="metric">
                        <span class="metric-label">ML Iterations</span>
                        <span class="metric-value">{data['metalearning_iterations']}</span>
                    </div>
                </div>
                <div class="metric-row">
                    <div class="metric">
                        <span class="metric-label">Programs Synthesized</span>
                        <span class="metric-value">{data['synthesis_count']}</span>
                    </div>
                </div>
            </div>
        </div>
        """

    def _cmp_tracker_html(self) -> str:
        """Generate CMP tracker panel HTML."""
        data = self.panels[DialogosPanelType.CMP_TRACKER]
        if not data:
            return "<div>No data</div>"

        fitness = data['fitness']
        fitness_pct = min(100, (fitness / 5.0) * 100)  # Scale to percentage

        return f"""
        <div class="dialogos-cmp-tracker" data-theme="{self.config.theme}">
            <div class="panel-header holographic">
                <h2>CMP Fitness Tracker</h2>
            </div>
            <div class="panel-content">
                <div class="fitness-display">
                    <div class="fitness-value">{fitness:.3f}</div>
                    <div class="fitness-label">Current Fitness</div>
                </div>
                <div class="progress-bar holographic">
                    <div class="progress-fill" style="width: {fitness_pct}%;"></div>
                </div>
                <div class="clade-stats">
                    <div class="clade-stat">
                        <span class="stat-label">Total Clades</span>
                        <span class="stat-value">{data['clade_count']}</span>
                    </div>
                    <div class="clade-stat">
                        <span class="stat-label">Active</span>
                        <span class="stat-value">{data['active_clades']}</span>
                    </div>
                    <div class="clade-stat">
                        <span class="stat-label">Converging</span>
                        <span class="stat-value">{data['converging_clades']}</span>
                    </div>
                </div>
            </div>
        </div>
        """

    def _icl_visualizer_html(self) -> str:
        """Generate ICL visualizer panel HTML."""
        data = self.panels[DialogosPanelType.ICL_VISUALIZER]
        if not data:
            return "<div>No data</div>"

        return f"""
        <div class="dialogos-icl-visualizer" data-theme="{self.config.theme}">
            <div class="panel-header holographic">
                <h2>ICL Buffer</h2>
            </div>
            <div class="panel-content">
                <div class="buffer-stats">
                    <div class="buffer-metric">
                        <span class="metric-label">Examples</span>
                        <span class="metric-value">{data['buffer_size']}</span>
                    </div>
                    <div class="buffer-metric">
                        <span class="metric-label">Utilization</span>
                        <span class="metric-value">{data['buffer_utilization']:.0%}</span>
                    </div>
                    <div class="buffer-metric">
                        <span class="metric-label">Avg Novelty</span>
                        <span class="metric-value">{data['avg_novelty']:.2f}</span>
                    </div>
                    <div class="buffer-metric">
                        <span class="metric-label">Avg Entropy</span>
                        <span class="metric-value">{data['avg_entropy']:.2f}</span>
                    </div>
                </div>
                <div class="geometric-embedding-view holographic">
                    <!-- Placeholder for geometric visualization -->
                    <canvas id="icl-embeddings-canvas"></canvas>
                </div>
            </div>
        </div>
        """

    def _event_stream_html(self) -> str:
        """Generate event stream panel HTML."""
        events_html = ""
        for event in reversed(self.events[-10:]):
            severity_class = f"severity-{event.severity}"
            progress_bar = ""
            if event.progress is not None:
                progress_bar = f'<div class="event-progress"><div style="width: {event.progress * 100}%"></div></div>'

            events_html += f"""
            <div class="event-row {severity_class}" data-type="{event.event_type}">
                <span class="event-icon">{event.icon or ''}</span>
                <div class="event-content">
                    <div class="event-title">{event.title}</div>
                    <div class="event-message">{event.message}</div>
                    {progress_bar}
                </div>
            </div>
            """

        return f"""
        <div class="dialogos-event-stream" data-theme="{self.config.theme}">
            <div class="panel-header holographic">
                <h2>Event Stream</h2>
            </div>
            <div class="panel-content events-list">
                {events_html if events_html else '<div class="no-events">No events yet</div>'}
            </div>
        </div>
        """

    def _control_panel_html(self) -> str:
        """Generate control panel HTML."""
        return """
        <div class="dialogos-control-panel" data-theme="cyberpunk">
            <div class="panel-header holographic">
                <h2>VIL Controls</h2>
            </div>
            <div class="panel-content">
                <div class="control-group">
                    <h3>Vision Pipeline</h3>
                    <button class="control-btn holographic" data-action="start_vision">Start Vision</button>
                    <button class="control-btn holographic" data-action="stop_vision">Stop Vision</button>
                </div>
                <div class="control-group">
                    <h3>Metalearning</h3>
                    <button class="control-btn holographic" data-action="start_learning">Start Learning</button>
                    <button class="control-btn holographic" data-action="stop_learning">Stop Learning</button>
                </div>
                <div class="control-group">
                    <h3>ICL Buffer</h3>
                    <button class="control-btn holographic" data-action="clear_buffer">Clear Buffer</button>
                    <button class="control-btn holographic" data-action="export_buffer">Export Buffer</button>
                </div>
                <div class="control-group">
                    <h3>Parameters</h3>
                    <div class="control-row">
                        <label>Temperature</label>
                        <input type="range" min="0" max="1" step="0.1" value="0.5" data-param="temperature">
                    </div>
                    <div class="control-row">
                        <label>Mutation Rate</label>
                        <input type="range" min="0" max="1" step="0.05" value="0.1" data-param="mutation_rate">
                    </div>
                </div>
            </div>
        </div>
        """

    def generate_css(self) -> str:
        """Generate CSS for Dialogos VIL enhancements."""
        colors = self.config.color_scheme

        return f"""
        .dialogos-vil-overview, .dialogos-cmp-tracker, .dialogos-icl-visualizer,
        .dialogos-event-stream, .dialogos-control-panel {{
            background: {colors['panel_bg']};
            border: 1px solid {colors['primary']};
            border-radius: 8px;
            padding: 16px;
            margin: 8px;
            box-shadow: 0 0 20px rgba(0, 255, 255, 0.2);
        }}

        .panel-header {{
            display: flex;
            justify-content: space-between;
            align-items: center;
            margin-bottom: 16px;
            padding-bottom: 8px;
            border-bottom: 1px solid {colors['primary']};
        }}

        .panel-header.holographic {{
            animation: holographic-glow 2s ease-in-out infinite;
        }}

        .status-indicator {{
            width: 12px;
            height: 12px;
            border-radius: 50%;
            animation: pulse 2s ease-in-out infinite;
        }}

        .status-active {{ background: {colors['success']}; }}
        .status-idle {{ background: {colors['warning']}; }}
        .status-error {{ background: {colors['error']}; }}

        .metric-row, .buffer-stats, .clade-stats {{
            display: flex;
            justify-content: space-around;
            margin: 12px 0;
        }}

        .metric, .buffer-metric, .clade-stat {{
            text-align: center;
        }}

        .metric-label, .stat-label {{
            color: {colors['primary']};
            font-size: 0.8em;
            text-transform: uppercase;
        }}

        .metric-value, .stat-value {{
            color: {colors['text']};
            font-size: 1.2em;
            font-weight: bold;
        }}

        .progress-bar {{
            height: 24px;
            background: rgba(0, 0, 0, 0.3);
            border-radius: 12px;
            overflow: hidden;
            margin: 12px 0;
        }}

        .progress-fill {{
            height: 100%;
            background: linear-gradient(90deg, {colors['primary']}, {colors['secondary']});
            transition: width 0.5s ease;
        }}

        .event-row {{
            padding: 8px;
            margin: 4px 0;
            border-left: 3px solid;
            background: rgba(255, 255, 255, 0.05);
        }}

        .severity-info {{ border-color: {colors['primary']}; }}
        .severity-success {{ border-color: {colors['success']}; }}
        .severity-warning {{ border-color: {colors['warning']}; }}
        .severity-error {{ border-color: {colors['error']}; }}

        .control-btn {{
            background: rgba(0, 255, 255, 0.1);
            border: 1px solid {colors['primary']};
            color: {colors['text']};
            padding: 8px 16px;
            margin: 4px;
            border-radius: 4px;
            cursor: pointer;
            transition: all 0.3s ease;
        }}

        .control-btn:hover {{
            background: rgba(0, 255, 255, 0.2);
            box-shadow: 0 0 10px {colors['primary']};
        }}

        @keyframes pulse {{
            0%, 100% {{ opacity: 1; }}
            50% {{ opacity: 0.5; }}
        }}

        @keyframes holographic-glow {{
            0%, 100% {{ box-shadow: 0 0 5px {colors['primary']}; }}
            50% {{ box-shadow: 0 0 20px {colors['secondary']}; }}
        }}
        """

    def export_config(self) -> str:
        """Export Dialogos configuration as JSON."""
        return json.dumps({
            "theme": self.config.theme,
            "color_scheme": self.config.color_scheme,
            "panels": {pt.value: data for pt, data in self.panels.items()},
        }, indent=2)


def create_dialogos_vil_enhancements(
    config: Optional[DialogosConfig] = None,
    bus_emitter: Optional[Callable] = None,
) -> DialogosVILEnhancements:
    """Create Dialogos VIL enhancements with default config."""
    return DialogosVILEnhancements(
        config=config,
        bus_emitter=bus_emitter,
    )


__all__ = [
    "DialogosPanelType",
    "DialogosConfig",
    "DialogosEvent",
    "DialogosVILEnhancements",
    "create_dialogos_vil_enhancements",
]
