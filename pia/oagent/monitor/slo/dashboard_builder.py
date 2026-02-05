#!/usr/bin/env python3
"""
Dashboard Builder - Step 269

Builds real-time monitoring dashboards.

PBTSO Phase: DISTILL

Bus Topics:
- monitor.dashboard.build (emitted)
- monitor.dashboard.update (emitted)
- monitor.dashboard.data (emitted)

Protocol: DKIN v30, CITIZEN v2, HOLON v2
"""

from __future__ import annotations

import json
import os
import socket
import time
import uuid
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional


class PanelType(Enum):
    """Dashboard panel types."""
    METRIC = "metric"
    GAUGE = "gauge"
    CHART = "chart"
    TABLE = "table"
    STATUS = "status"
    ALERT_LIST = "alert_list"
    LOG_VIEW = "log_view"
    HEATMAP = "heatmap"
    TEXT = "text"


class ChartType(Enum):
    """Chart types."""
    LINE = "line"
    AREA = "area"
    BAR = "bar"
    PIE = "pie"
    SCATTER = "scatter"


@dataclass
class DashboardLayout:
    """Dashboard layout configuration.

    Attributes:
        columns: Number of columns in grid
        row_height: Height of each row in pixels
        auto_refresh_s: Auto-refresh interval in seconds
    """
    columns: int = 12
    row_height: int = 80
    auto_refresh_s: int = 30

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return asdict(self)


@dataclass
class PanelPosition:
    """Panel position in dashboard grid.

    Attributes:
        x: X position (column)
        y: Y position (row)
        width: Width in columns
        height: Height in rows
    """
    x: int = 0
    y: int = 0
    width: int = 4
    height: int = 3

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return asdict(self)


@dataclass
class DataQuery:
    """Data query for panel.

    Attributes:
        source: Data source name
        metric: Metric name
        aggregation: Aggregation function
        window_s: Query window in seconds
        labels: Label filters
        group_by: Group by fields
    """
    source: str
    metric: str
    aggregation: str = "avg"
    window_s: int = 300
    labels: Dict[str, str] = field(default_factory=dict)
    group_by: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return asdict(self)


@dataclass
class ThresholdConfig:
    """Threshold configuration for panels.

    Attributes:
        warning: Warning threshold
        critical: Critical threshold
        direction: Threshold direction (above/below)
    """
    warning: float
    critical: float
    direction: str = "above"  # above or below

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return asdict(self)

    def get_state(self, value: float) -> str:
        """Get state based on value.

        Args:
            value: Current value

        Returns:
            State: ok, warning, or critical
        """
        if self.direction == "above":
            if value >= self.critical:
                return "critical"
            elif value >= self.warning:
                return "warning"
        else:
            if value <= self.critical:
                return "critical"
            elif value <= self.warning:
                return "warning"
        return "ok"


@dataclass
class DashboardPanel:
    """Dashboard panel.

    Attributes:
        panel_id: Unique panel ID
        title: Panel title
        panel_type: Type of panel
        position: Position in grid
        query: Data query
        chart_type: Chart type (for chart panels)
        thresholds: Threshold configuration
        unit: Display unit
        description: Panel description
    """
    panel_id: str
    title: str
    panel_type: PanelType
    position: PanelPosition = field(default_factory=PanelPosition)
    query: Optional[DataQuery] = None
    chart_type: ChartType = ChartType.LINE
    thresholds: Optional[ThresholdConfig] = None
    unit: str = ""
    description: str = ""

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "panel_id": self.panel_id,
            "title": self.title,
            "panel_type": self.panel_type.value,
            "position": self.position.to_dict(),
            "query": self.query.to_dict() if self.query else None,
            "chart_type": self.chart_type.value,
            "thresholds": self.thresholds.to_dict() if self.thresholds else None,
            "unit": self.unit,
            "description": self.description,
        }


@dataclass
class Dashboard:
    """Dashboard definition.

    Attributes:
        dashboard_id: Unique dashboard ID
        name: Dashboard name
        description: Dashboard description
        layout: Layout configuration
        panels: List of panels
        tags: Dashboard tags
        owner: Dashboard owner
        created_at: Creation timestamp
        updated_at: Update timestamp
    """
    dashboard_id: str
    name: str
    description: str = ""
    layout: DashboardLayout = field(default_factory=DashboardLayout)
    panels: List[DashboardPanel] = field(default_factory=list)
    tags: List[str] = field(default_factory=list)
    owner: str = ""
    created_at: float = field(default_factory=time.time)
    updated_at: float = field(default_factory=time.time)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "dashboard_id": self.dashboard_id,
            "name": self.name,
            "description": self.description,
            "layout": self.layout.to_dict(),
            "panels": [p.to_dict() for p in self.panels],
            "tags": self.tags,
            "owner": self.owner,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
        }

    def add_panel(self, panel: DashboardPanel) -> None:
        """Add a panel.

        Args:
            panel: Panel to add
        """
        self.panels.append(panel)
        self.updated_at = time.time()

    def remove_panel(self, panel_id: str) -> bool:
        """Remove a panel.

        Args:
            panel_id: Panel ID to remove

        Returns:
            True if removed
        """
        for i, panel in enumerate(self.panels):
            if panel.panel_id == panel_id:
                self.panels.pop(i)
                self.updated_at = time.time()
                return True
        return False

    def get_panel(self, panel_id: str) -> Optional[DashboardPanel]:
        """Get a panel by ID.

        Args:
            panel_id: Panel ID

        Returns:
            Panel or None
        """
        for panel in self.panels:
            if panel.panel_id == panel_id:
                return panel
        return None


@dataclass
class PanelData:
    """Data for a dashboard panel.

    Attributes:
        panel_id: Panel ID
        value: Current value (for metric panels)
        series: Time series data (for chart panels)
        rows: Table rows (for table panels)
        state: Current state (ok, warning, critical)
        timestamp: Data timestamp
    """
    panel_id: str
    value: Optional[float] = None
    series: List[Dict[str, Any]] = field(default_factory=list)
    rows: List[Dict[str, Any]] = field(default_factory=list)
    state: str = "ok"
    timestamp: float = field(default_factory=time.time)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return asdict(self)


class DashboardBuilder:
    """
    Build and manage monitoring dashboards.

    The builder:
    - Creates dashboard definitions
    - Provides panel templates
    - Manages dashboard storage
    - Supports real-time updates

    Example:
        builder = DashboardBuilder()

        # Create a dashboard
        dashboard = builder.create_dashboard(
            name="System Overview",
            description="System health and performance"
        )

        # Add panels
        builder.add_metric_panel(dashboard, "CPU Usage", "cpu.usage")
        builder.add_chart_panel(dashboard, "Request Rate", "requests.count")

        # Save dashboard
        builder.save_dashboard(dashboard)
    """

    BUS_TOPICS = {
        "build": "monitor.dashboard.build",
        "update": "monitor.dashboard.update",
        "data": "monitor.dashboard.data",
    }

    def __init__(
        self,
        storage_dir: Optional[str] = None,
        bus_dir: Optional[str] = None,
    ):
        """Initialize dashboard builder.

        Args:
            storage_dir: Directory for dashboard storage
            bus_dir: Bus directory
        """
        pluribus_root = os.environ.get("PLURIBUS_ROOT", "/pluribus")

        self._storage_dir = Path(storage_dir or os.path.join(pluribus_root, ".pluribus", "dashboards"))
        self._storage_dir.mkdir(parents=True, exist_ok=True)

        # Dashboard registry
        self._dashboards: Dict[str, Dashboard] = {}

        # Data providers
        self._data_providers: Dict[str, Callable[[DataQuery], Any]] = {}

        # Load existing dashboards
        self._load_dashboards()

        # Bus path
        self._bus_dir = bus_dir or os.path.join(pluribus_root, ".pluribus", "bus")
        self._bus_path = Path(self._bus_dir) / "events.ndjson"
        self._bus_path.parent.mkdir(parents=True, exist_ok=True)

    def create_dashboard(
        self,
        name: str,
        description: str = "",
        layout: Optional[DashboardLayout] = None,
        tags: Optional[List[str]] = None,
        owner: str = "",
    ) -> Dashboard:
        """Create a new dashboard.

        Args:
            name: Dashboard name
            description: Description
            layout: Layout configuration
            tags: Tags
            owner: Owner

        Returns:
            New dashboard
        """
        dashboard_id = str(uuid.uuid4())[:8]

        dashboard = Dashboard(
            dashboard_id=dashboard_id,
            name=name,
            description=description,
            layout=layout or DashboardLayout(),
            tags=tags or [],
            owner=owner,
        )

        self._dashboards[dashboard_id] = dashboard

        self._emit_bus_event(
            self.BUS_TOPICS["build"],
            {
                "dashboard_id": dashboard_id,
                "name": name,
                "action": "created",
            }
        )

        return dashboard

    def get_dashboard(self, dashboard_id: str) -> Optional[Dashboard]:
        """Get a dashboard by ID.

        Args:
            dashboard_id: Dashboard ID

        Returns:
            Dashboard or None
        """
        return self._dashboards.get(dashboard_id)

    def list_dashboards(self) -> List[Dict[str, Any]]:
        """List all dashboards.

        Returns:
            Dashboard summaries
        """
        return [
            {
                "dashboard_id": d.dashboard_id,
                "name": d.name,
                "description": d.description,
                "panel_count": len(d.panels),
                "tags": d.tags,
                "updated_at": d.updated_at,
            }
            for d in self._dashboards.values()
        ]

    def delete_dashboard(self, dashboard_id: str) -> bool:
        """Delete a dashboard.

        Args:
            dashboard_id: Dashboard ID

        Returns:
            True if deleted
        """
        if dashboard_id not in self._dashboards:
            return False

        del self._dashboards[dashboard_id]

        # Remove from storage
        filepath = self._storage_dir / f"{dashboard_id}.json"
        if filepath.exists():
            filepath.unlink()

        return True

    def save_dashboard(self, dashboard: Dashboard) -> str:
        """Save dashboard to storage.

        Args:
            dashboard: Dashboard to save

        Returns:
            File path
        """
        dashboard.updated_at = time.time()
        filepath = self._storage_dir / f"{dashboard.dashboard_id}.json"
        filepath.write_text(json.dumps(dashboard.to_dict(), indent=2))

        self._dashboards[dashboard.dashboard_id] = dashboard

        return str(filepath)

    def add_metric_panel(
        self,
        dashboard: Dashboard,
        title: str,
        metric: str,
        position: Optional[PanelPosition] = None,
        thresholds: Optional[ThresholdConfig] = None,
        unit: str = "",
    ) -> DashboardPanel:
        """Add a metric panel.

        Args:
            dashboard: Dashboard to add to
            title: Panel title
            metric: Metric name
            position: Panel position
            thresholds: Thresholds
            unit: Display unit

        Returns:
            New panel
        """
        panel = DashboardPanel(
            panel_id=str(uuid.uuid4())[:8],
            title=title,
            panel_type=PanelType.METRIC,
            position=position or self._auto_position(dashboard),
            query=DataQuery(source="metrics", metric=metric),
            thresholds=thresholds,
            unit=unit,
        )

        dashboard.add_panel(panel)
        return panel

    def add_gauge_panel(
        self,
        dashboard: Dashboard,
        title: str,
        metric: str,
        position: Optional[PanelPosition] = None,
        thresholds: Optional[ThresholdConfig] = None,
        unit: str = "%",
    ) -> DashboardPanel:
        """Add a gauge panel.

        Args:
            dashboard: Dashboard to add to
            title: Panel title
            metric: Metric name
            position: Panel position
            thresholds: Thresholds
            unit: Display unit

        Returns:
            New panel
        """
        panel = DashboardPanel(
            panel_id=str(uuid.uuid4())[:8],
            title=title,
            panel_type=PanelType.GAUGE,
            position=position or self._auto_position(dashboard),
            query=DataQuery(source="metrics", metric=metric),
            thresholds=thresholds or ThresholdConfig(warning=75.0, critical=90.0),
            unit=unit,
        )

        dashboard.add_panel(panel)
        return panel

    def add_chart_panel(
        self,
        dashboard: Dashboard,
        title: str,
        metric: str,
        chart_type: ChartType = ChartType.LINE,
        position: Optional[PanelPosition] = None,
        window_s: int = 3600,
    ) -> DashboardPanel:
        """Add a chart panel.

        Args:
            dashboard: Dashboard to add to
            title: Panel title
            metric: Metric name
            chart_type: Chart type
            position: Panel position
            window_s: Query window

        Returns:
            New panel
        """
        panel = DashboardPanel(
            panel_id=str(uuid.uuid4())[:8],
            title=title,
            panel_type=PanelType.CHART,
            position=position or self._auto_position(dashboard, width=6, height=4),
            query=DataQuery(source="metrics", metric=metric, window_s=window_s),
            chart_type=chart_type,
        )

        dashboard.add_panel(panel)
        return panel

    def add_table_panel(
        self,
        dashboard: Dashboard,
        title: str,
        source: str,
        position: Optional[PanelPosition] = None,
    ) -> DashboardPanel:
        """Add a table panel.

        Args:
            dashboard: Dashboard to add to
            title: Panel title
            source: Data source
            position: Panel position

        Returns:
            New panel
        """
        panel = DashboardPanel(
            panel_id=str(uuid.uuid4())[:8],
            title=title,
            panel_type=PanelType.TABLE,
            position=position or self._auto_position(dashboard, width=6, height=4),
            query=DataQuery(source=source, metric=""),
        )

        dashboard.add_panel(panel)
        return panel

    def add_status_panel(
        self,
        dashboard: Dashboard,
        title: str,
        services: List[str],
        position: Optional[PanelPosition] = None,
    ) -> DashboardPanel:
        """Add a status panel.

        Args:
            dashboard: Dashboard to add to
            title: Panel title
            services: Services to monitor
            position: Panel position

        Returns:
            New panel
        """
        panel = DashboardPanel(
            panel_id=str(uuid.uuid4())[:8],
            title=title,
            panel_type=PanelType.STATUS,
            position=position or self._auto_position(dashboard, width=3, height=2),
            query=DataQuery(
                source="service_health",
                metric="",
                labels={"services": ",".join(services)},
            ),
        )

        dashboard.add_panel(panel)
        return panel

    def add_alert_list_panel(
        self,
        dashboard: Dashboard,
        title: str = "Active Alerts",
        position: Optional[PanelPosition] = None,
    ) -> DashboardPanel:
        """Add an alert list panel.

        Args:
            dashboard: Dashboard to add to
            title: Panel title
            position: Panel position

        Returns:
            New panel
        """
        panel = DashboardPanel(
            panel_id=str(uuid.uuid4())[:8],
            title=title,
            panel_type=PanelType.ALERT_LIST,
            position=position or self._auto_position(dashboard, width=4, height=5),
            query=DataQuery(source="alerts", metric=""),
        )

        dashboard.add_panel(panel)
        return panel

    def add_text_panel(
        self,
        dashboard: Dashboard,
        title: str,
        content: str,
        position: Optional[PanelPosition] = None,
    ) -> DashboardPanel:
        """Add a text panel.

        Args:
            dashboard: Dashboard to add to
            title: Panel title
            content: Text content
            position: Panel position

        Returns:
            New panel
        """
        panel = DashboardPanel(
            panel_id=str(uuid.uuid4())[:8],
            title=title,
            panel_type=PanelType.TEXT,
            position=position or self._auto_position(dashboard, width=3, height=2),
            description=content,
        )

        dashboard.add_panel(panel)
        return panel

    def get_panel_data(
        self,
        dashboard_id: str,
        panel_id: str
    ) -> Optional[PanelData]:
        """Get data for a panel.

        Args:
            dashboard_id: Dashboard ID
            panel_id: Panel ID

        Returns:
            Panel data or None
        """
        dashboard = self.get_dashboard(dashboard_id)
        if not dashboard:
            return None

        panel = dashboard.get_panel(panel_id)
        if not panel or not panel.query:
            return None

        # Get data from provider
        provider = self._data_providers.get(panel.query.source)
        if not provider:
            return PanelData(panel_id=panel_id)

        raw_data = provider(panel.query)

        # Build panel data
        data = PanelData(panel_id=panel_id)

        if panel.panel_type in (PanelType.METRIC, PanelType.GAUGE):
            if isinstance(raw_data, (int, float)):
                data.value = float(raw_data)
            elif isinstance(raw_data, dict):
                data.value = raw_data.get("value", 0.0)

            # Determine state from thresholds
            if panel.thresholds and data.value is not None:
                data.state = panel.thresholds.get_state(data.value)

        elif panel.panel_type == PanelType.CHART:
            if isinstance(raw_data, list):
                data.series = raw_data

        elif panel.panel_type == PanelType.TABLE:
            if isinstance(raw_data, list):
                data.rows = raw_data

        return data

    def get_dashboard_data(
        self,
        dashboard_id: str
    ) -> Dict[str, PanelData]:
        """Get data for all panels in a dashboard.

        Args:
            dashboard_id: Dashboard ID

        Returns:
            Dictionary of panel data
        """
        dashboard = self.get_dashboard(dashboard_id)
        if not dashboard:
            return {}

        result = {}
        for panel in dashboard.panels:
            data = self.get_panel_data(dashboard_id, panel.panel_id)
            if data:
                result[panel.panel_id] = data

        # Emit data event
        self._emit_bus_event(
            self.BUS_TOPICS["data"],
            {
                "dashboard_id": dashboard_id,
                "panel_count": len(result),
            }
        )

        return result

    def register_data_provider(
        self,
        source: str,
        provider: Callable[[DataQuery], Any]
    ) -> None:
        """Register a data provider.

        Args:
            source: Source name
            provider: Provider function
        """
        self._data_providers[source] = provider

    def create_system_overview_dashboard(self) -> Dashboard:
        """Create a system overview dashboard.

        Returns:
            System overview dashboard
        """
        dashboard = self.create_dashboard(
            name="System Overview",
            description="System health and performance overview",
            tags=["system", "overview"],
        )

        # Row 1: Key metrics
        self.add_gauge_panel(dashboard, "CPU Usage", "cpu.usage",
                            PanelPosition(0, 0, 3, 3))
        self.add_gauge_panel(dashboard, "Memory Usage", "memory.usage",
                            PanelPosition(3, 0, 3, 3))
        self.add_gauge_panel(dashboard, "Disk Usage", "disk.usage",
                            PanelPosition(6, 0, 3, 3))
        self.add_metric_panel(dashboard, "Load Average", "system.load",
                             PanelPosition(9, 0, 3, 3), unit="")

        # Row 2: Charts
        self.add_chart_panel(dashboard, "Request Rate", "requests.rate",
                            ChartType.LINE, PanelPosition(0, 3, 6, 4))
        self.add_chart_panel(dashboard, "Error Rate", "errors.rate",
                            ChartType.LINE, PanelPosition(6, 3, 6, 4))

        # Row 3: Status and alerts
        self.add_status_panel(dashboard, "Service Status", ["api", "db", "cache"],
                             PanelPosition(0, 7, 4, 3))
        self.add_alert_list_panel(dashboard, "Active Alerts",
                                 PanelPosition(4, 7, 8, 3))

        self.save_dashboard(dashboard)
        return dashboard

    def create_slo_dashboard(self) -> Dashboard:
        """Create an SLO tracking dashboard.

        Returns:
            SLO dashboard
        """
        dashboard = self.create_dashboard(
            name="SLO Dashboard",
            description="Service Level Objective tracking",
            tags=["slo", "compliance"],
        )

        # SLO compliance gauges
        self.add_gauge_panel(dashboard, "API Availability", "slo.api.availability",
                            PanelPosition(0, 0, 4, 3),
                            ThresholdConfig(warning=99.5, critical=99.0, direction="below"),
                            unit="%")
        self.add_gauge_panel(dashboard, "API Latency P99", "slo.api.latency_p99",
                            PanelPosition(4, 0, 4, 3),
                            ThresholdConfig(warning=200, critical=500),
                            unit="ms")
        self.add_metric_panel(dashboard, "Error Budget", "slo.error_budget",
                             PanelPosition(8, 0, 4, 3), unit="%")

        # Compliance charts
        self.add_chart_panel(dashboard, "Availability Trend", "slo.availability.history",
                            ChartType.LINE, PanelPosition(0, 3, 6, 4))
        self.add_chart_panel(dashboard, "Latency Trend", "slo.latency.history",
                            ChartType.LINE, PanelPosition(6, 3, 6, 4))

        # SLO table
        self.add_table_panel(dashboard, "SLO Status", "slo_status",
                            PanelPosition(0, 7, 12, 4))

        self.save_dashboard(dashboard)
        return dashboard

    def _auto_position(
        self,
        dashboard: Dashboard,
        width: int = 3,
        height: int = 3
    ) -> PanelPosition:
        """Calculate auto-position for new panel.

        Args:
            dashboard: Dashboard
            width: Panel width
            height: Panel height

        Returns:
            Panel position
        """
        if not dashboard.panels:
            return PanelPosition(0, 0, width, height)

        # Find next available position
        max_y = 0
        for panel in dashboard.panels:
            panel_bottom = panel.position.y + panel.position.height
            if panel_bottom > max_y:
                max_y = panel_bottom

        return PanelPosition(0, max_y, width, height)

    def _load_dashboards(self) -> None:
        """Load dashboards from storage."""
        for filepath in self._storage_dir.glob("*.json"):
            try:
                data = json.loads(filepath.read_text())
                panels = []
                for panel_data in data.get("panels", []):
                    panel = DashboardPanel(
                        panel_id=panel_data["panel_id"],
                        title=panel_data["title"],
                        panel_type=PanelType(panel_data["panel_type"]),
                        position=PanelPosition(**panel_data.get("position", {})),
                        query=DataQuery(**panel_data["query"]) if panel_data.get("query") else None,
                        chart_type=ChartType(panel_data.get("chart_type", "line")),
                        thresholds=ThresholdConfig(**panel_data["thresholds"]) if panel_data.get("thresholds") else None,
                        unit=panel_data.get("unit", ""),
                        description=panel_data.get("description", ""),
                    )
                    panels.append(panel)

                dashboard = Dashboard(
                    dashboard_id=data["dashboard_id"],
                    name=data["name"],
                    description=data.get("description", ""),
                    layout=DashboardLayout(**data.get("layout", {})),
                    panels=panels,
                    tags=data.get("tags", []),
                    owner=data.get("owner", ""),
                    created_at=data.get("created_at", time.time()),
                    updated_at=data.get("updated_at", time.time()),
                )
                self._dashboards[dashboard.dashboard_id] = dashboard
            except Exception:
                continue

    def _emit_bus_event(
        self,
        topic: str,
        data: Dict[str, Any],
        level: str = "info",
        kind: str = "event"
    ) -> str:
        """Emit event to bus."""
        event_id = str(uuid.uuid4())
        event = {
            "id": event_id,
            "ts": time.time(),
            "iso": datetime.now(timezone.utc).isoformat() + "Z",
            "topic": topic,
            "kind": kind,
            "level": level,
            "actor": "monitor-agent",
            "host": socket.gethostname(),
            "pid": os.getpid(),
            "data": data,
        }

        try:
            with open(self._bus_path, "a") as f:
                f.write(json.dumps(event) + "\n")
        except Exception:
            pass

        return event_id


# Singleton
_builder: Optional[DashboardBuilder] = None


def get_dashboard_builder() -> DashboardBuilder:
    """Get or create the dashboard builder singleton."""
    global _builder
    if _builder is None:
        _builder = DashboardBuilder()
    return _builder


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Dashboard Builder (Step 269)")
    parser.add_argument("--list", action="store_true", help="List dashboards")
    parser.add_argument("--show", metavar="ID", help="Show dashboard details")
    parser.add_argument("--create-overview", action="store_true", help="Create system overview dashboard")
    parser.add_argument("--create-slo", action="store_true", help="Create SLO dashboard")
    parser.add_argument("--json", action="store_true", help="JSON output")

    args = parser.parse_args()

    builder = get_dashboard_builder()

    if args.list:
        dashboards = builder.list_dashboards()
        if args.json:
            print(json.dumps(dashboards, indent=2))
        else:
            print("Dashboards:")
            for d in dashboards:
                print(f"  [{d['dashboard_id']}] {d['name']} ({d['panel_count']} panels)")

    if args.show:
        dashboard = builder.get_dashboard(args.show)
        if dashboard:
            if args.json:
                print(json.dumps(dashboard.to_dict(), indent=2))
            else:
                print(f"Dashboard: {dashboard.name}")
                print(f"  ID: {dashboard.dashboard_id}")
                print(f"  Description: {dashboard.description}")
                print(f"  Panels:")
                for panel in dashboard.panels:
                    print(f"    - {panel.title} ({panel.panel_type.value})")
        else:
            print(f"Dashboard not found: {args.show}")

    if args.create_overview:
        dashboard = builder.create_system_overview_dashboard()
        print(f"Created System Overview dashboard: {dashboard.dashboard_id}")

    if args.create_slo:
        dashboard = builder.create_slo_dashboard()
        print(f"Created SLO dashboard: {dashboard.dashboard_id}")
