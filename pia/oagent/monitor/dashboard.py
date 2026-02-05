#!/usr/bin/env python3
"""
Monitor Metrics Dashboard - Step 271

Monitoring analytics and real-time metrics dashboard.

PBTSO Phase: REPORT

Bus Topics:
- monitor.dashboard.update (emitted)
- monitor.dashboard.refresh (subscribed)
- telemetry.* (subscribed)

Protocol: DKIN v30, PAIP v16, CITIZEN v2, HOLON v2
"""

from __future__ import annotations

import asyncio
import fcntl
import json
import os
import socket
import time
import uuid
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Set


class WidgetType(Enum):
    """Dashboard widget types."""
    GAUGE = "gauge"
    LINE_CHART = "line_chart"
    BAR_CHART = "bar_chart"
    PIE_CHART = "pie_chart"
    TABLE = "table"
    STAT = "stat"
    HEATMAP = "heatmap"
    LOG_STREAM = "log_stream"
    ALERT_LIST = "alert_list"


class TimeRange(Enum):
    """Time range options for dashboard."""
    LAST_5M = "5m"
    LAST_15M = "15m"
    LAST_1H = "1h"
    LAST_6H = "6h"
    LAST_24H = "24h"
    LAST_7D = "7d"
    LAST_30D = "30d"


@dataclass
class DashboardWidget:
    """A dashboard widget.

    Attributes:
        widget_id: Unique widget ID
        title: Widget title
        widget_type: Type of widget
        metric_query: Metric query expression
        config: Widget configuration
        position: Widget position (x, y, width, height)
        refresh_interval_s: Refresh interval
        thresholds: Alert thresholds
    """
    widget_id: str
    title: str
    widget_type: WidgetType
    metric_query: str
    config: Dict[str, Any] = field(default_factory=dict)
    position: Dict[str, int] = field(default_factory=lambda: {"x": 0, "y": 0, "w": 4, "h": 3})
    refresh_interval_s: int = 30
    thresholds: Dict[str, float] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "widget_id": self.widget_id,
            "title": self.title,
            "widget_type": self.widget_type.value,
            "metric_query": self.metric_query,
            "config": self.config,
            "position": self.position,
            "refresh_interval_s": self.refresh_interval_s,
            "thresholds": self.thresholds,
        }


@dataclass
class DashboardPanel:
    """A dashboard panel containing widgets.

    Attributes:
        panel_id: Unique panel ID
        title: Panel title
        widgets: Panel widgets
        collapsed: Whether panel is collapsed
        row: Panel row position
    """
    panel_id: str
    title: str
    widgets: List[DashboardWidget] = field(default_factory=list)
    collapsed: bool = False
    row: int = 0

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "panel_id": self.panel_id,
            "title": self.title,
            "widgets": [w.to_dict() for w in self.widgets],
            "collapsed": self.collapsed,
            "row": self.row,
        }


@dataclass
class Dashboard:
    """A metrics dashboard.

    Attributes:
        dashboard_id: Unique dashboard ID
        name: Dashboard name
        description: Dashboard description
        panels: Dashboard panels
        time_range: Default time range
        refresh_interval_s: Auto-refresh interval
        variables: Dashboard variables
        tags: Dashboard tags
        created_at: Creation timestamp
        updated_at: Last update timestamp
    """
    dashboard_id: str
    name: str
    description: str = ""
    panels: List[DashboardPanel] = field(default_factory=list)
    time_range: TimeRange = TimeRange.LAST_1H
    refresh_interval_s: int = 30
    variables: Dict[str, Any] = field(default_factory=dict)
    tags: List[str] = field(default_factory=list)
    created_at: float = field(default_factory=time.time)
    updated_at: float = field(default_factory=time.time)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "dashboard_id": self.dashboard_id,
            "name": self.name,
            "description": self.description,
            "panels": [p.to_dict() for p in self.panels],
            "time_range": self.time_range.value,
            "refresh_interval_s": self.refresh_interval_s,
            "variables": self.variables,
            "tags": self.tags,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
        }

    def add_panel(self, panel: DashboardPanel) -> None:
        """Add a panel to the dashboard."""
        self.panels.append(panel)
        self.updated_at = time.time()

    def add_widget(self, panel_id: str, widget: DashboardWidget) -> bool:
        """Add a widget to a panel."""
        for panel in self.panels:
            if panel.panel_id == panel_id:
                panel.widgets.append(widget)
                self.updated_at = time.time()
                return True
        return False


@dataclass
class WidgetData:
    """Data for a widget.

    Attributes:
        widget_id: Widget ID
        data: Widget data
        timestamp: Data timestamp
        error: Error if any
    """
    widget_id: str
    data: Any
    timestamp: float = field(default_factory=time.time)
    error: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "widget_id": self.widget_id,
            "data": self.data,
            "timestamp": self.timestamp,
            "error": self.error,
        }


class MetricsDashboard:
    """
    Monitor metrics dashboard for real-time analytics.

    The dashboard:
    - Creates and manages dashboards
    - Provides real-time metric visualization data
    - Supports multiple widget types
    - Handles dashboard persistence

    Example:
        dashboard = MetricsDashboard()

        # Create a dashboard
        db = dashboard.create_dashboard(
            name="System Overview",
            description="System health metrics"
        )

        # Add a panel
        panel = dashboard.add_panel(db.dashboard_id, "CPU Metrics")

        # Add widgets
        dashboard.add_widget(
            db.dashboard_id,
            panel.panel_id,
            DashboardWidget(
                widget_id="cpu-gauge",
                title="CPU Usage",
                widget_type=WidgetType.GAUGE,
                metric_query="system.cpu.usage",
            )
        )

        # Get widget data
        data = await dashboard.get_widget_data(db.dashboard_id, "cpu-gauge")
    """

    BUS_TOPICS = {
        "update": "monitor.dashboard.update",
        "refresh": "monitor.dashboard.refresh",
    }

    # A2A heartbeat settings
    HEARTBEAT_INTERVAL = 300  # 300s
    HEARTBEAT_TIMEOUT = 900   # 900s

    def __init__(
        self,
        metric_collector: Optional[Any] = None,
        bus_dir: Optional[str] = None,
    ):
        """Initialize metrics dashboard.

        Args:
            metric_collector: Optional metric collector instance
            bus_dir: Bus directory
        """
        self._metric_collector = metric_collector
        self._dashboards: Dict[str, Dashboard] = {}
        self._widget_data_cache: Dict[str, WidgetData] = {}
        self._refresh_callbacks: List[Callable] = []
        self._last_heartbeat = time.time()

        # Bus path with file locking support
        pluribus_root = os.environ.get("PLURIBUS_ROOT", "/pluribus")
        self._bus_dir = bus_dir or os.path.join(pluribus_root, ".pluribus", "bus")
        self._bus_path = Path(self._bus_dir) / "events.ndjson"
        self._bus_path.parent.mkdir(parents=True, exist_ok=True)

        # Create default dashboards
        self._create_default_dashboards()

    def create_dashboard(
        self,
        name: str,
        description: str = "",
        time_range: TimeRange = TimeRange.LAST_1H,
        tags: Optional[List[str]] = None,
    ) -> Dashboard:
        """Create a new dashboard.

        Args:
            name: Dashboard name
            description: Description
            time_range: Default time range
            tags: Dashboard tags

        Returns:
            Created dashboard
        """
        dashboard_id = f"dash-{uuid.uuid4().hex[:8]}"

        dashboard = Dashboard(
            dashboard_id=dashboard_id,
            name=name,
            description=description,
            time_range=time_range,
            tags=tags or [],
        )

        self._dashboards[dashboard_id] = dashboard

        self._emit_bus_event(
            self.BUS_TOPICS["update"],
            {
                "action": "created",
                "dashboard_id": dashboard_id,
                "name": name,
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
                "widget_count": sum(len(p.widgets) for p in d.panels),
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
        if dashboard_id in self._dashboards:
            del self._dashboards[dashboard_id]

            self._emit_bus_event(
                self.BUS_TOPICS["update"],
                {
                    "action": "deleted",
                    "dashboard_id": dashboard_id,
                }
            )
            return True
        return False

    def add_panel(
        self,
        dashboard_id: str,
        title: str,
        row: int = 0,
    ) -> Optional[DashboardPanel]:
        """Add a panel to a dashboard.

        Args:
            dashboard_id: Dashboard ID
            title: Panel title
            row: Row position

        Returns:
            Created panel or None
        """
        dashboard = self._dashboards.get(dashboard_id)
        if not dashboard:
            return None

        panel = DashboardPanel(
            panel_id=f"panel-{uuid.uuid4().hex[:8]}",
            title=title,
            row=row,
        )

        dashboard.add_panel(panel)
        return panel

    def add_widget(
        self,
        dashboard_id: str,
        panel_id: str,
        widget: DashboardWidget,
    ) -> bool:
        """Add a widget to a dashboard panel.

        Args:
            dashboard_id: Dashboard ID
            panel_id: Panel ID
            widget: Widget to add

        Returns:
            True if added
        """
        dashboard = self._dashboards.get(dashboard_id)
        if not dashboard:
            return False

        return dashboard.add_widget(panel_id, widget)

    async def get_widget_data(
        self,
        dashboard_id: str,
        widget_id: str,
        time_range: Optional[TimeRange] = None,
    ) -> Optional[WidgetData]:
        """Get data for a widget.

        Args:
            dashboard_id: Dashboard ID
            widget_id: Widget ID
            time_range: Optional time range override

        Returns:
            Widget data or None
        """
        dashboard = self._dashboards.get(dashboard_id)
        if not dashboard:
            return None

        # Find widget
        widget = None
        for panel in dashboard.panels:
            for w in panel.widgets:
                if w.widget_id == widget_id:
                    widget = w
                    break
            if widget:
                break

        if not widget:
            return None

        # Check cache
        cache_key = f"{dashboard_id}:{widget_id}"
        cached = self._widget_data_cache.get(cache_key)
        if cached and time.time() - cached.timestamp < widget.refresh_interval_s:
            return cached

        # Query metrics
        data = await self._query_widget_data(widget, time_range or dashboard.time_range)

        widget_data = WidgetData(
            widget_id=widget_id,
            data=data,
        )

        self._widget_data_cache[cache_key] = widget_data
        return widget_data

    async def refresh_dashboard(self, dashboard_id: str) -> Dict[str, WidgetData]:
        """Refresh all widgets in a dashboard.

        Args:
            dashboard_id: Dashboard ID

        Returns:
            Dictionary of widget_id -> WidgetData
        """
        dashboard = self._dashboards.get(dashboard_id)
        if not dashboard:
            return {}

        results: Dict[str, WidgetData] = {}

        for panel in dashboard.panels:
            for widget in panel.widgets:
                data = await self.get_widget_data(dashboard_id, widget.widget_id)
                if data:
                    results[widget.widget_id] = data

        self._emit_bus_event(
            self.BUS_TOPICS["refresh"],
            {
                "dashboard_id": dashboard_id,
                "widgets_refreshed": len(results),
            }
        )

        return results

    def get_dashboard_snapshot(self, dashboard_id: str) -> Dict[str, Any]:
        """Get a snapshot of dashboard data.

        Args:
            dashboard_id: Dashboard ID

        Returns:
            Dashboard snapshot
        """
        dashboard = self._dashboards.get(dashboard_id)
        if not dashboard:
            return {}

        snapshot = dashboard.to_dict()
        snapshot["widget_data"] = {
            k.split(":")[-1]: v.to_dict()
            for k, v in self._widget_data_cache.items()
            if k.startswith(f"{dashboard_id}:")
        }
        snapshot["snapshot_time"] = time.time()

        return snapshot

    def register_refresh_callback(self, callback: Callable) -> None:
        """Register a callback for dashboard refresh events.

        Args:
            callback: Callback function
        """
        self._refresh_callbacks.append(callback)

    def get_analytics_summary(self) -> Dict[str, Any]:
        """Get analytics summary across all dashboards.

        Returns:
            Analytics summary
        """
        total_widgets = 0
        widget_types: Dict[str, int] = {}

        for dashboard in self._dashboards.values():
            for panel in dashboard.panels:
                for widget in panel.widgets:
                    total_widgets += 1
                    wtype = widget.widget_type.value
                    widget_types[wtype] = widget_types.get(wtype, 0) + 1

        return {
            "dashboards": len(self._dashboards),
            "total_widgets": total_widgets,
            "widget_types": widget_types,
            "cached_data_points": len(self._widget_data_cache),
            "last_heartbeat": self._last_heartbeat,
        }

    def emit_heartbeat(self) -> bool:
        """Emit heartbeat for A2A protocol.

        Returns:
            True if heartbeat was emitted
        """
        now = time.time()
        if now - self._last_heartbeat < self.HEARTBEAT_INTERVAL - 30:
            return False

        self._last_heartbeat = now

        self._emit_bus_event(
            "a2a.heartbeat",
            {
                "component": "dashboard",
                "status": "healthy",
                "dashboards": len(self._dashboards),
                "uptime_s": now,
            }
        )

        return True

    async def _query_widget_data(
        self,
        widget: DashboardWidget,
        time_range: TimeRange,
    ) -> Any:
        """Query data for a widget.

        Args:
            widget: Widget to query
            time_range: Time range

        Returns:
            Widget data
        """
        # Parse time range to seconds
        range_seconds = self._parse_time_range(time_range)

        # If we have a metric collector, use it
        if self._metric_collector:
            try:
                if widget.widget_type == WidgetType.GAUGE:
                    return self._metric_collector.query(
                        widget.metric_query,
                        aggregation="last",
                        window_s=range_seconds
                    )
                elif widget.widget_type == WidgetType.LINE_CHART:
                    points = self._metric_collector.query_series(
                        widget.metric_query,
                        window_s=range_seconds
                    )
                    return [{"x": p.timestamp, "y": p.value} for p in points]
                elif widget.widget_type == WidgetType.STAT:
                    return {
                        "current": self._metric_collector.query(widget.metric_query, "last", range_seconds),
                        "avg": self._metric_collector.query(widget.metric_query, "avg", range_seconds),
                        "max": self._metric_collector.query(widget.metric_query, "max", range_seconds),
                        "min": self._metric_collector.query(widget.metric_query, "min", range_seconds),
                    }
            except Exception as e:
                return {"error": str(e)}

        # Return placeholder data
        return self._generate_placeholder_data(widget)

    def _parse_time_range(self, time_range: TimeRange) -> int:
        """Parse time range to seconds."""
        ranges = {
            TimeRange.LAST_5M: 300,
            TimeRange.LAST_15M: 900,
            TimeRange.LAST_1H: 3600,
            TimeRange.LAST_6H: 21600,
            TimeRange.LAST_24H: 86400,
            TimeRange.LAST_7D: 604800,
            TimeRange.LAST_30D: 2592000,
        }
        return ranges.get(time_range, 3600)

    def _generate_placeholder_data(self, widget: DashboardWidget) -> Any:
        """Generate placeholder data for a widget."""
        if widget.widget_type == WidgetType.GAUGE:
            return 0.0
        elif widget.widget_type in (WidgetType.LINE_CHART, WidgetType.BAR_CHART):
            return []
        elif widget.widget_type == WidgetType.STAT:
            return {"current": 0, "avg": 0, "max": 0, "min": 0}
        elif widget.widget_type == WidgetType.TABLE:
            return {"columns": [], "rows": []}
        elif widget.widget_type == WidgetType.ALERT_LIST:
            return []
        return None

    def _create_default_dashboards(self) -> None:
        """Create default monitoring dashboards."""
        # System Overview dashboard
        system_dashboard = self.create_dashboard(
            name="System Overview",
            description="System health and resource metrics",
            tags=["system", "default"],
        )

        # Add resources panel
        resources_panel = self.add_panel(system_dashboard.dashboard_id, "Resources", row=0)
        if resources_panel:
            self.add_widget(
                system_dashboard.dashboard_id,
                resources_panel.panel_id,
                DashboardWidget(
                    widget_id="cpu-usage",
                    title="CPU Usage",
                    widget_type=WidgetType.GAUGE,
                    metric_query="system.cpu.usage",
                    thresholds={"warning": 70, "critical": 90},
                )
            )
            self.add_widget(
                system_dashboard.dashboard_id,
                resources_panel.panel_id,
                DashboardWidget(
                    widget_id="memory-usage",
                    title="Memory Usage",
                    widget_type=WidgetType.GAUGE,
                    metric_query="system.memory.usage",
                    thresholds={"warning": 80, "critical": 95},
                )
            )

        # Agent Health dashboard
        agent_dashboard = self.create_dashboard(
            name="Agent Health",
            description="OAGENT health metrics",
            tags=["agents", "default"],
        )

        # Add health panel
        health_panel = self.add_panel(agent_dashboard.dashboard_id, "Agent Status", row=0)
        if health_panel:
            self.add_widget(
                agent_dashboard.dashboard_id,
                health_panel.panel_id,
                DashboardWidget(
                    widget_id="agent-health",
                    title="Agent Health",
                    widget_type=WidgetType.TABLE,
                    metric_query="agent.health.*",
                )
            )
            self.add_widget(
                agent_dashboard.dashboard_id,
                health_panel.panel_id,
                DashboardWidget(
                    widget_id="active-alerts",
                    title="Active Alerts",
                    widget_type=WidgetType.ALERT_LIST,
                    metric_query="alert.active.*",
                )
            )

    def _emit_bus_event(
        self,
        topic: str,
        data: Dict[str, Any],
        level: str = "info",
        kind: str = "event"
    ) -> str:
        """Emit event to bus with file locking.

        Args:
            topic: Event topic
            data: Event data
            level: Log level
            kind: Event kind

        Returns:
            Event ID
        """
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

        # Use file locking for bus writes (fcntl.flock)
        try:
            with open(self._bus_path, "a") as f:
                fcntl.flock(f.fileno(), fcntl.LOCK_EX)
                try:
                    f.write(json.dumps(event) + "\n")
                finally:
                    fcntl.flock(f.fileno(), fcntl.LOCK_UN)
        except Exception:
            pass

        return event_id


# Singleton instance
_dashboard: Optional[MetricsDashboard] = None


def get_dashboard() -> MetricsDashboard:
    """Get or create the metrics dashboard singleton.

    Returns:
        MetricsDashboard instance
    """
    global _dashboard
    if _dashboard is None:
        _dashboard = MetricsDashboard()
    return _dashboard


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Monitor Metrics Dashboard (Step 271)")
    parser.add_argument("--list", action="store_true", help="List dashboards")
    parser.add_argument("--create", metavar="NAME", help="Create dashboard")
    parser.add_argument("--show", metavar="ID", help="Show dashboard details")
    parser.add_argument("--delete", metavar="ID", help="Delete dashboard")
    parser.add_argument("--summary", action="store_true", help="Show analytics summary")
    parser.add_argument("--json", action="store_true", help="JSON output")

    args = parser.parse_args()

    dashboard = get_dashboard()

    if args.list:
        dashboards = dashboard.list_dashboards()
        if args.json:
            print(json.dumps(dashboards, indent=2))
        else:
            print("Dashboards:")
            for d in dashboards:
                print(f"  [{d['dashboard_id']}] {d['name']} ({d['widget_count']} widgets)")

    if args.create:
        db = dashboard.create_dashboard(name=args.create)
        if args.json:
            print(json.dumps(db.to_dict(), indent=2))
        else:
            print(f"Created dashboard: {db.dashboard_id}")

    if args.show:
        db = dashboard.get_dashboard(args.show)
        if db:
            if args.json:
                print(json.dumps(db.to_dict(), indent=2))
            else:
                print(f"Dashboard: {db.name}")
                print(f"  ID: {db.dashboard_id}")
                print(f"  Panels: {len(db.panels)}")
                for panel in db.panels:
                    print(f"    - {panel.title}: {len(panel.widgets)} widgets")
        else:
            print(f"Dashboard not found: {args.show}")

    if args.delete:
        success = dashboard.delete_dashboard(args.delete)
        print(f"Deleted: {success}")

    if args.summary:
        summary = dashboard.get_analytics_summary()
        if args.json:
            print(json.dumps(summary, indent=2))
        else:
            print("Analytics Summary:")
            for k, v in summary.items():
                print(f"  {k}: {v}")
