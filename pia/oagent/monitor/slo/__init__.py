#!/usr/bin/env python3
"""
SLO/SLI Systems - Monitor Agent Steps 261-270

Provides comprehensive SLO/SLI tracking, resource monitoring, capacity planning,
and monitoring orchestration.

Architecture:
- Resource Monitor: CPU/memory/disk tracking (Step 261)
- Network Monitor: Network health/latency (Step 262)
- Service Health: Service availability tracking (Step 263)
- SLO Tracker: SLO/SLA compliance (Step 264)
- Budget Monitor: Cost/resource budget tracking (Step 265)
- Capacity Planner: Capacity planning/forecasting (Step 266)
- Trend Analyzer: Long-term trend analysis (Step 267)
- Report Generator: Automated reporting (Step 268)
- Dashboard Builder: Real-time dashboard generation (Step 269)
- Monitor Orchestrator v2: Enhanced monitoring coordination (Step 270)

PBTSO Phases:
- SKILL: Resource monitoring initialization
- ITERATE: Continuous metric collection and analysis
- VERIFY: SLO compliance verification
- PLAN: Capacity planning
- DISTILL: Report generation

Bus Topics:
- monitor.resources.track
- monitor.network.health
- monitor.service.health
- monitor.slo.track
- monitor.budget.track
- monitor.capacity.plan
- monitor.trends.analyze
- monitor.report.generate
- monitor.dashboard.build
- a2a.monitor.orchestrate

Protocol: DKIN v30, PAIP v16, CITIZEN v2, HOLON v2
"""

from .resource_monitor import (
    ResourceMonitor,
    ResourceMetrics,
    CPUMetrics,
    MemoryMetrics,
    DiskMetrics,
    ResourceAlert,
)
from .network_monitor import (
    NetworkMonitor,
    NetworkMetrics,
    LatencyMetrics,
    BandwidthMetrics,
    ConnectionHealth,
)
from .service_health import (
    ServiceHealthMonitor,
    ServiceStatus,
    HealthCheckResult,
    ServiceDependency,
    HealthState,
)
from .slo_tracker import (
    SLOTracker,
    SLO,
    SLI,
    SLOTarget,
    SLOCompliance,
    ComplianceWindow,
)
from .budget_monitor import (
    BudgetMonitor,
    Budget,
    BudgetUsage,
    CostAllocation,
    BudgetAlert,
)
from .capacity_planner import (
    CapacityPlanner,
    CapacityForecast,
    ResourceProjection,
    ScalingRecommendation,
    CapacityThreshold,
)
from .trend_analyzer import (
    TrendAnalyzer,
    Trend,
    TrendType,
    TrendPeriod,
    SeasonalPattern,
)
from .report_generator import (
    ReportGenerator,
    Report,
    ReportType,
    ReportSection,
    ReportFormat,
)
from .dashboard_builder import (
    DashboardBuilder,
    Dashboard,
    DashboardPanel,
    PanelType,
    DashboardLayout,
)
from .orchestrator import (
    MonitorOrchestrator,
    MonitoringPipeline,
    PipelineStage,
    OrchestratorConfig,
)

__all__ = [
    # Resource Monitor (Step 261)
    "ResourceMonitor",
    "ResourceMetrics",
    "CPUMetrics",
    "MemoryMetrics",
    "DiskMetrics",
    "ResourceAlert",
    # Network Monitor (Step 262)
    "NetworkMonitor",
    "NetworkMetrics",
    "LatencyMetrics",
    "BandwidthMetrics",
    "ConnectionHealth",
    # Service Health (Step 263)
    "ServiceHealthMonitor",
    "ServiceStatus",
    "HealthCheckResult",
    "ServiceDependency",
    "HealthState",
    # SLO Tracker (Step 264)
    "SLOTracker",
    "SLO",
    "SLI",
    "SLOTarget",
    "SLOCompliance",
    "ComplianceWindow",
    # Budget Monitor (Step 265)
    "BudgetMonitor",
    "Budget",
    "BudgetUsage",
    "CostAllocation",
    "BudgetAlert",
    # Capacity Planner (Step 266)
    "CapacityPlanner",
    "CapacityForecast",
    "ResourceProjection",
    "ScalingRecommendation",
    "CapacityThreshold",
    # Trend Analyzer (Step 267)
    "TrendAnalyzer",
    "Trend",
    "TrendType",
    "TrendPeriod",
    "SeasonalPattern",
    # Report Generator (Step 268)
    "ReportGenerator",
    "Report",
    "ReportType",
    "ReportSection",
    "ReportFormat",
    # Dashboard Builder (Step 269)
    "DashboardBuilder",
    "Dashboard",
    "DashboardPanel",
    "PanelType",
    "DashboardLayout",
    # Monitor Orchestrator (Step 270)
    "MonitorOrchestrator",
    "MonitoringPipeline",
    "PipelineStage",
    "OrchestratorConfig",
]

__version__ = "0.1.0"
__step_range__ = "261-270"
