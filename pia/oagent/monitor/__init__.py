#!/usr/bin/env python3
"""
Monitor Agent - OAGENT Subagent 6 (Steps 251-300)

Provides observability, metrics collection, anomaly detection, and alerting systems.

Architecture:
- Bootstrap: Agent initialization and A2A registration
- Metric Collector: Telemetry aggregation from all agents
- Metric Aggregator: Statistical rollups and windowed calculations
- Log Collector: Centralized log ingestion
- Log Analyzer: Pattern detection and classification
- Log Correlator: Cross-service correlation
- Anomaly Detector: Z-score and statistical anomaly detection
- Alert Router: Alert routing to appropriate channels
- Alert Manager: Alert lifecycle management
- Incident Automator: Automated incident response
- Dashboard: Real-time metrics visualization
- Report Generator: Automated report generation
- Scheduler: Scheduled monitoring tasks
- Correlation Engine: Event correlation and pattern detection
- Root Cause Analyzer: Automated RCA
- Prediction Engine: Predictive alerting
- Integration Hub: External service integrations
- Notification System: Alert routing and delivery
- API: REST API for monitor operations
- CLI: Command-line interface

PBTSO Phases:
- SKILL: Bootstrap initialization, API, CLI
- SEQUESTER: Security sandboxing for log access
- ITERATE: Metric/log collection and alert routing
- VERIFY: Anomaly detection and log analysis
- DISTILL: Metric aggregation
- RESEARCH: Log correlation, RCA, Prediction
- REPORT: Dashboards, Report generation
- DISTRIBUTE: Integration hub, Notifications
- PLAN: Orchestration, Scheduling

Bus Topics:
- a2a.monitor.bootstrap.start
- a2a.monitor.bootstrap.complete
- telemetry.*
- monitor.metrics.collected
- monitor.metrics.aggregate
- monitor.metrics.aggregated
- monitor.logs.collected
- monitor.logs.analyze
- monitor.logs.patterns
- monitor.logs.correlate
- monitor.logs.correlated
- monitor.anomaly.detect
- monitor.anomaly.detected
- monitor.alert.route
- monitor.alert.sent
- monitor.alert.create
- monitor.alert.acknowledge
- monitor.alert.resolve
- monitor.incident.create
- monitor.incident.escalate
- monitor.incident.resolve
- monitor.dashboard.update
- monitor.dashboard.refresh
- monitor.report.generate
- monitor.report.complete
- monitor.schedule.create
- monitor.schedule.execute
- monitor.schedule.complete
- monitor.correlation.detect
- monitor.correlation.found
- monitor.rca.analyze
- monitor.rca.result
- monitor.prediction.forecast
- monitor.prediction.alert
- monitor.integration.send
- monitor.integration.receive
- monitor.notification.send
- monitor.notification.delivered
- monitor.api.request
- monitor.api.response
- monitor.cli.command

Protocol: DKIN v30, CITIZEN v2, PAIP v16, HOLON v2
"""

from .bootstrap import MonitorAgentBootstrap, MonitorAgentConfig

# Metrics (Steps 252-253)
from .metrics.collector import MetricCollector, MetricPoint, MetricType
from .metrics.aggregator import MetricAggregator, AggregatedMetric, AggregationType

# Logs (Steps 254-256)
from .logs.collector import LogCollector, LogEntry, LogLevel
from .logs.analyzer import LogAnalyzer, LogPattern, PatternType
from .logs.correlator import LogCorrelator, CorrelatedEvent, CorrelationChain

# Anomaly Detection (Step 257)
from .anomaly.detector import AnomalyDetector, Anomaly, AnomalySeverity, DetectionMethod

# Alerts (Steps 258-259)
from .alerts.router import AlertRouter, AlertRoute, AlertChannel, RoutingRule
from .alerts.manager import AlertManager, Alert, AlertState, AlertPriority

# Incidents (Step 260)
from .incident.automator import (
    IncidentAutomator,
    Incident,
    IncidentState,
    IncidentSeverity,
    ResponseAction,
    ResponsePlaybook,
)

# Dashboard (Step 271)
from .dashboard import (
    MetricsDashboard,
    Dashboard,
    DashboardPanel,
    DashboardWidget,
    WidgetType,
    TimeRange,
)

# Report Generator (Step 272)
from .report import (
    ReportGenerator,
    Report,
    ReportSection,
    ReportTemplate,
    ReportFormat,
    ReportPeriod,
    ReportType,
)

# Scheduler (Step 273)
from .scheduler import (
    MonitorScheduler,
    ScheduledTask,
    TaskExecution,
    ScheduleType,
    TaskState,
    TaskPriority,
)

# Correlation Engine (Step 274)
from .correlation import (
    CorrelationEngine,
    Correlation,
    CorrelationRule,
    Event,
    CorrelationType,
    CorrelationStrength,
)

# Root Cause Analyzer (Step 275)
from .rca import (
    RootCauseAnalyzer,
    RCAResult,
    CauseHypothesis,
    Evidence,
    RCAStatus,
    CauseCategory,
    ConfidenceLevel,
)

# Prediction Engine (Step 276)
from .prediction import (
    PredictionEngine,
    Prediction,
    PredictiveAlert,
    MetricDataPoint,
    ThresholdConfig,
    PredictionType,
    AlertLevel,
)

# Integration Hub (Step 277)
from .integration import (
    IntegrationHub,
    IntegrationConfig,
    IntegrationMessage,
    DeliveryResult,
    IntegrationType,
    IntegrationStatus,
)

# Notification System (Step 278)
from .notification import (
    NotificationSystem,
    Notification,
    NotificationRecipient,
    RoutingRule as NotificationRoutingRule,
    NotificationChannel,
    NotificationPriority,
    NotificationStatus,
)

# API (Step 279)
from .api import (
    MonitorAPI,
    APIRequest,
    APIResponse,
    APIEndpoint,
    HTTPMethod,
    APIStatus,
)

# CLI (Step 280)
from .cli import (
    MonitorCLI,
    CLIContext,
    OutputFormat,
)

__all__ = [
    # Bootstrap (Step 251)
    "MonitorAgentBootstrap",
    "MonitorAgentConfig",
    # Metrics (Steps 252-253)
    "MetricCollector",
    "MetricPoint",
    "MetricType",
    "MetricAggregator",
    "AggregatedMetric",
    "AggregationType",
    # Logs (Steps 254-256)
    "LogCollector",
    "LogEntry",
    "LogLevel",
    "LogAnalyzer",
    "LogPattern",
    "PatternType",
    "LogCorrelator",
    "CorrelatedEvent",
    "CorrelationChain",
    # Anomaly (Step 257)
    "AnomalyDetector",
    "Anomaly",
    "AnomalySeverity",
    "DetectionMethod",
    # Alerts (Steps 258-259)
    "AlertRouter",
    "AlertRoute",
    "AlertChannel",
    "RoutingRule",
    "AlertManager",
    "Alert",
    "AlertState",
    "AlertPriority",
    # Incidents (Step 260)
    "IncidentAutomator",
    "Incident",
    "IncidentState",
    "IncidentSeverity",
    "ResponseAction",
    "ResponsePlaybook",
    # Dashboard (Step 271)
    "MetricsDashboard",
    "Dashboard",
    "DashboardPanel",
    "DashboardWidget",
    "WidgetType",
    "TimeRange",
    # Report Generator (Step 272)
    "ReportGenerator",
    "Report",
    "ReportSection",
    "ReportTemplate",
    "ReportFormat",
    "ReportPeriod",
    "ReportType",
    # Scheduler (Step 273)
    "MonitorScheduler",
    "ScheduledTask",
    "TaskExecution",
    "ScheduleType",
    "TaskState",
    "TaskPriority",
    # Correlation Engine (Step 274)
    "CorrelationEngine",
    "Correlation",
    "CorrelationRule",
    "Event",
    "CorrelationType",
    "CorrelationStrength",
    # Root Cause Analyzer (Step 275)
    "RootCauseAnalyzer",
    "RCAResult",
    "CauseHypothesis",
    "Evidence",
    "RCAStatus",
    "CauseCategory",
    "ConfidenceLevel",
    # Prediction Engine (Step 276)
    "PredictionEngine",
    "Prediction",
    "PredictiveAlert",
    "MetricDataPoint",
    "ThresholdConfig",
    "PredictionType",
    "AlertLevel",
    # Integration Hub (Step 277)
    "IntegrationHub",
    "IntegrationConfig",
    "IntegrationMessage",
    "DeliveryResult",
    "IntegrationType",
    "IntegrationStatus",
    # Notification System (Step 278)
    "NotificationSystem",
    "Notification",
    "NotificationRecipient",
    "NotificationRoutingRule",
    "NotificationChannel",
    "NotificationPriority",
    "NotificationStatus",
    # API (Step 279)
    "MonitorAPI",
    "APIRequest",
    "APIResponse",
    "APIEndpoint",
    "HTTPMethod",
    "APIStatus",
    # CLI (Step 280)
    "MonitorCLI",
    "CLIContext",
    "OutputFormat",
]

__version__ = "0.2.0"
__step_range__ = "251-280"
