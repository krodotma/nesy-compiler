"""
VIL UI (User Interface) Module
Floating overlay UI and voice interface for VIL control.

Features:
1. VIL Control Center - floating overlay panels
2. Auralux Voice UI - voice command interface
3. Dialogos Enhancements - holographic UI panels
4. Real-time status monitoring
5. Interactive parameter controls

Integration points:
- Dashboard → VIL Control Center overlay
- Auralux → Voice command interface
- Dialogos → Holographic VIL visualization
- Bus Events → Real-time UI updates

Version: 1.0
Date: 2026-01-25
"""

from .control_center import (
    UIPanelType,
    UIPanelConfig,
    VILStatusData,
    UIEvent,
    VILControlCenter,
    create_vil_control_center,
)
from .auralux import (
    VoiceCommand,
    VoiceIntent,
    VoiceFeedback,
    AuraluxVILInterface,
    create_auralux_vil_interface,
)
from .dialogos import (
    DialogosPanelType,
    DialogosConfig,
    DialogosEvent,
    DialogosVILEnhancements,
    create_dialogos_vil_enhancements,
)

__all__ = [
    "UIPanelType",
    "UIPanelConfig",
    "VILStatusData",
    "UIEvent",
    "VILControlCenter",
    "create_vil_control_center",
    "VoiceCommand",
    "VoiceIntent",
    "VoiceFeedback",
    "AuraluxVILInterface",
    "create_auralux_vil_interface",
    "DialogosPanelType",
    "DialogosConfig",
    "DialogosEvent",
    "DialogosVILEnhancements",
    "create_dialogos_vil_enhancements",
]
