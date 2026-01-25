"""
Theia — Vision-First Neurosymbolic Agent
=========================================

Titaness of Sight: Endowing AI with celestial vision.

Theia consolidates all Pluribus vision, browser automation, program synthesis,
and self-teaching capabilities into a unified subproject.

Architecture:
    L5: Ω Reflexive Domain     — Self-modeling, metacognition
    L4: DNA Automata           — Coalgebraic reentry, self-modification
    L3: Birkhoff Polytope      — Sinkhorn crystallization to discrete
    L2: mHC                    — Energy landscape, attractors
    L1: Geometric Substrate    — S^n ⊣ H^n, fiber bundles
    L0: Vision + Browser       — Capture, agent-browser, ingest

Usage:
    from theia import Theia
    
    t = Theia()
    t.capture.start()           # Start screen capture
    t.browser.open("https://claude.ai")
    t.browser.fill("textarea", "Hello")
    snapshot = t.browser.snapshot()
"""

__version__ = "0.1.0"
__author__ = "Pluribus Team"

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from theia.capture import CaptureModule
    from theia.browser import BrowserModule


class Theia:
    """
    Main Theia interface — unified vision and browser automation.
    
    Attributes:
        capture: Vision capture module (ring buffer, screen, ingest)
        browser: Browser automation module (agent-browser wrapper)
    """
    
    def __init__(self):
        # Lazy imports to avoid circular dependencies
        from theia.capture import CaptureModule
        from theia.browser import BrowserModule
        
        self.capture = CaptureModule()
        self.browser = BrowserModule()
    
    def status(self) -> dict:
        """Return status of all Theia subsystems."""
        return {
            "version": __version__,
            "capture": self.capture.status() if hasattr(self.capture, 'status') else "not implemented",
            "browser": self.browser.status() if hasattr(self.browser, 'status') else "not implemented",
        }


__all__ = ["Theia", "__version__"]
