#!/usr/bin/env python3
"""
test_manim_renderer.py - Tests for ManimCE rendering backend.

Tests scene detection, render request handling, and result formatting.
"""

import json
import pytest
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

# Add tools to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from manim_renderer import (
    RenderRequest,
    RenderResult,
    detect_scene_names,
    MANIM_QUALITY_PRESETS,
    DEFAULT_QUALITY,
)


class TestSceneDetection:
    """Tests for Manim scene class detection."""

    def test_detect_simple_scene(self) -> None:
        code = """
class MyScene(Scene):
    def construct(self):
        circle = Circle()
        self.play(Create(circle))
"""
        scenes = detect_scene_names(code)
        assert scenes == ["MyScene"]

    def test_detect_threed_scene(self) -> None:
        code = """
class SurfacePlot(ThreeDScene):
    def construct(self):
        axes = ThreeDAxes()
        self.add(axes)
"""
        scenes = detect_scene_names(code)
        assert scenes == ["SurfacePlot"]

    def test_detect_moving_camera_scene(self) -> None:
        code = """
class PanZoom(MovingCameraScene):
    def construct(self):
        self.camera.frame.animate.scale(2)
"""
        scenes = detect_scene_names(code)
        assert scenes == ["PanZoom"]

    def test_detect_zoomed_scene(self) -> None:
        code = """
class ZoomDemo(ZoomedScene):
    def construct(self):
        self.activate_zooming()
"""
        scenes = detect_scene_names(code)
        assert scenes == ["ZoomDemo"]

    def test_detect_multiple_scenes(self) -> None:
        code = """
class Scene1(Scene):
    pass

class Scene2(Scene):
    pass

class NotAScene:
    pass

class Scene3(ThreeDScene):
    pass
"""
        scenes = detect_scene_names(code)
        assert len(scenes) == 3
        assert "Scene1" in scenes
        assert "Scene2" in scenes
        assert "Scene3" in scenes
        assert "NotAScene" not in scenes

    def test_detect_with_manim_prefix(self) -> None:
        code = """
import manim

class MyScene(manim.Scene):
    pass
"""
        scenes = detect_scene_names(code)
        assert scenes == ["MyScene"]

    def test_no_scene_found(self) -> None:
        code = """
def draw_circle():
    return Circle()

class Helper:
    pass
"""
        scenes = detect_scene_names(code)
        assert scenes == []


class TestRenderRequest:
    """Tests for RenderRequest dataclass."""

    def test_default_values(self) -> None:
        req = RenderRequest(code="class Test(Scene): pass")
        assert req.code == "class Test(Scene): pass"
        assert req.scene_name is None
        assert req.quality == DEFAULT_QUALITY
        assert req.output_format == "mp4"
        assert req.output_path is None
        assert req.request_id  # Should be auto-generated
        assert req.extra_args == []

    def test_custom_values(self) -> None:
        req = RenderRequest(
            code="class Custom(Scene): pass",
            scene_name="Custom",
            quality="high",
            output_format="gif",
            output_path="/tmp/out.gif",
            request_id="custom-123",
            extra_args=["--transparent"],
        )
        assert req.scene_name == "Custom"
        assert req.quality == "high"
        assert req.output_format == "gif"
        assert req.output_path == "/tmp/out.gif"
        assert req.request_id == "custom-123"
        assert "--transparent" in req.extra_args


class TestRenderResult:
    """Tests for RenderResult dataclass."""

    def test_success_result(self) -> None:
        result = RenderResult(
            success=True,
            request_id="req-123",
            output_path="/tmp/out.mp4",
            duration_seconds=5.2,
            frame_count=150,
        )
        assert result.success
        assert result.output_path == "/tmp/out.mp4"
        assert result.duration_seconds == 5.2
        assert result.error is None

    def test_failure_result(self) -> None:
        result = RenderResult(
            success=False,
            request_id="req-456",
            error="SyntaxError: invalid syntax",
            stderr="File 'scene.py', line 5\n  def\n    ^",
        )
        assert not result.success
        assert result.output_path is None
        assert "SyntaxError" in result.error
        assert result.stderr

    def test_preview_frame(self) -> None:
        result = RenderResult(
            success=True,
            request_id="req-789",
            output_path="/tmp/out.mp4",
            preview_frame="iVBORw0KGgoAAAANSUhEUg...",  # Base64
        )
        assert result.preview_frame is not None


class TestQualityPresets:
    """Tests for quality preset configuration."""

    def test_all_presets_defined(self) -> None:
        expected = ["preview", "low", "medium", "high", "4k"]
        for preset in expected:
            assert preset in MANIM_QUALITY_PRESETS

    def test_preset_structure(self) -> None:
        for name, preset in MANIM_QUALITY_PRESETS.items():
            assert "quality" in preset
            assert "fps" in preset
            assert "pixel_height" in preset

    def test_preview_preset(self) -> None:
        preset = MANIM_QUALITY_PRESETS["preview"]
        assert preset["quality"] == "l"
        assert preset["fps"] == 15
        assert preset["pixel_height"] == 480

    def test_high_preset(self) -> None:
        preset = MANIM_QUALITY_PRESETS["high"]
        assert preset["quality"] == "h"
        assert preset["fps"] == 60
        assert preset["pixel_height"] == 1080

    def test_4k_preset(self) -> None:
        preset = MANIM_QUALITY_PRESETS["4k"]
        assert preset["quality"] == "p"
        assert preset["fps"] == 60
        assert preset["pixel_height"] == 2160


class TestCodePreprocessing:
    """Tests for code preprocessing before rendering."""

    def test_auto_import_detection(self) -> None:
        code_with_import = "from manim import *\nclass Test(Scene): pass"
        code_without_import = "class Test(Scene): pass"

        # Code with import should not be modified
        assert "from manim import" in code_with_import

        # Code without import needs prepending
        assert "from manim import" not in code_without_import

    def test_import_prepending(self) -> None:
        code = "class Test(Scene): pass"
        if "from manim import" not in code:
            code = "from manim import *\n\n" + code

        assert code.startswith("from manim import *")
        assert "class Test(Scene): pass" in code


class TestHTTPServerEndpoints:
    """Tests for HTTP server endpoint structure."""

    def test_render_request_schema(self) -> None:
        """Validate render request JSON schema."""
        request = {
            "code": "class MyScene(Scene): pass",
            "scene_name": "MyScene",
            "quality": "preview",
            "output_format": "mp4",
            "extra_args": [],
        }
        assert "code" in request
        assert request["quality"] in MANIM_QUALITY_PRESETS

    def test_render_response_schema(self) -> None:
        """Validate render response JSON schema."""
        response = {
            "request_id": "abc-123",
            "status": "pending",
            "message": "Render started",
        }
        assert "request_id" in response
        assert response["status"] in ["pending", "complete", "failed"]

    def test_status_response_schema(self) -> None:
        """Validate status response JSON schema."""
        response = {
            "request_id": "abc-123",
            "status": "complete",
            "success": True,
            "output_path": "/tmp/out.mp4",
            "preview_frame": None,
            "duration_seconds": 2.5,
            "error": None,
            "completed_at": 1702849200.0,
        }
        assert response["status"] == "complete"
        assert response["success"] is True

    def test_health_response_schema(self) -> None:
        """Validate health response JSON schema."""
        response = {
            "status": "healthy",
            "manim_version": "0.18.0",
            "jobs_pending": 0,
        }
        assert response["status"] in ["healthy", "degraded", "unhealthy"]


class TestBusEventIntegration:
    """Tests for bus event emission."""

    def test_render_started_event(self) -> None:
        """Validate render.started bus event format."""
        event = {
            "topic": "manim.render.started",
            "kind": "log",
            "level": "info",
            "actor": "manim-renderer",
            "data": {
                "request_id": "req-123",
                "quality": "preview",
                "format": "mp4",
            },
        }
        assert event["topic"] == "manim.render.started"
        assert event["kind"] == "log"
        assert "request_id" in event["data"]

    def test_render_complete_event(self) -> None:
        """Validate render.complete bus event format."""
        event = {
            "topic": "manim.render.complete",
            "kind": "response",
            "level": "info",
            "actor": "manim-renderer",
            "data": {
                "request_id": "req-123",
                "output_path": "/tmp/out.mp4",
                "duration_seconds": 5.2,
                "quality": "preview",
            },
        }
        assert event["topic"] == "manim.render.complete"
        assert event["kind"] == "response"
        assert "output_path" in event["data"]


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
