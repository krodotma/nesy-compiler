#!/usr/bin/env python3
"""Tests for Live2Diff Stream module."""

import base64
import io
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from live2diff_stream import (
    AVAILABLE_STYLES,
    StyleConfig,
    FrameResult,
    StreamStats,
    check_gpu_available,
    get_fallback_filter,
    stylize_frame,
    StreamProcessor,
    build_parser,
)


class TestStyleConfig(unittest.TestCase):
    """Tests for StyleConfig dataclass."""

    def test_default_values(self):
        config = StyleConfig()
        self.assertEqual(config.style, "anime")
        self.assertEqual(config.strength, 1.0)
        self.assertEqual(config.temporal_smoothing, 0.5)
        self.assertEqual(config.resolution_scale, 1.0)

    def test_custom_values(self):
        config = StyleConfig(
            style="sketch",
            strength=0.7,
            temporal_smoothing=0.3,
            resolution_scale=0.5,
        )
        self.assertEqual(config.style, "sketch")
        self.assertAlmostEqual(config.strength, 0.7)


class TestFrameResult(unittest.TestCase):
    """Tests for FrameResult dataclass."""

    def test_to_dict_success(self):
        result = FrameResult(
            frame_idx=10,
            original_bytes=b"orig",
            styled_bytes=b"styled",
            style="anime",
            latency_ms=50.5,
            status="success",
        )
        d = result.to_dict()

        self.assertEqual(d["frame_idx"], 10)
        self.assertEqual(d["original_b64"], base64.b64encode(b"orig").decode())
        self.assertEqual(d["styled_b64"], base64.b64encode(b"styled").decode())
        self.assertEqual(d["style"], "anime")
        self.assertAlmostEqual(d["latency_ms"], 50.5)
        self.assertEqual(d["status"], "success")

    def test_to_dict_error(self):
        result = FrameResult(
            frame_idx=0,
            status="error",
            error="Decode failed",
        )
        d = result.to_dict()
        self.assertEqual(d["status"], "error")
        self.assertEqual(d["error"], "Decode failed")


class TestStreamStats(unittest.TestCase):
    """Tests for StreamStats dataclass."""

    def test_default_values(self):
        stats = StreamStats()
        self.assertEqual(stats.frames_processed, 0)
        self.assertEqual(stats.dropped_frames, 0)

    def test_to_dict(self):
        stats = StreamStats(
            frames_processed=100,
            total_latency_ms=5000.0,
            avg_latency_ms=50.0,
            current_fps=16.5,
            dropped_frames=2,
            mode="mock",
        )
        d = stats.to_dict()
        self.assertEqual(d["frames_processed"], 100)
        self.assertAlmostEqual(d["current_fps"], 16.5)
        self.assertEqual(d["mode"], "mock")


class TestAvailableStyles(unittest.TestCase):
    """Tests for available styles."""

    def test_all_styles_have_descriptions(self):
        expected_styles = [
            "anime", "watercolor", "oil_paint", "sketch", "neon",
            "vintage", "comic", "pixel", "impressionist", "none",
        ]
        for style in expected_styles:
            self.assertIn(style, AVAILABLE_STYLES)
            self.assertIsInstance(AVAILABLE_STYLES[style], str)

    def test_style_count(self):
        self.assertGreaterEqual(len(AVAILABLE_STYLES), 10)


class TestFallbackFilters(unittest.TestCase):
    """Tests for fallback filter functions."""

    def test_get_fallback_filter_known_styles(self):
        for style in AVAILABLE_STYLES:
            filter_fn = get_fallback_filter(style)
            self.assertTrue(callable(filter_fn))

    def test_get_fallback_filter_unknown_style(self):
        filter_fn = get_fallback_filter("unknown_style")
        self.assertTrue(callable(filter_fn))

    def test_filters_produce_output(self):
        try:
            from PIL import Image
            import numpy as np
        except ImportError:
            self.skipTest("PIL not available")

        # Create test image
        arr = np.zeros((64, 64, 3), dtype=np.uint8)
        arr[:, :, 0] = 128
        arr[:, :, 1] = 64
        arr[:, :, 2] = 200
        img = Image.fromarray(arr)

        for style in ["anime", "sketch", "vintage", "pixel", "none"]:
            filter_fn = get_fallback_filter(style)
            output = filter_fn(img)
            self.assertIsInstance(output, Image.Image)
            self.assertEqual(output.size, img.size)


class TestStylizeFrame(unittest.TestCase):
    """Tests for frame stylization."""

    def test_stylize_invalid_frame(self):
        config = StyleConfig(style="anime")
        result = stylize_frame(b"not_an_image", config, 0)
        self.assertEqual(result.status, "error")

    def test_stylize_valid_frame(self):
        try:
            from PIL import Image
            import numpy as np
        except ImportError:
            self.skipTest("PIL not available")

        # Create test frame
        arr = np.zeros((64, 64, 3), dtype=np.uint8)
        arr[:, :, 0] = 255  # Red
        img = Image.fromarray(arr)
        buf = io.BytesIO()
        img.save(buf, format="PNG")
        frame_bytes = buf.getvalue()

        config = StyleConfig(style="anime", strength=1.0)
        result = stylize_frame(frame_bytes, config, 0)

        self.assertEqual(result.status, "success")
        self.assertIsNotNone(result.styled_bytes)
        self.assertEqual(result.style, "anime")
        self.assertGreater(result.latency_ms, 0)
        # Output should be valid PNG
        self.assertTrue(result.styled_bytes.startswith(b'\x89PNG'))

    def test_stylize_different_styles(self):
        try:
            from PIL import Image
            import numpy as np
        except ImportError:
            self.skipTest("PIL not available")

        # Create test frame
        arr = np.full((32, 32, 3), 128, dtype=np.uint8)
        img = Image.fromarray(arr)
        buf = io.BytesIO()
        img.save(buf, format="PNG")
        frame_bytes = buf.getvalue()

        for style in ["sketch", "neon", "vintage"]:
            config = StyleConfig(style=style)
            result = stylize_frame(frame_bytes, config, 0)
            self.assertEqual(result.status, "success", f"Style {style} failed")
            self.assertEqual(result.style, style)

    def test_stylize_with_strength(self):
        try:
            from PIL import Image
            import numpy as np
        except ImportError:
            self.skipTest("PIL not available")

        arr = np.full((32, 32, 3), 100, dtype=np.uint8)
        img = Image.fromarray(arr)
        buf = io.BytesIO()
        img.save(buf, format="PNG")
        frame_bytes = buf.getvalue()

        # Full strength
        config_full = StyleConfig(style="vintage", strength=1.0)
        result_full = stylize_frame(frame_bytes, config_full, 0)

        # Half strength
        config_half = StyleConfig(style="vintage", strength=0.5)
        result_half = stylize_frame(frame_bytes, config_half, 0)

        self.assertEqual(result_full.status, "success")
        self.assertEqual(result_half.status, "success")
        # Both should produce output
        self.assertIsNotNone(result_full.styled_bytes)
        self.assertIsNotNone(result_half.styled_bytes)


class TestStreamProcessor(unittest.TestCase):
    """Tests for StreamProcessor class."""

    def test_processor_lifecycle(self):
        config = StyleConfig(style="none")
        processor = StreamProcessor(config)

        processor.start()
        self.assertTrue(processor.running)
        self.assertEqual(processor.stats.mode, "mock")

        processor.stop()
        self.assertFalse(processor.running)

    def test_processor_stats_initial(self):
        config = StyleConfig()
        processor = StreamProcessor(config)
        processor.start()

        self.assertEqual(processor.stats.frames_processed, 0)
        self.assertEqual(processor.stats.dropped_frames, 0)

        processor.stop()


class TestParser(unittest.TestCase):
    """Tests for CLI parser."""

    def test_stylize_command(self):
        parser = build_parser()
        args = parser.parse_args([
            "stylize",
            "--input", "/tmp/video.mp4",
            "--style", "anime",
            "--strength", "0.8",
        ])
        self.assertEqual(args.cmd, "stylize")
        self.assertEqual(args.input, "/tmp/video.mp4")
        self.assertEqual(args.style, "anime")
        self.assertAlmostEqual(args.strength, 0.8)

    def test_stream_command(self):
        parser = build_parser()
        args = parser.parse_args([
            "stream",
            "--source", "0",
            "--style", "sketch",
        ])
        self.assertEqual(args.cmd, "stream")
        self.assertEqual(args.source, "0")
        self.assertEqual(args.style, "sketch")

    def test_serve_command(self):
        parser = build_parser()
        args = parser.parse_args(["serve", "--port", "7777"])
        self.assertEqual(args.cmd, "serve")
        self.assertEqual(args.port, 7777)

    def test_daemon_command(self):
        parser = build_parser()
        args = parser.parse_args(["daemon"])
        self.assertEqual(args.cmd, "daemon")

    def test_test_command(self):
        parser = build_parser()
        args = parser.parse_args(["test", "--style", "neon"])
        self.assertEqual(args.cmd, "test")
        self.assertEqual(args.style, "neon")

    def test_all_styles_valid_choices(self):
        parser = build_parser()
        for style in AVAILABLE_STYLES:
            args = parser.parse_args(["stylize", "--input", "/test.mp4", "--style", style])
            self.assertEqual(args.style, style)


class TestGPUDetection(unittest.TestCase):
    """Tests for GPU detection."""

    def test_check_gpu_available_returns_bool(self):
        result = check_gpu_available()
        self.assertIsInstance(result, bool)


if __name__ == "__main__":
    unittest.main()
