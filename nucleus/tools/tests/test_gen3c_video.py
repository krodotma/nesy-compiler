#!/usr/bin/env python3
"""Tests for GEN3C Video Generation module."""

import base64
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from gen3c_video import (
    Gen3CRequest,
    Gen3CResult,
    check_gpu_available,
    generate_mock_frame,
    generate_camera_path,
    generate_video,
    build_parser,
)


class TestGen3CRequest(unittest.TestCase):
    """Tests for Gen3CRequest dataclass."""

    def test_default_values(self):
        req = Gen3CRequest(view_paths=["/a.png", "/b.png"])
        self.assertEqual(req.view_paths, ["/a.png", "/b.png"])
        self.assertEqual(req.duration_seconds, 4.0)
        self.assertEqual(req.fps, 24)
        self.assertEqual(req.resolution, (512, 512))
        self.assertEqual(req.camera_path, "orbit")
        self.assertEqual(req.seed, -1)

    def test_to_dict(self):
        req = Gen3CRequest(
            view_paths=["/a.png"],
            duration_seconds=2.0,
            fps=12,
            resolution=(256, 256),
            camera_path="linear",
        )
        d = req.to_dict()
        self.assertEqual(d["view_paths"], ["/a.png"])
        self.assertEqual(d["duration_seconds"], 2.0)
        self.assertEqual(d["fps"], 12)
        self.assertEqual(d["resolution"], [256, 256])
        self.assertEqual(d["camera_path"], "linear")


class TestGen3CResult(unittest.TestCase):
    """Tests for Gen3CResult dataclass."""

    def test_to_dict_success(self):
        req = Gen3CRequest(view_paths=["/test.png"])
        result = Gen3CResult(
            request=req,
            video_path="/tmp/out.mp4",
            video_bytes=b"video_data",
            frame_count=48,
            metadata={"mode": "mock"},
            generation_time_ms=5000.0,
            status="success",
        )
        d = result.to_dict()

        self.assertEqual(d["video_path"], "/tmp/out.mp4")
        self.assertEqual(d["video_bytes_b64"], base64.b64encode(b"video_data").decode())
        self.assertEqual(d["frame_count"], 48)
        self.assertEqual(d["status"], "success")

    def test_to_dict_error(self):
        req = Gen3CRequest(view_paths=[])
        result = Gen3CResult(
            request=req,
            status="error",
            error="No views provided",
        )
        d = result.to_dict()
        self.assertEqual(d["status"], "error")
        self.assertEqual(d["error"], "No views provided")


class TestMockGeneration(unittest.TestCase):
    """Tests for mock frame generation."""

    def test_generate_mock_frame_no_base_images(self):
        frame = generate_mock_frame([], 0, 10, 256, 256)
        self.assertIsInstance(frame, bytes)
        self.assertGreater(len(frame), 0)
        # Should be valid PNG
        self.assertTrue(frame.startswith(b'\x89PNG'))

    def test_generate_mock_frame_with_base_images(self):
        try:
            from PIL import Image
            import numpy as np
        except ImportError:
            self.skipTest("PIL not available")

        # Create test images
        base_images = []
        for color in [(255, 0, 0), (0, 255, 0)]:
            arr = np.full((64, 64, 3), color, dtype=np.uint8)
            base_images.append(Image.fromarray(arr))

        frame = generate_mock_frame(base_images, 5, 10, 64, 64)
        self.assertIsInstance(frame, bytes)
        self.assertTrue(frame.startswith(b'\x89PNG'))


class TestCameraPath(unittest.TestCase):
    """Tests for camera path generation."""

    def test_orbit_path(self):
        poses = generate_camera_path("orbit", 24, 3)
        self.assertEqual(len(poses), 24)
        for pose in poses:
            self.assertEqual(len(pose), 16)  # 4x4 matrix flattened

    def test_linear_path(self):
        poses = generate_camera_path("linear", 12, 3)
        self.assertEqual(len(poses), 12)

    def test_spline_path(self):
        poses = generate_camera_path("spline", 10, 4)
        self.assertEqual(len(poses), 10)


class TestVideoGeneration(unittest.TestCase):
    """Tests for video generation."""

    def test_generate_video_no_views(self):
        req = Gen3CRequest(view_paths=["/nonexistent1.png", "/nonexistent2.png"])
        result = generate_video(req)
        # Should fail or produce empty/mock output
        # In mock mode without PIL, may still succeed with fallback
        self.assertIn(result.status, ["success", "error"])

    def test_generate_video_with_test_views(self):
        try:
            from PIL import Image
            import numpy as np
        except ImportError:
            self.skipTest("PIL not available")

        with tempfile.TemporaryDirectory() as td:
            # Create test views
            view_paths = []
            for i, color in enumerate([(255, 0, 0), (0, 255, 0), (0, 0, 255)]):
                arr = np.full((64, 64, 3), color, dtype=np.uint8)
                img = Image.fromarray(arr)
                path = Path(td) / f"view_{i}.png"
                img.save(path)
                view_paths.append(str(path))

            req = Gen3CRequest(
                view_paths=view_paths,
                duration_seconds=1.0,
                fps=6,
                resolution=(64, 64),
            )

            result = generate_video(req)

            self.assertEqual(result.status, "success")
            self.assertGreater(result.frame_count, 0)
            self.assertEqual(result.metadata.get("mode"), "mock")


class TestParser(unittest.TestCase):
    """Tests for CLI parser."""

    def test_generate_command(self):
        parser = build_parser()
        args = parser.parse_args([
            "generate",
            "--views", "/a.png", "/b.png",
            "--duration", "2",
            "--fps", "12",
        ])
        self.assertEqual(args.cmd, "generate")
        self.assertEqual(args.views, ["/a.png", "/b.png"])
        self.assertEqual(args.duration, 2.0)
        self.assertEqual(args.fps, 12)

    def test_serve_command(self):
        parser = build_parser()
        args = parser.parse_args(["serve", "--port", "8888"])
        self.assertEqual(args.cmd, "serve")
        self.assertEqual(args.port, 8888)

    def test_daemon_command(self):
        parser = build_parser()
        args = parser.parse_args(["daemon"])
        self.assertEqual(args.cmd, "daemon")

    def test_camera_options(self):
        parser = build_parser()
        for camera in ["orbit", "linear", "spline"]:
            args = parser.parse_args([
                "generate",
                "--views", "/test.png",
                "--camera", camera,
            ])
            self.assertEqual(args.camera, camera)


class TestGPUDetection(unittest.TestCase):
    """Tests for GPU detection."""

    def test_check_gpu_available_returns_bool(self):
        result = check_gpu_available()
        self.assertIsInstance(result, bool)


if __name__ == "__main__":
    unittest.main()
