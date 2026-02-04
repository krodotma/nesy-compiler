#!/usr/bin/env python3
"""Tests for VGGT Inference module."""

import base64
import io
import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from vggt_inference import (
    VGGTResult,
    check_gpu_available,
    generate_mock_depth_map,
    generate_mock_normal_map,
    generate_mock_mesh,
    infer_vggt,
    build_parser,
)


class TestVGGTResult(unittest.TestCase):
    """Tests for VGGTResult dataclass."""

    def test_to_dict_success(self):
        result = VGGTResult(
            image_path="/tmp/test.png",
            depth_map=b"depth_data",
            normal_map=b"normal_data",
            mesh_obj="# mesh",
            metadata={"mode": "mock"},
            inference_time_ms=100.5,
            status="success",
        )
        d = result.to_dict()

        self.assertEqual(d["image_path"], "/tmp/test.png")
        self.assertEqual(d["depth_map_b64"], base64.b64encode(b"depth_data").decode())
        self.assertEqual(d["normal_map_b64"], base64.b64encode(b"normal_data").decode())
        self.assertEqual(d["mesh_obj"], "# mesh")
        self.assertEqual(d["metadata"]["mode"], "mock")
        self.assertAlmostEqual(d["inference_time_ms"], 100.5)
        self.assertEqual(d["status"], "success")
        self.assertIsNone(d["error"])

    def test_to_dict_error(self):
        result = VGGTResult(
            image_path="/tmp/missing.png",
            status="error",
            error="File not found",
        )
        d = result.to_dict()

        self.assertEqual(d["status"], "error")
        self.assertEqual(d["error"], "File not found")
        self.assertIsNone(d["depth_map_b64"])


class TestMockGeneration(unittest.TestCase):
    """Tests for mock depth/normal/mesh generation."""

    def test_generate_mock_depth_map(self):
        depth = generate_mock_depth_map(256, 256)
        self.assertIsInstance(depth, bytes)
        self.assertGreater(len(depth), 0)
        # Should be valid PNG (magic bytes)
        self.assertTrue(depth.startswith(b'\x89PNG'))

    def test_generate_mock_normal_map(self):
        normals = generate_mock_normal_map(256, 256)
        self.assertIsInstance(normals, bytes)
        self.assertGreater(len(normals), 0)
        self.assertTrue(normals.startswith(b'\x89PNG'))

    def test_generate_mock_mesh(self):
        mesh = generate_mock_mesh(256, 256)
        self.assertIsInstance(mesh, str)
        self.assertIn("# VGGT Mock Mesh", mesh)
        self.assertIn("v ", mesh)  # vertices
        self.assertIn("f ", mesh)  # faces
        self.assertIn("vn ", mesh)  # normals


class TestInferVGGT(unittest.TestCase):
    """Tests for main inference function."""

    def test_infer_missing_image(self):
        result = infer_vggt("/nonexistent/image.png")
        self.assertEqual(result.status, "error")
        self.assertIn("not found", result.error.lower())

    def test_infer_with_test_image(self):
        try:
            from PIL import Image
            import numpy as np
        except ImportError:
            self.skipTest("PIL not available")

        with tempfile.TemporaryDirectory() as td:
            # Create test image
            arr = np.zeros((128, 128, 3), dtype=np.uint8)
            arr[:, :, 0] = 255  # Red image
            img = Image.fromarray(arr)
            test_path = Path(td) / "test_input.png"
            img.save(test_path)

            # Run inference
            result = infer_vggt(str(test_path))

            self.assertEqual(result.status, "success")
            self.assertIsNotNone(result.depth_map)
            self.assertIsNotNone(result.normal_map)
            self.assertIsNotNone(result.mesh_obj)
            self.assertGreater(result.inference_time_ms, 0)
            self.assertEqual(result.metadata.get("mode"), "mock")


class TestParser(unittest.TestCase):
    """Tests for CLI parser."""

    def test_infer_command(self):
        parser = build_parser()
        args = parser.parse_args(["infer", "--image", "/tmp/test.png"])
        self.assertEqual(args.cmd, "infer")
        self.assertEqual(args.image, "/tmp/test.png")

    def test_serve_command(self):
        parser = build_parser()
        args = parser.parse_args(["serve", "--port", "9999"])
        self.assertEqual(args.cmd, "serve")
        self.assertEqual(args.port, 9999)

    def test_daemon_command(self):
        parser = build_parser()
        args = parser.parse_args(["daemon"])
        self.assertEqual(args.cmd, "daemon")

    def test_test_command(self):
        parser = build_parser()
        args = parser.parse_args(["test"])
        self.assertEqual(args.cmd, "test")

    def test_emit_bus_flag(self):
        parser = build_parser()
        args = parser.parse_args(["--emit-bus", "test"])
        self.assertTrue(args.emit_bus)


class TestGPUDetection(unittest.TestCase):
    """Tests for GPU detection."""

    def test_check_gpu_available_returns_bool(self):
        result = check_gpu_available()
        self.assertIsInstance(result, bool)


if __name__ == "__main__":
    unittest.main()
