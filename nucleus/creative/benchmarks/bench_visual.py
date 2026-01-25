"""
Visual Subsystem Benchmarks
===========================

Benchmarks for image generation, style transfer, and upscaling.
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field
from typing import Callable, Any, Optional
import math

from .bench_runner import BenchmarkSuite


@dataclass
class ImageBuffer:
    """Mock image buffer for benchmarking."""
    width: int
    height: int
    channels: int
    data: list[float] = field(default_factory=list)

    def __post_init__(self):
        if not self.data:
            size = self.width * self.height * self.channels
            self.data = [0.0] * size

    @classmethod
    def random(cls, width: int, height: int, channels: int = 3) -> "ImageBuffer":
        """Create random image buffer."""
        size = width * height * channels
        data = [random.random() for _ in range(size)]
        return cls(width=width, height=height, channels=channels, data=data)

    @classmethod
    def gradient(cls, width: int, height: int) -> "ImageBuffer":
        """Create gradient image."""
        data = []
        for y in range(height):
            for x in range(width):
                r = x / width
                g = y / height
                b = (x + y) / (width + height)
                data.extend([r, g, b])
        return cls(width=width, height=height, channels=3, data=data)

    def get_pixel(self, x: int, y: int) -> tuple[float, ...]:
        """Get pixel value at (x, y)."""
        idx = (y * self.width + x) * self.channels
        return tuple(self.data[idx:idx + self.channels])

    def set_pixel(self, x: int, y: int, value: tuple[float, ...]) -> None:
        """Set pixel value at (x, y)."""
        idx = (y * self.width + x) * self.channels
        for i, v in enumerate(value):
            if idx + i < len(self.data):
                self.data[idx + i] = v

    def resize(self, new_width: int, new_height: int) -> "ImageBuffer":
        """Bilinear resize."""
        result = ImageBuffer(new_width, new_height, self.channels)

        x_ratio = self.width / new_width
        y_ratio = self.height / new_height

        for y in range(new_height):
            for x in range(new_width):
                src_x = x * x_ratio
                src_y = y * y_ratio

                x0 = min(int(src_x), self.width - 1)
                y0 = min(int(src_y), self.height - 1)

                pixel = self.get_pixel(x0, y0)
                result.set_pixel(x, y, pixel)

        return result

    def apply_kernel(self, kernel: list[list[float]]) -> "ImageBuffer":
        """Apply convolution kernel."""
        result = ImageBuffer(self.width, self.height, self.channels)
        k_size = len(kernel)
        k_half = k_size // 2

        for y in range(self.height):
            for x in range(self.width):
                new_pixel = [0.0] * self.channels

                for ky in range(k_size):
                    for kx in range(k_size):
                        px = max(0, min(self.width - 1, x + kx - k_half))
                        py = max(0, min(self.height - 1, y + ky - k_half))

                        pixel = self.get_pixel(px, py)
                        weight = kernel[ky][kx]

                        for c in range(self.channels):
                            new_pixel[c] += pixel[c] * weight

                result.set_pixel(x, y, tuple(max(0.0, min(1.0, v)) for v in new_pixel))

        return result


@dataclass
class StyleTransferConfig:
    """Style transfer configuration."""
    style_weight: float = 1e6
    content_weight: float = 1.0
    iterations: int = 100
    learning_rate: float = 0.01


class MockImageGenerator:
    """Mock image generator for benchmarking."""

    def __init__(self, seed: int = 42):
        self.seed = seed
        random.seed(seed)

    def generate_noise(self, width: int, height: int) -> ImageBuffer:
        """Generate noise image."""
        return ImageBuffer.random(width, height)

    def generate_pattern(self, width: int, height: int, pattern: str = "checkerboard") -> ImageBuffer:
        """Generate patterned image."""
        img = ImageBuffer(width, height, 3)

        for y in range(height):
            for x in range(width):
                if pattern == "checkerboard":
                    v = 1.0 if (x // 8 + y // 8) % 2 == 0 else 0.0
                    img.set_pixel(x, y, (v, v, v))
                elif pattern == "stripes":
                    v = 1.0 if x % 16 < 8 else 0.0
                    img.set_pixel(x, y, (v, v, v))
                elif pattern == "circles":
                    cx, cy = width // 2, height // 2
                    r = math.sqrt((x - cx) ** 2 + (y - cy) ** 2)
                    v = 1.0 if int(r / 10) % 2 == 0 else 0.0
                    img.set_pixel(x, y, (v, v, v))

        return img

    def text_to_image(self, prompt: str, width: int, height: int, steps: int = 50) -> ImageBuffer:
        """Mock text-to-image generation."""
        # Simulate diffusion steps
        img = self.generate_noise(width, height)

        for step in range(steps):
            # Simulated denoising step
            noise_level = 1.0 - (step / steps)
            for i in range(len(img.data)):
                img.data[i] = img.data[i] * (1 - noise_level * 0.1) + random.random() * noise_level * 0.1

        return img

    def img2img(
        self,
        image: ImageBuffer,
        prompt: str,
        strength: float = 0.5,
        steps: int = 30,
    ) -> ImageBuffer:
        """Mock image-to-image generation."""
        # Add noise proportional to strength
        result = ImageBuffer(image.width, image.height, image.channels, image.data.copy())

        for step in range(int(steps * strength)):
            noise_level = strength * (1.0 - step / (steps * strength))
            for i in range(len(result.data)):
                result.data[i] += random.gauss(0, noise_level * 0.1)
                result.data[i] = max(0.0, min(1.0, result.data[i]))

        return result


class MockStyleTransfer:
    """Mock neural style transfer."""

    def __init__(self, config: StyleTransferConfig | None = None):
        self.config = config or StyleTransferConfig()

    def extract_features(self, image: ImageBuffer) -> list[list[float]]:
        """Extract mock features from image."""
        # Simulate feature extraction at multiple scales
        features = []

        for scale in [1, 2, 4, 8]:
            scaled = image.resize(image.width // scale, image.height // scale)
            features.append(scaled.data[:1000])  # Take first 1000 values

        return features

    def compute_gram_matrix(self, features: list[float]) -> list[list[float]]:
        """Compute Gram matrix from features."""
        n = min(32, len(features))
        gram = [[0.0] * n for _ in range(n)]

        for i in range(n):
            for j in range(n):
                gram[i][j] = features[i] * features[j] if i < len(features) and j < len(features) else 0.0

        return gram

    def transfer(
        self,
        content: ImageBuffer,
        style: ImageBuffer,
        iterations: int | None = None,
    ) -> ImageBuffer:
        """Perform style transfer."""
        iterations = iterations or self.config.iterations

        # Initialize with content image
        result = ImageBuffer(
            content.width, content.height, content.channels,
            content.data.copy(),
        )

        # Extract style features
        style_features = self.extract_features(style)
        style_gram = [self.compute_gram_matrix(f) for f in style_features]

        # Optimization loop
        for _ in range(iterations):
            # Compute content loss gradient
            content_grad = [
                (r - c) * self.config.content_weight
                for r, c in zip(result.data, content.data)
            ]

            # Update result
            for i in range(len(result.data)):
                result.data[i] -= content_grad[i] * self.config.learning_rate
                result.data[i] = max(0.0, min(1.0, result.data[i]))

        return result


class MockUpscaler:
    """Mock image upscaler."""

    def __init__(self, model: str = "real-esrgan"):
        self.model = model

    def upscale(self, image: ImageBuffer, scale: int = 4) -> ImageBuffer:
        """Upscale image by factor."""
        new_width = image.width * scale
        new_height = image.height * scale

        # Bilinear upscale as base
        result = image.resize(new_width, new_height)

        # Simulate detail enhancement
        sharpen_kernel = [
            [0, -1, 0],
            [-1, 5, -1],
            [0, -1, 0],
        ]

        return result.apply_kernel(sharpen_kernel)

    def enhance_faces(self, image: ImageBuffer) -> ImageBuffer:
        """Mock face enhancement."""
        # Simulate face detection and enhancement
        result = ImageBuffer(
            image.width, image.height, image.channels,
            image.data.copy(),
        )

        # Apply smoothing kernel to simulate face restoration
        smooth_kernel = [
            [1/16, 1/8, 1/16],
            [1/8, 1/4, 1/8],
            [1/16, 1/8, 1/16],
        ]

        return result.apply_kernel(smooth_kernel)


class VisualBenchmark(BenchmarkSuite):
    """Benchmark suite for visual subsystem."""

    @property
    def name(self) -> str:
        return "visual"

    @property
    def description(self) -> str:
        return "Image generation, style transfer, upscaling benchmarks"

    def __init__(self):
        self._generator: Optional[MockImageGenerator] = None
        self._style_transfer: Optional[MockStyleTransfer] = None
        self._upscaler: Optional[MockUpscaler] = None
        self._test_images: dict[str, ImageBuffer] = {}

    def setup(self) -> None:
        """Setup test data and mock objects."""
        self._generator = MockImageGenerator(seed=42)
        self._style_transfer = MockStyleTransfer()
        self._upscaler = MockUpscaler()

        # Pre-generate test images
        self._test_images = {
            "small": ImageBuffer.random(64, 64),
            "medium": ImageBuffer.random(256, 256),
            "large": ImageBuffer.random(512, 512),
            "content": ImageBuffer.gradient(256, 256),
            "style": self._generator.generate_pattern(256, 256, "circles"),
        }

    def get_benchmarks(self) -> list[tuple[str, Callable[[], Any]]]:
        """Get all visual benchmarks."""
        return [
            # Image creation
            ("image_create_small", self._image_create_small),
            ("image_create_medium", self._image_create_medium),
            ("image_create_large", self._image_create_large),
            ("image_gradient", self._image_gradient),
            ("image_pattern_checkerboard", self._image_pattern_checkerboard),
            ("image_pattern_circles", self._image_pattern_circles),

            # Image operations
            ("image_resize_down", self._image_resize_down),
            ("image_resize_up", self._image_resize_up),
            ("image_kernel_blur", self._image_kernel_blur),
            ("image_kernel_sharpen", self._image_kernel_sharpen),

            # Generation
            ("generate_noise", self._generate_noise),
            ("generate_text_to_image_fast", self._generate_t2i_fast),
            ("generate_text_to_image_quality", self._generate_t2i_quality),
            ("generate_img2img", self._generate_img2img),

            # Style transfer
            ("style_extract_features", self._style_extract_features),
            ("style_gram_matrix", self._style_gram_matrix),
            ("style_transfer_fast", self._style_transfer_fast),
            ("style_transfer_quality", self._style_transfer_quality),

            # Upscaling
            ("upscale_2x", self._upscale_2x),
            ("upscale_4x", self._upscale_4x),
            ("upscale_face_enhance", self._upscale_face_enhance),
        ]

    def _image_create_small(self) -> ImageBuffer:
        """Create 64x64 image."""
        return ImageBuffer.random(64, 64)

    def _image_create_medium(self) -> ImageBuffer:
        """Create 256x256 image."""
        return ImageBuffer.random(256, 256)

    def _image_create_large(self) -> ImageBuffer:
        """Create 512x512 image."""
        return ImageBuffer.random(512, 512)

    def _image_gradient(self) -> ImageBuffer:
        """Create gradient image."""
        return ImageBuffer.gradient(256, 256)

    def _image_pattern_checkerboard(self) -> ImageBuffer:
        """Create checkerboard pattern."""
        return self._generator.generate_pattern(256, 256, "checkerboard")

    def _image_pattern_circles(self) -> ImageBuffer:
        """Create circles pattern."""
        return self._generator.generate_pattern(256, 256, "circles")

    def _image_resize_down(self) -> ImageBuffer:
        """Resize image down."""
        return self._test_images["large"].resize(128, 128)

    def _image_resize_up(self) -> ImageBuffer:
        """Resize image up."""
        return self._test_images["small"].resize(256, 256)

    def _image_kernel_blur(self) -> ImageBuffer:
        """Apply blur kernel."""
        kernel = [[1/9] * 3 for _ in range(3)]
        return self._test_images["medium"].apply_kernel(kernel)

    def _image_kernel_sharpen(self) -> ImageBuffer:
        """Apply sharpen kernel."""
        kernel = [
            [0, -1, 0],
            [-1, 5, -1],
            [0, -1, 0],
        ]
        return self._test_images["medium"].apply_kernel(kernel)

    def _generate_noise(self) -> ImageBuffer:
        """Generate noise image."""
        return self._generator.generate_noise(256, 256)

    def _generate_t2i_fast(self) -> ImageBuffer:
        """Fast text-to-image (10 steps)."""
        return self._generator.text_to_image("sunset mountains", 128, 128, steps=10)

    def _generate_t2i_quality(self) -> ImageBuffer:
        """Quality text-to-image (50 steps)."""
        return self._generator.text_to_image("sunset mountains", 256, 256, steps=50)

    def _generate_img2img(self) -> ImageBuffer:
        """Image-to-image generation."""
        return self._generator.img2img(
            self._test_images["content"],
            "oil painting style",
            strength=0.5,
            steps=20,
        )

    def _style_extract_features(self) -> list[list[float]]:
        """Extract style features."""
        return self._style_transfer.extract_features(self._test_images["style"])

    def _style_gram_matrix(self) -> list[list[float]]:
        """Compute Gram matrix."""
        features = self._test_images["style"].data[:100]
        return self._style_transfer.compute_gram_matrix(features)

    def _style_transfer_fast(self) -> ImageBuffer:
        """Fast style transfer (10 iterations)."""
        return self._style_transfer.transfer(
            self._test_images["content"],
            self._test_images["style"],
            iterations=10,
        )

    def _style_transfer_quality(self) -> ImageBuffer:
        """Quality style transfer (50 iterations)."""
        return self._style_transfer.transfer(
            self._test_images["content"],
            self._test_images["style"],
            iterations=50,
        )

    def _upscale_2x(self) -> ImageBuffer:
        """Upscale 2x."""
        return self._upscaler.upscale(self._test_images["small"], scale=2)

    def _upscale_4x(self) -> ImageBuffer:
        """Upscale 4x."""
        return self._upscaler.upscale(self._test_images["small"], scale=4)

    def _upscale_face_enhance(self) -> ImageBuffer:
        """Face enhancement."""
        return self._upscaler.enhance_faces(self._test_images["medium"])
