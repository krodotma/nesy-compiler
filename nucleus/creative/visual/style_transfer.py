"""
Neural Style Transfer Module
============================

Provides neural style transfer capabilities for applying artistic styles
to images. Supports multiple backends and style presets.
"""

from __future__ import annotations

import asyncio
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Optional, Union
import time

import numpy as np

try:
    from PIL import Image
    HAS_PIL = True
except ImportError:
    HAS_PIL = False


class StylePreset(str, Enum):
    """Built-in style presets."""
    STARRY_NIGHT = "starry_night"
    WAVE = "wave"
    MOSAIC = "mosaic"
    PENCIL_SKETCH = "pencil_sketch"
    OIL_PAINTING = "oil_painting"
    WATERCOLOR = "watercolor"
    CYBERPUNK = "cyberpunk"
    VINTAGE = "vintage"
    POP_ART = "pop_art"
    ABSTRACT = "abstract"


class TransferMethod(str, Enum):
    """Style transfer methods."""
    NEURAL = "neural"
    FAST_NEURAL = "fast_neural"
    COLOR_TRANSFER = "color_transfer"
    HISTOGRAM_MATCHING = "histogram_matching"


# Style preset configurations
STYLE_PRESETS: dict[str, dict] = {
    StylePreset.STARRY_NIGHT: {
        "name": "Starry Night",
        "description": "Van Gogh's swirling night sky style",
        "content_weight": 1.0,
        "style_weight": 100000.0,
        "color_preserve": 0.0,
    },
    StylePreset.WAVE: {
        "name": "The Great Wave",
        "description": "Hokusai's iconic wave style",
        "content_weight": 1.0,
        "style_weight": 80000.0,
        "color_preserve": 0.2,
    },
    StylePreset.MOSAIC: {
        "name": "Mosaic",
        "description": "Stained glass mosaic effect",
        "content_weight": 1.0,
        "style_weight": 50000.0,
        "color_preserve": 0.5,
    },
    StylePreset.PENCIL_SKETCH: {
        "name": "Pencil Sketch",
        "description": "Hand-drawn pencil sketch effect",
        "content_weight": 2.0,
        "style_weight": 30000.0,
        "color_preserve": 0.0,
        "grayscale": True,
    },
    StylePreset.OIL_PAINTING: {
        "name": "Oil Painting",
        "description": "Classical oil painting style",
        "content_weight": 1.0,
        "style_weight": 60000.0,
        "color_preserve": 0.3,
    },
    StylePreset.WATERCOLOR: {
        "name": "Watercolor",
        "description": "Soft watercolor painting effect",
        "content_weight": 1.5,
        "style_weight": 40000.0,
        "color_preserve": 0.4,
    },
    StylePreset.CYBERPUNK: {
        "name": "Cyberpunk",
        "description": "Neon-lit cyberpunk aesthetic",
        "content_weight": 1.0,
        "style_weight": 70000.0,
        "color_preserve": 0.1,
        "neon_boost": True,
    },
    StylePreset.VINTAGE: {
        "name": "Vintage",
        "description": "Aged, nostalgic photo effect",
        "content_weight": 2.0,
        "style_weight": 20000.0,
        "color_preserve": 0.6,
        "sepia": True,
    },
    StylePreset.POP_ART: {
        "name": "Pop Art",
        "description": "Bold, colorful pop art style",
        "content_weight": 1.0,
        "style_weight": 90000.0,
        "color_preserve": 0.0,
        "posterize": True,
    },
    StylePreset.ABSTRACT: {
        "name": "Abstract",
        "description": "Abstract expressionist style",
        "content_weight": 0.5,
        "style_weight": 120000.0,
        "color_preserve": 0.2,
    },
}


@dataclass
class StyleConfig:
    """Configuration for style transfer.

    Attributes:
        content_weight: Weight for content preservation
        style_weight: Weight for style application
        total_variation_weight: Weight for smoothness
        color_preserve: Amount of original color to preserve (0-1)
        iterations: Number of optimization iterations
        output_size: Output image size (width, height)
        method: Transfer method to use
        preset: Optional style preset
        style_layers: Which network layers to use for style
        content_layers: Which network layers to use for content
    """
    content_weight: float = 1.0
    style_weight: float = 100000.0
    total_variation_weight: float = 1e-4
    color_preserve: float = 0.0
    iterations: int = 300
    output_size: Optional[tuple[int, int]] = None
    method: TransferMethod = TransferMethod.NEURAL
    preset: Optional[StylePreset] = None
    style_layers: list[str] = field(default_factory=lambda: [
        "conv1_1", "conv2_1", "conv3_1", "conv4_1", "conv5_1"
    ])
    content_layers: list[str] = field(default_factory=lambda: ["conv4_2"])
    extra_params: dict = field(default_factory=dict)

    def __post_init__(self):
        """Apply preset if specified."""
        if self.preset and self.preset in STYLE_PRESETS:
            preset_config = STYLE_PRESETS[self.preset]
            if "content_weight" in preset_config:
                self.content_weight = preset_config["content_weight"]
            if "style_weight" in preset_config:
                self.style_weight = preset_config["style_weight"]
            if "color_preserve" in preset_config:
                self.color_preserve = preset_config["color_preserve"]

    def validate(self) -> list[str]:
        """Validate configuration."""
        errors = []
        if self.content_weight < 0:
            errors.append("Content weight must be non-negative")
        if self.style_weight < 0:
            errors.append("Style weight must be non-negative")
        if not (0 <= self.color_preserve <= 1):
            errors.append("Color preserve must be between 0 and 1")
        if self.iterations < 1:
            errors.append("Iterations must be at least 1")
        if self.output_size and (self.output_size[0] <= 0 or self.output_size[1] <= 0):
            errors.append("Output size dimensions must be positive")
        return errors


@dataclass
class StyleResult:
    """Result of a style transfer operation.

    Attributes:
        output_image: The stylized image as numpy array
        content_image: Original content image
        style_image: Style reference image (if provided)
        config: Configuration used
        transfer_time: Time taken in seconds
        method_used: The transfer method that was used
        loss_history: Loss values during optimization
        metadata: Additional metadata
    """
    output_image: np.ndarray
    content_image: np.ndarray
    style_image: Optional[np.ndarray]
    config: StyleConfig
    transfer_time: float
    method_used: str
    loss_history: list[float] = field(default_factory=list)
    metadata: dict = field(default_factory=dict)

    @property
    def content_shape(self) -> tuple[int, ...]:
        """Shape of the content image."""
        return self.content_image.shape

    @property
    def output_shape(self) -> tuple[int, ...]:
        """Shape of the output image."""
        return self.output_image.shape


class StyleTransferBackend(ABC):
    """Abstract base class for style transfer backends."""

    @property
    @abstractmethod
    def name(self) -> str:
        """Backend name."""
        pass

    @property
    @abstractmethod
    def supported_methods(self) -> list[TransferMethod]:
        """List of supported transfer methods."""
        pass

    @abstractmethod
    async def transfer(
        self,
        content: np.ndarray,
        style: np.ndarray,
        config: StyleConfig,
        progress_callback: Optional[Callable[[float, str], None]] = None,
    ) -> StyleResult:
        """Apply style transfer.

        Args:
            content: Content image as numpy array (H, W, C)
            style: Style image as numpy array (H, W, C)
            config: Style transfer configuration
            progress_callback: Optional progress callback

        Returns:
            StyleResult with the stylized image
        """
        pass

    @abstractmethod
    async def is_available(self) -> bool:
        """Check if backend is available."""
        pass


class PILStyleBackend(StyleTransferBackend):
    """PIL-based style transfer backend.

    Uses simple image processing techniques for fast, CPU-friendly
    style transfer approximations.
    """

    @property
    def name(self) -> str:
        return "PIL Style Backend"

    @property
    def supported_methods(self) -> list[TransferMethod]:
        return [TransferMethod.COLOR_TRANSFER, TransferMethod.HISTOGRAM_MATCHING]

    async def transfer(
        self,
        content: np.ndarray,
        style: np.ndarray,
        config: StyleConfig,
        progress_callback: Optional[Callable[[float, str], None]] = None,
    ) -> StyleResult:
        """Apply color/histogram-based style transfer."""
        start_time = time.time()

        if not HAS_PIL:
            raise RuntimeError("PIL is required for PILStyleBackend")

        if progress_callback:
            progress_callback(10, "Processing images...")

        # Convert to float
        content_f = content.astype(np.float32) / 255.0
        style_f = style.astype(np.float32) / 255.0

        if config.method == TransferMethod.HISTOGRAM_MATCHING:
            output = await self._histogram_match(content_f, style_f, progress_callback)
        else:  # COLOR_TRANSFER
            output = await self._color_transfer(content_f, style_f, progress_callback)

        # Apply color preservation if specified
        if config.color_preserve > 0:
            output = self._preserve_color(content_f, output, config.color_preserve)

        if progress_callback:
            progress_callback(90, "Finalizing...")

        # Convert back to uint8
        output = np.clip(output * 255, 0, 255).astype(np.uint8)

        transfer_time = time.time() - start_time

        if progress_callback:
            progress_callback(100, "Complete")

        return StyleResult(
            output_image=output,
            content_image=content,
            style_image=style,
            config=config,
            transfer_time=transfer_time,
            method_used=config.method.value,
            metadata={"backend": "pil"},
        )

    async def _histogram_match(
        self,
        content: np.ndarray,
        style: np.ndarray,
        progress_callback: Optional[Callable[[float, str], None]] = None,
    ) -> np.ndarray:
        """Match histogram of content to style."""
        output = np.zeros_like(content)

        for c in range(3):
            if progress_callback:
                progress_callback(20 + c * 20, f"Processing channel {c + 1}/3...")

            # Get sorted pixel values
            content_sorted = np.sort(content[:, :, c].flatten())
            style_sorted = np.sort(style[:, :, c].flatten())

            # Create mapping
            content_vals = content[:, :, c].flatten()
            indices = np.searchsorted(content_sorted, content_vals)
            indices = np.clip(indices, 0, len(style_sorted) - 1)

            # Map to style distribution
            output[:, :, c] = style_sorted[indices].reshape(content.shape[:2])

            await asyncio.sleep(0)  # Yield to event loop

        return output

    async def _color_transfer(
        self,
        content: np.ndarray,
        style: np.ndarray,
        progress_callback: Optional[Callable[[float, str], None]] = None,
    ) -> np.ndarray:
        """Transfer color statistics from style to content."""
        if progress_callback:
            progress_callback(30, "Computing color statistics...")

        # Convert to LAB-like space (simplified)
        content_mean = np.mean(content, axis=(0, 1))
        content_std = np.std(content, axis=(0, 1)) + 1e-6

        style_mean = np.mean(style, axis=(0, 1))
        style_std = np.std(style, axis=(0, 1)) + 1e-6

        if progress_callback:
            progress_callback(60, "Applying color transfer...")

        # Normalize and denormalize
        output = (content - content_mean) / content_std
        output = output * style_std + style_mean

        return np.clip(output, 0, 1)

    def _preserve_color(
        self,
        original: np.ndarray,
        stylized: np.ndarray,
        amount: float,
    ) -> np.ndarray:
        """Blend original colors back into stylized image."""
        # Simple luminance transfer approach
        orig_lum = 0.299 * original[:, :, 0] + 0.587 * original[:, :, 1] + 0.114 * original[:, :, 2]
        style_lum = 0.299 * stylized[:, :, 0] + 0.587 * stylized[:, :, 1] + 0.114 * stylized[:, :, 2]

        # Ratio of luminances
        ratio = (style_lum + 1e-6) / (orig_lum + 1e-6)
        ratio = ratio[:, :, np.newaxis]

        # Apply stylized luminance to original colors
        color_preserved = np.clip(original * ratio, 0, 1)

        # Blend
        return stylized * (1 - amount) + color_preserved * amount

    async def is_available(self) -> bool:
        return HAS_PIL


class MockStyleBackend(StyleTransferBackend):
    """Mock backend for testing without neural network dependencies."""

    @property
    def name(self) -> str:
        return "Mock Style Backend"

    @property
    def supported_methods(self) -> list[TransferMethod]:
        return list(TransferMethod)

    async def transfer(
        self,
        content: np.ndarray,
        style: np.ndarray,
        config: StyleConfig,
        progress_callback: Optional[Callable[[float, str], None]] = None,
    ) -> StyleResult:
        """Generate mock stylized output."""
        start_time = time.time()

        # Simulate progress
        for i in range(10):
            if progress_callback:
                progress_callback(i * 10, f"Simulating style transfer step {i + 1}/10...")
            await asyncio.sleep(0.05)

        # Simple blend as mock output
        style_resized = self._resize_to_match(style, content.shape[:2])
        alpha = config.style_weight / (config.content_weight + config.style_weight)
        output = (content * (1 - alpha * 0.3) + style_resized * (alpha * 0.3)).astype(np.uint8)

        # Apply some color shift to make it look "stylized"
        output = np.clip(output.astype(np.float32) * 1.1, 0, 255).astype(np.uint8)

        transfer_time = time.time() - start_time

        if progress_callback:
            progress_callback(100, "Complete")

        return StyleResult(
            output_image=output,
            content_image=content,
            style_image=style,
            config=config,
            transfer_time=transfer_time,
            method_used=f"mock_{config.method.value}",
            loss_history=[1000 - i * 100 for i in range(10)],
            metadata={"backend": "mock", "simulated": True},
        )

    def _resize_to_match(self, img: np.ndarray, target_shape: tuple[int, int]) -> np.ndarray:
        """Resize image to match target dimensions."""
        if not HAS_PIL:
            # Simple nearest-neighbor resize
            h, w = target_shape
            indices_h = (np.arange(h) * img.shape[0] / h).astype(int)
            indices_w = (np.arange(w) * img.shape[1] / w).astype(int)
            return img[indices_h][:, indices_w]

        from PIL import Image
        pil_img = Image.fromarray(img)
        pil_img = pil_img.resize((target_shape[1], target_shape[0]), Image.Resampling.LANCZOS)
        return np.array(pil_img)

    async def is_available(self) -> bool:
        return True


class NeuralStyleTransfer:
    """Neural style transfer orchestrator.

    Manages backends and provides high-level API for style transfer.

    Example:
        >>> nst = NeuralStyleTransfer()
        >>> config = StyleConfig(preset=StylePreset.STARRY_NIGHT)
        >>> result = await nst.transfer(content_img, style_img, config)
        >>> stylized = result.output_image
    """

    def __init__(self, default_method: TransferMethod = TransferMethod.NEURAL):
        """Initialize neural style transfer.

        Args:
            default_method: Default transfer method to use
        """
        self.default_method = default_method
        self._backends: dict[TransferMethod, StyleTransferBackend] = {
            TransferMethod.COLOR_TRANSFER: PILStyleBackend(),
            TransferMethod.HISTOGRAM_MATCHING: PILStyleBackend(),
            TransferMethod.NEURAL: MockStyleBackend(),
            TransferMethod.FAST_NEURAL: MockStyleBackend(),
        }
        self._transfer_history: list[StyleResult] = []

    def register_backend(self, method: TransferMethod, backend: StyleTransferBackend):
        """Register a custom backend for a transfer method.

        Args:
            method: The transfer method
            backend: The backend implementation
        """
        self._backends[method] = backend

    def get_backend(self, method: TransferMethod) -> StyleTransferBackend:
        """Get the backend for a transfer method.

        Args:
            method: The transfer method

        Returns:
            The backend instance

        Raises:
            ValueError: If no backend registered for method
        """
        if method not in self._backends:
            raise ValueError(f"No backend registered for method {method}")
        return self._backends[method]

    async def transfer(
        self,
        content: np.ndarray,
        style: np.ndarray,
        config: Optional[StyleConfig] = None,
        progress_callback: Optional[Callable[[float, str], None]] = None,
    ) -> StyleResult:
        """Apply style transfer from style image to content image.

        Args:
            content: Content image (H, W, C) numpy array
            style: Style image (H, W, C) numpy array
            config: Style transfer configuration
            progress_callback: Optional progress callback

        Returns:
            StyleResult with stylized image

        Raises:
            ValueError: If configuration is invalid
            RuntimeError: If backend is unavailable
        """
        config = config or StyleConfig()

        # Validate
        errors = config.validate()
        if errors:
            raise ValueError(f"Invalid configuration: {'; '.join(errors)}")

        # Get backend
        backend = self.get_backend(config.method)

        # Check availability
        if not await backend.is_available():
            raise RuntimeError(f"Backend for {config.method} is not available")

        # Apply transfer
        result = await backend.transfer(content, style, config, progress_callback)

        # Store history
        self._transfer_history.append(result)

        return result

    async def transfer_with_preset(
        self,
        content: np.ndarray,
        preset: StylePreset,
        style: Optional[np.ndarray] = None,
        progress_callback: Optional[Callable[[float, str], None]] = None,
    ) -> StyleResult:
        """Apply style transfer using a preset.

        Args:
            content: Content image
            preset: Style preset to apply
            style: Optional custom style image (uses preset default if not provided)
            progress_callback: Optional progress callback

        Returns:
            StyleResult
        """
        config = StyleConfig(preset=preset)

        # If no style image provided, generate a placeholder
        if style is None:
            style = self._generate_preset_style(preset, content.shape)

        return await self.transfer(content, style, config, progress_callback)

    def _generate_preset_style(self, preset: StylePreset, shape: tuple) -> np.ndarray:
        """Generate a style image for a preset."""
        h, w = shape[:2]
        rng = np.random.default_rng(hash(preset.value))

        # Generate texture-like pattern
        style = rng.integers(0, 255, (h, w, 3), dtype=np.uint8)

        # Apply preset-specific modifications
        if preset == StylePreset.STARRY_NIGHT:
            # Blue-yellow gradient
            style[:, :, 0] = np.linspace(0, 100, h).reshape(-1, 1).repeat(w, axis=1)
            style[:, :, 2] = np.linspace(100, 255, h).reshape(-1, 1).repeat(w, axis=1)
        elif preset == StylePreset.PENCIL_SKETCH:
            # Grayscale
            gray = np.mean(style, axis=2, keepdims=True)
            style = np.broadcast_to(gray, shape).astype(np.uint8)
        elif preset == StylePreset.VINTAGE:
            # Sepia tones
            style[:, :, 0] = np.clip(style[:, :, 0] * 0.9, 0, 255)
            style[:, :, 1] = np.clip(style[:, :, 1] * 0.7, 0, 255)
            style[:, :, 2] = np.clip(style[:, :, 2] * 0.5, 0, 255)

        return style

    @property
    def history(self) -> list[StyleResult]:
        """Get transfer history."""
        return self._transfer_history.copy()

    def clear_history(self):
        """Clear transfer history."""
        self._transfer_history.clear()

    def list_presets(self) -> dict[str, dict]:
        """Get all available style presets with descriptions."""
        return STYLE_PRESETS.copy()

    async def list_available_methods(self) -> list[tuple[TransferMethod, bool]]:
        """List all methods and their availability.

        Returns:
            List of (method, is_available) tuples
        """
        results = []
        for method in TransferMethod:
            try:
                backend = self.get_backend(method)
                available = await backend.is_available()
                results.append((method, available))
            except ValueError:
                results.append((method, False))
        return results


# Convenience aliases
StyleTransferEngine = NeuralStyleTransfer


# Convenience functions

async def apply_style(
    content: np.ndarray,
    style: np.ndarray,
    preset: Optional[StylePreset] = None,
    **kwargs,
) -> np.ndarray:
    """Quick style transfer function.

    Args:
        content: Content image
        style: Style image
        preset: Optional style preset
        **kwargs: Additional StyleConfig parameters

    Returns:
        Stylized image as numpy array
    """
    nst = NeuralStyleTransfer()
    config = StyleConfig(preset=preset, **kwargs) if kwargs or preset else None
    result = await nst.transfer(content, style, config)
    return result.output_image


def create_style_config(
    preset: Optional[StylePreset] = None,
    **kwargs,
) -> StyleConfig:
    """Create a StyleConfig with optional preset.

    Args:
        preset: Style preset
        **kwargs: Override default values

    Returns:
        Configured StyleConfig
    """
    return StyleConfig(preset=preset, **kwargs)
