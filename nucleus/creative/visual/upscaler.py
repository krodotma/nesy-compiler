"""
Image Upscaler Module
=====================

Provides image upscaling capabilities using various methods including
CPU-based algorithms and GPU-accelerated neural networks.
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


class UpscaleMethod(str, Enum):
    """Supported upscaling methods."""
    LANCZOS = "lanczos"
    BICUBIC = "bicubic"
    BILINEAR = "bilinear"
    NEAREST = "nearest"
    REAL_ESRGAN = "real_esrgan"
    GFPGAN = "gfpgan"


@dataclass
class UpscaleResult:
    """Result of an upscale operation.

    Attributes:
        output_image: The upscaled image as numpy array
        output_size: Output dimensions as (width, height) tuple
        scale_factor: The scale factor applied
        model_used: Identifier of the method/model used
        upscale_time: Time taken in seconds
        input_size: Original input dimensions (width, height)
        metadata: Additional metadata
    """
    output_image: np.ndarray = field(default=None, repr=False)
    output_size: tuple[int, int] = (0, 0)
    scale_factor: int = 1
    model_used: str = "unknown"
    upscale_time: float = 0.0
    input_size: tuple[int, int] = (0, 0)
    metadata: dict = field(default_factory=dict)

    @property
    def width(self) -> int:
        """Output image width."""
        return self.output_size[0]

    @property
    def height(self) -> int:
        """Output image height."""
        return self.output_size[1]


class UpscalerBackend(ABC):
    """Abstract base class for upscaler backends."""

    @property
    @abstractmethod
    def name(self) -> str:
        """Backend name."""
        pass

    @property
    @abstractmethod
    def supported_methods(self) -> list[UpscaleMethod]:
        """List of supported upscale methods."""
        pass

    @property
    @abstractmethod
    def max_scale_factor(self) -> int:
        """Maximum supported scale factor."""
        pass

    @abstractmethod
    async def upscale(
        self,
        image: np.ndarray,
        scale_factor: int,
        method: str,
        progress_callback: Optional[Callable[[float, str], None]] = None,
    ) -> UpscaleResult:
        """Upscale an image.

        Args:
            image: Input image as numpy array (H, W, C)
            scale_factor: Factor to scale by (2, 4, etc.)
            method: Upscaling method to use
            progress_callback: Optional progress callback

        Returns:
            UpscaleResult with the upscaled image
        """
        pass

    @abstractmethod
    async def is_available(self) -> bool:
        """Check if backend is available."""
        pass


class SimpleUpscaler(UpscalerBackend):
    """Simple CPU-based upscaler using PIL/Lanczos.

    This is a lightweight upscaler that doesn't require GPU or
    neural network dependencies. Suitable for basic upscaling needs.

    Example:
        >>> upscaler = SimpleUpscaler()
        >>> result = await upscaler.upscale(image, scale_factor=2, method="lanczos")
        >>> print(f"Output size: {result.output_size}")
    """

    @property
    def name(self) -> str:
        return "Simple Upscaler"

    @property
    def supported_methods(self) -> list[UpscaleMethod]:
        return [
            UpscaleMethod.LANCZOS,
            UpscaleMethod.BICUBIC,
            UpscaleMethod.BILINEAR,
            UpscaleMethod.NEAREST,
        ]

    @property
    def max_scale_factor(self) -> int:
        return 8

    async def upscale(
        self,
        image: np.ndarray,
        scale_factor: int = 2,
        method: str = "lanczos",
        progress_callback: Optional[Callable[[float, str], None]] = None,
    ) -> UpscaleResult:
        """Upscale image using PIL resampling methods.

        Args:
            image: Input image as numpy array (H, W, C) with uint8 dtype
            scale_factor: Factor to scale by (1-8)
            method: Resampling method ('lanczos', 'bicubic', 'bilinear', 'nearest')
            progress_callback: Optional progress callback

        Returns:
            UpscaleResult with upscaled image

        Raises:
            ValueError: If method is not supported
            RuntimeError: If PIL is not available
        """
        start_time = time.time()

        if not HAS_PIL:
            raise RuntimeError("PIL is required for SimpleUpscaler")

        # Validate scale factor
        scale_factor = max(1, min(scale_factor, self.max_scale_factor))

        # Get input dimensions
        h, w = image.shape[:2]
        input_size = (w, h)

        if progress_callback:
            progress_callback(10, "Preparing image...")

        # Map method names to PIL resampling filters
        method_lower = method.lower()
        resampling_map = {
            "lanczos": Image.Resampling.LANCZOS,
            "bicubic": Image.Resampling.BICUBIC,
            "bilinear": Image.Resampling.BILINEAR,
            "nearest": Image.Resampling.NEAREST,
        }

        if method_lower not in resampling_map:
            raise ValueError(f"Unsupported method: {method}. Supported: {list(resampling_map.keys())}")

        resampling = resampling_map[method_lower]

        if progress_callback:
            progress_callback(30, "Converting to PIL...")

        # Convert numpy to PIL
        pil_image = Image.fromarray(image)

        # Calculate new size
        new_width = w * scale_factor
        new_height = h * scale_factor
        output_size = (new_width, new_height)

        if progress_callback:
            progress_callback(50, f"Upscaling to {new_width}x{new_height}...")

        # Perform upscale
        # Use asyncio.to_thread for CPU-bound operation
        def _resize():
            return pil_image.resize((new_width, new_height), resampling)

        upscaled_pil = await asyncio.to_thread(_resize)

        if progress_callback:
            progress_callback(80, "Converting back to array...")

        # Convert back to numpy
        output_image = np.array(upscaled_pil)

        upscale_time = time.time() - start_time

        if progress_callback:
            progress_callback(100, "Complete")

        # Determine model name for result
        model_used = f"PIL_{method_lower}"

        return UpscaleResult(
            output_image=output_image,
            output_size=output_size,
            scale_factor=scale_factor,
            model_used=model_used,
            upscale_time=upscale_time,
            input_size=input_size,
            metadata={
                "backend": "simple",
                "resampling": method_lower,
            },
        )

    async def is_available(self) -> bool:
        """Check if PIL is available."""
        return HAS_PIL


class RealESRGANUpscaler(UpscalerBackend):
    """Real-ESRGAN based upscaler for high-quality AI upscaling.

    Requires the real-esrgan package and appropriate GPU resources.
    Falls back to mock implementation if not available.
    """

    def __init__(self, model_name: str = "RealESRGAN_x4plus", device: str = "auto"):
        """Initialize Real-ESRGAN upscaler.

        Args:
            model_name: Name of the Real-ESRGAN model
            device: Device to run on ('cuda', 'cpu', 'auto')
        """
        self.model_name = model_name
        self.device = device
        self._model = None
        self._available: Optional[bool] = None

    @property
    def name(self) -> str:
        return "Real-ESRGAN Upscaler"

    @property
    def supported_methods(self) -> list[UpscaleMethod]:
        return [UpscaleMethod.REAL_ESRGAN]

    @property
    def max_scale_factor(self) -> int:
        return 4

    async def _load_model(self):
        """Lazy-load the Real-ESRGAN model."""
        if self._model is not None:
            return True

        try:
            from basicsr.archs.rrdbnet_arch import RRDBNet
            from realesrgan import RealESRGANer
            import torch

            device = self.device
            if device == "auto":
                device = "cuda" if torch.cuda.is_available() else "cpu"

            # Initialize model architecture
            model = RRDBNet(
                num_in_ch=3,
                num_out_ch=3,
                num_feat=64,
                num_block=23,
                num_grow_ch=32,
                scale=4,
            )

            # Create upscaler
            self._model = RealESRGANer(
                scale=4,
                model_path=None,  # Will use default
                model=model,
                device=device,
            )
            return True

        except ImportError:
            return False

    async def upscale(
        self,
        image: np.ndarray,
        scale_factor: int = 4,
        method: str = "real_esrgan",
        progress_callback: Optional[Callable[[float, str], None]] = None,
    ) -> UpscaleResult:
        """Upscale using Real-ESRGAN.

        Args:
            image: Input image (H, W, C)
            scale_factor: Scale factor (2 or 4)
            method: Method name (ignored, always uses Real-ESRGAN)
            progress_callback: Progress callback

        Returns:
            UpscaleResult with upscaled image
        """
        start_time = time.time()

        if progress_callback:
            progress_callback(10, "Loading model...")

        loaded = await self._load_model()
        if not loaded:
            raise RuntimeError("Real-ESRGAN not available, install with: pip install realesrgan")

        h, w = image.shape[:2]
        input_size = (w, h)

        if progress_callback:
            progress_callback(30, "Processing...")

        # Run inference
        def _enhance():
            output, _ = self._model.enhance(image, outscale=scale_factor)
            return output

        output_image = await asyncio.to_thread(_enhance)

        new_h, new_w = output_image.shape[:2]
        output_size = (new_w, new_h)

        upscale_time = time.time() - start_time

        if progress_callback:
            progress_callback(100, "Complete")

        return UpscaleResult(
            output_image=output_image,
            output_size=output_size,
            scale_factor=scale_factor,
            model_used=f"RealESRGAN_{self.model_name}",
            upscale_time=upscale_time,
            input_size=input_size,
            metadata={
                "backend": "realesrgan",
                "model": self.model_name,
            },
        )

    async def is_available(self) -> bool:
        """Check if Real-ESRGAN dependencies are installed."""
        if self._available is not None:
            return self._available

        try:
            from basicsr.archs.rrdbnet_arch import RRDBNet
            from realesrgan import RealESRGANer
            self._available = True
        except ImportError:
            self._available = False

        return self._available


class GFPGANFaceEnhancer(UpscalerBackend):
    """GFPGAN-based face enhancement and restoration.

    Specialized for face images, provides better results than
    general upscalers for portraits.
    """

    def __init__(self, model_name: str = "GFPGANv1.4", device: str = "auto"):
        """Initialize GFPGAN enhancer.

        Args:
            model_name: Name of the GFPGAN model
            device: Device to run on
        """
        self.model_name = model_name
        self.device = device
        self._model = None
        self._available: Optional[bool] = None

    @property
    def name(self) -> str:
        return "GFPGAN Face Enhancer"

    @property
    def supported_methods(self) -> list[UpscaleMethod]:
        return [UpscaleMethod.GFPGAN]

    @property
    def max_scale_factor(self) -> int:
        return 4

    async def upscale(
        self,
        image: np.ndarray,
        scale_factor: int = 2,
        method: str = "gfpgan",
        progress_callback: Optional[Callable[[float, str], None]] = None,
    ) -> UpscaleResult:
        """Enhance face image using GFPGAN.

        Args:
            image: Input face image (H, W, C)
            scale_factor: Scale factor
            method: Method name (ignored)
            progress_callback: Progress callback

        Returns:
            UpscaleResult with enhanced image
        """
        start_time = time.time()

        if progress_callback:
            progress_callback(10, "Initializing face enhancement...")

        # Check availability
        if not await self.is_available():
            raise RuntimeError("GFPGAN not available, install with: pip install gfpgan")

        h, w = image.shape[:2]
        input_size = (w, h)

        if progress_callback:
            progress_callback(50, "Enhancing faces...")

        # For now, fall back to simple upscale if GFPGAN not loaded
        simple = SimpleUpscaler()
        result = await simple.upscale(image, scale_factor, "lanczos", progress_callback)

        # Update metadata
        result.model_used = f"GFPGAN_{self.model_name}_fallback"
        result.metadata["backend"] = "gfpgan_fallback"

        return result

    async def is_available(self) -> bool:
        """Check if GFPGAN is available."""
        if self._available is not None:
            return self._available

        try:
            from gfpgan import GFPGANer
            self._available = True
        except ImportError:
            self._available = False

        return self._available


class Upscaler:
    """Main upscaler orchestrator.

    Manages multiple backends and provides high-level API for image upscaling.

    Example:
        >>> upscaler = Upscaler()
        >>> result = await upscaler.upscale(image, scale_factor=2)
        >>> print(f"Upscaled to {result.output_size}")

        # Use specific method
        >>> result = await upscaler.upscale(image, scale_factor=4, method="real_esrgan")
    """

    def __init__(self, default_method: UpscaleMethod = UpscaleMethod.LANCZOS):
        """Initialize upscaler.

        Args:
            default_method: Default upscaling method
        """
        self.default_method = default_method
        self._backends: dict[UpscaleMethod, UpscalerBackend] = {
            UpscaleMethod.LANCZOS: SimpleUpscaler(),
            UpscaleMethod.BICUBIC: SimpleUpscaler(),
            UpscaleMethod.BILINEAR: SimpleUpscaler(),
            UpscaleMethod.NEAREST: SimpleUpscaler(),
        }
        self._upscale_history: list[UpscaleResult] = []

    def register_backend(self, method: UpscaleMethod, backend: UpscalerBackend):
        """Register a backend for a method.

        Args:
            method: The upscale method
            backend: The backend implementation
        """
        self._backends[method] = backend

    def get_backend(self, method: UpscaleMethod) -> UpscalerBackend:
        """Get backend for a method.

        Args:
            method: The upscale method

        Returns:
            The backend instance
        """
        if method not in self._backends:
            # Try to create default backends
            if method == UpscaleMethod.REAL_ESRGAN:
                self._backends[method] = RealESRGANUpscaler()
            elif method == UpscaleMethod.GFPGAN:
                self._backends[method] = GFPGANFaceEnhancer()
            else:
                raise ValueError(f"No backend for method {method}")
        return self._backends[method]

    async def upscale(
        self,
        image: np.ndarray,
        scale_factor: int = 2,
        method: Optional[Union[str, UpscaleMethod]] = None,
        progress_callback: Optional[Callable[[float, str], None]] = None,
    ) -> UpscaleResult:
        """Upscale an image.

        Args:
            image: Input image (H, W, C) numpy array
            scale_factor: Factor to scale by
            method: Upscaling method (uses default if not specified)
            progress_callback: Progress callback

        Returns:
            UpscaleResult with upscaled image

        Raises:
            ValueError: If method is invalid
            RuntimeError: If backend is unavailable
        """
        # Resolve method
        if method is None:
            resolved_method = self.default_method
        elif isinstance(method, str):
            try:
                resolved_method = UpscaleMethod(method.lower())
            except ValueError:
                raise ValueError(f"Unknown method: {method}")
        else:
            resolved_method = method

        # Get backend
        backend = self.get_backend(resolved_method)

        # Check availability
        if not await backend.is_available():
            raise RuntimeError(f"Backend for {resolved_method} is not available")

        # Validate scale factor
        if scale_factor > backend.max_scale_factor:
            raise ValueError(
                f"Scale factor {scale_factor} exceeds max {backend.max_scale_factor} "
                f"for {backend.name}"
            )

        # Perform upscale
        result = await backend.upscale(image, scale_factor, resolved_method.value, progress_callback)

        # Store history
        self._upscale_history.append(result)

        return result

    @property
    def history(self) -> list[UpscaleResult]:
        """Get upscale history."""
        return self._upscale_history.copy()

    def clear_history(self):
        """Clear upscale history."""
        self._upscale_history.clear()

    async def list_available_methods(self) -> list[tuple[UpscaleMethod, bool]]:
        """List all methods and their availability.

        Returns:
            List of (method, is_available) tuples
        """
        results = []
        for method in UpscaleMethod:
            try:
                backend = self.get_backend(method)
                available = await backend.is_available()
                results.append((method, available))
            except ValueError:
                results.append((method, False))
        return results


# Convenience functions

async def upscale_image(
    image: np.ndarray,
    scale_factor: int = 2,
    method: str = "lanczos",
) -> np.ndarray:
    """Quick upscale function.

    Args:
        image: Input image
        scale_factor: Scale factor
        method: Upscaling method

    Returns:
        Upscaled image as numpy array
    """
    upscaler = Upscaler()
    result = await upscaler.upscale(image, scale_factor, method)
    return result.output_image


async def upscale_2x(image: np.ndarray) -> np.ndarray:
    """Upscale image by 2x using Lanczos.

    Args:
        image: Input image

    Returns:
        Upscaled image
    """
    return await upscale_image(image, scale_factor=2, method="lanczos")


async def upscale_4x(image: np.ndarray) -> np.ndarray:
    """Upscale image by 4x using Lanczos.

    Args:
        image: Input image

    Returns:
        Upscaled image
    """
    return await upscale_image(image, scale_factor=4, method="lanczos")
