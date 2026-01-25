"""
Image Generator Module
======================

Provides image generation capabilities using various backends and providers.
Supports text-to-image, image-to-image, and inpainting workflows.
"""

from __future__ import annotations

import asyncio
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Optional, Union
import uuid
from datetime import datetime, timezone

import numpy as np

try:
    from PIL import Image
    HAS_PIL = True
except ImportError:
    HAS_PIL = False


class ImageProviderType(str, Enum):
    """Supported image generation providers."""
    LOCAL_DIFFUSION = "local_diffusion"
    STABLE_DIFFUSION_API = "sd_api"
    DALLE = "dalle"
    MIDJOURNEY = "midjourney"
    REPLICATE = "replicate"
    MOCK = "mock"


@dataclass
class ImageGenerationConfig:
    """Configuration for image generation.

    Attributes:
        prompt: Text prompt describing the desired image
        negative_prompt: Text describing what to avoid
        width: Output image width in pixels
        height: Output image height in pixels
        num_inference_steps: Number of diffusion steps
        guidance_scale: Classifier-free guidance scale
        seed: Random seed for reproducibility
        num_images: Number of images to generate
        scheduler: Diffusion scheduler type
        model_id: Specific model identifier
        provider: Which generation provider to use
    """
    prompt: str
    negative_prompt: str = ""
    width: int = 512
    height: int = 512
    num_inference_steps: int = 50
    guidance_scale: float = 7.5
    seed: Optional[int] = None
    num_images: int = 1
    scheduler: str = "euler_a"
    model_id: str = "stable-diffusion-xl"
    provider: ImageProviderType = ImageProviderType.MOCK
    extra_params: dict = field(default_factory=dict)

    def validate(self) -> list[str]:
        """Validate configuration and return list of errors."""
        errors = []
        if not self.prompt:
            errors.append("Prompt cannot be empty")
        if self.width <= 0 or self.height <= 0:
            errors.append("Width and height must be positive")
        if self.width > 4096 or self.height > 4096:
            errors.append("Maximum dimension is 4096 pixels")
        if self.num_inference_steps < 1 or self.num_inference_steps > 500:
            errors.append("Inference steps must be between 1 and 500")
        if self.guidance_scale < 1.0 or self.guidance_scale > 30.0:
            errors.append("Guidance scale must be between 1.0 and 30.0")
        if self.num_images < 1 or self.num_images > 16:
            errors.append("Number of images must be between 1 and 16")
        return errors


@dataclass
class GenerationResult:
    """Result of an image generation operation.

    Attributes:
        images: List of generated images as numpy arrays
        prompt: The prompt used for generation
        config: The configuration used
        generation_time: Time taken in seconds
        model_used: Identifier of the model used
        metadata: Additional metadata about the generation
    """
    images: list[np.ndarray]
    prompt: str
    config: ImageGenerationConfig
    generation_time: float
    model_used: str
    seed_used: int
    metadata: dict = field(default_factory=dict)

    @property
    def image(self) -> np.ndarray:
        """Get the first generated image."""
        if not self.images:
            raise ValueError("No images generated")
        return self.images[0]

    @property
    def count(self) -> int:
        """Number of generated images."""
        return len(self.images)


class ImageProvider(ABC):
    """Abstract base class for image generation providers.

    Implement this class to add support for new image generation backends.
    """

    @property
    @abstractmethod
    def name(self) -> str:
        """Provider name."""
        pass

    @property
    @abstractmethod
    def supported_models(self) -> list[str]:
        """List of supported model identifiers."""
        pass

    @abstractmethod
    async def generate(
        self,
        config: ImageGenerationConfig,
        progress_callback: Optional[Callable[[float, str], None]] = None,
    ) -> GenerationResult:
        """Generate images from the configuration.

        Args:
            config: Generation configuration
            progress_callback: Optional callback for progress updates

        Returns:
            GenerationResult with generated images
        """
        pass

    @abstractmethod
    async def is_available(self) -> bool:
        """Check if the provider is available and properly configured."""
        pass

    def validate_config(self, config: ImageGenerationConfig) -> list[str]:
        """Validate configuration for this specific provider.

        Override to add provider-specific validation.
        """
        return config.validate()


class MockImageProvider(ImageProvider):
    """Mock provider for testing without actual generation.

    Generates random colored images with the prompt text overlaid.
    """

    @property
    def name(self) -> str:
        return "Mock Provider"

    @property
    def supported_models(self) -> list[str]:
        return ["mock-v1", "mock-v2"]

    async def generate(
        self,
        config: ImageGenerationConfig,
        progress_callback: Optional[Callable[[float, str], None]] = None,
    ) -> GenerationResult:
        """Generate mock images."""
        import time
        start_time = time.time()

        seed = config.seed or int(time.time() * 1000) % (2**32)
        rng = np.random.default_rng(seed)

        images = []
        for i in range(config.num_images):
            if progress_callback:
                progress = (i + 1) / config.num_images * 100
                progress_callback(progress, f"Generating image {i + 1}/{config.num_images}")

            # Generate random colored image
            img = rng.integers(0, 255, (config.height, config.width, 3), dtype=np.uint8)

            # Add some structure by blending with gradient
            gradient_h = np.linspace(0, 255, config.height).reshape(-1, 1, 1)
            gradient_w = np.linspace(0, 255, config.width).reshape(1, -1, 1)
            gradient = (gradient_h + gradient_w) / 2
            gradient = np.broadcast_to(gradient, img.shape).astype(np.uint8)

            img = ((img.astype(np.float32) + gradient.astype(np.float32)) / 2).astype(np.uint8)
            images.append(img)

            # Simulate processing time
            await asyncio.sleep(0.1)

        generation_time = time.time() - start_time

        return GenerationResult(
            images=images,
            prompt=config.prompt,
            config=config,
            generation_time=generation_time,
            model_used="mock-v1",
            seed_used=seed,
            metadata={"provider": "mock", "simulated": True},
        )

    async def is_available(self) -> bool:
        return True


class LocalDiffusionProvider(ImageProvider):
    """Provider for local diffusion model inference.

    Requires diffusers library and appropriate GPU resources.
    """

    def __init__(self, model_path: Optional[str] = None, device: str = "auto"):
        """Initialize local diffusion provider.

        Args:
            model_path: Path to local model or HuggingFace model ID
            device: Device to run on ('cuda', 'cpu', 'auto')
        """
        self.model_path = model_path
        self.device = device
        self._pipeline = None

    @property
    def name(self) -> str:
        return "Local Diffusion"

    @property
    def supported_models(self) -> list[str]:
        return [
            "stable-diffusion-xl",
            "stable-diffusion-2.1",
            "stable-diffusion-1.5",
            "sdxl-turbo",
        ]

    async def _load_pipeline(self, model_id: str):
        """Lazy-load the diffusion pipeline."""
        if self._pipeline is not None:
            return

        try:
            from diffusers import AutoPipelineForText2Image
            import torch

            device = self.device
            if device == "auto":
                device = "cuda" if torch.cuda.is_available() else "cpu"

            path = self.model_path or model_id
            self._pipeline = AutoPipelineForText2Image.from_pretrained(
                path,
                torch_dtype=torch.float16 if device == "cuda" else torch.float32,
            )
            self._pipeline = self._pipeline.to(device)

        except ImportError:
            raise RuntimeError("diffusers library required for local generation")

    async def generate(
        self,
        config: ImageGenerationConfig,
        progress_callback: Optional[Callable[[float, str], None]] = None,
    ) -> GenerationResult:
        """Generate images using local diffusion model."""
        import time
        start_time = time.time()

        await self._load_pipeline(config.model_id)

        if progress_callback:
            progress_callback(10, "Model loaded, generating...")

        import torch

        generator = None
        seed = config.seed or int(time.time() * 1000) % (2**32)
        if config.seed is not None:
            generator = torch.Generator(device=self._pipeline.device).manual_seed(seed)

        # Run generation
        output = self._pipeline(
            prompt=config.prompt,
            negative_prompt=config.negative_prompt or None,
            width=config.width,
            height=config.height,
            num_inference_steps=config.num_inference_steps,
            guidance_scale=config.guidance_scale,
            num_images_per_prompt=config.num_images,
            generator=generator,
        )

        if progress_callback:
            progress_callback(90, "Post-processing...")

        # Convert PIL images to numpy
        images = [np.array(img) for img in output.images]

        generation_time = time.time() - start_time

        if progress_callback:
            progress_callback(100, "Complete")

        return GenerationResult(
            images=images,
            prompt=config.prompt,
            config=config,
            generation_time=generation_time,
            model_used=config.model_id,
            seed_used=seed,
            metadata={"provider": "local_diffusion", "device": str(self._pipeline.device)},
        )

    async def is_available(self) -> bool:
        """Check if diffusers is installed and GPU available."""
        try:
            import torch
            from diffusers import AutoPipelineForText2Image
            return True
        except ImportError:
            return False


class ImageGenerator:
    """Main image generation orchestrator.

    Manages multiple providers and handles generation workflows.

    Example:
        >>> generator = ImageGenerator()
        >>> config = ImageGenerationConfig(
        ...     prompt="A sunset over mountains",
        ...     width=512,
        ...     height=512,
        ... )
        >>> result = await generator.generate(config)
        >>> print(f"Generated {result.count} images in {result.generation_time:.2f}s")
    """

    def __init__(self, default_provider: Optional[ImageProviderType] = None):
        """Initialize the image generator.

        Args:
            default_provider: Default provider to use if not specified in config
        """
        self.default_provider = default_provider or ImageProviderType.MOCK
        self._providers: dict[ImageProviderType, ImageProvider] = {
            ImageProviderType.MOCK: MockImageProvider(),
        }
        self._generation_history: list[GenerationResult] = []

    def register_provider(self, provider_type: ImageProviderType, provider: ImageProvider):
        """Register a custom provider.

        Args:
            provider_type: The provider type identifier
            provider: The provider instance
        """
        self._providers[provider_type] = provider

    def get_provider(self, provider_type: ImageProviderType) -> ImageProvider:
        """Get a registered provider.

        Args:
            provider_type: The provider type to get

        Returns:
            The provider instance

        Raises:
            ValueError: If provider is not registered
        """
        if provider_type not in self._providers:
            # Try to create default provider
            if provider_type == ImageProviderType.LOCAL_DIFFUSION:
                self._providers[provider_type] = LocalDiffusionProvider()
            else:
                raise ValueError(f"Provider {provider_type} not registered")
        return self._providers[provider_type]

    async def generate(
        self,
        config: ImageGenerationConfig,
        progress_callback: Optional[Callable[[float, str], None]] = None,
    ) -> GenerationResult:
        """Generate images using the configured provider.

        Args:
            config: Generation configuration
            progress_callback: Optional callback for progress updates

        Returns:
            GenerationResult with generated images

        Raises:
            ValueError: If configuration validation fails
            RuntimeError: If generation fails
        """
        # Validate configuration
        errors = config.validate()
        if errors:
            raise ValueError(f"Invalid configuration: {'; '.join(errors)}")

        # Get provider
        provider_type = config.provider or self.default_provider
        provider = self.get_provider(provider_type)

        # Check availability
        if not await provider.is_available():
            raise RuntimeError(f"Provider {provider_type} is not available")

        # Provider-specific validation
        provider_errors = provider.validate_config(config)
        if provider_errors:
            raise ValueError(f"Provider validation failed: {'; '.join(provider_errors)}")

        # Generate
        result = await provider.generate(config, progress_callback)

        # Store in history
        self._generation_history.append(result)

        return result

    async def generate_from_prompt(
        self,
        prompt: str,
        width: int = 512,
        height: int = 512,
        **kwargs,
    ) -> GenerationResult:
        """Convenience method to generate from a simple prompt.

        Args:
            prompt: Text prompt
            width: Image width
            height: Image height
            **kwargs: Additional config parameters

        Returns:
            GenerationResult
        """
        config = ImageGenerationConfig(
            prompt=prompt,
            width=width,
            height=height,
            **kwargs,
        )
        return await self.generate(config)

    @property
    def history(self) -> list[GenerationResult]:
        """Get generation history."""
        return self._generation_history.copy()

    def clear_history(self):
        """Clear generation history."""
        self._generation_history.clear()

    async def list_available_providers(self) -> list[tuple[ImageProviderType, bool]]:
        """List all providers and their availability status.

        Returns:
            List of (provider_type, is_available) tuples
        """
        results = []
        for pt in ImageProviderType:
            try:
                provider = self.get_provider(pt)
                available = await provider.is_available()
                results.append((pt, available))
            except ValueError:
                results.append((pt, False))
        return results


# Convenience functions

async def generate_image(prompt: str, **kwargs) -> np.ndarray:
    """Quick image generation from prompt.

    Args:
        prompt: Text prompt
        **kwargs: Additional configuration options

    Returns:
        Generated image as numpy array
    """
    generator = ImageGenerator()
    result = await generator.generate_from_prompt(prompt, **kwargs)
    return result.image


def create_config(prompt: str, **kwargs) -> ImageGenerationConfig:
    """Create an ImageGenerationConfig with defaults.

    Args:
        prompt: Text prompt
        **kwargs: Override default values

    Returns:
        Configured ImageGenerationConfig
    """
    return ImageGenerationConfig(prompt=prompt, **kwargs)
