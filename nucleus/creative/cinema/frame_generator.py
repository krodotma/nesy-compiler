"""
Frame Generator Module
======================

Video frame generation based on storyboard panels.
Interfaces with image generation models to create frames.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
from dataclasses import dataclass, field
from enum import Enum, auto
from pathlib import Path
from typing import List, Optional, Dict, Any, Callable, Union
from datetime import datetime
import uuid

from .storyboard import StoryboardPanel, Storyboard, ShotType


class GenerationModel(Enum):
    """Available image generation models."""
    PLACEHOLDER = auto()       # Returns placeholder frames
    STABLE_DIFFUSION = auto()  # Stable Diffusion
    DALL_E = auto()            # DALL-E 3
    MIDJOURNEY = auto()        # Midjourney
    FLUX = auto()              # Flux models
    CUSTOM = auto()            # Custom model


class AspectRatio(Enum):
    """Standard video aspect ratios."""
    RATIO_16_9 = (1920, 1080)   # HD widescreen
    RATIO_4_3 = (1440, 1080)    # Standard
    RATIO_21_9 = (2560, 1080)   # Ultrawide
    RATIO_1_1 = (1080, 1080)    # Square
    RATIO_9_16 = (1080, 1920)   # Vertical
    RATIO_2_35 = (2560, 1090)   # Cinemascope


@dataclass
class FrameGenerationConfig:
    """
    Configuration for frame generation.

    Attributes:
        model: The generation model to use
        width: Output frame width in pixels
        height: Output frame height in pixels
        aspect_ratio: Aspect ratio preset (overrides width/height if set)
        fps: Target frames per second
        quality: Quality setting (0.0 to 1.0)
        style_prompt: Global style prompt for all frames
        negative_prompt: Negative prompt for all frames
        seed: Random seed for reproducibility
        batch_size: Number of frames to generate in parallel
        output_dir: Directory for output frames
        cache_enabled: Whether to cache generated frames
        cache_dir: Directory for cached frames
        metadata: Additional configuration metadata
    """
    model: GenerationModel = GenerationModel.PLACEHOLDER
    width: int = 1920
    height: int = 1080
    aspect_ratio: Optional[AspectRatio] = None
    fps: int = 24
    quality: float = 0.9
    style_prompt: str = ""
    negative_prompt: str = "blurry, low quality, distorted"
    seed: Optional[int] = None
    batch_size: int = 4
    output_dir: Path = field(default_factory=lambda: Path("./frames"))
    cache_enabled: bool = True
    cache_dir: Path = field(default_factory=lambda: Path("./.frame_cache"))
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        """Apply aspect ratio if set."""
        if self.aspect_ratio:
            self.width, self.height = self.aspect_ratio.value

    def to_dict(self) -> Dict[str, Any]:
        """Convert config to dictionary."""
        return {
            'model': self.model.name,
            'width': self.width,
            'height': self.height,
            'aspect_ratio': self.aspect_ratio.name if self.aspect_ratio else None,
            'fps': self.fps,
            'quality': self.quality,
            'style_prompt': self.style_prompt,
            'negative_prompt': self.negative_prompt,
            'seed': self.seed,
            'batch_size': self.batch_size,
            'output_dir': str(self.output_dir),
            'cache_enabled': self.cache_enabled,
            'cache_dir': str(self.cache_dir),
            'metadata': self.metadata,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'FrameGenerationConfig':
        """Create config from dictionary."""
        config = cls(
            model=GenerationModel[data.get('model', 'PLACEHOLDER')],
            width=data.get('width', 1920),
            height=data.get('height', 1080),
            fps=data.get('fps', 24),
            quality=data.get('quality', 0.9),
            style_prompt=data.get('style_prompt', ''),
            negative_prompt=data.get('negative_prompt', ''),
            seed=data.get('seed'),
            batch_size=data.get('batch_size', 4),
            output_dir=Path(data.get('output_dir', './frames')),
            cache_enabled=data.get('cache_enabled', True),
            cache_dir=Path(data.get('cache_dir', './.frame_cache')),
            metadata=data.get('metadata', {}),
        )
        if data.get('aspect_ratio'):
            config.aspect_ratio = AspectRatio[data['aspect_ratio']]
        return config


@dataclass
class GenerationResult:
    """
    Result of frame generation.

    Attributes:
        success: Whether generation was successful
        panel_id: The panel this frame was generated for
        frame_paths: Paths to generated frame files
        prompt_used: The actual prompt used for generation
        generation_time: Time taken to generate in seconds
        model_used: The model that was used
        error: Error message if generation failed
        metadata: Additional result metadata
    """
    success: bool
    panel_id: str
    frame_paths: List[Path] = field(default_factory=list)
    prompt_used: str = ""
    generation_time: float = 0.0
    model_used: GenerationModel = GenerationModel.PLACEHOLDER
    error: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert result to dictionary."""
        return {
            'success': self.success,
            'panel_id': self.panel_id,
            'frame_paths': [str(p) for p in self.frame_paths],
            'prompt_used': self.prompt_used,
            'generation_time': self.generation_time,
            'model_used': self.model_used.name,
            'error': self.error,
            'metadata': self.metadata,
        }


class FrameGenerator:
    """
    Generates video frames based on storyboard panels.

    Provides an abstraction layer over various image generation
    models to produce consistent video frames.
    """

    def __init__(self, config: Optional[FrameGenerationConfig] = None):
        """
        Initialize the frame generator.

        Args:
            config: Frame generation configuration
        """
        self.config = config or FrameGenerationConfig()
        self._ensure_directories()
        self._model_handlers: Dict[GenerationModel, Callable] = {
            GenerationModel.PLACEHOLDER: self._generate_placeholder,
            GenerationModel.STABLE_DIFFUSION: self._generate_stable_diffusion,
            GenerationModel.DALL_E: self._generate_dalle,
            GenerationModel.FLUX: self._generate_flux,
        }

    def _ensure_directories(self) -> None:
        """Ensure output and cache directories exist."""
        self.config.output_dir.mkdir(parents=True, exist_ok=True)
        if self.config.cache_enabled:
            self.config.cache_dir.mkdir(parents=True, exist_ok=True)

    def generate_frame(self, panel: StoryboardPanel) -> GenerationResult:
        """
        Generate a frame for a single storyboard panel.

        Args:
            panel: The storyboard panel to generate a frame for

        Returns:
            GenerationResult with paths to generated frames.
        """
        import time
        start_time = time.time()

        # Build the prompt
        prompt = self._build_prompt(panel)

        # Check cache first
        if self.config.cache_enabled:
            cached = self._check_cache(prompt)
            if cached:
                return GenerationResult(
                    success=True,
                    panel_id=panel.panel_id,
                    frame_paths=cached,
                    prompt_used=prompt,
                    generation_time=time.time() - start_time,
                    model_used=self.config.model,
                    metadata={'cached': True}
                )

        # Generate frames
        try:
            handler = self._model_handlers.get(self.config.model, self._generate_placeholder)
            frame_paths = handler(panel, prompt)

            # Cache the results
            if self.config.cache_enabled:
                self._cache_result(prompt, frame_paths)

            return GenerationResult(
                success=True,
                panel_id=panel.panel_id,
                frame_paths=frame_paths,
                prompt_used=prompt,
                generation_time=time.time() - start_time,
                model_used=self.config.model,
                metadata={'cached': False}
            )
        except Exception as e:
            return GenerationResult(
                success=False,
                panel_id=panel.panel_id,
                prompt_used=prompt,
                generation_time=time.time() - start_time,
                model_used=self.config.model,
                error=str(e)
            )

    async def generate_frame_async(self, panel: StoryboardPanel) -> GenerationResult:
        """
        Asynchronously generate a frame for a storyboard panel.

        Args:
            panel: The storyboard panel to generate a frame for

        Returns:
            GenerationResult with paths to generated frames.
        """
        # For now, run sync version in executor
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, self.generate_frame, panel)

    def generate_storyboard(self, storyboard: Storyboard,
                            progress_callback: Optional[Callable[[int, int], None]] = None) -> List[GenerationResult]:
        """
        Generate frames for an entire storyboard.

        Args:
            storyboard: The storyboard to generate frames for
            progress_callback: Optional callback(current, total) for progress

        Returns:
            List of GenerationResult for each panel.
        """
        results = []
        total = len(storyboard)

        for i, panel in enumerate(storyboard):
            result = self.generate_frame(panel)
            results.append(result)

            if progress_callback:
                progress_callback(i + 1, total)

        return results

    async def generate_storyboard_async(self, storyboard: Storyboard,
                                        progress_callback: Optional[Callable[[int, int], None]] = None) -> List[GenerationResult]:
        """
        Asynchronously generate frames for a storyboard with batching.

        Args:
            storyboard: The storyboard to generate frames for
            progress_callback: Optional callback for progress

        Returns:
            List of GenerationResult for each panel.
        """
        results = []
        panels = list(storyboard)
        total = len(panels)

        # Process in batches
        for batch_start in range(0, total, self.config.batch_size):
            batch_end = min(batch_start + self.config.batch_size, total)
            batch = panels[batch_start:batch_end]

            # Generate batch concurrently
            tasks = [self.generate_frame_async(panel) for panel in batch]
            batch_results = await asyncio.gather(*tasks)
            results.extend(batch_results)

            if progress_callback:
                progress_callback(batch_end, total)

        return results

    def _build_prompt(self, panel: StoryboardPanel) -> str:
        """Build a generation prompt from panel data."""
        parts = []

        # Add style prompt
        if self.config.style_prompt:
            parts.append(self.config.style_prompt)

        # Add shot type context
        shot_type_descriptions = {
            ShotType.EXTREME_WIDE: "extreme wide shot, establishing shot",
            ShotType.WIDE: "wide shot, full scene visible",
            ShotType.MEDIUM_WIDE: "medium wide shot, cowboy shot",
            ShotType.MEDIUM: "medium shot, waist up",
            ShotType.MEDIUM_CLOSE: "medium close-up, chest up",
            ShotType.CLOSE: "close-up shot, face and head",
            ShotType.EXTREME_CLOSE: "extreme close-up, detail shot",
            ShotType.OVER_SHOULDER: "over the shoulder shot",
            ShotType.TWO_SHOT: "two shot, two characters framed together",
            ShotType.GROUP: "group shot, multiple characters",
            ShotType.POV: "point of view shot",
            ShotType.INSERT: "insert shot, detail",
            ShotType.REACTION: "reaction shot",
            ShotType.CUTAWAY: "cutaway shot",
        }
        parts.append(shot_type_descriptions.get(panel.shot_type, "medium shot"))

        # Add description
        if panel.description:
            parts.append(panel.description)

        # Add action context
        if panel.action:
            parts.append(f"Action: {panel.action[:100]}")

        # Add characters
        if panel.characters:
            parts.append(f"Characters: {', '.join(panel.characters)}")

        # Combine and add quality markers
        prompt = ", ".join(parts)
        prompt += ", cinematic, film still, high quality, detailed"

        return prompt

    def _generate_placeholder(self, panel: StoryboardPanel, prompt: str) -> List[Path]:
        """Generate placeholder frames (solid color with text overlay)."""
        # Calculate frames needed for panel duration
        num_frames = max(1, int(panel.duration_seconds * self.config.fps))

        frame_paths = []
        for frame_idx in range(num_frames):
            # Create a simple placeholder file
            frame_name = f"scene{panel.scene_number:03d}_shot{panel.shot_number:03d}_frame{frame_idx:04d}.txt"
            frame_path = self.config.output_dir / frame_name

            # Write placeholder data
            placeholder_data = {
                'panel_id': panel.panel_id,
                'scene': panel.scene_number,
                'shot': panel.shot_number,
                'frame': frame_idx,
                'shot_type': panel.shot_type.name,
                'prompt': prompt,
                'width': self.config.width,
                'height': self.config.height,
                'timestamp': datetime.now().isoformat(),
            }
            frame_path.write_text(json.dumps(placeholder_data, indent=2))
            frame_paths.append(frame_path)

        return frame_paths

    def _generate_stable_diffusion(self, panel: StoryboardPanel, prompt: str) -> List[Path]:
        """Generate frames using Stable Diffusion."""
        # Stub implementation - would integrate with SD API/local model
        # For now, falls back to placeholder
        return self._generate_placeholder(panel, prompt)

    def _generate_dalle(self, panel: StoryboardPanel, prompt: str) -> List[Path]:
        """Generate frames using DALL-E."""
        # Stub implementation - would integrate with OpenAI API
        # For now, falls back to placeholder
        return self._generate_placeholder(panel, prompt)

    def _generate_flux(self, panel: StoryboardPanel, prompt: str) -> List[Path]:
        """Generate frames using Flux models."""
        # Stub implementation - would integrate with Flux API
        # For now, falls back to placeholder
        return self._generate_placeholder(panel, prompt)

    def _get_cache_key(self, prompt: str) -> str:
        """Generate a cache key from prompt and config."""
        config_str = f"{self.config.width}x{self.config.height}_{self.config.model.name}_{self.config.seed}"
        full_key = f"{prompt}_{config_str}"
        return hashlib.sha256(full_key.encode()).hexdigest()[:16]

    def _check_cache(self, prompt: str) -> Optional[List[Path]]:
        """Check if frames exist in cache."""
        cache_key = self._get_cache_key(prompt)
        cache_meta_path = self.config.cache_dir / f"{cache_key}.json"

        if cache_meta_path.exists():
            try:
                meta = json.loads(cache_meta_path.read_text())
                frame_paths = [Path(p) for p in meta.get('frame_paths', [])]
                if all(p.exists() for p in frame_paths):
                    return frame_paths
            except (json.JSONDecodeError, KeyError):
                pass

        return None

    def _cache_result(self, prompt: str, frame_paths: List[Path]) -> None:
        """Cache generated frame paths."""
        cache_key = self._get_cache_key(prompt)
        cache_meta_path = self.config.cache_dir / f"{cache_key}.json"

        meta = {
            'prompt': prompt,
            'frame_paths': [str(p) for p in frame_paths],
            'timestamp': datetime.now().isoformat(),
            'config': self.config.to_dict(),
        }
        cache_meta_path.write_text(json.dumps(meta, indent=2))

    def clear_cache(self) -> int:
        """Clear all cached frames. Returns number of items cleared."""
        if not self.config.cache_dir.exists():
            return 0

        count = 0
        for cache_file in self.config.cache_dir.glob("*.json"):
            cache_file.unlink()
            count += 1

        return count


__all__ = [
    'GenerationModel',
    'AspectRatio',
    'FrameGenerationConfig',
    'GenerationResult',
    'FrameGenerator',
]
