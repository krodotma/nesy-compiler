"""
Auralux Voice Synthesizer Module
================================

Provides text-to-speech synthesis using multiple backends:
- EdgeTTSSynthesizer: Microsoft Edge TTS (online, high quality)
- CoquiTTSSynthesizer: Coqui TTS (local, open source)
- VoiceSynthesizer: Unified interface with fallback support
"""

from __future__ import annotations

import asyncio
import hashlib
import io
import os
import tempfile
import time
import uuid
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum, auto
from pathlib import Path
from typing import (
    Any,
    Callable,
    Dict,
    List,
    Optional,
    Protocol,
    Tuple,
    Union,
    runtime_checkable,
)

import numpy as np
import numpy.typing as npt

# Type aliases
AudioArray = npt.NDArray[np.float32]  # Samples x Channels float32
ProgressCallback = Callable[[str, float], None]


# =============================================================================
# VOICE CONFIGURATION
# =============================================================================


class VoiceGender(Enum):
    """Voice gender options."""
    MALE = "male"
    FEMALE = "female"
    NEUTRAL = "neutral"


class VoiceStyle(Enum):
    """Voice style presets."""
    NEUTRAL = "neutral"
    CHEERFUL = "cheerful"
    SAD = "sad"
    ANGRY = "angry"
    FEARFUL = "fearful"
    EXCITED = "excited"
    FRIENDLY = "friendly"
    TERRIFIED = "terrified"
    SHOUTING = "shouting"
    WHISPERING = "whispering"
    NARRATION = "narration"
    NEWSCAST = "newscast"


@dataclass
class VoiceConfig:
    """Configuration for voice synthesis.

    Attributes:
        voice_id: Identifier for the voice (backend-specific)
        language: Language code (e.g., 'en-US', 'fr-FR')
        gender: Voice gender preference
        style: Voice style/emotion
        rate: Speaking rate multiplier (0.5 = slow, 2.0 = fast)
        pitch: Pitch adjustment in Hz offset
        volume: Volume level (0.0 to 1.0)
        sample_rate: Output sample rate in Hz
        output_format: Audio format ('wav', 'mp3', 'ogg')
        cache_enabled: Whether to cache synthesis results
        ssml_enabled: Whether SSML input is supported
        custom_params: Backend-specific parameters
    """
    voice_id: str = "en-US-AriaNeural"
    language: str = "en-US"
    gender: VoiceGender = VoiceGender.FEMALE
    style: VoiceStyle = VoiceStyle.NEUTRAL
    rate: float = 1.0
    pitch: float = 0.0
    volume: float = 1.0
    sample_rate: int = 24000
    output_format: str = "wav"
    cache_enabled: bool = True
    ssml_enabled: bool = True
    custom_params: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        """Validate configuration values."""
        if not 0.25 <= self.rate <= 4.0:
            raise ValueError(f"Rate must be between 0.25 and 4.0, got {self.rate}")
        if not -100 <= self.pitch <= 100:
            raise ValueError(f"Pitch must be between -100 and 100, got {self.pitch}")
        if not 0.0 <= self.volume <= 1.0:
            raise ValueError(f"Volume must be between 0.0 and 1.0, got {self.volume}")
        if self.sample_rate not in (8000, 16000, 22050, 24000, 44100, 48000):
            raise ValueError(f"Invalid sample rate: {self.sample_rate}")

    def cache_key(self, text: str) -> str:
        """Generate a cache key for the given text and config."""
        config_str = (
            f"{self.voice_id}:{self.language}:{self.rate}:"
            f"{self.pitch}:{self.volume}:{self.sample_rate}"
        )
        combined = f"{config_str}:{text}"
        return hashlib.sha256(combined.encode()).hexdigest()[:16]

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            "voice_id": self.voice_id,
            "language": self.language,
            "gender": self.gender.value,
            "style": self.style.value,
            "rate": self.rate,
            "pitch": self.pitch,
            "volume": self.volume,
            "sample_rate": self.sample_rate,
            "output_format": self.output_format,
            "cache_enabled": self.cache_enabled,
            "ssml_enabled": self.ssml_enabled,
            "custom_params": self.custom_params,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "VoiceConfig":
        """Create config from dictionary."""
        data = data.copy()
        if "gender" in data:
            data["gender"] = VoiceGender(data["gender"])
        if "style" in data:
            data["style"] = VoiceStyle(data["style"])
        return cls(**data)


# =============================================================================
# SYNTHESIS RESULT
# =============================================================================


@dataclass
class SynthesisResult:
    """Result of a voice synthesis operation.

    Attributes:
        audio: Audio data as numpy array (float32)
        sample_rate: Sample rate of the audio
        duration: Duration in seconds
        text: Original text that was synthesized
        voice_id: Voice ID that was used
        cached: Whether result was from cache
        synthesis_time_ms: Time taken to synthesize (ms)
        provider: Name of the synthesis backend
        metadata: Additional metadata
    """
    audio: AudioArray
    sample_rate: int
    duration: float
    text: str
    voice_id: str
    cached: bool = False
    synthesis_time_ms: float = 0.0
    provider: str = "unknown"
    metadata: Dict[str, Any] = field(default_factory=dict)
    result_id: str = field(default_factory=lambda: str(uuid.uuid4())[:8])
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))

    def __post_init__(self) -> None:
        """Validate result."""
        if self.duration <= 0:
            # Calculate duration from audio if not provided
            if len(self.audio) > 0:
                self.duration = len(self.audio) / self.sample_rate

    def to_bytes(self, format: str = "wav") -> bytes:
        """Convert audio to bytes in specified format."""
        try:
            import soundfile as sf
        except ImportError:
            raise ImportError("soundfile required for audio export")

        buffer = io.BytesIO()
        sf.write(buffer, self.audio, self.sample_rate, format=format)
        buffer.seek(0)
        return buffer.read()

    def save(self, path: Union[str, Path]) -> Path:
        """Save audio to file."""
        try:
            import soundfile as sf
        except ImportError:
            raise ImportError("soundfile required for audio export")

        path = Path(path)
        sf.write(str(path), self.audio, self.sample_rate)
        return path

    def resample(self, target_rate: int) -> "SynthesisResult":
        """Resample audio to target sample rate."""
        try:
            import librosa
        except ImportError:
            raise ImportError("librosa required for resampling")

        if target_rate == self.sample_rate:
            return self

        resampled = librosa.resample(
            self.audio, orig_sr=self.sample_rate, target_sr=target_rate
        )

        return SynthesisResult(
            audio=resampled.astype(np.float32),
            sample_rate=target_rate,
            duration=self.duration,
            text=self.text,
            voice_id=self.voice_id,
            cached=self.cached,
            synthesis_time_ms=self.synthesis_time_ms,
            provider=self.provider,
            metadata={**self.metadata, "resampled_from": self.sample_rate},
        )

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary (without audio data)."""
        return {
            "result_id": self.result_id,
            "sample_rate": self.sample_rate,
            "duration": self.duration,
            "text": self.text[:100] + "..." if len(self.text) > 100 else self.text,
            "voice_id": self.voice_id,
            "cached": self.cached,
            "synthesis_time_ms": self.synthesis_time_ms,
            "provider": self.provider,
            "timestamp": self.timestamp.isoformat(),
            "metadata": self.metadata,
        }


# =============================================================================
# SYNTHESIZER PROTOCOL
# =============================================================================


@runtime_checkable
class SynthesizerProtocol(Protocol):
    """Protocol for TTS synthesizer implementations."""

    async def synthesize(
        self,
        text: str,
        config: Optional[VoiceConfig] = None,
        progress_callback: Optional[ProgressCallback] = None,
    ) -> SynthesisResult:
        """Synthesize speech from text."""
        ...

    def list_voices(self, language: Optional[str] = None) -> List[Dict[str, Any]]:
        """List available voices."""
        ...

    def is_available(self) -> bool:
        """Check if the synthesizer is available."""
        ...


# =============================================================================
# BASE SYNTHESIZER
# =============================================================================


class BaseSynthesizer(ABC):
    """Abstract base class for TTS synthesizers."""

    def __init__(
        self,
        cache_dir: Optional[Path] = None,
        default_config: Optional[VoiceConfig] = None,
    ):
        """Initialize synthesizer.

        Args:
            cache_dir: Directory for caching synthesis results
            default_config: Default voice configuration
        """
        self.cache_dir = cache_dir or Path(tempfile.gettempdir()) / "auralux_cache"
        self.default_config = default_config or VoiceConfig()
        self._cache: Dict[str, SynthesisResult] = {}

        # Ensure cache directory exists
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    @abstractmethod
    async def _synthesize_impl(
        self,
        text: str,
        config: VoiceConfig,
        progress_callback: Optional[ProgressCallback] = None,
    ) -> SynthesisResult:
        """Implementation-specific synthesis."""
        ...

    @abstractmethod
    def list_voices(self, language: Optional[str] = None) -> List[Dict[str, Any]]:
        """List available voices for the given language."""
        ...

    @abstractmethod
    def is_available(self) -> bool:
        """Check if the synthesizer backend is available."""
        ...

    async def synthesize(
        self,
        text: str,
        config: Optional[VoiceConfig] = None,
        progress_callback: Optional[ProgressCallback] = None,
    ) -> SynthesisResult:
        """Synthesize speech from text.

        Args:
            text: Text to synthesize
            config: Voice configuration (uses default if not provided)
            progress_callback: Optional callback for progress updates

        Returns:
            SynthesisResult containing the generated audio
        """
        config = config or self.default_config

        # Check cache if enabled
        if config.cache_enabled:
            cache_key = config.cache_key(text)
            if cache_key in self._cache:
                result = self._cache[cache_key]
                result.cached = True
                return result

        # Synthesize
        start_time = time.time()
        result = await self._synthesize_impl(text, config, progress_callback)
        result.synthesis_time_ms = (time.time() - start_time) * 1000

        # Cache result if enabled
        if config.cache_enabled:
            cache_key = config.cache_key(text)
            self._cache[cache_key] = result

        return result

    def clear_cache(self) -> int:
        """Clear the synthesis cache. Returns number of items cleared."""
        count = len(self._cache)
        self._cache.clear()
        return count


# =============================================================================
# EDGE TTS SYNTHESIZER
# =============================================================================


class EdgeTTSSynthesizer(BaseSynthesizer):
    """Microsoft Edge TTS synthesizer using edge-tts library.

    Provides high-quality neural TTS voices via Microsoft's Edge browser API.
    Requires internet connection.

    Example:
        >>> synth = EdgeTTSSynthesizer()
        >>> result = await synth.synthesize("Hello, world!")
        >>> result.save("output.wav")
    """

    # Common Edge TTS voices
    VOICES = {
        "en-US-AriaNeural": {"gender": "female", "language": "en-US"},
        "en-US-GuyNeural": {"gender": "male", "language": "en-US"},
        "en-US-JennyNeural": {"gender": "female", "language": "en-US"},
        "en-GB-SoniaNeural": {"gender": "female", "language": "en-GB"},
        "en-GB-RyanNeural": {"gender": "male", "language": "en-GB"},
        "fr-FR-DeniseNeural": {"gender": "female", "language": "fr-FR"},
        "fr-FR-HenriNeural": {"gender": "male", "language": "fr-FR"},
        "de-DE-KatjaNeural": {"gender": "female", "language": "de-DE"},
        "de-DE-ConradNeural": {"gender": "male", "language": "de-DE"},
        "es-ES-ElviraNeural": {"gender": "female", "language": "es-ES"},
        "es-ES-AlvaroNeural": {"gender": "male", "language": "es-ES"},
        "ja-JP-NanamiNeural": {"gender": "female", "language": "ja-JP"},
        "ja-JP-KeitaNeural": {"gender": "male", "language": "ja-JP"},
        "zh-CN-XiaoxiaoNeural": {"gender": "female", "language": "zh-CN"},
        "zh-CN-YunxiNeural": {"gender": "male", "language": "zh-CN"},
    }

    def __init__(
        self,
        cache_dir: Optional[Path] = None,
        default_config: Optional[VoiceConfig] = None,
    ):
        """Initialize Edge TTS synthesizer."""
        super().__init__(cache_dir, default_config)
        self._edge_tts_available: Optional[bool] = None

    def is_available(self) -> bool:
        """Check if edge-tts is available."""
        if self._edge_tts_available is None:
            try:
                import edge_tts
                self._edge_tts_available = True
            except ImportError:
                self._edge_tts_available = False
        return self._edge_tts_available

    def list_voices(self, language: Optional[str] = None) -> List[Dict[str, Any]]:
        """List available Edge TTS voices."""
        voices = []
        for voice_id, info in self.VOICES.items():
            if language is None or info["language"].startswith(language):
                voices.append({
                    "voice_id": voice_id,
                    "name": voice_id.replace("-", " ").replace("Neural", ""),
                    **info,
                })
        return voices

    async def _synthesize_impl(
        self,
        text: str,
        config: VoiceConfig,
        progress_callback: Optional[ProgressCallback] = None,
    ) -> SynthesisResult:
        """Synthesize using Edge TTS."""
        if not self.is_available():
            raise ImportError("edge-tts package is required for EdgeTTSSynthesizer")

        import edge_tts

        if progress_callback:
            progress_callback("Initializing Edge TTS", 0.0)

        # Build rate and pitch strings
        rate_str = f"{'+' if config.rate >= 1 else ''}{int((config.rate - 1) * 100)}%"
        pitch_str = f"{'+' if config.pitch >= 0 else ''}{int(config.pitch)}Hz"
        volume_str = f"{'+' if config.volume >= 0.5 else ''}{int((config.volume - 0.5) * 200)}%"

        # Create communicate instance
        communicate = edge_tts.Communicate(
            text,
            voice=config.voice_id,
            rate=rate_str,
            pitch=pitch_str,
            volume=volume_str,
        )

        if progress_callback:
            progress_callback("Synthesizing speech", 0.3)

        # Write to temporary file
        with tempfile.NamedTemporaryFile(suffix=".mp3", delete=False) as tmp:
            tmp_path = tmp.name

        try:
            await communicate.save(tmp_path)

            if progress_callback:
                progress_callback("Loading audio", 0.7)

            # Load audio
            try:
                import soundfile as sf
                audio, sr = sf.read(tmp_path)
            except Exception:
                # Fallback to pydub for MP3
                try:
                    from pydub import AudioSegment
                    segment = AudioSegment.from_mp3(tmp_path)
                    audio = np.array(segment.get_array_of_samples()).astype(np.float32)
                    audio = audio / 32768.0  # Normalize
                    sr = segment.frame_rate
                except ImportError:
                    raise ImportError("soundfile or pydub required for audio loading")

            # Ensure mono
            if len(audio.shape) > 1:
                audio = np.mean(audio, axis=1)

            audio = audio.astype(np.float32)
            duration = len(audio) / sr

            if progress_callback:
                progress_callback("Complete", 1.0)

            return SynthesisResult(
                audio=audio,
                sample_rate=sr,
                duration=duration,
                text=text,
                voice_id=config.voice_id,
                provider="edge-tts",
                metadata={
                    "rate": config.rate,
                    "pitch": config.pitch,
                    "volume": config.volume,
                },
            )
        finally:
            # Clean up temp file
            try:
                os.unlink(tmp_path)
            except Exception:
                pass


# =============================================================================
# COQUI TTS SYNTHESIZER
# =============================================================================


class CoquiTTSSynthesizer(BaseSynthesizer):
    """Coqui TTS synthesizer using TTS library.

    Provides local neural TTS with various models including XTTS.
    No internet required after model download.

    Example:
        >>> synth = CoquiTTSSynthesizer(model_name="tts_models/en/ljspeech/tacotron2-DDC")
        >>> result = await synth.synthesize("Hello, world!")
    """

    # Common Coqui TTS models
    MODELS = {
        "tacotron2-ddc": "tts_models/en/ljspeech/tacotron2-DDC",
        "glow-tts": "tts_models/en/ljspeech/glow-tts",
        "speedy-speech": "tts_models/en/ljspeech/speedy-speech",
        "vits": "tts_models/en/ljspeech/vits",
        "jenny": "tts_models/en/jenny/jenny",
        "xtts-v2": "tts_models/multilingual/multi-dataset/xtts_v2",
    }

    def __init__(
        self,
        model_name: str = "tts_models/en/ljspeech/tacotron2-DDC",
        cache_dir: Optional[Path] = None,
        default_config: Optional[VoiceConfig] = None,
        gpu: bool = False,
    ):
        """Initialize Coqui TTS synthesizer.

        Args:
            model_name: TTS model name or path
            cache_dir: Cache directory
            default_config: Default voice configuration
            gpu: Whether to use GPU acceleration
        """
        super().__init__(cache_dir, default_config)
        self.model_name = model_name
        self.gpu = gpu
        self._tts: Optional[Any] = None
        self._tts_available: Optional[bool] = None

    def is_available(self) -> bool:
        """Check if Coqui TTS is available."""
        if self._tts_available is None:
            try:
                from TTS.api import TTS
                self._tts_available = True
            except ImportError:
                self._tts_available = False
        return self._tts_available

    def _get_tts(self) -> Any:
        """Lazy-load TTS model."""
        if self._tts is None:
            from TTS.api import TTS
            self._tts = TTS(model_name=self.model_name, gpu=self.gpu)
        return self._tts

    def list_voices(self, language: Optional[str] = None) -> List[Dict[str, Any]]:
        """List available Coqui TTS models."""
        voices = []
        for name, model_path in self.MODELS.items():
            lang = model_path.split("/")[1]
            if language is None or lang.startswith(language):
                voices.append({
                    "voice_id": name,
                    "model_path": model_path,
                    "language": lang,
                    "local": True,
                })
        return voices

    async def _synthesize_impl(
        self,
        text: str,
        config: VoiceConfig,
        progress_callback: Optional[ProgressCallback] = None,
    ) -> SynthesisResult:
        """Synthesize using Coqui TTS."""
        if not self.is_available():
            raise ImportError("TTS package is required for CoquiTTSSynthesizer")

        if progress_callback:
            progress_callback("Loading TTS model", 0.0)

        # Run synthesis in thread pool to avoid blocking
        loop = asyncio.get_event_loop()

        def _synthesize():
            tts = self._get_tts()

            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
                tmp_path = tmp.name

            try:
                # Synthesize to file
                tts.tts_to_file(
                    text=text,
                    file_path=tmp_path,
                    speed=config.rate,
                )

                # Load audio
                import soundfile as sf
                audio, sr = sf.read(tmp_path)

                # Ensure mono
                if len(audio.shape) > 1:
                    audio = np.mean(audio, axis=1)

                return audio.astype(np.float32), sr
            finally:
                try:
                    os.unlink(tmp_path)
                except Exception:
                    pass

        if progress_callback:
            progress_callback("Synthesizing speech", 0.3)

        audio, sr = await loop.run_in_executor(None, _synthesize)
        duration = len(audio) / sr

        if progress_callback:
            progress_callback("Complete", 1.0)

        return SynthesisResult(
            audio=audio,
            sample_rate=sr,
            duration=duration,
            text=text,
            voice_id=self.model_name,
            provider="coqui-tts",
            metadata={
                "model": self.model_name,
                "gpu": self.gpu,
            },
        )


# =============================================================================
# UNIFIED VOICE SYNTHESIZER
# =============================================================================


class SynthesizerBackend(Enum):
    """Available synthesizer backends."""
    EDGE_TTS = auto()
    COQUI_TTS = auto()
    AUTO = auto()


class VoiceSynthesizer:
    """Unified voice synthesizer with multiple backend support.

    Provides a single interface for TTS with automatic fallback between
    backends when one is unavailable.

    Example:
        >>> synth = VoiceSynthesizer()
        >>> config = VoiceConfig(voice_id="en-US-AriaNeural", rate=1.2)
        >>> result = await synth.synthesize("Hello, world!", config)
        >>> result.save("output.wav")
    """

    def __init__(
        self,
        backend: SynthesizerBackend = SynthesizerBackend.AUTO,
        cache_dir: Optional[Path] = None,
        default_config: Optional[VoiceConfig] = None,
        fallback_enabled: bool = True,
    ):
        """Initialize unified synthesizer.

        Args:
            backend: Preferred backend (AUTO selects best available)
            cache_dir: Cache directory
            default_config: Default voice configuration
            fallback_enabled: Whether to fall back to other backends on failure
        """
        self.cache_dir = cache_dir or Path(tempfile.gettempdir()) / "auralux_cache"
        self.default_config = default_config or VoiceConfig()
        self.fallback_enabled = fallback_enabled

        self._edge_tts: Optional[EdgeTTSSynthesizer] = None
        self._coqui_tts: Optional[CoquiTTSSynthesizer] = None

        # Determine backend
        if backend == SynthesizerBackend.AUTO:
            self._backend = self._detect_best_backend()
        else:
            self._backend = backend

    def _detect_best_backend(self) -> SynthesizerBackend:
        """Detect the best available backend."""
        # Prefer Edge TTS (online, high quality)
        try:
            import edge_tts
            return SynthesizerBackend.EDGE_TTS
        except ImportError:
            pass

        # Fall back to Coqui TTS (local)
        try:
            from TTS.api import TTS
            return SynthesizerBackend.COQUI_TTS
        except ImportError:
            pass

        # Default to Edge TTS (will fail with helpful error)
        return SynthesizerBackend.EDGE_TTS

    def _get_edge_tts(self) -> EdgeTTSSynthesizer:
        """Get or create Edge TTS synthesizer."""
        if self._edge_tts is None:
            self._edge_tts = EdgeTTSSynthesizer(
                cache_dir=self.cache_dir,
                default_config=self.default_config,
            )
        return self._edge_tts

    def _get_coqui_tts(self) -> CoquiTTSSynthesizer:
        """Get or create Coqui TTS synthesizer."""
        if self._coqui_tts is None:
            self._coqui_tts = CoquiTTSSynthesizer(
                cache_dir=self.cache_dir,
                default_config=self.default_config,
            )
        return self._coqui_tts

    def _get_synthesizer(self, backend: SynthesizerBackend) -> BaseSynthesizer:
        """Get synthesizer for the given backend."""
        if backend == SynthesizerBackend.EDGE_TTS:
            return self._get_edge_tts()
        elif backend == SynthesizerBackend.COQUI_TTS:
            return self._get_coqui_tts()
        else:
            raise ValueError(f"Unknown backend: {backend}")

    def is_available(self, backend: Optional[SynthesizerBackend] = None) -> bool:
        """Check if a backend is available."""
        backend = backend or self._backend
        synth = self._get_synthesizer(backend)
        return synth.is_available()

    def list_voices(self, language: Optional[str] = None) -> List[Dict[str, Any]]:
        """List all available voices across backends."""
        voices = []

        # Edge TTS voices
        if self._get_edge_tts().is_available():
            for voice in self._get_edge_tts().list_voices(language):
                voice["backend"] = "edge-tts"
                voices.append(voice)

        # Coqui TTS voices
        if self._get_coqui_tts().is_available():
            for voice in self._get_coqui_tts().list_voices(language):
                voice["backend"] = "coqui-tts"
                voices.append(voice)

        return voices

    async def synthesize(
        self,
        text: str,
        config: Optional[VoiceConfig] = None,
        backend: Optional[SynthesizerBackend] = None,
        progress_callback: Optional[ProgressCallback] = None,
    ) -> SynthesisResult:
        """Synthesize speech from text.

        Args:
            text: Text to synthesize
            config: Voice configuration
            backend: Specific backend to use (overrides default)
            progress_callback: Progress callback

        Returns:
            SynthesisResult with generated audio

        Raises:
            RuntimeError: If synthesis fails on all backends
        """
        config = config or self.default_config
        backend = backend or self._backend

        errors = []
        backends_to_try = [backend]

        if self.fallback_enabled:
            # Add fallback backends
            for b in SynthesizerBackend:
                if b not in backends_to_try and b != SynthesizerBackend.AUTO:
                    backends_to_try.append(b)

        for b in backends_to_try:
            try:
                synth = self._get_synthesizer(b)
                if not synth.is_available():
                    continue

                result = await synth.synthesize(text, config, progress_callback)
                return result

            except Exception as e:
                errors.append((b.name, str(e)))
                if not self.fallback_enabled:
                    raise

        error_msg = "; ".join(f"{b}: {e}" for b, e in errors)
        raise RuntimeError(f"All synthesis backends failed: {error_msg}")

    def synthesize_sync(
        self,
        text: str,
        config: Optional[VoiceConfig] = None,
        backend: Optional[SynthesizerBackend] = None,
        progress_callback: Optional[ProgressCallback] = None,
    ) -> SynthesisResult:
        """Synchronous wrapper for synthesize()."""
        try:
            loop = asyncio.get_event_loop()
        except RuntimeError:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)

        return loop.run_until_complete(
            self.synthesize(text, config, backend, progress_callback)
        )

    async def synthesize_ssml(
        self,
        ssml: str,
        config: Optional[VoiceConfig] = None,
    ) -> SynthesisResult:
        """Synthesize from SSML markup.

        Args:
            ssml: SSML markup string
            config: Voice configuration

        Returns:
            SynthesisResult with generated audio
        """
        config = config or self.default_config
        if not config.ssml_enabled:
            raise ValueError("SSML is not enabled in config")

        # Edge TTS supports SSML natively
        return await self.synthesize(ssml, config, SynthesizerBackend.EDGE_TTS)

    async def synthesize_batch(
        self,
        texts: List[str],
        config: Optional[VoiceConfig] = None,
        max_concurrent: int = 3,
        progress_callback: Optional[ProgressCallback] = None,
    ) -> List[SynthesisResult]:
        """Synthesize multiple texts in batch.

        Args:
            texts: List of texts to synthesize
            config: Voice configuration
            max_concurrent: Maximum concurrent synthesis operations
            progress_callback: Progress callback

        Returns:
            List of SynthesisResult objects
        """
        semaphore = asyncio.Semaphore(max_concurrent)

        async def _synth_one(i: int, text: str) -> SynthesisResult:
            async with semaphore:
                if progress_callback:
                    progress_callback(f"Synthesizing {i+1}/{len(texts)}", i / len(texts))
                return await self.synthesize(text, config)

        results = await asyncio.gather(*[
            _synth_one(i, text) for i, text in enumerate(texts)
        ])

        if progress_callback:
            progress_callback("Batch complete", 1.0)

        return list(results)

    def clear_cache(self) -> int:
        """Clear all synthesis caches."""
        count = 0
        if self._edge_tts:
            count += self._edge_tts.clear_cache()
        if self._coqui_tts:
            count += self._coqui_tts.clear_cache()
        return count


# =============================================================================
# MODULE EXPORTS
# =============================================================================

__all__ = [
    # Enums
    "VoiceGender",
    "VoiceStyle",
    "SynthesizerBackend",
    # Config and Result
    "VoiceConfig",
    "SynthesisResult",
    # Synthesizers
    "BaseSynthesizer",
    "EdgeTTSSynthesizer",
    "CoquiTTSSynthesizer",
    "VoiceSynthesizer",
    # Protocol
    "SynthesizerProtocol",
]
