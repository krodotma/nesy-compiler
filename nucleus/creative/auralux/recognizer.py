"""
Auralux Speech Recognition Module
=================================

Provides speech-to-text recognition using multiple backends:
- WhisperRecognizer: OpenAI Whisper ASR (local or API)
- SileroVAD: Voice Activity Detection
- SpeechRecognizer: Unified interface with VAD and streaming support
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
    AsyncIterator,
    Callable,
    Dict,
    Iterator,
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
# WORD TIMING AND SEGMENTS
# =============================================================================


@dataclass
class WordTiming:
    """Word-level timing information from transcription.

    Attributes:
        word: The transcribed word
        start: Start time in seconds
        end: End time in seconds
        confidence: Confidence score (0-1)
        speaker: Speaker ID if diarization is enabled
    """
    word: str
    start: float
    end: float
    confidence: float = 1.0
    speaker: Optional[str] = None

    @property
    def duration(self) -> float:
        """Duration of the word in seconds."""
        return self.end - self.start

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "word": self.word,
            "start": self.start,
            "end": self.end,
            "confidence": self.confidence,
            "speaker": self.speaker,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "WordTiming":
        """Create from dictionary."""
        return cls(**data)


@dataclass
class Segment:
    """A segment of transcribed speech.

    Segments typically correspond to sentences or utterances.

    Attributes:
        text: The transcribed text
        start: Start time in seconds
        end: End time in seconds
        words: Word-level timing (if available)
        confidence: Average confidence score
        language: Detected language code
        speaker: Speaker ID if diarization is enabled
        is_final: Whether this segment is final (for streaming)
    """
    text: str
    start: float
    end: float
    words: List[WordTiming] = field(default_factory=list)
    confidence: float = 1.0
    language: Optional[str] = None
    speaker: Optional[str] = None
    is_final: bool = True
    segment_id: str = field(default_factory=lambda: str(uuid.uuid4())[:8])

    @property
    def duration(self) -> float:
        """Duration of the segment in seconds."""
        return self.end - self.start

    @property
    def word_count(self) -> int:
        """Number of words in the segment."""
        return len(self.words) if self.words else len(self.text.split())

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "segment_id": self.segment_id,
            "text": self.text,
            "start": self.start,
            "end": self.end,
            "words": [w.to_dict() for w in self.words],
            "confidence": self.confidence,
            "language": self.language,
            "speaker": self.speaker,
            "is_final": self.is_final,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Segment":
        """Create from dictionary."""
        data = data.copy()
        if "words" in data:
            data["words"] = [WordTiming.from_dict(w) for w in data["words"]]
        return cls(**data)


# =============================================================================
# TRANSCRIPTION RESULT
# =============================================================================


@dataclass
class TranscriptionResult:
    """Result of a transcription operation.

    Attributes:
        text: Full transcribed text
        segments: List of transcription segments
        language: Detected language code
        duration: Audio duration in seconds
        confidence: Overall confidence score
        processing_time_ms: Time taken to transcribe
        provider: Name of the recognition backend
        word_timestamps: Whether word timestamps are available
        metadata: Additional metadata
    """
    text: str
    segments: List[Segment]
    language: str
    duration: float
    confidence: float = 1.0
    processing_time_ms: float = 0.0
    provider: str = "unknown"
    word_timestamps: bool = False
    metadata: Dict[str, Any] = field(default_factory=dict)
    result_id: str = field(default_factory=lambda: str(uuid.uuid4())[:8])
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))

    @property
    def word_count(self) -> int:
        """Total word count."""
        return sum(s.word_count for s in self.segments)

    @property
    def words_per_minute(self) -> float:
        """Words per minute (speaking rate)."""
        if self.duration <= 0:
            return 0.0
        return (self.word_count / self.duration) * 60

    def get_text_at_time(self, time_s: float) -> Optional[str]:
        """Get the text being spoken at a given time."""
        for segment in self.segments:
            if segment.start <= time_s <= segment.end:
                if segment.words:
                    for word in segment.words:
                        if word.start <= time_s <= word.end:
                            return word.word
                return segment.text
        return None

    def get_segment_at_time(self, time_s: float) -> Optional[Segment]:
        """Get the segment containing the given time."""
        for segment in self.segments:
            if segment.start <= time_s <= segment.end:
                return segment
        return None

    def to_srt(self) -> str:
        """Export to SRT subtitle format."""
        lines = []
        for i, segment in enumerate(self.segments, 1):
            start = self._format_srt_time(segment.start)
            end = self._format_srt_time(segment.end)
            lines.append(f"{i}")
            lines.append(f"{start} --> {end}")
            lines.append(segment.text)
            lines.append("")
        return "\n".join(lines)

    def to_vtt(self) -> str:
        """Export to WebVTT subtitle format."""
        lines = ["WEBVTT", ""]
        for segment in self.segments:
            start = self._format_vtt_time(segment.start)
            end = self._format_vtt_time(segment.end)
            lines.append(f"{start} --> {end}")
            lines.append(segment.text)
            lines.append("")
        return "\n".join(lines)

    @staticmethod
    def _format_srt_time(seconds: float) -> str:
        """Format time for SRT (HH:MM:SS,mmm)."""
        hours = int(seconds // 3600)
        minutes = int((seconds % 3600) // 60)
        secs = int(seconds % 60)
        millis = int((seconds % 1) * 1000)
        return f"{hours:02d}:{minutes:02d}:{secs:02d},{millis:03d}"

    @staticmethod
    def _format_vtt_time(seconds: float) -> str:
        """Format time for VTT (HH:MM:SS.mmm)."""
        hours = int(seconds // 3600)
        minutes = int((seconds % 3600) // 60)
        secs = int(seconds % 60)
        millis = int((seconds % 1) * 1000)
        return f"{hours:02d}:{minutes:02d}:{secs:02d}.{millis:03d}"

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "result_id": self.result_id,
            "text": self.text,
            "segments": [s.to_dict() for s in self.segments],
            "language": self.language,
            "duration": self.duration,
            "confidence": self.confidence,
            "processing_time_ms": self.processing_time_ms,
            "provider": self.provider,
            "word_timestamps": self.word_timestamps,
            "word_count": self.word_count,
            "words_per_minute": self.words_per_minute,
            "timestamp": self.timestamp.isoformat(),
            "metadata": self.metadata,
        }


# =============================================================================
# RECOGNIZER PROTOCOL
# =============================================================================


@runtime_checkable
class RecognizerProtocol(Protocol):
    """Protocol for speech recognizer implementations."""

    async def transcribe(
        self,
        audio: Union[AudioArray, Path, str, bytes],
        language: Optional[str] = None,
        progress_callback: Optional[ProgressCallback] = None,
    ) -> TranscriptionResult:
        """Transcribe audio to text."""
        ...

    def is_available(self) -> bool:
        """Check if the recognizer is available."""
        ...


# =============================================================================
# BASE RECOGNIZER
# =============================================================================


class BaseRecognizer(ABC):
    """Abstract base class for speech recognizers."""

    def __init__(
        self,
        model_size: str = "base",
        language: Optional[str] = None,
        device: str = "cpu",
    ):
        """Initialize recognizer.

        Args:
            model_size: Model size (tiny, base, small, medium, large)
            language: Default language for transcription
            device: Device to run model on (cpu, cuda)
        """
        self.model_size = model_size
        self.default_language = language
        self.device = device

    @abstractmethod
    async def transcribe(
        self,
        audio: Union[AudioArray, Path, str, bytes],
        language: Optional[str] = None,
        word_timestamps: bool = False,
        progress_callback: Optional[ProgressCallback] = None,
    ) -> TranscriptionResult:
        """Transcribe audio to text."""
        ...

    @abstractmethod
    def is_available(self) -> bool:
        """Check if the recognizer backend is available."""
        ...

    def _load_audio(
        self,
        audio: Union[AudioArray, Path, str, bytes],
        target_sr: int = 16000,
    ) -> Tuple[AudioArray, int]:
        """Load audio from various sources.

        Args:
            audio: Audio data (array, path, or bytes)
            target_sr: Target sample rate

        Returns:
            Tuple of (audio array, sample rate)
        """
        try:
            import soundfile as sf
        except ImportError:
            raise ImportError("soundfile required for audio loading")

        if isinstance(audio, np.ndarray):
            return audio.astype(np.float32), target_sr

        if isinstance(audio, (str, Path)):
            audio_data, sr = sf.read(str(audio))
        elif isinstance(audio, bytes):
            audio_data, sr = sf.read(io.BytesIO(audio))
        else:
            raise TypeError(f"Unsupported audio type: {type(audio)}")

        # Convert to mono if needed
        if len(audio_data.shape) > 1:
            audio_data = np.mean(audio_data, axis=1)

        # Resample if needed
        if sr != target_sr:
            try:
                import librosa
                audio_data = librosa.resample(audio_data, orig_sr=sr, target_sr=target_sr)
                sr = target_sr
            except ImportError:
                pass  # Use original sample rate

        return audio_data.astype(np.float32), sr


# =============================================================================
# WHISPER RECOGNIZER
# =============================================================================


class WhisperModel(Enum):
    """Available Whisper model sizes."""
    TINY = "tiny"
    BASE = "base"
    SMALL = "small"
    MEDIUM = "medium"
    LARGE = "large"
    LARGE_V2 = "large-v2"
    LARGE_V3 = "large-v3"


class WhisperRecognizer(BaseRecognizer):
    """OpenAI Whisper speech recognizer.

    Supports both local Whisper models and OpenAI API.

    Example:
        >>> recognizer = WhisperRecognizer(model_size="base")
        >>> result = await recognizer.transcribe("audio.wav")
        >>> print(result.text)
    """

    def __init__(
        self,
        model_size: str = "base",
        language: Optional[str] = None,
        device: str = "cpu",
        use_api: bool = False,
        api_key: Optional[str] = None,
        compute_type: str = "float32",
    ):
        """Initialize Whisper recognizer.

        Args:
            model_size: Model size (tiny, base, small, medium, large)
            language: Default language
            device: Device (cpu, cuda)
            use_api: Whether to use OpenAI API instead of local model
            api_key: OpenAI API key (if using API)
            compute_type: Compute type for faster-whisper (float32, float16, int8)
        """
        super().__init__(model_size, language, device)
        self.use_api = use_api
        self.api_key = api_key or os.environ.get("OPENAI_API_KEY")
        self.compute_type = compute_type
        self._model: Optional[Any] = None
        self._whisper_available: Optional[bool] = None

    def is_available(self) -> bool:
        """Check if Whisper is available."""
        if self._whisper_available is not None:
            return self._whisper_available

        if self.use_api:
            self._whisper_available = self.api_key is not None
        else:
            try:
                # Try faster-whisper first
                import faster_whisper
                self._whisper_available = True
            except ImportError:
                try:
                    import whisper
                    self._whisper_available = True
                except ImportError:
                    self._whisper_available = False

        return self._whisper_available

    def _get_model(self) -> Any:
        """Lazy-load Whisper model."""
        if self._model is not None:
            return self._model

        if self.use_api:
            raise ValueError("API mode does not use local model")

        # Try faster-whisper first (more efficient)
        try:
            from faster_whisper import WhisperModel
            self._model = WhisperModel(
                self.model_size,
                device=self.device,
                compute_type=self.compute_type,
            )
            self._model_type = "faster-whisper"
            return self._model
        except ImportError:
            pass

        # Fall back to original whisper
        import whisper
        self._model = whisper.load_model(self.model_size, device=self.device)
        self._model_type = "whisper"
        return self._model

    async def transcribe(
        self,
        audio: Union[AudioArray, Path, str, bytes],
        language: Optional[str] = None,
        word_timestamps: bool = False,
        progress_callback: Optional[ProgressCallback] = None,
    ) -> TranscriptionResult:
        """Transcribe audio using Whisper.

        Args:
            audio: Audio data
            language: Language code (auto-detect if None)
            word_timestamps: Whether to include word-level timestamps
            progress_callback: Progress callback

        Returns:
            TranscriptionResult
        """
        if not self.is_available():
            raise ImportError("Whisper or faster-whisper package required")

        language = language or self.default_language
        start_time = time.time()

        if progress_callback:
            progress_callback("Loading audio", 0.0)

        if self.use_api:
            result = await self._transcribe_api(audio, language, word_timestamps, progress_callback)
        else:
            result = await self._transcribe_local(audio, language, word_timestamps, progress_callback)

        result.processing_time_ms = (time.time() - start_time) * 1000
        return result

    async def _transcribe_api(
        self,
        audio: Union[AudioArray, Path, str, bytes],
        language: Optional[str],
        word_timestamps: bool,
        progress_callback: Optional[ProgressCallback],
    ) -> TranscriptionResult:
        """Transcribe using OpenAI API."""
        try:
            import openai
        except ImportError:
            raise ImportError("openai package required for API mode")

        if progress_callback:
            progress_callback("Preparing audio", 0.2)

        # Prepare audio file
        if isinstance(audio, np.ndarray):
            try:
                import soundfile as sf
            except ImportError:
                raise ImportError("soundfile required for audio handling")

            buffer = io.BytesIO()
            sf.write(buffer, audio, 16000, format="WAV")
            buffer.seek(0)
            audio_file = buffer
        elif isinstance(audio, (str, Path)):
            audio_file = open(str(audio), "rb")
        elif isinstance(audio, bytes):
            audio_file = io.BytesIO(audio)
        else:
            raise TypeError(f"Unsupported audio type: {type(audio)}")

        if progress_callback:
            progress_callback("Sending to API", 0.4)

        client = openai.OpenAI(api_key=self.api_key)

        try:
            # Use verbose_json for timestamps
            response_format = "verbose_json" if word_timestamps else "json"

            transcript = client.audio.transcriptions.create(
                model="whisper-1",
                file=audio_file,
                language=language,
                response_format=response_format,
                timestamp_granularities=["word", "segment"] if word_timestamps else None,
            )
        finally:
            if hasattr(audio_file, 'close'):
                audio_file.close()

        if progress_callback:
            progress_callback("Processing response", 0.8)

        # Parse response
        if isinstance(transcript, str):
            text = transcript
            segments = [Segment(text=text, start=0.0, end=0.0)]
            detected_language = language or "en"
            duration = 0.0
        else:
            text = transcript.text
            detected_language = getattr(transcript, "language", language or "en")
            duration = getattr(transcript, "duration", 0.0)

            segments = []
            if hasattr(transcript, "segments"):
                for seg in transcript.segments:
                    words = []
                    if word_timestamps and hasattr(transcript, "words"):
                        for w in transcript.words:
                            if seg.start <= w.start <= seg.end:
                                words.append(WordTiming(
                                    word=w.word,
                                    start=w.start,
                                    end=w.end,
                                ))

                    segments.append(Segment(
                        text=seg.text.strip(),
                        start=seg.start,
                        end=seg.end,
                        words=words,
                    ))
            else:
                segments = [Segment(text=text, start=0.0, end=duration)]

        if progress_callback:
            progress_callback("Complete", 1.0)

        return TranscriptionResult(
            text=text,
            segments=segments,
            language=detected_language,
            duration=duration,
            provider="openai-whisper-api",
            word_timestamps=word_timestamps,
        )

    async def _transcribe_local(
        self,
        audio: Union[AudioArray, Path, str, bytes],
        language: Optional[str],
        word_timestamps: bool,
        progress_callback: Optional[ProgressCallback],
    ) -> TranscriptionResult:
        """Transcribe using local Whisper model."""
        loop = asyncio.get_event_loop()

        # Load audio
        audio_data, sr = self._load_audio(audio, target_sr=16000)
        duration = len(audio_data) / sr

        if progress_callback:
            progress_callback("Loading model", 0.2)

        def _transcribe():
            model = self._get_model()

            if self._model_type == "faster-whisper":
                return self._transcribe_faster_whisper(
                    model, audio_data, language, word_timestamps
                )
            else:
                return self._transcribe_whisper(
                    model, audio_data, language, word_timestamps
                )

        if progress_callback:
            progress_callback("Transcribing", 0.4)

        text, segments, detected_language = await loop.run_in_executor(None, _transcribe)

        if progress_callback:
            progress_callback("Complete", 1.0)

        return TranscriptionResult(
            text=text,
            segments=segments,
            language=detected_language,
            duration=duration,
            provider=f"whisper-{self.model_size}",
            word_timestamps=word_timestamps,
        )

    def _transcribe_faster_whisper(
        self,
        model: Any,
        audio: AudioArray,
        language: Optional[str],
        word_timestamps: bool,
    ) -> Tuple[str, List[Segment], str]:
        """Transcribe with faster-whisper."""
        segments_gen, info = model.transcribe(
            audio,
            language=language,
            word_timestamps=word_timestamps,
            vad_filter=True,
        )

        segments = []
        full_text = []

        for seg in segments_gen:
            words = []
            if word_timestamps and seg.words:
                for w in seg.words:
                    words.append(WordTiming(
                        word=w.word,
                        start=w.start,
                        end=w.end,
                        confidence=w.probability,
                    ))

            segments.append(Segment(
                text=seg.text.strip(),
                start=seg.start,
                end=seg.end,
                words=words,
                confidence=getattr(seg, 'avg_logprob', 1.0),
            ))
            full_text.append(seg.text.strip())

        return " ".join(full_text), segments, info.language

    def _transcribe_whisper(
        self,
        model: Any,
        audio: AudioArray,
        language: Optional[str],
        word_timestamps: bool,
    ) -> Tuple[str, List[Segment], str]:
        """Transcribe with original whisper."""
        result = model.transcribe(
            audio,
            language=language,
            word_timestamps=word_timestamps,
        )

        segments = []
        for seg in result.get("segments", []):
            words = []
            if word_timestamps and "words" in seg:
                for w in seg["words"]:
                    words.append(WordTiming(
                        word=w["word"],
                        start=w["start"],
                        end=w["end"],
                        confidence=w.get("probability", 1.0),
                    ))

            segments.append(Segment(
                text=seg["text"].strip(),
                start=seg["start"],
                end=seg["end"],
                words=words,
            ))

        return result["text"].strip(), segments, result.get("language", "en")


# =============================================================================
# SILERO VAD
# =============================================================================


@dataclass
class VADSegment:
    """Voice activity detection segment."""
    start: float  # seconds
    end: float  # seconds
    confidence: float = 1.0

    @property
    def duration(self) -> float:
        return self.end - self.start


class SileroVAD:
    """Silero Voice Activity Detection.

    Detects speech segments in audio for pre-processing before ASR.

    Example:
        >>> vad = SileroVAD()
        >>> segments = vad.detect(audio_array)
        >>> for seg in segments:
        ...     print(f"Speech from {seg.start:.2f}s to {seg.end:.2f}s")
    """

    def __init__(
        self,
        threshold: float = 0.5,
        min_speech_duration: float = 0.25,
        min_silence_duration: float = 0.1,
        window_size_samples: int = 512,
        sample_rate: int = 16000,
    ):
        """Initialize Silero VAD.

        Args:
            threshold: Speech probability threshold
            min_speech_duration: Minimum speech segment duration (seconds)
            min_silence_duration: Minimum silence duration to split (seconds)
            window_size_samples: Window size for VAD
            sample_rate: Expected sample rate (must be 8000 or 16000)
        """
        if sample_rate not in (8000, 16000):
            raise ValueError("Silero VAD requires 8000 or 16000 Hz sample rate")

        self.threshold = threshold
        self.min_speech_duration = min_speech_duration
        self.min_silence_duration = min_silence_duration
        self.window_size_samples = window_size_samples
        self.sample_rate = sample_rate
        self._model: Optional[Any] = None
        self._available: Optional[bool] = None

    def is_available(self) -> bool:
        """Check if Silero VAD is available."""
        if self._available is not None:
            return self._available

        try:
            import torch
            self._available = True
        except ImportError:
            self._available = False

        return self._available

    def _get_model(self) -> Any:
        """Load Silero VAD model."""
        if self._model is not None:
            return self._model

        import torch

        model, utils = torch.hub.load(
            repo_or_dir='snakers4/silero-vad',
            model='silero_vad',
            force_reload=False,
            onnx=False,
        )
        self._model = model
        self._get_speech_timestamps = utils[0]
        return self._model

    def detect(
        self,
        audio: Union[AudioArray, Path, str, bytes],
        return_seconds: bool = True,
    ) -> List[VADSegment]:
        """Detect voice activity in audio.

        Args:
            audio: Audio data
            return_seconds: If True, return times in seconds (else samples)

        Returns:
            List of VADSegment with detected speech regions
        """
        if not self.is_available():
            raise ImportError("torch required for Silero VAD")

        import torch

        # Load audio
        if isinstance(audio, np.ndarray):
            audio_data = audio.astype(np.float32)
        elif isinstance(audio, (str, Path)):
            try:
                import soundfile as sf
                audio_data, sr = sf.read(str(audio))
                if sr != self.sample_rate:
                    try:
                        import librosa
                        audio_data = librosa.resample(audio_data, orig_sr=sr, target_sr=self.sample_rate)
                    except ImportError:
                        raise ValueError(f"Audio sample rate {sr} != {self.sample_rate}")
            except ImportError:
                raise ImportError("soundfile required for audio loading")
            audio_data = audio_data.astype(np.float32)
        elif isinstance(audio, bytes):
            try:
                import soundfile as sf
                audio_data, sr = sf.read(io.BytesIO(audio))
            except ImportError:
                raise ImportError("soundfile required for audio loading")
            audio_data = audio_data.astype(np.float32)
        else:
            raise TypeError(f"Unsupported audio type: {type(audio)}")

        # Ensure mono
        if len(audio_data.shape) > 1:
            audio_data = np.mean(audio_data, axis=1)

        # Get model
        model = self._get_model()

        # Convert to torch tensor
        audio_tensor = torch.from_numpy(audio_data)

        # Get speech timestamps
        speech_timestamps = self._get_speech_timestamps(
            audio_tensor,
            model,
            threshold=self.threshold,
            sampling_rate=self.sample_rate,
            min_speech_duration_ms=int(self.min_speech_duration * 1000),
            min_silence_duration_ms=int(self.min_silence_duration * 1000),
            window_size_samples=self.window_size_samples,
            return_seconds=return_seconds,
        )

        segments = []
        for ts in speech_timestamps:
            segments.append(VADSegment(
                start=ts['start'],
                end=ts['end'],
            ))

        return segments

    def get_speech_audio(
        self,
        audio: AudioArray,
        segments: Optional[List[VADSegment]] = None,
        padding: float = 0.1,
    ) -> List[Tuple[AudioArray, float]]:
        """Extract speech segments from audio.

        Args:
            audio: Audio array
            segments: VAD segments (detect if not provided)
            padding: Padding around segments in seconds

        Returns:
            List of (audio_chunk, start_time) tuples
        """
        if segments is None:
            segments = self.detect(audio)

        chunks = []
        for seg in segments:
            start_sample = max(0, int((seg.start - padding) * self.sample_rate))
            end_sample = min(len(audio), int((seg.end + padding) * self.sample_rate))
            chunk = audio[start_sample:end_sample]
            chunks.append((chunk, seg.start - padding))

        return chunks


# =============================================================================
# UNIFIED SPEECH RECOGNIZER
# =============================================================================


class RecognizerBackend(Enum):
    """Available recognizer backends."""
    WHISPER_LOCAL = auto()
    WHISPER_API = auto()
    AUTO = auto()


class SpeechRecognizer:
    """Unified speech recognizer with VAD and streaming support.

    Provides a single interface for speech recognition with:
    - Multiple backend support (local Whisper, API)
    - Voice Activity Detection preprocessing
    - Streaming transcription
    - Batch processing

    Example:
        >>> recognizer = SpeechRecognizer()
        >>> result = await recognizer.transcribe("audio.wav")
        >>> print(result.text)

        # With VAD preprocessing
        >>> result = await recognizer.transcribe("audio.wav", use_vad=True)

        # Streaming
        >>> async for segment in recognizer.stream_transcribe(audio_chunks):
        ...     print(segment.text)
    """

    def __init__(
        self,
        backend: RecognizerBackend = RecognizerBackend.AUTO,
        model_size: str = "base",
        language: Optional[str] = None,
        device: str = "cpu",
        use_vad: bool = False,
        vad_threshold: float = 0.5,
    ):
        """Initialize unified recognizer.

        Args:
            backend: Recognition backend
            model_size: Whisper model size
            language: Default language
            device: Device (cpu, cuda)
            use_vad: Enable VAD preprocessing by default
            vad_threshold: VAD speech threshold
        """
        self.backend = backend
        self.model_size = model_size
        self.default_language = language
        self.device = device
        self.use_vad = use_vad

        self._whisper: Optional[WhisperRecognizer] = None
        self._vad: Optional[SileroVAD] = None
        self._vad_threshold = vad_threshold

    def _detect_backend(self) -> RecognizerBackend:
        """Detect best available backend."""
        # Check for local whisper
        try:
            import faster_whisper
            return RecognizerBackend.WHISPER_LOCAL
        except ImportError:
            pass

        try:
            import whisper
            return RecognizerBackend.WHISPER_LOCAL
        except ImportError:
            pass

        # Check for API
        if os.environ.get("OPENAI_API_KEY"):
            return RecognizerBackend.WHISPER_API

        return RecognizerBackend.WHISPER_LOCAL  # Will fail with helpful error

    def _get_whisper(self) -> WhisperRecognizer:
        """Get or create Whisper recognizer."""
        if self._whisper is not None:
            return self._whisper

        backend = self.backend
        if backend == RecognizerBackend.AUTO:
            backend = self._detect_backend()

        use_api = backend == RecognizerBackend.WHISPER_API

        self._whisper = WhisperRecognizer(
            model_size=self.model_size,
            language=self.default_language,
            device=self.device,
            use_api=use_api,
        )
        return self._whisper

    def _get_vad(self) -> SileroVAD:
        """Get or create VAD."""
        if self._vad is None:
            self._vad = SileroVAD(threshold=self._vad_threshold)
        return self._vad

    def is_available(self, backend: Optional[RecognizerBackend] = None) -> bool:
        """Check if backend is available."""
        backend = backend or self.backend
        if backend == RecognizerBackend.AUTO:
            backend = self._detect_backend()

        whisper = WhisperRecognizer(
            use_api=backend == RecognizerBackend.WHISPER_API
        )
        return whisper.is_available()

    async def transcribe(
        self,
        audio: Union[AudioArray, Path, str, bytes],
        language: Optional[str] = None,
        word_timestamps: bool = False,
        use_vad: Optional[bool] = None,
        progress_callback: Optional[ProgressCallback] = None,
    ) -> TranscriptionResult:
        """Transcribe audio to text.

        Args:
            audio: Audio data
            language: Language code (auto-detect if None)
            word_timestamps: Include word-level timestamps
            use_vad: Use VAD preprocessing (None uses default)
            progress_callback: Progress callback

        Returns:
            TranscriptionResult
        """
        use_vad = use_vad if use_vad is not None else self.use_vad

        if use_vad and self._get_vad().is_available():
            return await self._transcribe_with_vad(
                audio, language, word_timestamps, progress_callback
            )

        return await self._get_whisper().transcribe(
            audio, language, word_timestamps, progress_callback
        )

    async def _transcribe_with_vad(
        self,
        audio: Union[AudioArray, Path, str, bytes],
        language: Optional[str],
        word_timestamps: bool,
        progress_callback: Optional[ProgressCallback],
    ) -> TranscriptionResult:
        """Transcribe with VAD preprocessing."""
        vad = self._get_vad()
        whisper = self._get_whisper()

        if progress_callback:
            progress_callback("Running VAD", 0.1)

        # Load audio for VAD
        if isinstance(audio, np.ndarray):
            audio_data = audio
        else:
            audio_data, sr = whisper._load_audio(audio, target_sr=16000)

        # Detect speech segments
        vad_segments = vad.detect(audio_data)

        if not vad_segments:
            return TranscriptionResult(
                text="",
                segments=[],
                language=language or "en",
                duration=len(audio_data) / 16000,
                provider="whisper-vad",
            )

        if progress_callback:
            progress_callback("Transcribing speech segments", 0.3)

        # Transcribe each speech segment
        all_segments = []
        full_text = []

        for i, vad_seg in enumerate(vad_segments):
            start_sample = int(vad_seg.start * 16000)
            end_sample = int(vad_seg.end * 16000)
            chunk = audio_data[start_sample:end_sample]

            result = await whisper.transcribe(
                chunk, language, word_timestamps
            )

            # Adjust timestamps
            for seg in result.segments:
                seg.start += vad_seg.start
                seg.end += vad_seg.start
                for word in seg.words:
                    word.start += vad_seg.start
                    word.end += vad_seg.start
                all_segments.append(seg)

            full_text.append(result.text)

        if progress_callback:
            progress_callback("Complete", 1.0)

        return TranscriptionResult(
            text=" ".join(full_text),
            segments=all_segments,
            language=language or "en",
            duration=len(audio_data) / 16000,
            word_timestamps=word_timestamps,
            provider="whisper-vad",
            metadata={"vad_segments": len(vad_segments)},
        )

    def transcribe_sync(
        self,
        audio: Union[AudioArray, Path, str, bytes],
        language: Optional[str] = None,
        word_timestamps: bool = False,
        use_vad: Optional[bool] = None,
        progress_callback: Optional[ProgressCallback] = None,
    ) -> TranscriptionResult:
        """Synchronous wrapper for transcribe()."""
        try:
            loop = asyncio.get_event_loop()
        except RuntimeError:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)

        return loop.run_until_complete(
            self.transcribe(audio, language, word_timestamps, use_vad, progress_callback)
        )

    async def stream_transcribe(
        self,
        audio_stream: AsyncIterator[AudioArray],
        language: Optional[str] = None,
        chunk_duration: float = 2.0,
    ) -> AsyncIterator[Segment]:
        """Stream transcription of audio chunks.

        Args:
            audio_stream: Async iterator of audio chunks
            language: Language code
            chunk_duration: Minimum chunk duration before transcribing

        Yields:
            Segment objects as they are transcribed
        """
        whisper = self._get_whisper()
        buffer = np.array([], dtype=np.float32)
        sample_rate = 16000
        min_samples = int(chunk_duration * sample_rate)
        time_offset = 0.0

        async for chunk in audio_stream:
            buffer = np.concatenate([buffer, chunk.astype(np.float32)])

            if len(buffer) >= min_samples:
                # Transcribe buffer
                result = await whisper.transcribe(buffer, language)

                for seg in result.segments:
                    seg.start += time_offset
                    seg.end += time_offset
                    seg.is_final = False
                    yield seg

                time_offset += len(buffer) / sample_rate
                buffer = np.array([], dtype=np.float32)

        # Final chunk
        if len(buffer) > 0:
            result = await whisper.transcribe(buffer, language)
            for seg in result.segments:
                seg.start += time_offset
                seg.end += time_offset
                seg.is_final = True
                yield seg

    async def transcribe_batch(
        self,
        audio_files: List[Union[Path, str]],
        language: Optional[str] = None,
        max_concurrent: int = 2,
        progress_callback: Optional[ProgressCallback] = None,
    ) -> List[TranscriptionResult]:
        """Transcribe multiple audio files in batch.

        Args:
            audio_files: List of audio file paths
            language: Language code
            max_concurrent: Maximum concurrent transcriptions
            progress_callback: Progress callback

        Returns:
            List of TranscriptionResult
        """
        semaphore = asyncio.Semaphore(max_concurrent)

        async def _transcribe_one(i: int, path: Union[Path, str]) -> TranscriptionResult:
            async with semaphore:
                if progress_callback:
                    progress_callback(f"Transcribing {i+1}/{len(audio_files)}", i / len(audio_files))
                return await self.transcribe(path, language)

        results = await asyncio.gather(*[
            _transcribe_one(i, path) for i, path in enumerate(audio_files)
        ])

        if progress_callback:
            progress_callback("Batch complete", 1.0)

        return list(results)


# =============================================================================
# MODULE EXPORTS
# =============================================================================

__all__ = [
    # Data classes
    "WordTiming",
    "Segment",
    "TranscriptionResult",
    "VADSegment",
    # Enums
    "WhisperModel",
    "RecognizerBackend",
    # Recognizers
    "BaseRecognizer",
    "WhisperRecognizer",
    "SileroVAD",
    "SpeechRecognizer",
    # Protocol
    "RecognizerProtocol",
]
