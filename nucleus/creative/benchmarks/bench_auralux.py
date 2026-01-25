"""
Auralux Subsystem Benchmarks
============================

Benchmarks for TTS (text-to-speech) and STT (speech-to-text) operations.
"""

from __future__ import annotations

import random
import math
from dataclasses import dataclass, field
from typing import Callable, Any, Optional

from .bench_runner import BenchmarkSuite


@dataclass
class AudioBuffer:
    """Mock audio buffer for benchmarking."""
    samples: list[float]
    sample_rate: int = 22050
    channels: int = 1

    @property
    def duration_seconds(self) -> float:
        """Get duration in seconds."""
        return len(self.samples) / self.sample_rate / self.channels

    @classmethod
    def silence(cls, duration_sec: float, sample_rate: int = 22050) -> "AudioBuffer":
        """Create silent audio buffer."""
        n_samples = int(duration_sec * sample_rate)
        return cls(samples=[0.0] * n_samples, sample_rate=sample_rate)

    @classmethod
    def noise(cls, duration_sec: float, sample_rate: int = 22050) -> "AudioBuffer":
        """Create white noise audio buffer."""
        n_samples = int(duration_sec * sample_rate)
        samples = [random.gauss(0, 0.1) for _ in range(n_samples)]
        return cls(samples=samples, sample_rate=sample_rate)

    @classmethod
    def sine_wave(
        cls,
        frequency: float,
        duration_sec: float,
        sample_rate: int = 22050,
        amplitude: float = 0.5,
    ) -> "AudioBuffer":
        """Create sine wave audio."""
        n_samples = int(duration_sec * sample_rate)
        samples = [
            amplitude * math.sin(2 * math.pi * frequency * i / sample_rate)
            for i in range(n_samples)
        ]
        return cls(samples=samples, sample_rate=sample_rate)

    def resample(self, target_rate: int) -> "AudioBuffer":
        """Resample to target sample rate."""
        if target_rate == self.sample_rate:
            return AudioBuffer(self.samples.copy(), self.sample_rate, self.channels)

        ratio = target_rate / self.sample_rate
        new_length = int(len(self.samples) * ratio)

        # Linear interpolation
        new_samples = []
        for i in range(new_length):
            src_idx = i / ratio
            idx0 = int(src_idx)
            idx1 = min(idx0 + 1, len(self.samples) - 1)
            frac = src_idx - idx0

            sample = self.samples[idx0] * (1 - frac) + self.samples[idx1] * frac
            new_samples.append(sample)

        return AudioBuffer(new_samples, target_rate, self.channels)

    def normalize(self) -> "AudioBuffer":
        """Normalize audio to [-1, 1] range."""
        if not self.samples:
            return AudioBuffer([], self.sample_rate, self.channels)

        max_val = max(abs(s) for s in self.samples)
        if max_val == 0:
            return AudioBuffer(self.samples.copy(), self.sample_rate, self.channels)

        normalized = [s / max_val for s in self.samples]
        return AudioBuffer(normalized, self.sample_rate, self.channels)

    def apply_gain(self, gain_db: float) -> "AudioBuffer":
        """Apply gain in dB."""
        multiplier = 10 ** (gain_db / 20)
        new_samples = [s * multiplier for s in self.samples]
        return AudioBuffer(new_samples, self.sample_rate, self.channels)


@dataclass
class VoiceConfig:
    """Voice synthesis configuration."""
    voice_id: str = "default"
    language: str = "en"
    speaking_rate: float = 1.0
    pitch: float = 0.0
    volume_gain_db: float = 0.0


@dataclass
class SynthesisResult:
    """Result of TTS synthesis."""
    audio: AudioBuffer
    duration_seconds: float
    phoneme_timings: list[tuple[str, float, float]] = field(default_factory=list)
    word_timings: list[tuple[str, float, float]] = field(default_factory=list)


@dataclass
class WordTiming:
    """Word timing information."""
    word: str
    start_time: float
    end_time: float
    confidence: float = 1.0


@dataclass
class TranscriptionResult:
    """Result of STT transcription."""
    text: str
    confidence: float
    language: str = "en"
    word_timings: list[WordTiming] = field(default_factory=list)


@dataclass
class SpeakerEmbedding:
    """Speaker embedding vector."""
    embedding: list[float]
    speaker_id: Optional[str] = None

    @classmethod
    def random(cls, dim: int = 256) -> "SpeakerEmbedding":
        """Create random embedding."""
        embedding = [random.gauss(0, 1) for _ in range(dim)]
        # Normalize to unit vector
        norm = math.sqrt(sum(e ** 2 for e in embedding))
        embedding = [e / norm for e in embedding]
        return cls(embedding=embedding)

    def cosine_similarity(self, other: "SpeakerEmbedding") -> float:
        """Compute cosine similarity with another embedding."""
        dot = sum(a * b for a, b in zip(self.embedding, other.embedding))
        return dot  # Already normalized


class MockTTSSynthesizer:
    """Mock TTS synthesizer for benchmarking."""

    def __init__(self, config: VoiceConfig | None = None):
        self.config = config or VoiceConfig()
        self._phoneme_map = {
            "a": (0.08, 440), "e": (0.07, 480), "i": (0.06, 520),
            "o": (0.08, 400), "u": (0.07, 380), " ": (0.05, 0),
            "b": (0.04, 200), "c": (0.03, 2000), "d": (0.04, 300),
            "f": (0.06, 3000), "g": (0.04, 250), "h": (0.03, 3500),
        }

    def synthesize(self, text: str) -> SynthesisResult:
        """Synthesize speech from text."""
        samples: list[float] = []
        phoneme_timings: list[tuple[str, float, float]] = []
        word_timings: list[tuple[str, float, float]] = []

        current_time = 0.0
        current_word_start = 0.0
        current_word = ""

        for char in text.lower():
            duration, freq = self._phoneme_map.get(char, (0.05, 500))
            duration /= self.config.speaking_rate

            # Generate audio for this phoneme
            n_samples = int(duration * 22050)

            if freq > 0:
                # Generate voiced sound
                for i in range(n_samples):
                    t = i / 22050
                    sample = 0.3 * math.sin(2 * math.pi * freq * t)
                    sample += 0.1 * math.sin(2 * math.pi * freq * 2 * t)
                    sample += 0.05 * random.gauss(0, 0.1)
                    samples.append(sample)
            else:
                # Silence
                samples.extend([0.0] * n_samples)

            # Record phoneme timing
            phoneme_timings.append((char, current_time, current_time + duration))

            # Track words
            if char == " ":
                if current_word:
                    word_timings.append((current_word, current_word_start, current_time))
                    current_word = ""
                current_word_start = current_time + duration
            else:
                current_word += char

            current_time += duration

        # Final word
        if current_word:
            word_timings.append((current_word, current_word_start, current_time))

        audio = AudioBuffer(samples, sample_rate=22050)

        if self.config.volume_gain_db != 0:
            audio = audio.apply_gain(self.config.volume_gain_db)

        return SynthesisResult(
            audio=audio,
            duration_seconds=current_time,
            phoneme_timings=phoneme_timings,
            word_timings=word_timings,
        )

    def synthesize_phonemes(self, phonemes: list[str]) -> SynthesisResult:
        """Synthesize from phoneme sequence."""
        text = "".join(p[0] if p else "" for p in phonemes)
        return self.synthesize(text)


class MockSTTRecognizer:
    """Mock STT recognizer for benchmarking."""

    def __init__(self, model: str = "whisper-base"):
        self.model = model
        self._model_params = {
            "whisper-tiny": {"chunk_size": 512, "vocab_size": 1000},
            "whisper-base": {"chunk_size": 1024, "vocab_size": 5000},
            "whisper-small": {"chunk_size": 2048, "vocab_size": 10000},
            "whisper-medium": {"chunk_size": 4096, "vocab_size": 50000},
        }

    def transcribe(self, audio: AudioBuffer) -> TranscriptionResult:
        """Transcribe audio to text."""
        params = self._model_params.get(self.model, self._model_params["whisper-base"])

        # Simulate processing
        chunks = len(audio.samples) // params["chunk_size"]

        # Generate mock transcription
        words = ["the", "quick", "brown", "fox", "jumps", "over", "lazy", "dog"]
        n_words = max(1, int(audio.duration_seconds * 2))  # ~2 words per second

        selected_words = [random.choice(words) for _ in range(n_words)]
        text = " ".join(selected_words)

        # Generate word timings
        word_timings = []
        current_time = 0.0
        word_duration = audio.duration_seconds / n_words

        for word in selected_words:
            word_timings.append(WordTiming(
                word=word,
                start_time=current_time,
                end_time=current_time + word_duration * 0.9,
                confidence=random.uniform(0.85, 0.99),
            ))
            current_time += word_duration

        return TranscriptionResult(
            text=text,
            confidence=random.uniform(0.9, 0.99),
            word_timings=word_timings,
        )

    def detect_voice_activity(self, audio: AudioBuffer) -> list[tuple[float, float]]:
        """Detect voice activity segments."""
        segments = []
        chunk_size = int(0.1 * audio.sample_rate)  # 100ms chunks

        is_speech = False
        segment_start = 0.0

        for i in range(0, len(audio.samples), chunk_size):
            chunk = audio.samples[i:i + chunk_size]
            energy = sum(s ** 2 for s in chunk) / len(chunk) if chunk else 0

            current_time = i / audio.sample_rate

            if energy > 0.001 and not is_speech:
                is_speech = True
                segment_start = current_time
            elif energy <= 0.001 and is_speech:
                is_speech = False
                segments.append((segment_start, current_time))

        if is_speech:
            segments.append((segment_start, audio.duration_seconds))

        return segments


class MockSpeakerEncoder:
    """Mock speaker encoder for benchmarking."""

    def __init__(self, embedding_dim: int = 256):
        self.embedding_dim = embedding_dim

    def encode(self, audio: AudioBuffer) -> SpeakerEmbedding:
        """Extract speaker embedding from audio."""
        # Simulate feature extraction
        features = []
        chunk_size = int(0.025 * audio.sample_rate)  # 25ms frames

        for i in range(0, min(len(audio.samples), chunk_size * 100), chunk_size):
            chunk = audio.samples[i:i + chunk_size]
            if chunk:
                # Mock MFCC-like features
                features.append(sum(chunk) / len(chunk))
                features.append(max(chunk) - min(chunk))

        # Generate embedding from features
        random.seed(int(sum(features[:10]) * 1000) if features else 0)
        return SpeakerEmbedding.random(self.embedding_dim)

    def compare(self, embedding1: SpeakerEmbedding, embedding2: SpeakerEmbedding) -> float:
        """Compare two speaker embeddings."""
        return embedding1.cosine_similarity(embedding2)


class AuraluxBenchmark(BenchmarkSuite):
    """Benchmark suite for auralux subsystem."""

    @property
    def name(self) -> str:
        return "auralux"

    @property
    def description(self) -> str:
        return "TTS/STT synthesis and recognition benchmarks"

    def __init__(self):
        self._tts: Optional[MockTTSSynthesizer] = None
        self._stt: Optional[MockSTTRecognizer] = None
        self._encoder: Optional[MockSpeakerEncoder] = None
        self._test_audio: dict[str, AudioBuffer] = {}
        self._test_embeddings: list[SpeakerEmbedding] = []

    def setup(self) -> None:
        """Setup test data and mock objects."""
        self._tts = MockTTSSynthesizer()
        self._stt = MockSTTRecognizer()
        self._encoder = MockSpeakerEncoder()

        # Pre-generate test audio
        self._test_audio = {
            "short": AudioBuffer.noise(1.0),
            "medium": AudioBuffer.noise(5.0),
            "long": AudioBuffer.noise(30.0),
            "sine": AudioBuffer.sine_wave(440, 2.0),
            "speech": self._tts.synthesize("Hello world, this is a test.").audio,
        }

        # Pre-generate embeddings
        self._test_embeddings = [SpeakerEmbedding.random() for _ in range(10)]

    def get_benchmarks(self) -> list[tuple[str, Callable[[], Any]]]:
        """Get all auralux benchmarks."""
        return [
            # Audio buffer operations
            ("audio_create_silence", self._audio_create_silence),
            ("audio_create_noise", self._audio_create_noise),
            ("audio_create_sine", self._audio_create_sine),
            ("audio_resample_up", self._audio_resample_up),
            ("audio_resample_down", self._audio_resample_down),
            ("audio_normalize", self._audio_normalize),
            ("audio_apply_gain", self._audio_apply_gain),

            # TTS benchmarks
            ("tts_short_text", self._tts_short_text),
            ("tts_medium_text", self._tts_medium_text),
            ("tts_long_text", self._tts_long_text),
            ("tts_with_timings", self._tts_with_timings),
            ("tts_phonemes", self._tts_phonemes),

            # STT benchmarks
            ("stt_short_audio", self._stt_short_audio),
            ("stt_medium_audio", self._stt_medium_audio),
            ("stt_long_audio", self._stt_long_audio),
            ("stt_vad_detect", self._stt_vad_detect),

            # Speaker embedding benchmarks
            ("speaker_encode", self._speaker_encode),
            ("speaker_compare", self._speaker_compare),
            ("speaker_batch_compare", self._speaker_batch_compare),
        ]

    def _audio_create_silence(self) -> AudioBuffer:
        """Create 5 second silence."""
        return AudioBuffer.silence(5.0)

    def _audio_create_noise(self) -> AudioBuffer:
        """Create 5 second noise."""
        return AudioBuffer.noise(5.0)

    def _audio_create_sine(self) -> AudioBuffer:
        """Create 5 second sine wave."""
        return AudioBuffer.sine_wave(440, 5.0)

    def _audio_resample_up(self) -> AudioBuffer:
        """Resample from 22050 to 44100 Hz."""
        return self._test_audio["medium"].resample(44100)

    def _audio_resample_down(self) -> AudioBuffer:
        """Resample from 22050 to 16000 Hz."""
        return self._test_audio["medium"].resample(16000)

    def _audio_normalize(self) -> AudioBuffer:
        """Normalize audio."""
        return self._test_audio["medium"].normalize()

    def _audio_apply_gain(self) -> AudioBuffer:
        """Apply gain to audio."""
        return self._test_audio["medium"].apply_gain(-6.0)

    def _tts_short_text(self) -> SynthesisResult:
        """Synthesize short text (~5 words)."""
        return self._tts.synthesize("Hello world.")

    def _tts_medium_text(self) -> SynthesisResult:
        """Synthesize medium text (~20 words)."""
        return self._tts.synthesize(
            "The quick brown fox jumps over the lazy dog. "
            "This is a longer sentence for testing."
        )

    def _tts_long_text(self) -> SynthesisResult:
        """Synthesize long text (~100 words)."""
        text = " ".join([
            "The quick brown fox jumps over the lazy dog."
        ] * 10)
        return self._tts.synthesize(text)

    def _tts_with_timings(self) -> SynthesisResult:
        """Synthesize with word timings."""
        result = self._tts.synthesize("Testing word timing extraction.")
        _ = result.word_timings  # Access timings
        return result

    def _tts_phonemes(self) -> SynthesisResult:
        """Synthesize from phonemes."""
        phonemes = ["h", "eh", "l", "ow", " ", "w", "er", "l", "d"]
        return self._tts.synthesize_phonemes(phonemes)

    def _stt_short_audio(self) -> TranscriptionResult:
        """Transcribe short audio (~1 second)."""
        return self._stt.transcribe(self._test_audio["short"])

    def _stt_medium_audio(self) -> TranscriptionResult:
        """Transcribe medium audio (~5 seconds)."""
        return self._stt.transcribe(self._test_audio["medium"])

    def _stt_long_audio(self) -> TranscriptionResult:
        """Transcribe long audio (~30 seconds)."""
        return self._stt.transcribe(self._test_audio["long"])

    def _stt_vad_detect(self) -> list[tuple[float, float]]:
        """Detect voice activity."""
        return self._stt.detect_voice_activity(self._test_audio["speech"])

    def _speaker_encode(self) -> SpeakerEmbedding:
        """Extract speaker embedding."""
        return self._encoder.encode(self._test_audio["speech"])

    def _speaker_compare(self) -> float:
        """Compare two speaker embeddings."""
        return self._encoder.compare(
            self._test_embeddings[0],
            self._test_embeddings[1],
        )

    def _speaker_batch_compare(self) -> list[float]:
        """Compare embedding against batch."""
        return [
            self._encoder.compare(self._test_embeddings[0], emb)
            for emb in self._test_embeddings[1:]
        ]
