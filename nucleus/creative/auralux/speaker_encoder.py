"""
Auralux Speaker Encoder Module
==============================

Provides speaker identification and embedding extraction:
- HuBERTEncoder: HuBERT-based speaker embeddings
- ECAPATDNNEncoder: ECAPA-TDNN speaker embeddings
- SpeakerEncoder: Unified interface with multiple backends
- SpeakerRegistry: Registry for managing known speakers
"""

from __future__ import annotations

import asyncio
import hashlib
import io
import json
import os
import pickle
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
AudioArray = npt.NDArray[np.float32]
EmbeddingArray = npt.NDArray[np.float32]
ProgressCallback = Callable[[str, float], None]


# =============================================================================
# SPEAKER EMBEDDING
# =============================================================================


@dataclass
class SpeakerEmbedding:
    """Speaker embedding vector with metadata.

    Represents a speaker's voice as a dense vector for comparison
    and identification purposes.

    Attributes:
        embedding: The embedding vector (typically 192-512 dimensions)
        speaker_id: Optional identifier for the speaker
        name: Optional human-readable name
        confidence: Confidence score of the embedding
        source: Source audio file or description
        model: Model used to generate embedding
        timestamp: When the embedding was created
        metadata: Additional metadata
    """
    embedding: EmbeddingArray
    speaker_id: Optional[str] = None
    name: Optional[str] = None
    confidence: float = 1.0
    source: Optional[str] = None
    model: str = "unknown"
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    metadata: Dict[str, Any] = field(default_factory=dict)
    embedding_id: str = field(default_factory=lambda: str(uuid.uuid4())[:8])

    def __post_init__(self) -> None:
        """Ensure embedding is properly formatted."""
        if not isinstance(self.embedding, np.ndarray):
            self.embedding = np.array(self.embedding, dtype=np.float32)
        self.embedding = self.embedding.astype(np.float32).flatten()

    @property
    def dimension(self) -> int:
        """Embedding dimension."""
        return len(self.embedding)

    @property
    def norm(self) -> float:
        """L2 norm of the embedding."""
        return float(np.linalg.norm(self.embedding))

    def normalize(self) -> "SpeakerEmbedding":
        """Return a normalized copy of the embedding."""
        norm = self.norm
        if norm > 0:
            normalized = self.embedding / norm
        else:
            normalized = self.embedding
        return SpeakerEmbedding(
            embedding=normalized,
            speaker_id=self.speaker_id,
            name=self.name,
            confidence=self.confidence,
            source=self.source,
            model=self.model,
            metadata={**self.metadata, "normalized": True},
        )

    def cosine_similarity(self, other: "SpeakerEmbedding") -> float:
        """Compute cosine similarity with another embedding."""
        if self.dimension != other.dimension:
            raise ValueError(
                f"Embedding dimensions must match: {self.dimension} vs {other.dimension}"
            )

        dot = np.dot(self.embedding, other.embedding)
        norm_product = self.norm * other.norm

        if norm_product == 0:
            return 0.0

        return float(dot / norm_product)

    def euclidean_distance(self, other: "SpeakerEmbedding") -> float:
        """Compute Euclidean distance to another embedding."""
        if self.dimension != other.dimension:
            raise ValueError(
                f"Embedding dimensions must match: {self.dimension} vs {other.dimension}"
            )
        return float(np.linalg.norm(self.embedding - other.embedding))

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary (without embedding data)."""
        return {
            "embedding_id": self.embedding_id,
            "speaker_id": self.speaker_id,
            "name": self.name,
            "dimension": self.dimension,
            "confidence": self.confidence,
            "source": self.source,
            "model": self.model,
            "timestamp": self.timestamp.isoformat(),
            "metadata": self.metadata,
        }

    def to_bytes(self) -> bytes:
        """Serialize embedding to bytes."""
        return pickle.dumps({
            "embedding": self.embedding.tobytes(),
            "embedding_shape": self.embedding.shape,
            "speaker_id": self.speaker_id,
            "name": self.name,
            "confidence": self.confidence,
            "source": self.source,
            "model": self.model,
            "timestamp": self.timestamp.isoformat(),
            "metadata": self.metadata,
            "embedding_id": self.embedding_id,
        })

    @classmethod
    def from_bytes(cls, data: bytes) -> "SpeakerEmbedding":
        """Deserialize embedding from bytes."""
        obj = pickle.loads(data)
        embedding = np.frombuffer(obj["embedding"], dtype=np.float32)
        embedding = embedding.reshape(obj["embedding_shape"])
        return cls(
            embedding=embedding,
            speaker_id=obj["speaker_id"],
            name=obj["name"],
            confidence=obj["confidence"],
            source=obj["source"],
            model=obj["model"],
            timestamp=datetime.fromisoformat(obj["timestamp"]),
            metadata=obj["metadata"],
            embedding_id=obj["embedding_id"],
        )

    def save(self, path: Union[str, Path]) -> Path:
        """Save embedding to file."""
        path = Path(path)
        path.write_bytes(self.to_bytes())
        return path

    @classmethod
    def load(cls, path: Union[str, Path]) -> "SpeakerEmbedding":
        """Load embedding from file."""
        return cls.from_bytes(Path(path).read_bytes())


# =============================================================================
# ENCODER PROTOCOL
# =============================================================================


@runtime_checkable
class EncoderProtocol(Protocol):
    """Protocol for speaker encoder implementations."""

    async def encode(
        self,
        audio: Union[AudioArray, Path, str, bytes],
        progress_callback: Optional[ProgressCallback] = None,
    ) -> SpeakerEmbedding:
        """Extract speaker embedding from audio."""
        ...

    def is_available(self) -> bool:
        """Check if the encoder is available."""
        ...


# =============================================================================
# BASE ENCODER
# =============================================================================


class BaseEncoder(ABC):
    """Abstract base class for speaker encoders."""

    def __init__(
        self,
        model_name: str = "default",
        device: str = "cpu",
        embedding_dim: int = 256,
    ):
        """Initialize encoder.

        Args:
            model_name: Name of the model
            device: Device to run model on (cpu, cuda)
            embedding_dim: Expected embedding dimension
        """
        self.model_name = model_name
        self.device = device
        self.embedding_dim = embedding_dim

    @abstractmethod
    async def encode(
        self,
        audio: Union[AudioArray, Path, str, bytes],
        progress_callback: Optional[ProgressCallback] = None,
    ) -> SpeakerEmbedding:
        """Extract speaker embedding from audio."""
        ...

    @abstractmethod
    def is_available(self) -> bool:
        """Check if the encoder backend is available."""
        ...

    def _load_audio(
        self,
        audio: Union[AudioArray, Path, str, bytes],
        target_sr: int = 16000,
    ) -> Tuple[AudioArray, int]:
        """Load audio from various sources."""
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

        # Convert to mono
        if len(audio_data.shape) > 1:
            audio_data = np.mean(audio_data, axis=1)

        # Resample if needed
        if sr != target_sr:
            try:
                import librosa
                audio_data = librosa.resample(audio_data, orig_sr=sr, target_sr=target_sr)
            except ImportError:
                pass  # Use original sample rate

        return audio_data.astype(np.float32), target_sr


# =============================================================================
# HUBERT ENCODER
# =============================================================================


class HuBERTEncoder(BaseEncoder):
    """HuBERT-based speaker encoder.

    Uses HuBERT (Hidden-Unit BERT) model for extracting speaker embeddings.
    Good for capturing both linguistic and speaker information.

    Example:
        >>> encoder = HuBERTEncoder()
        >>> embedding = await encoder.encode("audio.wav")
        >>> print(f"Embedding dimension: {embedding.dimension}")
    """

    # Available HuBERT models
    MODELS = {
        "hubert-base": "facebook/hubert-base-ls960",
        "hubert-large": "facebook/hubert-large-ls960-ft",
        "hubert-xlarge": "facebook/hubert-xlarge-ls960-ft",
    }

    def __init__(
        self,
        model_name: str = "hubert-base",
        device: str = "cpu",
        layer: int = -1,
    ):
        """Initialize HuBERT encoder.

        Args:
            model_name: HuBERT model variant
            device: Device (cpu, cuda)
            layer: Which layer to extract embeddings from (-1 = last)
        """
        super().__init__(model_name, device, embedding_dim=768)
        self.layer = layer
        self._model: Optional[Any] = None
        self._processor: Optional[Any] = None
        self._available: Optional[bool] = None

    def is_available(self) -> bool:
        """Check if HuBERT is available."""
        if self._available is not None:
            return self._available

        try:
            import transformers
            import torch
            self._available = True
        except ImportError:
            self._available = False

        return self._available

    def _get_model(self) -> Tuple[Any, Any]:
        """Load HuBERT model and processor."""
        if self._model is not None:
            return self._model, self._processor

        from transformers import HubertModel, Wav2Vec2Processor
        import torch

        model_path = self.MODELS.get(self.model_name, self.model_name)

        self._processor = Wav2Vec2Processor.from_pretrained(model_path)
        self._model = HubertModel.from_pretrained(model_path)
        self._model.to(self.device)
        self._model.eval()

        return self._model, self._processor

    async def encode(
        self,
        audio: Union[AudioArray, Path, str, bytes],
        progress_callback: Optional[ProgressCallback] = None,
    ) -> SpeakerEmbedding:
        """Extract speaker embedding using HuBERT."""
        if not self.is_available():
            raise ImportError("transformers and torch required for HuBERT")

        import torch

        if progress_callback:
            progress_callback("Loading audio", 0.0)

        # Load audio
        audio_data, sr = self._load_audio(audio, target_sr=16000)

        if progress_callback:
            progress_callback("Loading model", 0.2)

        loop = asyncio.get_event_loop()

        def _encode():
            model, processor = self._get_model()

            # Process audio
            inputs = processor(
                audio_data,
                sampling_rate=16000,
                return_tensors="pt",
                padding=True,
            )
            inputs = {k: v.to(self.device) for k, v in inputs.items()}

            # Extract embeddings
            with torch.no_grad():
                outputs = model(**inputs, output_hidden_states=True)

                # Get hidden states from specified layer
                if self.layer == -1:
                    hidden_states = outputs.last_hidden_state
                else:
                    hidden_states = outputs.hidden_states[self.layer]

                # Mean pooling over time dimension
                embedding = hidden_states.mean(dim=1).squeeze().cpu().numpy()

            return embedding

        if progress_callback:
            progress_callback("Extracting embedding", 0.5)

        embedding = await loop.run_in_executor(None, _encode)

        if progress_callback:
            progress_callback("Complete", 1.0)

        source = str(audio) if isinstance(audio, (str, Path)) else "audio_array"

        return SpeakerEmbedding(
            embedding=embedding,
            source=source,
            model=f"hubert-{self.model_name}",
            metadata={"layer": self.layer},
        )


# =============================================================================
# ECAPA-TDNN ENCODER
# =============================================================================


class ECAPATDNNEncoder(BaseEncoder):
    """ECAPA-TDNN speaker encoder.

    Uses ECAPA-TDNN (Emphasized Channel Attention, Propagation and Aggregation
    in Time Delay Neural Network) for speaker verification and identification.
    State-of-the-art for speaker recognition tasks.

    Example:
        >>> encoder = ECAPATDNNEncoder()
        >>> embedding = await encoder.encode("audio.wav")
        >>> similarity = embedding.cosine_similarity(other_embedding)
    """

    def __init__(
        self,
        model_name: str = "speechbrain/spkrec-ecapa-voxceleb",
        device: str = "cpu",
    ):
        """Initialize ECAPA-TDNN encoder.

        Args:
            model_name: Model name or path
            device: Device (cpu, cuda)
        """
        super().__init__(model_name, device, embedding_dim=192)
        self._model: Optional[Any] = None
        self._available: Optional[bool] = None

    def is_available(self) -> bool:
        """Check if ECAPA-TDNN is available."""
        if self._available is not None:
            return self._available

        try:
            import speechbrain
            import torch
            self._available = True
        except ImportError:
            self._available = False

        return self._available

    def _get_model(self) -> Any:
        """Load ECAPA-TDNN model."""
        if self._model is not None:
            return self._model

        from speechbrain.pretrained import EncoderClassifier
        import torch

        self._model = EncoderClassifier.from_hparams(
            source=self.model_name,
            savedir=os.path.join(tempfile.gettempdir(), "ecapa_model"),
            run_opts={"device": self.device},
        )

        return self._model

    async def encode(
        self,
        audio: Union[AudioArray, Path, str, bytes],
        progress_callback: Optional[ProgressCallback] = None,
    ) -> SpeakerEmbedding:
        """Extract speaker embedding using ECAPA-TDNN."""
        if not self.is_available():
            raise ImportError("speechbrain and torch required for ECAPA-TDNN")

        import torch

        if progress_callback:
            progress_callback("Loading audio", 0.0)

        # Load audio
        audio_data, sr = self._load_audio(audio, target_sr=16000)

        if progress_callback:
            progress_callback("Loading model", 0.2)

        loop = asyncio.get_event_loop()

        def _encode():
            model = self._get_model()

            # Convert to tensor
            waveform = torch.tensor(audio_data).unsqueeze(0)

            # Extract embedding
            with torch.no_grad():
                embedding = model.encode_batch(waveform)
                embedding = embedding.squeeze().cpu().numpy()

            return embedding

        if progress_callback:
            progress_callback("Extracting embedding", 0.5)

        embedding = await loop.run_in_executor(None, _encode)

        if progress_callback:
            progress_callback("Complete", 1.0)

        source = str(audio) if isinstance(audio, (str, Path)) else "audio_array"

        return SpeakerEmbedding(
            embedding=embedding,
            source=source,
            model="ecapa-tdnn",
        )


# =============================================================================
# UNIFIED SPEAKER ENCODER
# =============================================================================


class EncoderBackend(Enum):
    """Available encoder backends."""
    HUBERT = auto()
    ECAPA_TDNN = auto()
    AUTO = auto()


class SpeakerEncoder:
    """Unified speaker encoder with multiple backend support.

    Provides a single interface for speaker embedding extraction with:
    - Multiple model backends (HuBERT, ECAPA-TDNN)
    - Automatic backend selection
    - Batch processing
    - Speaker comparison utilities

    Example:
        >>> encoder = SpeakerEncoder()
        >>> emb1 = await encoder.encode("speaker1.wav")
        >>> emb2 = await encoder.encode("speaker2.wav")
        >>> similarity = encoder.compare(emb1, emb2)
        >>> print(f"Similarity: {similarity:.3f}")
    """

    def __init__(
        self,
        backend: EncoderBackend = EncoderBackend.AUTO,
        device: str = "cpu",
    ):
        """Initialize unified encoder.

        Args:
            backend: Encoder backend to use
            device: Device (cpu, cuda)
        """
        self.device = device
        self._backend = backend

        self._hubert: Optional[HuBERTEncoder] = None
        self._ecapa: Optional[ECAPATDNNEncoder] = None

        if backend == EncoderBackend.AUTO:
            self._backend = self._detect_backend()

    def _detect_backend(self) -> EncoderBackend:
        """Detect best available backend."""
        # Prefer ECAPA-TDNN for speaker recognition tasks
        try:
            import speechbrain
            return EncoderBackend.ECAPA_TDNN
        except ImportError:
            pass

        # Fall back to HuBERT
        try:
            import transformers
            return EncoderBackend.HUBERT
        except ImportError:
            pass

        return EncoderBackend.ECAPA_TDNN  # Will fail with helpful error

    def _get_hubert(self) -> HuBERTEncoder:
        """Get or create HuBERT encoder."""
        if self._hubert is None:
            self._hubert = HuBERTEncoder(device=self.device)
        return self._hubert

    def _get_ecapa(self) -> ECAPATDNNEncoder:
        """Get or create ECAPA-TDNN encoder."""
        if self._ecapa is None:
            self._ecapa = ECAPATDNNEncoder(device=self.device)
        return self._ecapa

    def _get_encoder(self, backend: EncoderBackend) -> BaseEncoder:
        """Get encoder for backend."""
        if backend == EncoderBackend.HUBERT:
            return self._get_hubert()
        elif backend == EncoderBackend.ECAPA_TDNN:
            return self._get_ecapa()
        else:
            raise ValueError(f"Unknown backend: {backend}")

    def is_available(self, backend: Optional[EncoderBackend] = None) -> bool:
        """Check if backend is available."""
        backend = backend or self._backend
        encoder = self._get_encoder(backend)
        return encoder.is_available()

    async def encode(
        self,
        audio: Union[AudioArray, Path, str, bytes],
        backend: Optional[EncoderBackend] = None,
        progress_callback: Optional[ProgressCallback] = None,
    ) -> SpeakerEmbedding:
        """Extract speaker embedding from audio.

        Args:
            audio: Audio data
            backend: Specific backend to use
            progress_callback: Progress callback

        Returns:
            SpeakerEmbedding
        """
        backend = backend or self._backend
        encoder = self._get_encoder(backend)

        if not encoder.is_available():
            raise ImportError(f"Backend {backend.name} is not available")

        return await encoder.encode(audio, progress_callback)

    def encode_sync(
        self,
        audio: Union[AudioArray, Path, str, bytes],
        backend: Optional[EncoderBackend] = None,
        progress_callback: Optional[ProgressCallback] = None,
    ) -> SpeakerEmbedding:
        """Synchronous wrapper for encode()."""
        try:
            loop = asyncio.get_event_loop()
        except RuntimeError:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)

        return loop.run_until_complete(
            self.encode(audio, backend, progress_callback)
        )

    async def encode_batch(
        self,
        audio_files: List[Union[Path, str]],
        backend: Optional[EncoderBackend] = None,
        max_concurrent: int = 2,
        progress_callback: Optional[ProgressCallback] = None,
    ) -> List[SpeakerEmbedding]:
        """Extract embeddings from multiple audio files.

        Args:
            audio_files: List of audio file paths
            backend: Encoder backend
            max_concurrent: Maximum concurrent encodings
            progress_callback: Progress callback

        Returns:
            List of SpeakerEmbedding
        """
        semaphore = asyncio.Semaphore(max_concurrent)

        async def _encode_one(i: int, path: Union[Path, str]) -> SpeakerEmbedding:
            async with semaphore:
                if progress_callback:
                    progress_callback(f"Encoding {i+1}/{len(audio_files)}", i / len(audio_files))
                return await self.encode(path, backend)

        results = await asyncio.gather(*[
            _encode_one(i, path) for i, path in enumerate(audio_files)
        ])

        if progress_callback:
            progress_callback("Batch complete", 1.0)

        return list(results)

    @staticmethod
    def compare(
        emb1: SpeakerEmbedding,
        emb2: SpeakerEmbedding,
        metric: str = "cosine",
    ) -> float:
        """Compare two speaker embeddings.

        Args:
            emb1: First embedding
            emb2: Second embedding
            metric: Comparison metric ("cosine" or "euclidean")

        Returns:
            Similarity/distance score
        """
        if metric == "cosine":
            return emb1.cosine_similarity(emb2)
        elif metric == "euclidean":
            return emb1.euclidean_distance(emb2)
        else:
            raise ValueError(f"Unknown metric: {metric}")

    @staticmethod
    def is_same_speaker(
        emb1: SpeakerEmbedding,
        emb2: SpeakerEmbedding,
        threshold: float = 0.7,
    ) -> Tuple[bool, float]:
        """Check if two embeddings are from the same speaker.

        Args:
            emb1: First embedding
            emb2: Second embedding
            threshold: Cosine similarity threshold

        Returns:
            Tuple of (is_same_speaker, similarity)
        """
        similarity = emb1.cosine_similarity(emb2)
        return similarity >= threshold, similarity


# =============================================================================
# SPEAKER REGISTRY
# =============================================================================


@dataclass
class RegisteredSpeaker:
    """A speaker registered in the registry."""
    speaker_id: str
    name: str
    embeddings: List[SpeakerEmbedding] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    updated_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))

    @property
    def mean_embedding(self) -> Optional[SpeakerEmbedding]:
        """Get mean embedding across all samples."""
        if not self.embeddings:
            return None

        mean_vec = np.mean([e.embedding for e in self.embeddings], axis=0)
        return SpeakerEmbedding(
            embedding=mean_vec.astype(np.float32),
            speaker_id=self.speaker_id,
            name=self.name,
            model="mean",
        )

    def add_embedding(self, embedding: SpeakerEmbedding) -> None:
        """Add an embedding for this speaker."""
        embedding.speaker_id = self.speaker_id
        embedding.name = self.name
        self.embeddings.append(embedding)
        self.updated_at = datetime.now(timezone.utc)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "speaker_id": self.speaker_id,
            "name": self.name,
            "num_embeddings": len(self.embeddings),
            "metadata": self.metadata,
            "created_at": self.created_at.isoformat(),
            "updated_at": self.updated_at.isoformat(),
        }


class SpeakerRegistry:
    """Registry for managing known speakers.

    Stores speaker embeddings and provides speaker identification
    and verification functionality.

    Example:
        >>> registry = SpeakerRegistry()
        >>> registry.register("speaker1", "Alice", embedding)
        >>> matches = registry.identify(new_embedding)
        >>> for speaker_id, score in matches:
        ...     print(f"{speaker_id}: {score:.3f}")
    """

    def __init__(
        self,
        storage_path: Optional[Path] = None,
        similarity_threshold: float = 0.7,
    ):
        """Initialize speaker registry.

        Args:
            storage_path: Path to persist registry
            similarity_threshold: Default threshold for identification
        """
        self.storage_path = storage_path
        self.similarity_threshold = similarity_threshold
        self._speakers: Dict[str, RegisteredSpeaker] = {}

        if storage_path and storage_path.exists():
            self._load()

    def _load(self) -> None:
        """Load registry from storage."""
        if not self.storage_path or not self.storage_path.exists():
            return

        data = json.loads(self.storage_path.read_text())

        for speaker_data in data.get("speakers", []):
            speaker = RegisteredSpeaker(
                speaker_id=speaker_data["speaker_id"],
                name=speaker_data["name"],
                metadata=speaker_data.get("metadata", {}),
                created_at=datetime.fromisoformat(speaker_data["created_at"]),
                updated_at=datetime.fromisoformat(speaker_data["updated_at"]),
            )

            # Load embeddings
            embeddings_dir = self.storage_path.parent / "embeddings" / speaker.speaker_id
            if embeddings_dir.exists():
                for emb_file in embeddings_dir.glob("*.emb"):
                    try:
                        embedding = SpeakerEmbedding.load(emb_file)
                        speaker.embeddings.append(embedding)
                    except Exception:
                        pass

            self._speakers[speaker.speaker_id] = speaker

    def _save(self) -> None:
        """Save registry to storage."""
        if not self.storage_path:
            return

        self.storage_path.parent.mkdir(parents=True, exist_ok=True)

        data = {
            "speakers": [s.to_dict() for s in self._speakers.values()],
        }

        self.storage_path.write_text(json.dumps(data, indent=2))

        # Save embeddings
        for speaker in self._speakers.values():
            embeddings_dir = self.storage_path.parent / "embeddings" / speaker.speaker_id
            embeddings_dir.mkdir(parents=True, exist_ok=True)

            for i, emb in enumerate(speaker.embeddings):
                emb_path = embeddings_dir / f"{i:04d}.emb"
                emb.save(emb_path)

    def register(
        self,
        speaker_id: str,
        name: str,
        embedding: Optional[SpeakerEmbedding] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> RegisteredSpeaker:
        """Register a new speaker.

        Args:
            speaker_id: Unique speaker identifier
            name: Human-readable name
            embedding: Initial embedding
            metadata: Additional metadata

        Returns:
            RegisteredSpeaker
        """
        if speaker_id in self._speakers:
            raise ValueError(f"Speaker {speaker_id} already registered")

        speaker = RegisteredSpeaker(
            speaker_id=speaker_id,
            name=name,
            metadata=metadata or {},
        )

        if embedding:
            speaker.add_embedding(embedding)

        self._speakers[speaker_id] = speaker

        if self.storage_path:
            self._save()

        return speaker

    def get(self, speaker_id: str) -> Optional[RegisteredSpeaker]:
        """Get a registered speaker by ID."""
        return self._speakers.get(speaker_id)

    def list_speakers(self) -> List[RegisteredSpeaker]:
        """List all registered speakers."""
        return list(self._speakers.values())

    def add_embedding(
        self,
        speaker_id: str,
        embedding: SpeakerEmbedding,
    ) -> None:
        """Add an embedding to an existing speaker.

        Args:
            speaker_id: Speaker ID
            embedding: Embedding to add
        """
        speaker = self._speakers.get(speaker_id)
        if not speaker:
            raise ValueError(f"Speaker {speaker_id} not found")

        speaker.add_embedding(embedding)

        if self.storage_path:
            self._save()

    def remove(self, speaker_id: str) -> bool:
        """Remove a speaker from the registry.

        Args:
            speaker_id: Speaker ID

        Returns:
            True if speaker was removed
        """
        if speaker_id not in self._speakers:
            return False

        del self._speakers[speaker_id]

        if self.storage_path:
            self._save()

        return True

    def identify(
        self,
        embedding: SpeakerEmbedding,
        threshold: Optional[float] = None,
        top_k: int = 5,
    ) -> List[Tuple[str, float]]:
        """Identify speaker from embedding.

        Args:
            embedding: Query embedding
            threshold: Similarity threshold (uses default if None)
            top_k: Maximum number of matches to return

        Returns:
            List of (speaker_id, similarity) tuples, sorted by similarity
        """
        threshold = threshold or self.similarity_threshold
        matches = []

        for speaker in self._speakers.values():
            mean_emb = speaker.mean_embedding
            if mean_emb is None:
                continue

            similarity = embedding.cosine_similarity(mean_emb)
            if similarity >= threshold:
                matches.append((speaker.speaker_id, similarity))

        # Sort by similarity descending
        matches.sort(key=lambda x: x[1], reverse=True)
        return matches[:top_k]

    def verify(
        self,
        speaker_id: str,
        embedding: SpeakerEmbedding,
        threshold: Optional[float] = None,
    ) -> Tuple[bool, float]:
        """Verify if embedding matches a specific speaker.

        Args:
            speaker_id: Speaker ID to verify against
            embedding: Query embedding
            threshold: Similarity threshold

        Returns:
            Tuple of (is_verified, similarity)
        """
        threshold = threshold or self.similarity_threshold
        speaker = self._speakers.get(speaker_id)

        if not speaker or not speaker.mean_embedding:
            return False, 0.0

        similarity = embedding.cosine_similarity(speaker.mean_embedding)
        return similarity >= threshold, similarity

    def find_similar(
        self,
        embedding: SpeakerEmbedding,
        top_k: int = 10,
    ) -> List[Tuple[str, float]]:
        """Find similar speakers (no threshold).

        Args:
            embedding: Query embedding
            top_k: Number of results

        Returns:
            List of (speaker_id, similarity) tuples
        """
        return self.identify(embedding, threshold=0.0, top_k=top_k)

    def cluster_embeddings(
        self,
        embeddings: List[SpeakerEmbedding],
        n_clusters: Optional[int] = None,
        min_cluster_size: int = 2,
    ) -> List[List[SpeakerEmbedding]]:
        """Cluster embeddings by speaker.

        Args:
            embeddings: List of embeddings to cluster
            n_clusters: Number of clusters (auto-detect if None)
            min_cluster_size: Minimum cluster size

        Returns:
            List of embedding clusters
        """
        if len(embeddings) < 2:
            return [embeddings] if embeddings else []

        try:
            from sklearn.cluster import AgglomerativeClustering
        except ImportError:
            raise ImportError("scikit-learn required for clustering")

        # Build feature matrix
        X = np.array([e.embedding for e in embeddings])

        if n_clusters is None:
            # Auto-detect using silhouette score
            from sklearn.metrics import silhouette_score

            best_score = -1
            best_n = 2

            for n in range(2, min(len(embeddings), 20)):
                clustering = AgglomerativeClustering(n_clusters=n)
                labels = clustering.fit_predict(X)

                score = silhouette_score(X, labels)
                if score > best_score:
                    best_score = score
                    best_n = n

            n_clusters = best_n

        # Perform clustering
        clustering = AgglomerativeClustering(n_clusters=n_clusters)
        labels = clustering.fit_predict(X)

        # Group embeddings by cluster
        clusters: Dict[int, List[SpeakerEmbedding]] = {}
        for emb, label in zip(embeddings, labels):
            if label not in clusters:
                clusters[label] = []
            clusters[label].append(emb)

        # Filter small clusters
        return [c for c in clusters.values() if len(c) >= min_cluster_size]

    def __len__(self) -> int:
        """Number of registered speakers."""
        return len(self._speakers)

    def __contains__(self, speaker_id: str) -> bool:
        """Check if speaker is registered."""
        return speaker_id in self._speakers


# =============================================================================
# MODULE EXPORTS
# =============================================================================

__all__ = [
    # Data classes
    "SpeakerEmbedding",
    "RegisteredSpeaker",
    # Enums
    "EncoderBackend",
    # Encoders
    "BaseEncoder",
    "HuBERTEncoder",
    "ECAPATDNNEncoder",
    "SpeakerEncoder",
    # Registry
    "SpeakerRegistry",
    # Protocol
    "EncoderProtocol",
]
