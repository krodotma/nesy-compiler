"""
Auralux Subsystem
=================

Voice synthesis, TTS/STT, speaker embeddings.

Components:
- synthesizer: Voice synthesis (TTS) with Edge TTS and Coqui backends
- recognizer: Speech recognition (STT) with Whisper and VAD
- speaker_encoder: Speaker embeddings with HuBERT and ECAPA-TDNN
"""

from __future__ import annotations

# Import submodules
from nucleus.creative.auralux import synthesizer
from nucleus.creative.auralux import recognizer
from nucleus.creative.auralux import speaker_encoder

# Export from synthesizer
from nucleus.creative.auralux.synthesizer import (
    VoiceConfig,
    SynthesisResult,
    EdgeTTSSynthesizer,
    CoquiTTSSynthesizer,
    VoiceSynthesizer,
    VoiceGender,
    VoiceStyle,
    SynthesizerBackend,
)

# Export from recognizer
from nucleus.creative.auralux.recognizer import (
    WordTiming,
    Segment,
    TranscriptionResult,
    WhisperRecognizer,
    SileroVAD,
    SpeechRecognizer,
    VADSegment,
    WhisperModel,
    RecognizerBackend,
)

# Export from speaker_encoder
from nucleus.creative.auralux.speaker_encoder import (
    SpeakerEmbedding,
    HuBERTEncoder,
    ECAPATDNNEncoder,
    SpeakerEncoder,
    SpeakerRegistry,
    RegisteredSpeaker,
    EncoderBackend,
)

__all__ = [
    # Synthesizer
    "VoiceConfig",
    "SynthesisResult",
    "EdgeTTSSynthesizer",
    "CoquiTTSSynthesizer",
    "VoiceSynthesizer",
    "VoiceGender",
    "VoiceStyle",
    "SynthesizerBackend",
    # Recognizer
    "WordTiming",
    "Segment",
    "TranscriptionResult",
    "WhisperRecognizer",
    "SileroVAD",
    "SpeechRecognizer",
    "VADSegment",
    "WhisperModel",
    "RecognizerBackend",
    # Speaker Encoder
    "SpeakerEmbedding",
    "HuBERTEncoder",
    "ECAPATDNNEncoder",
    "SpeakerEncoder",
    "SpeakerRegistry",
    "RegisteredSpeaker",
    "EncoderBackend",
    # Submodules
    "synthesizer",
    "recognizer",
    "speaker_encoder",
]
