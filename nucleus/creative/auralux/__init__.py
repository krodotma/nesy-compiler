"""
Auralux Subsystem
=================

Voice synthesis, TTS/STT, speaker embeddings.
"""

from __future__ import annotations
import importlib.util
import sys
from pathlib import Path

_cache_dir = Path(__file__).parent / "__pycache__"

def _load_from_pyc(name: str):
    """Load a module from its .pyc file."""
    pyc_pattern = f"{name}.cpython-*.pyc"
    pyc_files = list(_cache_dir.glob(pyc_pattern))
    if not pyc_files:
        return None

    pyc_path = pyc_files[0]
    module_name = f"nucleus.creative.auralux.{name}"

    spec = importlib.util.spec_from_file_location(module_name, pyc_path)
    if spec and spec.loader:
        module = importlib.util.module_from_spec(spec)
        sys.modules[module_name] = module
        try:
            spec.loader.exec_module(module)
            return module
        except Exception:
            del sys.modules[module_name]
            return None
    return None

# Load submodules
synthesizer = _load_from_pyc("synthesizer")
recognizer = _load_from_pyc("recognizer")
speaker_encoder = _load_from_pyc("speaker_encoder")

# Export from synthesizer
if synthesizer:
    VoiceConfig = getattr(synthesizer, "VoiceConfig", None)
    SynthesisResult = getattr(synthesizer, "SynthesisResult", None)
    EdgeTTSSynthesizer = getattr(synthesizer, "EdgeTTSSynthesizer", None)
    CoquiTTSSynthesizer = getattr(synthesizer, "CoquiTTSSynthesizer", None)
    VoiceSynthesizer = getattr(synthesizer, "VoiceSynthesizer", None)

# Export from recognizer
if recognizer:
    WordTiming = getattr(recognizer, "WordTiming", None)
    Segment = getattr(recognizer, "Segment", None)
    TranscriptionResult = getattr(recognizer, "TranscriptionResult", None)
    WhisperRecognizer = getattr(recognizer, "WhisperRecognizer", None)
    SileroVAD = getattr(recognizer, "SileroVAD", None)
    SpeechRecognizer = getattr(recognizer, "SpeechRecognizer", None)

# Export from speaker_encoder
if speaker_encoder:
    SpeakerEmbedding = getattr(speaker_encoder, "SpeakerEmbedding", None)
    HuBERTEncoder = getattr(speaker_encoder, "HuBERTEncoder", None)
    ECAPATDNNEncoder = getattr(speaker_encoder, "ECAPATDNNEncoder", None)
    SpeakerEncoder = getattr(speaker_encoder, "SpeakerEncoder", None)
    SpeakerRegistry = getattr(speaker_encoder, "SpeakerRegistry", None)

__all__ = [
    "VoiceConfig", "SynthesisResult", "EdgeTTSSynthesizer", "CoquiTTSSynthesizer", "VoiceSynthesizer",
    "WordTiming", "Segment", "TranscriptionResult", "WhisperRecognizer", "SileroVAD", "SpeechRecognizer",
    "SpeakerEmbedding", "HuBERTEncoder", "ECAPATDNNEncoder", "SpeakerEncoder", "SpeakerRegistry",
    "synthesizer", "recognizer", "speaker_encoder",
]
