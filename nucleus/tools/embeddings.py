#!/usr/bin/env python3
"""
Embedding helpers (local-first, optional deps)
=============================================

Goal: provide a deterministic, offline embedding function that upgrades to
sentence-transformers when available, without forcing heavyweight deps.

Default behavior:
- If `sentence_transformers` is available: use `all-MiniLM-L6-v2` (384-dim).
- Else: use a deterministic hashed bag-of-words embedding (384-dim).

Override with:
- `PLURIBUS_EMBED_MODE=sentence-transformers|hash|off`
"""
from __future__ import annotations

import hashlib
import math
import os
import re
import sys
from pathlib import Path
from typing import Any

DEFAULT_MODEL = "all-MiniLM-L6-v2"
DEFAULT_DIM = 384

_SENTENCE_MODEL: Any | None = None
_SENTENCE_OK: bool | None = None


def maybe_add_pluribus_venv_site() -> None:
    """Best-effort: allow optional deps installed into /pluribus/.pluribus/venv."""
    try:
        root = Path(__file__).resolve().parents[2]
        site = root / ".pluribus" / "venv" / "lib" / f"python{sys.version_info.major}.{sys.version_info.minor}" / "site-packages"
        if site.exists() and str(site) not in sys.path:
            sys.path.insert(0, str(site))
    except Exception:
        return


def _try_load_sentence_transformer(model_name: str = DEFAULT_MODEL):
    global _SENTENCE_MODEL, _SENTENCE_OK
    if _SENTENCE_OK is False:
        return None
    if _SENTENCE_MODEL is not None:
        return _SENTENCE_MODEL

    maybe_add_pluribus_venv_site()
    try:
        from sentence_transformers import SentenceTransformer  # type: ignore

        _SENTENCE_MODEL = SentenceTransformer(model_name)
        _SENTENCE_OK = True
        return _SENTENCE_MODEL
    except Exception:
        _SENTENCE_OK = False
        _SENTENCE_MODEL = None
        return None


_TOKEN_RE = re.compile(r"[a-z0-9_]{2,}", re.IGNORECASE)


def hash_embed(text: str, *, dim: int = DEFAULT_DIM) -> list[float]:
    """
    Deterministic hashed bag-of-words embedding (offline, no ML deps).

    Notes:
    - This is NOT a semantic embedding like transformer models, but it enables
      stable approximate similarity and can be replaced transparently.
    - Output is L2-normalized.
    """
    vec = [0.0] * int(dim)
    tokens = _TOKEN_RE.findall((text or "").lower())
    if not tokens:
        return vec

    for tok in tokens:
        h = hashlib.sha256(tok.encode("utf-8")).digest()
        idx = int.from_bytes(h[:4], "little") % dim
        sign = -1.0 if (h[4] & 1) else 1.0
        vec[idx] += sign

    norm = math.sqrt(sum(v * v for v in vec)) or 1.0
    return [v / norm for v in vec]


def embed_text(text: str, *, dim: int = DEFAULT_DIM, model: str = DEFAULT_MODEL) -> tuple[list[float] | None, dict]:
    """
    Returns (embedding, meta).
    - embedding is a list[float] of length dim, or None if disabled.
    - meta describes what embedder was used.
    """
    mode = (os.environ.get("PLURIBUS_EMBED_MODE") or "auto").strip().lower()
    if mode in {"off", "none", "0"}:
        return None, {"mode": "off", "dim": dim, "model": None}

    if mode in {"sentence-transformers", "st", "transformer"} or mode == "auto":
        st = _try_load_sentence_transformer(model)
        if st is not None:
            try:
                # Keep it fast + bounded (avoid extreme prompts).
                v = st.encode((text or "")[:4000])
                # sentence-transformers returns numpy arrays; tolist() yields floats.
                out = list(map(float, v.tolist()))
                if len(out) != dim:
                    # If dim mismatch, fall back to hash embedding with requested dim.
                    return hash_embed(text, dim=dim), {"mode": "hash", "dim": dim, "model": "hash_embed"}
                return out, {"mode": "sentence-transformers", "dim": dim, "model": model}
            except Exception:
                pass
        if mode != "auto":
            return None, {"mode": "sentence-transformers", "dim": dim, "model": model, "error": "unavailable"}

    # Default/fallback: hashed embedding
    return hash_embed(text, dim=dim), {"mode": "hash", "dim": dim, "model": "hash_embed"}

