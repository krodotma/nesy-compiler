#!/usr/bin/env python3
"""Post-Quantum Cryptography wrapper for Pluribus using pqcrypto (liboqs bindings).

Supports:
- CRYSTALS-Kyber (ML-KEM) for key encapsulation/exchange
- CRYSTALS-Dilithium (ML-DSA) for digital signatures
- FALCON (optional) for signatures

Per NIST FIPS 203 (ML-KEM) and FIPS 204 (ML-DSA).

Integration points:
- Bus event signing via PQC signatures
- iso_git.mjs quantum-safe commit signing
- Hexis buffer message authentication

Usage:
    # Generate keys
    python3 pqc_crypto.py keygen

    # Sign data
    python3 pqc_crypto.py sign "message to sign"

    # Verify signature
    python3 pqc_crypto.py verify <signature_json>

    # Key exchange (encapsulate)
    python3 pqc_crypto.py kem-encap <public_key_b64>

    # Key exchange (decapsulate)
    python3 pqc_crypto.py kem-decap <ciphertext_b64>
"""
from __future__ import annotations

import argparse
import base64
import hashlib
import json
import os
import sys
import time
import uuid
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Literal, TypedDict

sys.dont_write_bytecode = True


# ---------------------------------------------------------------------------
# PQC Algorithm Imports (pqcrypto wraps liboqs)
# ---------------------------------------------------------------------------
try:
    from pqcrypto.sign import ml_dsa_65
    from pqcrypto.sign import falcon_512
    from pqcrypto.kem import ml_kem_768
    PQC_AVAILABLE = True
except ImportError as e:
    PQC_AVAILABLE = False
    _IMPORT_ERROR = str(e)


# ---------------------------------------------------------------------------
# Constants and Configuration
# ---------------------------------------------------------------------------
DEFAULT_SIG_ALGO = "ML-DSA-65"  # FIPS 204 (formerly Dilithium3)
DEFAULT_KEM_ALGO = "ML-KEM-768"  # FIPS 203 (formerly Kyber768)
FALLBACK_SIG_ALGO = "FALCON-512"

# Key storage paths
def _secrets_dir() -> Path:
    return Path(os.environ.get("PLURIBUS_SECRETS_DIR", "")) or Path.home() / ".pluribus" / "secrets"


def _keys_path() -> Path:
    return _secrets_dir() / "pqc_keys.json"


# Algorithm metadata
ALGO_METADATA = {
    "ML-DSA-65": {
        "type": "signature",
        "fips": "FIPS 204",
        "security_level": 3,
        "classical_bits": 192,
        "quantum_bits": 128,
        "public_key_size": 1952,
        "secret_key_size": 4032,
        "signature_size": 3309,
    },
    "ML-KEM-768": {
        "type": "kem",
        "fips": "FIPS 203",
        "security_level": 3,
        "classical_bits": 192,
        "quantum_bits": 128,
        "public_key_size": 1184,
        "secret_key_size": 2400,
        "ciphertext_size": 1088,
        "shared_secret_size": 32,
    },
    "FALCON-512": {
        "type": "signature",
        "security_level": 1,
        "classical_bits": 128,
        "quantum_bits": 64,
        "public_key_size": 897,
        "secret_key_size": 1281,
        "signature_size": 690,  # Average, variable
    },
}


# ---------------------------------------------------------------------------
# Data Classes
# ---------------------------------------------------------------------------
@dataclass
class PQCKeyPair:
    """PQC key pair container."""
    algo: str
    public_key: str  # Base64 encoded
    secret_key: str  # Base64 encoded
    fingerprint: str
    created_iso: str
    key_sizes: dict[str, int]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class PQCSignature:
    """PQC signature container."""
    algo: str
    signer: str  # Fingerprint
    signature: str  # Base64 encoded
    payload_hash: str  # SHA3-256 of payload
    timestamp_iso: str


@dataclass
class KEMEncapsulation:
    """KEM encapsulation result."""
    algo: str
    ciphertext: str  # Base64 encoded
    shared_secret: str  # Base64 encoded (for sender)


@dataclass
class SignedEvent:
    """Bus event with PQC signature."""
    event: dict[str, Any]
    pqc_signature: PQCSignature


class PQCStatus(TypedDict):
    """PQC readiness status."""
    ready: bool
    sig_algo: str
    kem_algo: str
    sig_available: bool
    kem_available: bool
    library: str
    error: str | None


# ---------------------------------------------------------------------------
# Utility Functions
# ---------------------------------------------------------------------------
def _fingerprint(public_key_bytes: bytes) -> str:
    """Generate fingerprint from public key (SHA3-256, first 16 hex chars)."""
    h = hashlib.sha3_256(public_key_bytes).hexdigest()
    return h[:16]


def _now_iso() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def _hash_payload(payload: str | bytes) -> str:
    """SHA3-256 hash of payload."""
    if isinstance(payload, str):
        payload = payload.encode("utf-8")
    return hashlib.sha3_256(payload).hexdigest()


# ---------------------------------------------------------------------------
# Core PQC Operations
# ---------------------------------------------------------------------------
def check_pqc_status() -> PQCStatus:
    """Check PQC library availability and readiness."""
    if not PQC_AVAILABLE:
        return PQCStatus(
            ready=False,
            sig_algo=DEFAULT_SIG_ALGO,
            kem_algo=DEFAULT_KEM_ALGO,
            sig_available=False,
            kem_available=False,
            library="pqcrypto",
            error=_IMPORT_ERROR if "_IMPORT_ERROR" in dir() else "pqcrypto not installed",
        )

    return PQCStatus(
        ready=True,
        sig_algo=DEFAULT_SIG_ALGO,
        kem_algo=DEFAULT_KEM_ALGO,
        sig_available=True,
        kem_available=True,
        library="pqcrypto (liboqs bindings)",
        error=None,
    )


def generate_sig_keypair(algo: str = DEFAULT_SIG_ALGO) -> PQCKeyPair:
    """Generate PQC signature key pair."""
    if not PQC_AVAILABLE:
        raise RuntimeError("PQC library not available")

    if algo == "ML-DSA-65":
        pk, sk = ml_dsa_65.generate_keypair()
    elif algo == "FALCON-512":
        pk, sk = falcon_512.generate_keypair()
    else:
        raise ValueError(f"Unsupported signature algorithm: {algo}")

    pk_b64 = base64.b64encode(pk).decode("ascii")
    sk_b64 = base64.b64encode(sk).decode("ascii")
    fp = _fingerprint(pk)

    return PQCKeyPair(
        algo=algo,
        public_key=pk_b64,
        secret_key=sk_b64,
        fingerprint=fp,
        created_iso=_now_iso(),
        key_sizes={"public_key": len(pk), "secret_key": len(sk)},
    )


def generate_kem_keypair(algo: str = DEFAULT_KEM_ALGO) -> PQCKeyPair:
    """Generate PQC KEM key pair."""
    if not PQC_AVAILABLE:
        raise RuntimeError("PQC library not available")

    if algo == "ML-KEM-768":
        pk, sk = ml_kem_768.generate_keypair()
    else:
        raise ValueError(f"Unsupported KEM algorithm: {algo}")

    pk_b64 = base64.b64encode(pk).decode("ascii")
    sk_b64 = base64.b64encode(sk).decode("ascii")
    fp = _fingerprint(pk)

    return PQCKeyPair(
        algo=algo,
        public_key=pk_b64,
        secret_key=sk_b64,
        fingerprint=fp,
        created_iso=_now_iso(),
        key_sizes={"public_key": len(pk), "secret_key": len(sk)},
    )


def sign_data(
    data: str | bytes | dict[str, Any],
    secret_key_b64: str,
    algo: str = DEFAULT_SIG_ALGO,
    signer_fingerprint: str = "",
) -> PQCSignature:
    """Sign data with PQC signature algorithm."""
    if not PQC_AVAILABLE:
        raise RuntimeError("PQC library not available")

    # Normalize data to bytes
    if isinstance(data, dict):
        payload = json.dumps(data, sort_keys=True, separators=(",", ":"))
    elif isinstance(data, str):
        payload = data
    else:
        payload = data.decode("utf-8") if isinstance(data, bytes) else str(data)

    payload_bytes = payload.encode("utf-8")
    sk = base64.b64decode(secret_key_b64)

    if algo == "ML-DSA-65":
        sig = ml_dsa_65.sign(sk, payload_bytes)
    elif algo == "FALCON-512":
        sig = falcon_512.sign(sk, payload_bytes)
    else:
        raise ValueError(f"Unsupported signature algorithm: {algo}")

    return PQCSignature(
        algo=algo,
        signer=signer_fingerprint,
        signature=base64.b64encode(sig).decode("ascii"),
        payload_hash=_hash_payload(payload_bytes),
        timestamp_iso=_now_iso(),
    )


def verify_signature(
    data: str | bytes | dict[str, Any],
    signature_b64: str,
    public_key_b64: str,
    algo: str = DEFAULT_SIG_ALGO,
) -> bool:
    """Verify PQC signature."""
    if not PQC_AVAILABLE:
        raise RuntimeError("PQC library not available")

    # Normalize data to bytes
    if isinstance(data, dict):
        payload = json.dumps(data, sort_keys=True, separators=(",", ":"))
    elif isinstance(data, str):
        payload = data
    else:
        payload = data.decode("utf-8") if isinstance(data, bytes) else str(data)

    payload_bytes = payload.encode("utf-8")
    pk = base64.b64decode(public_key_b64)
    sig = base64.b64decode(signature_b64)

    try:
        if algo == "ML-DSA-65":
            return ml_dsa_65.verify(pk, payload_bytes, sig)
        elif algo == "FALCON-512":
            return falcon_512.verify(pk, payload_bytes, sig)
        else:
            raise ValueError(f"Unsupported signature algorithm: {algo}")
    except Exception:
        return False


def kem_encapsulate(public_key_b64: str, algo: str = DEFAULT_KEM_ALGO) -> KEMEncapsulation:
    """Encapsulate shared secret using recipient's public key (sender side)."""
    if not PQC_AVAILABLE:
        raise RuntimeError("PQC library not available")

    pk = base64.b64decode(public_key_b64)

    if algo == "ML-KEM-768":
        ciphertext, shared_secret = ml_kem_768.encrypt(pk)
    else:
        raise ValueError(f"Unsupported KEM algorithm: {algo}")

    return KEMEncapsulation(
        algo=algo,
        ciphertext=base64.b64encode(ciphertext).decode("ascii"),
        shared_secret=base64.b64encode(shared_secret).decode("ascii"),
    )


def kem_decapsulate(
    ciphertext_b64: str,
    secret_key_b64: str,
    algo: str = DEFAULT_KEM_ALGO,
) -> bytes:
    """Decapsulate shared secret using own secret key (receiver side)."""
    if not PQC_AVAILABLE:
        raise RuntimeError("PQC library not available")

    sk = base64.b64decode(secret_key_b64)
    ct = base64.b64decode(ciphertext_b64)

    if algo == "ML-KEM-768":
        shared_secret = ml_kem_768.decrypt(sk, ct)
    else:
        raise ValueError(f"Unsupported KEM algorithm: {algo}")

    return shared_secret


# ---------------------------------------------------------------------------
# Key Management
# ---------------------------------------------------------------------------
def save_keys(sig_keypair: PQCKeyPair, kem_keypair: PQCKeyPair | None = None) -> Path:
    """Save key pairs to secure storage."""
    secrets_dir = _secrets_dir()
    secrets_dir.mkdir(parents=True, exist_ok=True)
    keys_path = _keys_path()

    data = {
        "version": "1.0",
        "created_iso": _now_iso(),
        "signature": sig_keypair.to_dict(),
    }
    if kem_keypair:
        data["kem"] = kem_keypair.to_dict()

    # Write with restrictive permissions
    keys_path.write_text(json.dumps(data, indent=2), encoding="utf-8")
    os.chmod(keys_path, 0o600)

    return keys_path


def load_keys() -> dict[str, Any]:
    """Load key pairs from storage."""
    keys_path = _keys_path()
    if not keys_path.exists():
        raise FileNotFoundError(f"No keys found at {keys_path}. Run 'keygen' first.")
    return json.loads(keys_path.read_text(encoding="utf-8"))


def get_public_key(key_type: Literal["signature", "kem"] = "signature") -> str:
    """Get public key (base64) for sharing."""
    keys = load_keys()
    if key_type == "signature":
        return keys["signature"]["public_key"]
    elif key_type == "kem":
        if "kem" not in keys:
            raise KeyError("No KEM keys found. Regenerate with --with-kem")
        return keys["kem"]["public_key"]
    else:
        raise ValueError(f"Unknown key type: {key_type}")


def get_fingerprint(key_type: Literal["signature", "kem"] = "signature") -> str:
    """Get key fingerprint."""
    keys = load_keys()
    if key_type == "signature":
        return keys["signature"]["fingerprint"]
    elif key_type == "kem":
        if "kem" not in keys:
            raise KeyError("No KEM keys found")
        return keys["kem"]["fingerprint"]
    else:
        raise ValueError(f"Unknown key type: {key_type}")


# ---------------------------------------------------------------------------
# Bus Integration
# ---------------------------------------------------------------------------
def sign_bus_event(event: dict[str, Any]) -> SignedEvent:
    """Sign a bus event with PQC signature.

    Integrates with agent_bus.py event format.
    """
    keys = load_keys()
    sig_data = keys["signature"]

    signature = sign_data(
        data=event,
        secret_key_b64=sig_data["secret_key"],
        algo=sig_data["algo"],
        signer_fingerprint=sig_data["fingerprint"],
    )

    return SignedEvent(event=event, pqc_signature=signature)


def verify_bus_event(signed_event: SignedEvent | dict[str, Any], public_key_b64: str | None = None) -> bool:
    """Verify a signed bus event.

    If public_key_b64 is None, uses local keys.
    """
    if isinstance(signed_event, dict):
        event = signed_event.get("event", {})
        sig_data = signed_event.get("pqc_signature", {})
        signature_b64 = sig_data.get("signature", "")
        algo = sig_data.get("algo", DEFAULT_SIG_ALGO)
    else:
        event = signed_event.event
        signature_b64 = signed_event.pqc_signature.signature
        algo = signed_event.pqc_signature.algo

    if not public_key_b64:
        keys = load_keys()
        public_key_b64 = keys["signature"]["public_key"]

    return verify_signature(event, signature_b64, public_key_b64, algo)


def emit_pqc_signed_event(
    topic: str,
    data: dict[str, Any],
    kind: str = "artifact",
    level: str = "info",
) -> dict[str, Any]:
    """Create a PQC-signed event payload ready for bus emission.

    Returns dict that can be passed to agent_bus.py as --data.
    """
    event = {
        "topic": topic,
        "kind": kind,
        "level": level,
        "data": data,
        "timestamp_iso": _now_iso(),
        "event_id": str(uuid.uuid4()),
    }

    signed = sign_bus_event(event)

    return {
        "event": event,
        "pqc_signature": asdict(signed.pqc_signature),
    }


# ---------------------------------------------------------------------------
# CLI Interface
# ---------------------------------------------------------------------------
def cmd_keygen(args: argparse.Namespace) -> int:
    """Generate PQC key pairs."""
    sig_keypair = generate_sig_keypair(args.sig_algo)
    print(f"Generated {args.sig_algo} signature keys:")
    print(f"  Fingerprint: {sig_keypair.fingerprint}")
    print(f"  Public key:  {sig_keypair.key_sizes['public_key']} bytes")
    print(f"  Secret key:  {sig_keypair.key_sizes['secret_key']} bytes")

    kem_keypair = None
    if args.with_kem:
        kem_keypair = generate_kem_keypair(args.kem_algo)
        print(f"\nGenerated {args.kem_algo} KEM keys:")
        print(f"  Fingerprint: {kem_keypair.fingerprint}")
        print(f"  Public key:  {kem_keypair.key_sizes['public_key']} bytes")
        print(f"  Secret key:  {kem_keypair.key_sizes['secret_key']} bytes")

    path = save_keys(sig_keypair, kem_keypair)
    print(f"\nKeys saved to: {path}")
    return 0


def cmd_sign(args: argparse.Namespace) -> int:
    """Sign data."""
    keys = load_keys()
    sig_data = keys["signature"]

    signature = sign_data(
        data=args.data,
        secret_key_b64=sig_data["secret_key"],
        algo=sig_data["algo"],
        signer_fingerprint=sig_data["fingerprint"],
    )

    print(json.dumps(asdict(signature), indent=2))
    return 0


def cmd_verify(args: argparse.Namespace) -> int:
    """Verify signature."""
    sig_obj = json.loads(args.signature)
    keys = load_keys()

    # If external public key provided, use it
    pk_b64 = args.public_key or keys["signature"]["public_key"]

    valid = verify_signature(
        data=args.data,
        signature_b64=sig_obj["signature"],
        public_key_b64=pk_b64,
        algo=sig_obj.get("algo", DEFAULT_SIG_ALGO),
    )

    if valid:
        print("Verification: SUCCESS")
        return 0
    else:
        print("Verification: FAILED")
        return 1


def cmd_kem_encap(args: argparse.Namespace) -> int:
    """KEM encapsulation (sender side)."""
    result = kem_encapsulate(args.public_key)
    print(json.dumps(asdict(result), indent=2))
    return 0


def cmd_kem_decap(args: argparse.Namespace) -> int:
    """KEM decapsulation (receiver side)."""
    keys = load_keys()
    if "kem" not in keys:
        print("Error: No KEM keys found. Regenerate with --with-kem", file=sys.stderr)
        return 1

    shared_secret = kem_decapsulate(
        ciphertext_b64=args.ciphertext,
        secret_key_b64=keys["kem"]["secret_key"],
    )

    print(json.dumps({
        "algo": DEFAULT_KEM_ALGO,
        "shared_secret": base64.b64encode(shared_secret).decode("ascii"),
        "shared_secret_hex": shared_secret.hex(),
    }, indent=2))
    return 0


def cmd_status(args: argparse.Namespace) -> int:
    """Show PQC status."""
    status = check_pqc_status()
    print(json.dumps(status, indent=2))

    # Also show key status if keys exist
    try:
        keys = load_keys()
        print("\nKey Status:")
        print(f"  Signature algo: {keys['signature']['algo']}")
        print(f"  Signature fingerprint: {keys['signature']['fingerprint']}")
        if "kem" in keys:
            print(f"  KEM algo: {keys['kem']['algo']}")
            print(f"  KEM fingerprint: {keys['kem']['fingerprint']}")
    except FileNotFoundError:
        print("\nNo keys found. Run 'keygen' to generate.")

    return 0


def cmd_pubkey(args: argparse.Namespace) -> int:
    """Export public key."""
    try:
        pk = get_public_key(args.type)
        fp = get_fingerprint(args.type)
        print(json.dumps({
            "type": args.type,
            "fingerprint": fp,
            "public_key": pk,
        }, indent=2))
        return 0
    except (FileNotFoundError, KeyError) as e:
        print(f"Error: {e}", file=sys.stderr)
        return 1


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="pqc_crypto.py",
        description="Post-Quantum Cryptography wrapper for Pluribus (liboqs/pqcrypto)",
    )
    sub = p.add_subparsers(dest="cmd", required=True)

    # keygen
    kg = sub.add_parser("keygen", help="Generate PQC key pairs")
    kg.add_argument("--sig-algo", default=DEFAULT_SIG_ALGO, choices=["ML-DSA-65", "FALCON-512"])
    kg.add_argument("--kem-algo", default=DEFAULT_KEM_ALGO, choices=["ML-KEM-768"])
    kg.add_argument("--with-kem", action="store_true", help="Also generate KEM keys")
    kg.set_defaults(func=cmd_keygen)

    # sign
    sg = sub.add_parser("sign", help="Sign data")
    sg.add_argument("data", help="Data to sign (string or JSON)")
    sg.set_defaults(func=cmd_sign)

    # verify
    vf = sub.add_parser("verify", help="Verify signature")
    vf.add_argument("data", help="Original data")
    vf.add_argument("signature", help="Signature JSON object")
    vf.add_argument("--public-key", help="Public key (base64), uses local if not provided")
    vf.set_defaults(func=cmd_verify)

    # kem-encap
    ke = sub.add_parser("kem-encap", help="KEM encapsulation (sender)")
    ke.add_argument("public_key", help="Recipient's public key (base64)")
    ke.set_defaults(func=cmd_kem_encap)

    # kem-decap
    kd = sub.add_parser("kem-decap", help="KEM decapsulation (receiver)")
    kd.add_argument("ciphertext", help="Ciphertext (base64)")
    kd.set_defaults(func=cmd_kem_decap)

    # status
    st = sub.add_parser("status", help="Show PQC status")
    st.set_defaults(func=cmd_status)

    # pubkey
    pk = sub.add_parser("pubkey", help="Export public key")
    pk.add_argument("--type", choices=["signature", "kem"], default="signature")
    pk.set_defaults(func=cmd_pubkey)

    return p


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())
