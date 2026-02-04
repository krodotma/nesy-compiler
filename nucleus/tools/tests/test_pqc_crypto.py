#!/usr/bin/env python3
"""Tests for pqc_crypto.py - Post-Quantum Cryptography wrapper.

Tests cover:
- Key generation (ML-DSA-65, ML-KEM-768)
- Signature operations
- KEM encapsulation/decapsulation
- Bus event signing
- Round-trip verification
"""
from __future__ import annotations

import base64
import json
import os
import subprocess
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

# Module under test
TOOLS_DIR = Path(__file__).resolve().parents[1]
PQC_CRYPTO = TOOLS_DIR / "pqc_crypto.py"


def run_pqc(env: dict[str, str], *args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ["python3", str(PQC_CRYPTO), *args],
        check=False,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        env=env,
    )


class TestPQCCryptoImport(unittest.TestCase):
    """Test module imports and availability."""

    def test_module_imports(self):
        """Module should import successfully."""
        import sys
        sys.path.insert(0, str(TOOLS_DIR))
        try:
            import pqc_crypto
            self.assertTrue(hasattr(pqc_crypto, "check_pqc_status"))
            self.assertTrue(hasattr(pqc_crypto, "generate_sig_keypair"))
            self.assertTrue(hasattr(pqc_crypto, "sign_data"))
            self.assertTrue(hasattr(pqc_crypto, "verify_signature"))
        finally:
            sys.path.remove(str(TOOLS_DIR))

    def test_pqc_status(self):
        """check_pqc_status should return valid status."""
        import sys
        sys.path.insert(0, str(TOOLS_DIR))
        try:
            from pqc_crypto import check_pqc_status
            status = check_pqc_status()
            self.assertIn("ready", status)
            self.assertIn("sig_algo", status)
            self.assertIn("kem_algo", status)
            # If pqcrypto is available, should be ready
            if status["ready"]:
                self.assertEqual(status["sig_algo"], "ML-DSA-65")
                self.assertEqual(status["kem_algo"], "ML-KEM-768")
        finally:
            sys.path.remove(str(TOOLS_DIR))


class TestPQCSignature(unittest.TestCase):
    """Test ML-DSA-65 signature operations."""

    def setUp(self):
        import sys
        sys.path.insert(0, str(TOOLS_DIR))
        from pqc_crypto import check_pqc_status
        status = check_pqc_status()
        if not status["ready"]:
            self.skipTest("PQC library not available")

    def tearDown(self):
        import sys
        if str(TOOLS_DIR) in sys.path:
            sys.path.remove(str(TOOLS_DIR))

    def test_generate_sig_keypair(self):
        """Should generate ML-DSA-65 keypair."""
        from pqc_crypto import generate_sig_keypair, ALGO_METADATA

        kp = generate_sig_keypair("ML-DSA-65")

        self.assertEqual(kp.algo, "ML-DSA-65")
        self.assertIsNotNone(kp.public_key)
        self.assertIsNotNone(kp.secret_key)
        self.assertEqual(len(kp.fingerprint), 16)

        # Verify key sizes
        pk_bytes = base64.b64decode(kp.public_key)
        sk_bytes = base64.b64decode(kp.secret_key)
        expected = ALGO_METADATA["ML-DSA-65"]
        self.assertEqual(len(pk_bytes), expected["public_key_size"])
        self.assertEqual(len(sk_bytes), expected["secret_key_size"])

    def test_sign_and_verify_string(self):
        """Should sign and verify string data."""
        from pqc_crypto import generate_sig_keypair, sign_data, verify_signature

        kp = generate_sig_keypair("ML-DSA-65")
        message = "Hello, Post-Quantum World!"

        sig = sign_data(message, kp.secret_key, "ML-DSA-65", kp.fingerprint)

        self.assertEqual(sig.algo, "ML-DSA-65")
        self.assertEqual(sig.signer, kp.fingerprint)
        self.assertIsNotNone(sig.signature)
        self.assertIsNotNone(sig.payload_hash)

        # Verify
        valid = verify_signature(message, sig.signature, kp.public_key, "ML-DSA-65")
        self.assertTrue(valid)

    def test_sign_and_verify_dict(self):
        """Should sign and verify dict data (JSON)."""
        from pqc_crypto import generate_sig_keypair, sign_data, verify_signature

        kp = generate_sig_keypair("ML-DSA-65")
        data = {"topic": "test.event", "data": {"foo": "bar"}, "count": 42}

        sig = sign_data(data, kp.secret_key, "ML-DSA-65", kp.fingerprint)
        valid = verify_signature(data, sig.signature, kp.public_key, "ML-DSA-65")
        self.assertTrue(valid)

    def test_verify_wrong_key_fails(self):
        """Verification should fail with wrong public key."""
        from pqc_crypto import generate_sig_keypair, sign_data, verify_signature

        kp1 = generate_sig_keypair("ML-DSA-65")
        kp2 = generate_sig_keypair("ML-DSA-65")  # Different keys

        message = "Test message"
        sig = sign_data(message, kp1.secret_key, "ML-DSA-65", kp1.fingerprint)

        # Verify with wrong key
        valid = verify_signature(message, sig.signature, kp2.public_key, "ML-DSA-65")
        self.assertFalse(valid)

    def test_verify_tampered_data_fails(self):
        """Verification should fail if data was tampered."""
        from pqc_crypto import generate_sig_keypair, sign_data, verify_signature

        kp = generate_sig_keypair("ML-DSA-65")
        original = "Original message"
        tampered = "Tampered message"

        sig = sign_data(original, kp.secret_key, "ML-DSA-65", kp.fingerprint)

        # Verify with tampered data
        valid = verify_signature(tampered, sig.signature, kp.public_key, "ML-DSA-65")
        self.assertFalse(valid)


class TestPQCKEM(unittest.TestCase):
    """Test ML-KEM-768 key encapsulation."""

    def setUp(self):
        import sys
        sys.path.insert(0, str(TOOLS_DIR))
        from pqc_crypto import check_pqc_status
        status = check_pqc_status()
        if not status["ready"]:
            self.skipTest("PQC library not available")

    def tearDown(self):
        import sys
        if str(TOOLS_DIR) in sys.path:
            sys.path.remove(str(TOOLS_DIR))

    def test_generate_kem_keypair(self):
        """Should generate ML-KEM-768 keypair."""
        from pqc_crypto import generate_kem_keypair, ALGO_METADATA

        kp = generate_kem_keypair("ML-KEM-768")

        self.assertEqual(kp.algo, "ML-KEM-768")
        self.assertIsNotNone(kp.public_key)
        self.assertIsNotNone(kp.secret_key)

        # Verify key sizes
        pk_bytes = base64.b64decode(kp.public_key)
        sk_bytes = base64.b64decode(kp.secret_key)
        expected = ALGO_METADATA["ML-KEM-768"]
        self.assertEqual(len(pk_bytes), expected["public_key_size"])
        self.assertEqual(len(sk_bytes), expected["secret_key_size"])

    def test_kem_encapsulate_decapsulate(self):
        """KEM encapsulation and decapsulation should produce same shared secret."""
        from pqc_crypto import generate_kem_keypair, kem_encapsulate, kem_decapsulate

        # Receiver generates keypair
        receiver_kp = generate_kem_keypair("ML-KEM-768")

        # Sender encapsulates
        encap = kem_encapsulate(receiver_kp.public_key)
        sender_ss = base64.b64decode(encap.shared_secret)

        # Receiver decapsulates
        receiver_ss = kem_decapsulate(encap.ciphertext, receiver_kp.secret_key)

        # Shared secrets should match
        self.assertEqual(sender_ss, receiver_ss)
        self.assertEqual(len(sender_ss), 32)  # 256-bit shared secret


class TestBusIntegration(unittest.TestCase):
    """Test bus event signing integration."""

    def setUp(self):
        import sys
        sys.path.insert(0, str(TOOLS_DIR))
        from pqc_crypto import check_pqc_status
        status = check_pqc_status()
        if not status["ready"]:
            self.skipTest("PQC library not available")

        # Create temp directory for keys
        self.tmpdir = tempfile.mkdtemp()
        self.original_env = os.environ.get("PLURIBUS_SECRETS_DIR")
        os.environ["PLURIBUS_SECRETS_DIR"] = self.tmpdir

    def tearDown(self):
        import sys
        if str(TOOLS_DIR) in sys.path:
            sys.path.remove(str(TOOLS_DIR))
        if self.original_env:
            os.environ["PLURIBUS_SECRETS_DIR"] = self.original_env
        else:
            os.environ.pop("PLURIBUS_SECRETS_DIR", None)
        import shutil
        shutil.rmtree(self.tmpdir, ignore_errors=True)

    def test_sign_and_verify_bus_event(self):
        """Should sign and verify bus event."""
        from pqc_crypto import (
            generate_sig_keypair,
            save_keys,
            sign_bus_event,
            verify_bus_event,
        )

        # Generate and save keys
        kp = generate_sig_keypair("ML-DSA-65")
        save_keys(kp)

        # Create event
        event = {
            "topic": "test.pqc.event",
            "kind": "artifact",
            "data": {"message": "Hello PQC"},
        }

        # Sign
        signed = sign_bus_event(event)
        self.assertEqual(signed.event, event)
        self.assertEqual(signed.pqc_signature.algo, "ML-DSA-65")

        # Verify
        valid = verify_bus_event(signed)
        self.assertTrue(valid)

    def test_emit_pqc_signed_event(self):
        """Should create properly formatted signed event for bus."""
        from pqc_crypto import (
            generate_sig_keypair,
            save_keys,
            emit_pqc_signed_event,
            verify_bus_event,
        )

        # Generate and save keys
        kp = generate_sig_keypair("ML-DSA-65")
        save_keys(kp)

        # Emit signed event
        payload = emit_pqc_signed_event(
            topic="pqc.test.signed",
            data={"test": True, "value": 42},
        )

        # Should have expected structure
        self.assertIn("event", payload)
        self.assertIn("pqc_signature", payload)
        self.assertEqual(payload["event"]["topic"], "pqc.test.signed")
        self.assertEqual(payload["pqc_signature"]["algo"], "ML-DSA-65")

        # Should verify
        valid = verify_bus_event(payload)
        self.assertTrue(valid)


class TestCLI(unittest.TestCase):
    """Test CLI interface."""

    def test_status_command(self):
        """status command should return JSON."""
        with tempfile.TemporaryDirectory() as tmp:
            env = {
                **os.environ,
                "PLURIBUS_SECRETS_DIR": tmp,
                "PYTHONDONTWRITEBYTECODE": "1",
            }
            result = run_pqc(env, "status")
            self.assertEqual(result.returncode, 0, result.stderr)
            # Parse the JSON object (first 9 lines are the JSON)
            lines = result.stdout.strip().split("\n")
            json_lines = []
            for line in lines:
                json_lines.append(line)
                if line.strip() == "}":
                    break
            status = json.loads("\n".join(json_lines))
            self.assertIn("ready", status)

    def test_keygen_and_sign_verify_roundtrip(self):
        """Full keygen -> sign -> verify workflow via CLI."""
        with tempfile.TemporaryDirectory() as tmp:
            env = {
                **os.environ,
                "PLURIBUS_SECRETS_DIR": tmp,
                "PYTHONDONTWRITEBYTECODE": "1",
            }

            # Generate keys
            keygen = run_pqc(env, "keygen")
            self.assertEqual(keygen.returncode, 0, keygen.stderr)
            self.assertIn("ML-DSA-65", keygen.stdout)

            # Sign data
            message = "Test message for CLI"
            sign = run_pqc(env, "sign", message)
            self.assertEqual(sign.returncode, 0, sign.stderr)
            sig_obj = json.loads(sign.stdout)
            self.assertIn("signature", sig_obj)
            self.assertIn("algo", sig_obj)

            # Verify signature
            verify = run_pqc(env, "verify", message, json.dumps(sig_obj))
            self.assertEqual(verify.returncode, 0, verify.stderr)
            self.assertIn("SUCCESS", verify.stdout)

    def test_keygen_with_kem(self):
        """keygen --with-kem should generate both key types."""
        with tempfile.TemporaryDirectory() as tmp:
            env = {
                **os.environ,
                "PLURIBUS_SECRETS_DIR": tmp,
                "PYTHONDONTWRITEBYTECODE": "1",
            }

            keygen = run_pqc(env, "keygen", "--with-kem")
            self.assertEqual(keygen.returncode, 0, keygen.stderr)
            self.assertIn("ML-DSA-65", keygen.stdout)
            self.assertIn("ML-KEM-768", keygen.stdout)

            # Check keys file
            keys_path = Path(tmp) / "pqc_keys.json"
            self.assertTrue(keys_path.exists())
            keys = json.loads(keys_path.read_text())
            self.assertIn("signature", keys)
            self.assertIn("kem", keys)

    def test_kem_encap_decap_roundtrip(self):
        """KEM encapsulation/decapsulation via CLI."""
        with tempfile.TemporaryDirectory() as tmp:
            env = {
                **os.environ,
                "PLURIBUS_SECRETS_DIR": tmp,
                "PYTHONDONTWRITEBYTECODE": "1",
            }

            # Generate keys with KEM
            keygen = run_pqc(env, "keygen", "--with-kem")
            self.assertEqual(keygen.returncode, 0, keygen.stderr)

            # Get public key
            pubkey = run_pqc(env, "pubkey", "--type", "kem")
            self.assertEqual(pubkey.returncode, 0, pubkey.stderr)
            pk_obj = json.loads(pubkey.stdout)
            pk_b64 = pk_obj["public_key"]

            # Encapsulate
            encap = run_pqc(env, "kem-encap", pk_b64)
            self.assertEqual(encap.returncode, 0, encap.stderr)
            encap_obj = json.loads(encap.stdout)
            sender_ss = encap_obj["shared_secret"]
            ciphertext = encap_obj["ciphertext"]

            # Decapsulate
            decap = run_pqc(env, "kem-decap", ciphertext)
            self.assertEqual(decap.returncode, 0, decap.stderr)
            decap_obj = json.loads(decap.stdout)
            receiver_ss = decap_obj["shared_secret"]

            # Shared secrets should match
            self.assertEqual(sender_ss, receiver_ss)


class TestFALCON(unittest.TestCase):
    """Test FALCON-512 (optional) signature algorithm."""

    def setUp(self):
        import sys
        sys.path.insert(0, str(TOOLS_DIR))
        from pqc_crypto import check_pqc_status
        status = check_pqc_status()
        if not status["ready"]:
            self.skipTest("PQC library not available")

    def tearDown(self):
        import sys
        if str(TOOLS_DIR) in sys.path:
            sys.path.remove(str(TOOLS_DIR))

    def test_falcon_sign_verify(self):
        """FALCON-512 should sign and verify."""
        from pqc_crypto import generate_sig_keypair, sign_data, verify_signature

        kp = generate_sig_keypair("FALCON-512")
        self.assertEqual(kp.algo, "FALCON-512")

        message = "FALCON test message"
        sig = sign_data(message, kp.secret_key, "FALCON-512", kp.fingerprint)

        valid = verify_signature(message, sig.signature, kp.public_key, "FALCON-512")
        self.assertTrue(valid)


if __name__ == "__main__":
    unittest.main()
