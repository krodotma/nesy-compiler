import crypto from 'crypto';

export const pqc = {
  sign: (data) => {
    return `pqc_signed_${Buffer.from(data).toString('base64').slice(0, 16)}`;
  },
  verify: (data, signature) => {
    return signature.startsWith('pqc_signed_');
  }
};

export function signMessage(message) {
  // Placeholder: In a real PQC implementation, this would use the private key.
  // For the stub, we just return the message or a "signed" wrapper.
  // Ideally, iso_git might expect it to return just the signature or the signed payload.
  // Looking at iso_git usage: finalMessage = signMessage(message)
  // So it returns the signed message content.
  return `${message}\n\n-----BEGIN PQC SIGNATURE-----\n${pqc.sign(message)}\n-----END PQC SIGNATURE-----`;
}

export async function verifySignedCommit(dir, oid) {
  // Placeholder verification
  return {
    ok: true,
    status: 'verified_stub',
    signer: 'pluribus_stub',
    algo: 'Dilithium3-stub'
  };
}

export const identity = {
  get: () => {
    const actor = process.env.PLURIBUS_ACTOR || 'unknown';
    return {
      actor,
      timestamp: Date.now(),
      vps_id: '69.169.104.17'
    };
  }
};
