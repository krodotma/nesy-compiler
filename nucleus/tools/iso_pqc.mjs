import fs from 'fs'
import path from 'path'
import crypto from 'crypto'
import git from 'isomorphic-git'
import { pathToFileURL } from 'url'

// --- Real Post-Quantum Cryptography via ML-DSA-65 (FIPS 204) ---
// @noble/post-quantum provides audited ML-DSA-65 (formerly CRYSTALS-Dilithium3)
import { ml_dsa65 } from '@noble/post-quantum/ml-dsa.js'

// --- Configuration ---
function keyFilePath() {
  return path.join(process.env.HOME || '/root', '.pluribus', 'secrets', 'pqc_keys.json')
}

// --- PQC Readiness ---
export function checkPQCReadiness() {
  return {
    ready: true,
    algo: 'ML-DSA-65',
    mode: 'FIPS-204',
    library: '@noble/post-quantum',
    version: '0.5.2',
    note: 'Real Post-Quantum signatures via NIST FIPS 204 ML-DSA-65'
  }
}

// --- Crypto Abstraction Layer (CAL) for ML-DSA-65 ---

/**
 * Generate ML-DSA-65 keypair with cryptographically secure randomness.
 * ML-DSA-65 provides Level 3 security (~192-bit classical, ~128-bit quantum).
 *
 * Key sizes:
 * - Public key: 1952 bytes
 * - Secret key: 4032 bytes
 * - Signature: 3309 bytes
 */
function generateKeyPair() {
  // Use cryptographically secure random seed
  const seed = crypto.randomBytes(32)
  const { publicKey, secretKey } = ml_dsa65.keygen(seed)

  return {
    publicKey: Buffer.from(publicKey).toString('base64'),
    secretKey: Buffer.from(secretKey).toString('base64'),
    seed: seed.toString('base64')  // Store seed for deterministic regen if needed
  }
}

/**
 * Sign message using ML-DSA-65.
 * API: ml_dsa65.sign(message, secretKey) -> signature
 */
function sign(data, secretKeyBase64) {
  const secretKey = new Uint8Array(Buffer.from(secretKeyBase64, 'base64'))
  const message = new TextEncoder().encode(data)
  const signature = ml_dsa65.sign(message, secretKey)
  return Buffer.from(signature).toString('base64')
}

/**
 * Verify ML-DSA-65 signature.
 * API: ml_dsa65.verify(signature, message, publicKey) -> boolean
 */
function verify(data, signatureBase64, publicKeyBase64) {
  try {
    const publicKey = new Uint8Array(Buffer.from(publicKeyBase64, 'base64'))
    const signature = new Uint8Array(Buffer.from(signatureBase64, 'base64'))
    const message = new TextEncoder().encode(data)
    return ml_dsa65.verify(signature, message, publicKey)
  } catch (err) {
    console.error('Verify error:', err.message)
    return false
  }
}

/**
 * Generate fingerprint from public key.
 * Uses SHA-256 truncated to 16 hex chars for human readability.
 */
function getFingerprint(publicKeyBase64) {
  const hash = crypto.createHash('sha256')
  const publicKey = Buffer.from(publicKeyBase64, 'base64')
  hash.update(publicKey)
  return hash.digest('hex').slice(0, 16)
}

// --- Key Management ---

export function cmdKeygen() {
  const keyFile = keyFilePath()
  const secretsDir = path.dirname(keyFile)
  if (!fs.existsSync(secretsDir)) {
    fs.mkdirSync(secretsDir, { recursive: true })
  }

  const keypair = generateKeyPair()
  const keys = {
    algo: 'ML-DSA-65',
    version: 'FIPS-204',
    fingerprint: getFingerprint(keypair.publicKey),
    publicKey: keypair.publicKey,
    secretKey: keypair.secretKey,
    seed: keypair.seed,
    created: new Date().toISOString(),
    keySizes: {
      publicKey: Buffer.from(keypair.publicKey, 'base64').length,
      secretKey: Buffer.from(keypair.secretKey, 'base64').length
    }
  }

  fs.writeFileSync(keyFile, JSON.stringify(keys, null, 2), { mode: 0o600 })
  console.log(`ML-DSA-65 keys generated at ${keyFile}`)
  console.log(`Fingerprint: ${keys.fingerprint}`)
  console.log(`Public key size: ${keys.keySizes.publicKey} bytes`)
  console.log(`Secret key size: ${keys.keySizes.secretKey} bytes`)
  return keys
}

export function loadKeys() {
  const keyFile = keyFilePath()
  if (!fs.existsSync(keyFile)) {
    throw new Error(`No keys found at ${keyFile}. Run 'keygen' first.`)
  }
  return JSON.parse(fs.readFileSync(keyFile, 'utf-8'))
}

// --- Signing & Verification ---

export function signMessage(message) {
  const keys = loadKeys()
  const payloadToSign = `commit:${message}`
  const sig = sign(payloadToSign, keys.secretKey)

  return `${message}

X-PQC-Algo: ${keys.algo}
X-PQC-Signer: ${keys.fingerprint}
X-PQC-Signature: ${sig}`
}

/**
 * Sign arbitrary data (not just commit messages).
 * Returns signature object for embedding in events/artifacts.
 */
export function signData(data) {
  const keys = loadKeys()
  const payload = typeof data === 'string' ? data : JSON.stringify(data)
  const sig = sign(payload, keys.secretKey)

  return {
    algo: keys.algo,
    signer: keys.fingerprint,
    signature: sig,
    payload_hash: crypto.createHash('sha256').update(payload).digest('hex')
  }
}

/**
 * Verify arbitrary signed data.
 */
export function verifyData(data, signatureObj, publicKeyBase64) {
  const payload = typeof data === 'string' ? data : JSON.stringify(data)
  return verify(payload, signatureObj.signature, publicKeyBase64)
}

export async function verifySignedCommit(dir, ref = 'HEAD') {
  try {
    const commits = await git.log({ fs, dir, depth: 1, ref })
    if (!commits || commits.length === 0) {
      return { ok: false, status: 'NO_COMMITS' }
    }

    const commit = commits[0]
    const message = commit.commit.message

    const algoMatch = message.match(/X-PQC-Algo: (.*)/)
    const signerMatch = message.match(/X-PQC-Signer: (.*)/)
    const sigMatch = message.match(/X-PQC-Signature: (.*)/)

    if (!algoMatch || !sigMatch || !signerMatch) {
      return { ok: true, status: 'UNSEGMENTED' }
    }

    const algo = algoMatch[1].trim()
    const signer = signerMatch[1].trim()
    const signature = sigMatch[1].trim()

    const payloadEndIndex = message.indexOf('\nX-PQC-Algo:')
    const originalMessage = message.substring(0, payloadEndIndex).trim()
    const payloadToVerify = `commit:${originalMessage}`

    let keys
    try {
      keys = loadKeys()
    } catch (e) {
      return { ok: false, status: 'NO_KEYS' }
    }

    if (keys.fingerprint !== signer) {
      return { ok: false, status: 'UNTRUSTED', signer, algo }
    }

    const isValid = verify(payloadToVerify, signature, keys.publicKey)

    if (isValid) {
      return { ok: true, status: 'SUCCESS', signer, algo, payload: originalMessage }
    } else {
      return { ok: false, status: 'BAD_SIGNATURE', signer, algo }
    }

  } catch (err) {
    return { ok: false, status: 'ERROR', error: err?.message || String(err) }
  }
}

// --- CLI Entry ---

function printVerifyResult(r) {
  if (!r || typeof r !== 'object') {
    console.error('Verification: ERROR (Invalid result)')
    process.exit(1)
  }
  if (r.status === 'SUCCESS') {
    console.log(`Verification: SUCCESS (${r.algo})`)
    console.log(`Signer: ${r.signer}`)
    console.log(`Payload: "${r.payload}"`)
    return
  }
  if (r.status === 'UNTRUSTED') {
    console.log(`Verification: UNTRUSTED (Signer ${r.signer} not in local keyring)`)
    return
  }
  if (r.status === 'NO_KEYS') {
    console.log('Verification: FAILURE (No local keys to verify against)')
    return
  }
  if (r.status === 'UNSEGMENTED') {
    console.log('Verification: UNSEGMENTED (No PQC headers found)')
    return
  }
  if (r.status === 'BAD_SIGNATURE') {
    console.error('Verification: FAILURE (Signature Mismatch)')
    process.exit(1)
  }
  if (r.status === 'NO_COMMITS') {
    console.error('No commits found')
    process.exit(1)
  }
  console.error(`Verification Error: ${r.error || r.status}`)
  process.exit(1)
}

async function main(argv) {
  const args = argv || process.argv.slice(2)
  const command = args[0]

  if (command === 'keygen') {
    cmdKeygen()
    return
  }
  if (command === 'verify') {
    const dir = args[1] || '.'
    const r = await verifySignedCommit(dir)
    printVerifyResult(r)
    return
  }
  if (command === 'status') {
    const status = checkPQCReadiness()
    console.log(JSON.stringify(status, null, 2))
    return
  }
  if (command === 'sign') {
    const data = args[1]
    if (!data) {
      console.error('Usage: node iso_pqc.mjs sign <data>')
      process.exit(1)
    }
    const signed = signData(data)
    console.log(JSON.stringify(signed, null, 2))
    return
  }
  console.log('Usage: node iso_pqc.mjs <keygen|verify|status|sign> ...')
  console.log('')
  console.log('Commands:')
  console.log('  keygen         Generate new ML-DSA-65 keypair')
  console.log('  verify [dir]   Verify HEAD commit signature')
  console.log('  status         Show PQC readiness status')
  console.log('  sign <data>    Sign arbitrary data')
}

const isMain = import.meta.url === pathToFileURL(path.resolve(process.argv[1] || '')).href
if (isMain) {
  main()
}
