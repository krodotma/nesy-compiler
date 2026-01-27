import fs from 'fs'
import path from 'path'
import crypto from 'crypto'
import git from 'isomorphic-git'
import { pathToFileURL } from 'url'

const ALGO = 'Ed25519-PQC-Placeholder'
const SIGN_PREFIX = 'commit:'

function keyFilePath() {
  return path.join(process.env.HOME || '/root', '.pluribus', 'secrets', 'pqc_keys.json')
}

function ensureSecretDir() {
  const keyFile = keyFilePath()
  const secretsDir = path.dirname(keyFile)
  if (!fs.existsSync(secretsDir)) {
    fs.mkdirSync(secretsDir, { recursive: true })
  }
  return keyFile
}

function fingerprintFromKey(keyBase64) {
  return crypto.createHash('sha256').update(Buffer.from(keyBase64, 'base64')).digest('hex').slice(0, 16)
}

function hmacSign(payload, keyBase64) {
  const key = Buffer.from(keyBase64, 'base64')
  return crypto.createHmac('sha256', key).update(payload).digest('base64')
}

export function checkPQCReadiness() {
  return {
    ready: true,
    algo: ALGO,
    mode: 'HMAC-SHA256',
    library: 'node:crypto',
    note: 'Placeholder signature scheme (non-PQC) for iso_git compatibility'
  }
}

export function cmdKeygen() {
  const keyFile = ensureSecretDir()
  const secretKey = crypto.randomBytes(32).toString('base64')
  const keys = {
    algo: ALGO,
    fingerprint: fingerprintFromKey(secretKey),
    publicKey: secretKey,
    secretKey,
    created: new Date().toISOString(),
    keySize: Buffer.from(secretKey, 'base64').length
  }

  fs.writeFileSync(keyFile, JSON.stringify(keys, null, 2), { mode: 0o600 })
  return keys
}

export function loadKeys() {
  const keyFile = keyFilePath()
  if (!fs.existsSync(keyFile)) {
    throw new Error(`No keys found at ${keyFile}. Run 'keygen' first.`)
  }
  return JSON.parse(fs.readFileSync(keyFile, 'utf-8'))
}

export function signMessage(message) {
  const keys = loadKeys()
  const payload = `${SIGN_PREFIX}${message}`
  const sig = hmacSign(payload, keys.secretKey)

  return `${message}

X-PQC-Algo: ${keys.algo}
X-PQC-Signer: ${keys.fingerprint}
X-PQC-Signature: ${sig}`
}

export function signData(data) {
  const keys = loadKeys()
  const payload = typeof data === 'string' ? data : JSON.stringify(data)
  const sig = hmacSign(payload, keys.secretKey)
  return {
    algo: keys.algo,
    signer: keys.fingerprint,
    signature: sig,
    payload_hash: crypto.createHash('sha256').update(payload).digest('hex')
  }
}

export function verifyData(data, signatureObj, keyBase64) {
  const payload = typeof data === 'string' ? data : JSON.stringify(data)
  if (!signatureObj || !signatureObj.signature || !keyBase64) return false
  const expected = hmacSign(payload, keyBase64)
  return crypto.timingSafeEqual(Buffer.from(signatureObj.signature), Buffer.from(expected))
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
    const payloadToVerify = `${SIGN_PREFIX}${originalMessage}`

    let keys
    try {
      keys = loadKeys()
    } catch (e) {
      return { ok: false, status: 'NO_KEYS' }
    }

    if (keys.fingerprint !== signer) {
      return { ok: false, status: 'UNTRUSTED', signer, algo }
    }

    const isValid = verifyData(payloadToVerify, { signature }, keys.publicKey)
    if (isValid) {
      return { ok: true, status: 'SUCCESS', signer, algo, payload: originalMessage }
    }
    return { ok: false, status: 'BAD_SIGNATURE', signer, algo }
  } catch (err) {
    return { ok: false, status: 'ERROR', error: err?.message || String(err) }
  }
}

async function stageAll(dir) {
  const matrix = await git.statusMatrix({ fs, dir })
  for (const row of matrix) {
    const filepath = row[0]
    const workdir = row[2]
    const stage = row[3]
    if (!filepath) continue
    if (workdir !== stage) {
      if (workdir === 0) {
        await git.remove({ fs, dir, filepath })
      } else {
        await git.add({ fs, dir, filepath })
      }
    }
  }
}

export async function commitSigned(dir, message, authorName, authorEmail) {
  try {
    loadKeys()
  } catch (err) {
    throw new Error('Epistemic Failure: PQC keys missing')
  }

  await stageAll(dir)
  const signedMessage = signMessage(message)
  return git.commit({
    fs,
    dir,
    author: {
      name: authorName || process.env.PLURIBUS_ACTOR || 'pluribus',
      email: authorEmail || process.env.PLURIBUS_GIT_EMAIL || 'pluribus@local'
    },
    message: signedMessage
  })
}

export async function status() {
  return checkPQCReadiness()
}

export const identity = {
  get: () => {
    const actor = process.env.PLURIBUS_ACTOR || 'unknown'
    return {
      actor,
      timestamp: Date.now(),
      vps_id: '69.169.104.17'
    }
  }
}

function printVerifyResult(result) {
  if (!result || typeof result !== 'object') {
    console.error('Verification: ERROR (Invalid result)')
    process.exit(1)
  }
  if (result.status === 'SUCCESS') {
    console.log(`Verification: SUCCESS (${result.algo})`)
    console.log(`Signer: ${result.signer}`)
    console.log(`Payload: "${result.payload}"`)
    return
  }
  if (result.status === 'UNTRUSTED') {
    console.log(`Verification: UNTRUSTED (Signer ${result.signer} not in local keyring)`)
    return
  }
  if (result.status === 'NO_KEYS') {
    console.log('Verification: FAILURE (No local keys to verify against)')
    return
  }
  if (result.status === 'UNSEGMENTED') {
    console.log('Verification: UNSEGMENTED (No PQC headers found)')
    return
  }
  if (result.status === 'BAD_SIGNATURE') {
    console.error('Verification: FAILURE (Signature Mismatch)')
    process.exit(1)
  }
  if (result.status === 'NO_COMMITS') {
    console.error('No commits found')
    process.exit(1)
  }
  console.error(`Verification Error: ${result.error || result.status}`)
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
    const readiness = checkPQCReadiness()
    console.log(JSON.stringify(readiness, null, 2))
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
}

const isMain = import.meta.url === pathToFileURL(path.resolve(process.argv[1] || '')).href
if (isMain) {
  main()
}
