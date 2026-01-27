import assert from 'assert'
import fs from 'fs'
import path from 'path'
import { cmdKeygen, signMessage, verifySignedCommit, loadKeys } from '../iso_pqc.mjs'

// --- Test Framework ---
const TESTS = []

function test(name, fn) {
  TESTS.push({ name, fn })
}

async function runTests() {
  console.log(`Running ${TESTS.length} PQC tests...`)
  let passed = 0
  for (const t of TESTS) {
    try {
      console.log(`[TEST] ${t.name}...`)
      await t.fn()
      console.log(`[PASS] ${t.name}`)
      passed++
    } catch (err) {
      console.error(`[FAIL] ${t.name}`)
      console.error(err)
    }
  }
  console.log(`
Results: ${passed}/${TESTS.length} passed.`)
  if (passed !== TESTS.length) process.exit(1)
}

// --- Setup ---
// We need to point keyFilePath to a temp location to avoid overwriting real keys.
// iso_pqc.mjs uses: path.join(process.env.HOME || '/root', '.pluribus', 'secrets', 'pqc_keys.json')
// We can mock process.env.HOME.

const TEST_HOME = path.join(process.cwd(), '.tmp', `pqc-test-${Date.now()}`)
process.env.HOME = TEST_HOME

// --- Tests ---

test('should_generate_keys', async () => {
  cmdKeygen()
  const keys = loadKeys()
  assert.strictEqual(keys.algo, 'Ed25519-PQC-Placeholder')
  assert.ok(keys.publicKey, 'Missing public key')
  assert.ok(keys.secretKey, 'Missing secret key')
})

test('should_sign_and_verify_message', async () => {
  // Ensure keys exist
  try { loadKeys() } catch { cmdKeygen() }
  
  const message = 'Test Commit Message'
  const signedMessage = signMessage(message)
  
  assert.ok(signedMessage.includes(message), 'Message payload missing')
  assert.ok(signedMessage.includes('X-PQC-Algo: Ed25519-PQC-Placeholder'), 'Algo header missing')
  assert.ok(signedMessage.includes('X-PQC-Signature:'), 'Signature header missing')
  
  // Verification logic in iso_pqc.mjs is tied to git commits (verifySignedCommit)
  // or verifyData. Let's use verifyData if exported, or manually verify string.
  
  // iso_pqc exports verifyData!
  const { verifyData, loadKeys } = await import('../iso_pqc.mjs')
  const keys = loadKeys()
  
  // Parse the signed message to extract sig and payload
  const lines = signedMessage.split('\n')
  const sigLine = lines.find(l => l.startsWith('X-PQC-Signature: '))
  const signature = sigLine.split(': ')[1].trim()
  
  // The payload signed in signMessage is `commit:${message}`
  const payloadToVerify = `commit:${message}`
  
  const isValid = verifyData(payloadToVerify, { signature }, keys.publicKey)
  assert.strictEqual(isValid, true, 'Signature verification failed')
})

// --- Run ---
runTests()
