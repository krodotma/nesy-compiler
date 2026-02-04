#!/usr/bin/env node
/**
 * Tests for iso_pqc.mjs - PQC Git Router
 * Run: node pluribus_next/tools/tests/test_iso_pqc.mjs
 */

import fs from 'fs'
import os from 'os'
import path from 'path'
import { fileURLToPath } from 'url'
import git from 'isomorphic-git'
import { cmdKeygen, commitSigned, status as pqcStatus, verifySignedCommit } from '../iso_pqc.mjs'

const __dirname = path.dirname(fileURLToPath(import.meta.url))

let testDir = ''
let homeDir = '' // Mock home for keys
let passed = 0
let failed = 0

async function setup() {
  testDir = fs.mkdtempSync(path.join(os.tmpdir(), 'iso_pqc_test_'))
  homeDir = fs.mkdtempSync(path.join(os.tmpdir(), 'iso_pqc_home_'))
  
  // Initialize repo
  await git.init({ fs, dir: testDir })
  fs.writeFileSync(path.join(testDir, 'file.txt'), 'content')
  process.env.HOME = homeDir
  process.env.PLURIBUS_BUS_DIR = path.join(testDir, '.bus')
  
  console.log(`Test Env: Dir=${testDir}, Home=${homeDir}`)
}

function teardown() {
  if (testDir) fs.rmSync(testDir, { recursive: true, force: true })
  if (homeDir) fs.rmSync(homeDir, { recursive: true, force: true })
}

function assert(cond, msg) {
  if (!cond) throw new Error(msg || 'Assertion failed')
}

function assertIncludes(str, substr, msg) {
  if (!str.includes(substr)) {
    throw new Error(msg || `Expected output to include "${substr}", got "${str}"`) 
  }
}

// --- Tests ---

console.log('\n=== iso_pqc.mjs Tests (Aleatoric & Epistemic) ===\n')

await setup()

// 1. Initialization (Epistemic Foundation)
try {
  console.log('Test: Keygen (Epistemic Foundation)')
  cmdKeygen()
  assert(fs.existsSync(path.join(homeDir, '.pluribus/secrets/pqc_keys.json')), 'Keys not created')
  console.log('  [PASS]')
  passed++
} catch (e) { console.log(`  [FAIL] ${e.message}`); failed++ }

// 2. Commit Flow (Normal)
try {
  console.log('Test: Signed Commit')
  await commitSigned(testDir, 'feat: secure commit')
  
  // Verify it reached git
  const commits = await git.log({ fs, dir: testDir, depth: 10 })
  assert(commits.length > 0, 'No commits in log')
  assertIncludes(commits[0].commit.message, 'feat: secure commit', 'Commit not found in log')
  console.log('  [PASS]')
  passed++
} catch (e) { console.log(`  [FAIL] ${e.message}`); failed++ }

// 3. Verify Flow (Success)
try {
  console.log('Test: Verify Signature')
  const res = await verifySignedCommit(testDir)
  assert(res && res.status === 'SUCCESS', 'Verification failed')
  assert(res.algo === 'Ed25519-PQC-Placeholder', 'Wrong algo')
  console.log('  [PASS]')
  passed++
} catch (e) { console.log(`  [FAIL] ${e.message}`); failed++ }

// 4. Aleatoric Test: Tampering (Chaos)
try {
  console.log('Test: Tampered Signature (Aleatoric)')
  // We need to modify the commit message in .git directly to simulate tampering?
  // Or just modify the key file?
  // Easier: Create a commit, then tamper with the file content? 
  // No, git hash protects content. We verify the *signature* of the message.
  // If we change the message, git hash changes.
  
  // Let's simulate a "Wrong Key" scenario (Aleatoric key rotation/loss)
  // Rename the key file
  const keyPath = path.join(homeDir, '.pluribus/secrets/pqc_keys.json')
  const backup = path.join(homeDir, 'keys.bak')
  fs.renameSync(keyPath, backup)
  
  // Regenerate NEW keys (Attacker trying to verify or Signer lost keys)
  cmdKeygen()
  
  const res = await verifySignedCommit(testDir)
  // Should fail because the commit was signed by OLD key, but we have NEW key (fingerprint mismatch)
  // Our code says: "Verification: UNTRUSTED"
  assert(res && res.status === 'UNTRUSTED', 'Should detect unknown signer')
  
  // Restore keys
  fs.rmSync(keyPath)
  fs.renameSync(backup, keyPath)
  console.log('  [PASS]')
  passed++
} catch (e) { console.log(`  [FAIL] ${e.message}`); failed++ }

// 5. Epistemic Gap: Missing Keys
try {
  console.log('Test: Missing Keys (Epistemic Gap)')
  // Hide keys
  const keyPath = path.join(homeDir, '.pluribus/secrets/pqc_keys.json')
  const backup = path.join(homeDir, 'keys_hidden.json')
  fs.renameSync(keyPath, backup)
  
  // Try to commit
  let threw = false
  try {
    await commitSigned(testDir, 'fail me')
  } catch (err) {
    threw = true
    assertIncludes(String(err?.message || err), 'Epistemic Failure', 'Should report epistemic failure')
  }
  assert(threw, 'Commit should fail without keys')
  
  // Restore
  fs.renameSync(backup, keyPath)
  console.log('  [PASS]')
  passed++
} catch (e) { console.log(`  [FAIL] ${e.message}`); failed++ }

// 6. Read-Only Delegation (Status)
try {
  console.log('Test: Status Delegation')
  await pqcStatus(testDir)
  // We expect empty output or clean status
  console.log('  [PASS]')
  passed++
} catch (e) { console.log(`  [FAIL] ${e.message}`); failed++ }

teardown()

console.log(`\n=== Results: ${passed} passed, ${failed} failed ===\n`)
process.exit(failed > 0 ? 1 : 0)
