#!/usr/bin/env node
/**
 * Tests for iso_git.mjs - isomorphic-git wrapper
 * Run: node pluribus_next/tools/tests/test_iso_git.mjs
 */

import { spawnSync } from 'child_process'
import fs from 'fs'
import os from 'os'
import path from 'path'
import { fileURLToPath } from 'url'
import git from 'isomorphic-git'

const __dirname = path.dirname(fileURLToPath(import.meta.url))
const ISO_GIT = path.resolve(__dirname, '..', 'iso_git.mjs')

let testDir = ''
let busDir = ''
let passed = 0
let failed = 0

function setup() {
  testDir = fs.mkdtempSync(path.join(os.tmpdir(), 'iso_git_test_'))
  busDir = fs.mkdtempSync(path.join(os.tmpdir(), 'iso_git_bus_'))
  console.log(`Test directory: ${testDir}`)
}

function teardown() {
  if (testDir && fs.existsSync(testDir)) {
    fs.rmSync(testDir, { recursive: true, force: true })
  }
  if (busDir && fs.existsSync(busDir)) {
    fs.rmSync(busDir, { recursive: true, force: true })
  }
}

function run(cmd, args = []) {
  const result = spawnSync('node', [ISO_GIT, cmd, testDir, ...args], {
    encoding: 'utf-8',
    env: { ...process.env, PLURIBUS_BUS_DIR: busDir }
  })
  return {
    stdout: result.stdout?.trim() || '',
    stderr: result.stderr?.trim() || '',
    status: result.status
  }
}

function test(name, fn) {
  try {
    const r = fn()
    if (r && typeof r.then === 'function') {
      throw new Error('Test returned a Promise but was not awaited')
    }
    console.log(`  [PASS] ${name}`)
    passed++
  } catch (err) {
    console.log(`  [FAIL] ${name}: ${err.message}`)
    failed++
  }
}

async function testAsync(name, fn) {
  try {
    await fn()
    console.log(`  [PASS] ${name}`)
    passed++
  } catch (err) {
    console.log(`  [FAIL] ${name}: ${err.message}`)
    failed++
  }
}

function assert(cond, msg) {
  if (!cond) throw new Error(msg || 'Assertion failed')
}

function assertIncludes(str, substr, msg) {
  if (!str.includes(substr)) {
    throw new Error(msg || `Expected "${str}" to include "${substr}"`)
  }
}

function parseCommitSha(output) {
  const m = String(output || '').match(/\[([0-9a-f]{7,40})\]/i)
  return m ? m[1] : null
}

function readBusEvents(busDir) {
  const eventsFile = path.join(busDir, 'events.ndjson')
  if (!fs.existsSync(eventsFile)) return []
  const raw = fs.readFileSync(eventsFile, 'utf-8').trim().split('\n').filter(Boolean)
  return raw.map(line => JSON.parse(line))
}

function lastCommitShaFromBus(busDir, message) {
  const events = readBusEvents(busDir)
  const commits = events.filter(e => e.topic === 'git.commit' && (!message || e?.data?.message === message))
  const last = commits[commits.length - 1]
  return last?.data?.sha || null
}

function createIsolatedRepo() {
  const dir = fs.mkdtempSync(path.join(os.tmpdir(), 'iso_git_hgt_'))
  const bus = fs.mkdtempSync(path.join(os.tmpdir(), 'iso_git_hgt_bus_'))
  const runLocal = (cmd, args = []) => {
    const result = spawnSync('node', [ISO_GIT, cmd, dir, ...args], {
      encoding: 'utf-8',
      env: { ...process.env, PLURIBUS_BUS_DIR: bus, HOME: dir }
    })
    return {
      stdout: result.stdout?.trim() || '',
      stderr: result.stderr?.trim() || '',
      status: result.status
    }
  }
  // Initialize PQC keys for this isolated environment
  const PQC_TOOL = path.resolve(__dirname, '..', 'iso_pqc.mjs')
  spawnSync('node', [PQC_TOOL, 'keygen'], { env: { ...process.env, HOME: dir } })

  const cleanup = () => {
    try { fs.rmSync(dir, { recursive: true, force: true }) } catch {}
    try { fs.rmSync(bus, { recursive: true, force: true }) } catch {}
  }
  return { dir, bus, run: runLocal, cleanup }
}

// --- Tests ---

async function main() {
  console.log('\n=== iso_git.mjs Tests ===\n')
  setup()

test('init creates .git directory', () => {
  const result = run('init')
  assert(result.status === 0, `Exit code: ${result.status}`)
  assert(fs.existsSync(path.join(testDir, '.git')), '.git should exist')
})

await testAsync('commit creates a commit', async () => {
  // Create a file first
  fs.writeFileSync(path.join(testDir, 'test.txt'), 'hello world')

  const result = run('commit', ['Initial commit'])
  assert(result.status === 0, `Exit code: ${result.status}, stderr: ${result.stderr}`)
  const log = await git.log({ fs, dir: testDir, depth: 5 })
  assert(log.some(c => c.commit.message.includes('Initial commit')), 'Commit message should be present in repo history')
})

await testAsync('commit-paths only commits selected files', async () => {
  fs.writeFileSync(path.join(testDir, 'a.txt'), 'a1')
  fs.writeFileSync(path.join(testDir, 'b.txt'), 'b1')
  run('commit', ['base a+b'])

  fs.writeFileSync(path.join(testDir, 'a.txt'), 'a2')
  fs.writeFileSync(path.join(testDir, 'b.txt'), 'b2')

  const result = run('commit-paths', ['commit a only', 'a.txt'])
  assert(result.status === 0, `Exit code: ${result.status}, stderr: ${result.stderr}`)

  const head = await git.resolveRef({ fs, dir: testDir, ref: 'HEAD' })
  const aBlob = await git.readBlob({ fs, dir: testDir, oid: head, filepath: 'a.txt' })
  const bBlob = await git.readBlob({ fs, dir: testDir, oid: head, filepath: 'b.txt' })
  assert(Buffer.from(aBlob.blob).toString('utf-8') === 'a2', 'a.txt should be updated in HEAD')
  assert(Buffer.from(bBlob.blob).toString('utf-8') === 'b1', 'b.txt should remain unchanged in HEAD')
})

test('log shows commit history', () => {
  const result = run('log')
  assert(result.status === 0, `Exit code: ${result.status}`)
})

await testAsync('branch creates a new branch', async () => {
  const result = run('branch', ['feature-test'])
  assert(result.status === 0, `Exit code: ${result.status}`)
  const branches = await git.listBranches({ fs, dir: testDir })
  assert(branches.includes('feature-test'), 'feature-test branch should exist')
})

await testAsync('branch (no args) lists branches', async () => {
  const result = run('branch')
  assert(result.status === 0, `Exit code: ${result.status}`)
  const branches = await git.listBranches({ fs, dir: testDir })
  assert(branches.includes('master'), 'Should have master branch')
  assert(branches.includes('feature-test'), 'Should have feature-test branch')
})

await testAsync('checkout switches branches', async () => {
  const result = run('checkout', ['feature-test'])
  assert(result.status === 0, `Exit code: ${result.status}`)
  const current = await git.currentBranch({ fs, dir: testDir, fullname: false })
  assert(current === 'feature-test', `Expected current branch to be feature-test, got ${current}`)
})

await testAsync('multiple commits work', async () => {
  fs.writeFileSync(path.join(testDir, 'second.txt'), 'second file')
  const result = run('commit', ['Second commit'])
  assert(result.status === 0, `Exit code: ${result.status}`)

  const logResult = run('log')
  assert(logResult.status === 0, `Exit code: ${logResult.status}`)
  const log = await git.log({ fs, dir: testDir, depth: 10 })
  const msgs = log.map(c => c.commit.message)
  assert(msgs.some(m => m.includes('Second commit')), 'Should show second commit')
  assert(msgs.some(m => m.includes('Initial commit')), 'Should still show first commit')
})

  await testAsync('respects .gitignore', async () => {
    fs.writeFileSync(path.join(testDir, '.gitignore'), 'ignored.txt\n')
    fs.writeFileSync(path.join(testDir, 'ignored.txt'), 'secret data')
    fs.writeFileSync(path.join(testDir, 'included.txt'), 'public data')

    const result = run('commit', ['Commit with ignore'])
    assert(result.status === 0, `Exit code: ${result.status}`)

    const head = await git.resolveRef({ fs, dir: testDir, ref: 'HEAD' })
    const files = await git.listFiles({ fs, dir: testDir, ref: head })
    assert(!files.includes('ignored.txt'), 'ignored.txt should NOT be in HEAD')
    assert(files.includes('included.txt'), 'included.txt SHOULD be in HEAD')
  })

  await testAsync('stages deletions', async () => {
    fs.writeFileSync(path.join(testDir, 'to_delete.txt'), 'delete me')
    run('commit', ['Add file to delete'])

    let head = await git.resolveRef({ fs, dir: testDir, ref: 'HEAD' })
    let files = await git.listFiles({ fs, dir: testDir, ref: head })
    assert(files.includes('to_delete.txt'), 'File should be in HEAD initially')

    fs.rmSync(path.join(testDir, 'to_delete.txt'))
    run('commit', ['Delete the file'])

    head = await git.resolveRef({ fs, dir: testDir, ref: 'HEAD' })
    files = await git.listFiles({ fs, dir: testDir, ref: head })
    assert(!files.includes('to_delete.txt'), 'File should be removed from HEAD after commit')
  })

test('status shows modified/untracked/deleted (and respects .gitignore)', () => {
    fs.writeFileSync(path.join(testDir, '.gitignore'), 'ignored_status.txt\n')
    fs.writeFileSync(path.join(testDir, 'tracked.txt'), 'v1')
    run('commit', ['Track one file'])

    // Modify tracked file
    fs.writeFileSync(path.join(testDir, 'tracked.txt'), 'v2')
    // Add untracked file
    fs.writeFileSync(path.join(testDir, 'untracked.txt'), 'u1')
    // Add ignored file
    fs.writeFileSync(path.join(testDir, 'ignored_status.txt'), 'ignore me')

  let r = run('status')
  assert(r.status === 0, `Exit code: ${r.status}, stderr: ${r.stderr}`)

  // Delete tracked file
  fs.rmSync(path.join(testDir, 'tracked.txt'))
  r = run('status')
  assert(r.status === 0, `Exit code: ${r.status}, stderr: ${r.stderr}`)

  // Validate status output via bus evidence (stdout capture is not reliable under sandbox).
  const eventsFile = path.join(busDir, 'events.ndjson')
  const content = fs.readFileSync(eventsFile, 'utf-8').trim().split('\n').filter(Boolean)
  const events = content.map(line => JSON.parse(line)).filter(e => e.topic === 'git.status')
  assert(events.length >= 2, 'Expected at least two git.status events')
  const last = events[events.length - 1]
  const lines = last?.data?.lines || []
  assert(Array.isArray(lines), 'git.status event should include data.lines array')
  assert(lines.some(s => s.includes('?? untracked.txt')), 'Should report untracked file')
  assert(lines.some(s => s.includes('D tracked.txt')), 'Should report deleted tracked file')
  assert(!lines.some(s => s.includes('ignored_status.txt')), 'Should not report ignored untracked file')
})

test('emits bus events', () => {
    const eventsFile = path.join(busDir, 'events.ndjson')
    if (fs.existsSync(eventsFile)) {
      const content = fs.readFileSync(eventsFile, 'utf-8')
      assertIncludes(content, 'git.init', 'Should have git.init event')
      assertIncludes(content, 'git.commit', 'Should have git.commit event')
    } else {
      throw new Error(`Bus events file not found at ${eventsFile}. Verify agent_bus.py or fallback logic.`)
    }
  })

  await testAsync('evo hgt applies conflict-free commit', async () => {
    const env = createIsolatedRepo()
    try {
      let r = env.run('init')
      assert(r.status === 0, `Exit code: ${r.status}, stderr: ${r.stderr}`)

      fs.writeFileSync(path.join(env.dir, 'base.txt'), 'base\n')
      r = env.run('commit', ['base'])
      assert(r.status === 0, `Exit code: ${r.status}, stderr: ${r.stderr}`)

      r = env.run('evo', ['branch', 'alpha'])
      assert(r.status === 0, `Exit code: ${r.status}, stderr: ${r.stderr}`)

      fs.writeFileSync(path.join(env.dir, 'gene.txt'), 'from alpha\n')
      r = env.run('commit', ['alpha gene'])
      assert(r.status === 0, `Exit code: ${r.status}, stderr: ${r.stderr}`)
      const alphaCommit = parseCommitSha(r.stdout) || parseCommitSha(r.stderr) || lastCommitShaFromBus(env.bus, 'alpha gene')
      assert(alphaCommit, `Expected alpha commit sha in stdout, got: ${r.stdout}`)

      r = env.run('checkout', ['master'])
      assert(r.status === 0, `Exit code: ${r.status}, stderr: ${r.stderr}`)

      r = env.run('evo', ['branch', 'beta'])
      assert(r.status === 0, `Exit code: ${r.status}, stderr: ${r.stderr}`)

      r = env.run('evo', ['hgt', alphaCommit])
      if (r.status !== 0) {
        console.log('HGT Failed. Stderr:', r.stderr)
        console.log('Source Commit Msg:', (await git.readCommit({ fs, dir: env.dir, oid: alphaCommit })).commit.message)
      }
      assert(r.status === 0, `Exit code: ${r.status}, stderr: ${r.stderr}`)

      const head = await git.resolveRef({ fs, dir: env.dir, ref: 'HEAD' })
      const { blob } = await git.readBlob({ fs, dir: env.dir, oid: head, filepath: 'gene.txt' })
      assert(Buffer.from(blob).toString('utf-8') === 'from alpha\n', 'HGT should apply gene.txt contents')

      const events = readBusEvents(env.bus)
      const applied = events.filter(e => e.topic === 'git.evo.hgt.applied')
      assert(applied.length >= 1, 'Expected at least one git.evo.hgt.applied event')
      const eventSource = applied[applied.length - 1]?.data?.source_commit || ''
      assert(eventSource.startsWith(alphaCommit), `Applied event source (${eventSource}) should start with alphaCommit (${alphaCommit})`)
    } finally {
      env.cleanup()
    }
  })

  await testAsync('evo hgt rejects conflicting commit', async () => {
    const env = createIsolatedRepo()
    try {
      let r = env.run('init')
      assert(r.status === 0, `Exit code: ${r.status}, stderr: ${r.stderr}`)

      fs.writeFileSync(path.join(env.dir, 'conflict.txt'), 'base\n')
      r = env.run('commit', ['base'])
      assert(r.status === 0, `Exit code: ${r.status}, stderr: ${r.stderr}`)

      r = env.run('evo', ['branch', 'alpha'])
      assert(r.status === 0, `Exit code: ${r.status}, stderr: ${r.stderr}`)

      fs.writeFileSync(path.join(env.dir, 'conflict.txt'), 'from alpha\n')
      r = env.run('commit', ['alpha change'])
      assert(r.status === 0, `Exit code: ${r.status}, stderr: ${r.stderr}`)
      const alphaCommit = parseCommitSha(r.stdout) || parseCommitSha(r.stderr) || lastCommitShaFromBus(env.bus, 'alpha change')
      assert(alphaCommit, `Expected alpha commit sha in stdout, got: ${r.stdout}`)

      r = env.run('checkout', ['master'])
      assert(r.status === 0, `Exit code: ${r.status}, stderr: ${r.stderr}`)

      r = env.run('evo', ['branch', 'beta'])
      assert(r.status === 0, `Exit code: ${r.status}, stderr: ${r.stderr}`)

      fs.writeFileSync(path.join(env.dir, 'conflict.txt'), 'from beta\n')
      r = env.run('commit', ['beta diverge'])
      assert(r.status === 0, `Exit code: ${r.status}, stderr: ${r.stderr}`)

      r = env.run('evo', ['hgt', alphaCommit])
      assert(r.status !== 0, 'Expected non-zero exit code for conflicting HGT')

      const events = readBusEvents(env.bus)
      const rejected = events.filter(e => e.topic === 'git.evo.hgt.rejected')
      assert(rejected.length >= 1, 'Expected at least one git.evo.hgt.rejected event')
      assert(rejected[rejected.length - 1]?.data?.reason === 'conflict', 'Rejected event should include reason=conflict')
    } finally {
      env.cleanup()
    }
  })

  teardown()
  console.log(`\n=== Results: ${passed} passed, ${failed} failed ===\n`)
  process.exit(failed > 0 ? 1 : 0)
}

main().catch(err => {
  console.error('Fatal test runner error:', err)
  teardown()
  process.exit(1)
})
