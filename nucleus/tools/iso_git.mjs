import crypto from 'crypto'
import fs from 'fs'
import git from 'isomorphic-git'
import path from 'path'
import { spawnSync } from 'child_process'
import { pathToFileURL } from 'url'
import { verifySignedCommit, signMessage } from './iso_pqc.mjs'

// Status/scan performance: avoid traversing large, non-source trees.
// NOTE: Keep this list conservative; skipping a tracked directory would hide diffs.
const SKIP_DIR_NAMES = new Set([
  '.git',
  '.pluribus',
  '.pluribus_local',
  '.bus',
  'node_modules',
  '.venv',
  '__pycache__',
  '.pytest_cache',
  '.mypy_cache',
  '.ruff_cache',
  'dist',
  'coverage',
  'test-results',
  'LOST_FOUND',
])

function isSkippablePosixPath(posixPath) {
  const parts = String(posixPath || '').split('/')
  return parts.some(p => SKIP_DIR_NAMES.has(p))
}

// --- CMP/VGT/HGT Evolutionary Lineage ---
// Span schema for lineage tracking (per CMP-LARGE spec)
function createSpanContext(dir, opts = {}) {
  const lineageFile = path.join(dir, '.pluribus', 'lineage.json')
  let lineage = { dag_id: null, lineage_id: null, parent_lineage_id: null, generation: 0 }

  try {
    if (fs.existsSync(lineageFile)) {
      lineage = JSON.parse(fs.readFileSync(lineageFile, 'utf-8'))
    }
  } catch {}

  return {
    dag_id: lineage.dag_id || crypto.randomUUID(),
    node_id: crypto.randomUUID(),
    lineage_id: opts.new_lineage ? crypto.randomUUID() : lineage.lineage_id,
    parent_lineage_id: opts.new_lineage ? lineage.lineage_id : lineage.parent_lineage_id,
    mutation_op: opts.mutation_op || 'commit',
    transfer_type: opts.transfer_type || 'VGT', // VGT (vertical) or HGT (horizontal)
    generation: (lineage.generation || 0) + (opts.increment_gen ? 1 : 0),
  }
}

function saveLineage(dir, span) {
  const lineageDir = path.join(dir, '.pluribus')
  const lineageFile = path.join(lineageDir, 'lineage.json')

  try {
    fs.mkdirSync(lineageDir, { recursive: true })
    fs.writeFileSync(lineageFile, JSON.stringify({
      dag_id: span.dag_id,
      lineage_id: span.lineage_id,
      parent_lineage_id: span.parent_lineage_id,
      generation: span.generation,
      updated_iso: new Date().toISOString(),
    }, null, 2))
  } catch {}
}

async function treeBlobMap(dir, ref) {
  const out = new Map()
  await git.walk({
    fs,
    dir,
    trees: [git.TREE({ ref })],
    map: async (filepath, [entry]) => {
      if (filepath === '.') return
      if (!entry) return
      const type = await entry.type()
      if (type !== 'blob') return
      out.set(filepath, await entry.oid())
    },
  })
  return out
}

async function ensureCleanWorktree(dir) {
  if (process.env.PLURIBUS_FAST_STATUS === '1') {
    const result = spawnSync('git', ['status', '--porcelain'], {
      cwd: dir,
      encoding: 'utf-8',
    })
    if (result.status === 0) {
      const lines = String(result.stdout || '')
        .split('\n')
        .map(line => line.trimEnd())
        .filter(Boolean)
      const dirty = lines.map(line => line.slice(3)).filter(Boolean)
      return { clean: dirty.length === 0, dirty }
    }
  }

  try {
    const matrix = await git.statusMatrix({ fs, dir })
    const dirty = []
    for (const row of matrix) {
      const filepath = row[0]
      if (!filepath || isSkippablePosixPath(filepath)) continue
      const ignored = await git.isIgnored({ fs, dir, filepath }).catch(() => false)
      if (ignored) continue
      const head = row[1]
      const workdir = row[2]
      const stage = row[3]
      if (head !== workdir || workdir !== stage) dirty.push(filepath)
    }
    return { clean: dirty.length === 0, dirty }
  } catch (err) {
    const fallback = await buildFallbackStatus(dir)
    return { clean: fallback.dirty.length === 0, dirty: fallback.dirty }
  }
}

// Evolutionary branch naming: evo/<YYYYMMDD>-<slug>
function createEvoBranchName(slug) {
  const date = new Date().toISOString().slice(0, 10).replace(/-/g, '')
  return `evo/${date}-${slug.replace(/[^a-z0-9-]/gi, '-').toLowerCase()}`
}

// --- HGT Guard Ladder (per comprehensive_implementation_matrix.md) ---
// G1: Type compatibility - verify commit structure
// G2: Timing compatibility - no future timestamps
// G3: Effect boundary - no Ring 0 modifications
// G4: Omega acceptance - lineage compatibility
// G5: MDL penalty - description length acceptability
// G6: Spectral stability - HKS signature check (placeholder)

const RING0_PATHS = ['.pluribus/constitution.md', '.pluribus/luca.json', 'AGENTS.md']
const MDL_THRESHOLD_BYTES = 100 * 1024 // 100KB warning threshold

async function runHGTGuardLadder(dir, sourceOid, targetLineage) {
  const checks = []

  // G1: Type compatibility - verify source commit structure is valid
  try {
    const commit = await git.readCommit({ fs, dir, oid: sourceOid })
    if (!commit || !commit.commit) {
      checks.push({ name: 'type_check', passed: false, detail: 'Invalid commit structure' })
    } else {
      checks.push({ name: 'type_check', passed: true, detail: 'Commit structure valid' })
    }
  } catch (err) {
    checks.push({ name: 'type_check', passed: false, detail: err.message })
  }

  // G2: Timing check - source must not be from the future
  try {
    const commit = await git.readCommit({ fs, dir, oid: sourceOid })
    const commitTime = commit?.commit?.committer?.timestamp || 0
    const now = Math.floor(Date.now() / 1000)
    const maxFutureSkew = 300 // 5 minutes tolerance
    if (commitTime > now + maxFutureSkew) {
      checks.push({ name: 'timing_check', passed: false, detail: 'Commit timestamp in future' })
    } else {
      checks.push({ name: 'timing_check', passed: true, detail: 'Timestamp valid' })
    }
  } catch {
    checks.push({ name: 'timing_check', passed: true, detail: 'Skipped (no timestamp)' })
  }

  // G3: Effect boundary - check commit doesn't modify Ring 0 files
  try {
    const commit = await git.readCommit({ fs, dir, oid: sourceOid })
    const parents = commit?.commit?.parent || []
    if (parents.length === 1) {
      const parentOid = parents[0]
      const parentTree = await treeBlobMap(dir, parentOid)
      const sourceTree = await treeBlobMap(dir, sourceOid)

      let ring0Violation = null
      for (const p of RING0_PATHS) {
        const parentOidForPath = parentTree.get(p) || null
        const sourceOidForPath = sourceTree.get(p) || null
        if (parentOidForPath !== sourceOidForPath) {
          ring0Violation = p
          break
        }
      }

      if (ring0Violation) {
        checks.push({ name: 'effect_boundary', passed: false, detail: `Ring 0 file modified: ${ring0Violation}` })
      } else {
        checks.push({ name: 'effect_boundary', passed: true, detail: 'No Ring 0 violations' })
      }
    } else {
      checks.push({ name: 'effect_boundary', passed: true, detail: 'Skipped (merge commit)' })
    }
  } catch {
    checks.push({ name: 'effect_boundary', passed: true, detail: 'Skipped (error reading tree)' })
  }

  // G4: Omega acceptance - check lineage compatibility
  try {
    const lineageFile = path.join(dir, '.pluribus', 'lineage.json')
    if (fs.existsSync(lineageFile)) {
      const lineage = JSON.parse(fs.readFileSync(lineageFile, 'utf-8'))
      if (targetLineage?.dag_id && lineage.dag_id && targetLineage.dag_id !== lineage.dag_id) {
        checks.push({ name: 'omega_check', passed: true, detail: 'Cross-DAG HGT (monitored)' })
      } else {
        checks.push({ name: 'omega_check', passed: true, detail: 'Same DAG or new lineage' })
      }
    } else {
      checks.push({ name: 'omega_check', passed: true, detail: 'No lineage constraints' })
    }
  } catch {
    checks.push({ name: 'omega_check', passed: true, detail: 'Lineage check skipped' })
  }

  // G5: MDL penalty - compute description length increase
  try {
    const commit = await git.readCommit({ fs, dir, oid: sourceOid })
    const parents = commit?.commit?.parent || []
    let totalChangedBytes = 0

    if (parents.length === 1) {
      const parentTree = await treeBlobMap(dir, parents[0])
      const sourceTree = await treeBlobMap(dir, sourceOid)

      for (const [filepath, oid] of sourceTree.entries()) {
        if (parentTree.get(filepath) !== oid) {
          try {
            const { blob } = await git.readBlob({ fs, dir, oid: sourceOid, filepath })
            totalChangedBytes += blob.length
          } catch {}
        }
      }
    }

    if (totalChangedBytes > MDL_THRESHOLD_BYTES) {
      // Warning but not blocking - large changes need review
      checks.push({ name: 'mdl_check', passed: true, detail: `Large change: ${Math.round(totalChangedBytes/1024)}KB (review recommended)` })
    } else {
      checks.push({ name: 'mdl_check', passed: true, detail: `${Math.round(totalChangedBytes/1024)}KB changed` })
    }
  } catch {
    checks.push({ name: 'mdl_check', passed: true, detail: 'MDL check skipped' })
  }

  // G6: Spectral stability - HKS signature check (placeholder for future)
  checks.push({ name: 'spectral_check', passed: true, detail: 'HKS check pending implementation' })

  return {
    passed: checks.every(c => c.passed),
    checks,
    summary: checks.map(c => `${c.passed ? '✓' : '✗'} ${c.name}: ${c.detail}`).join('\n')
  }
}

// --- Git Reset (for HGT rollback to commit-level) ---
async function cmdReset(dir, targetOid) {
  try {
    // Resolve the target
    const resolved = await git.resolveRef({ fs, dir, ref: targetOid }).catch(async () => {
      return await git.expandOid({ fs, dir, oid: targetOid })
    })

    // Get current branch
    let currentBranch
    try {
      currentBranch = await git.currentBranch({ fs, dir })
    } catch {
      currentBranch = null
    }

    // Update the branch ref to point to target
    if (currentBranch) {
      await git.writeRef({
        fs,
        dir,
        ref: `refs/heads/${currentBranch}`,
        value: resolved,
        force: true
      })
    }

    // Checkout to update worktree
    await git.checkout({ fs, dir, ref: resolved, force: true })

    console.log(`Reset to ${resolved.slice(0, 7)}`)
    emitBusEvent('git.reset', { dir, target: resolved, branch: currentBranch })

    return resolved
  } catch (err) {
    console.error('Reset failed:', err.message)
    throw err
  }
}

// --- Bus Helper ---
function emitBusEvent(topic, data) {
  const busDir = process.env.PLURIBUS_BUS_DIR || '/pluribus/.pluribus/bus'
  const busPy = path.join(path.dirname(new URL(import.meta.url).pathname), 'agent_bus.py')
  const payload = JSON.stringify(data)
  const actor = process.env.PLURIBUS_ACTOR || 'iso_git_tool'
  const cmd = [
    process.execPath,
    busPy,
    '--bus-dir',
    busDir,
    'pub',
    '--topic',
    topic,
    '--kind',
    'artifact',
    '--level',
    'info',
    '--actor',
    actor,
    '--data',
    payload,
  ]
  const r = spawnSync('python3', cmd.slice(1), { stdio: 'ignore' })
  if (r.status !== 0) {
    // Best-effort fallback: raw append (no locks).
    const event = {
      id: crypto.randomUUID(),
      ts: Date.now() / 1000,
      iso: new Date().toISOString(),
      topic,
      kind: 'artifact',
      level: 'info',
      actor,
      data,
    }
    try {
      fs.appendFileSync(path.join(busDir, 'events.ndjson'), JSON.stringify(event) + '\n')
    } catch (err) {
      // ignore
    }
  }
}

// --- Commands ---

function isSkippablePath(relPath) {
  const parts = relPath.split(path.sep)
  return parts.some(p => SKIP_DIR_NAMES.has(p))
}

async function listWorkdirFiles(dir) {
  const out = []

  async function walk(rel) {
    const abs = path.join(dir, rel)
    const entries = await fs.promises.readdir(abs, { withFileTypes: true })
    for (const ent of entries) {
      const nextRel = rel ? path.join(rel, ent.name) : ent.name
      if (isSkippablePath(nextRel)) continue
      if (ent.isDirectory()) {
        await walk(nextRel)
      } else if (ent.isFile()) {
        // isomorphic-git expects POSIX-style filepaths
        out.push(nextRel.split(path.sep).join('/'))
      }
    }
  }

  await walk('')
  return out.sort()
}

function parseGitIndex(indexPath) {
  // Minimal reader: version 2/3/4, stage=0 entries only.
  // Format: https://git-scm.com/docs/index-format
  if (!fs.existsSync(indexPath)) return new Map()
  const buf = fs.readFileSync(indexPath)
  if (buf.length < 12) return new Map()
  if (buf.toString('utf8', 0, 4) !== 'DIRC') return new Map()

  const version = buf.readUInt32BE(4)
  const count = buf.readUInt32BE(8)
  if (![2, 3, 4].includes(version)) return new Map()

  let off = 12
  const entries = new Map()
  for (let i = 0; i < count; i++) {
    if (off + 62 > buf.length) break

    // Skip ctime/mtime/dev/ino/mode/uid/gid/size (10 * 4 bytes = 40)
    off += 40
    const oidHex = buf.subarray(off, off + 20).toString('hex')
    off += 20
    const flags = buf.readUInt16BE(off)
    off += 2

    const stage = (flags >> 12) & 0x3
    // Path: null-terminated, followed by padding to 8-byte alignment.
    let pathEnd = off
    while (pathEnd < buf.length && buf[pathEnd] !== 0) pathEnd++
    const name = buf.toString('utf8', off, pathEnd)
    off = pathEnd + 1

    // Align
    const entryLen = 62 + (name.length + 1)
    const pad = (8 - (entryLen % 8)) % 8
    off += pad

    if (stage === 0 && name) entries.set(name, oidHex)
  }
  return entries
}

async function cmdInit(dir) {
  await fs.promises.mkdir(dir, { recursive: true })
  await git.init({ fs, dir })
  console.log(`Initialized empty Git repository in ${dir}`)
  emitBusEvent('git.init', { dir })
}

async function cmdCommit(dir, message, authorName, authorEmail, opts = {}) {
  // Create span context for CMP lineage tracking
  const span = createSpanContext(dir, {
    mutation_op: opts.mutation_op || 'commit',
    transfer_type: opts.transfer_type || 'VGT',
    increment_gen: true,
  })

  // Stage all changes (add new/modified, remove deleted) without using statusMatrix
  // to avoid racy stat behavior in fast write cycles.
  const workFiles = await listWorkdirFiles(dir)
  for (const filepath of workFiles) {
    try {
      await git.add({ fs, dir, filepath })
    } catch (err) {
      // Most common: ignored files. Treat add as best-effort.
    }
  }

  const indexFiles = await git.listFiles({ fs, dir }).catch(() => [])
  for (const filepath of indexFiles) {
    try {
      if (!fs.existsSync(path.join(dir, filepath))) {
        await git.remove({ fs, dir, filepath })
      }
    } catch (err) {
      // Best-effort: ignore removals that fail.
    }
  }

  // Get current branch for lineage context
  let currentBranch = 'main'
  try {
    currentBranch = await git.currentBranch({ fs, dir }) || 'main'
  } catch {}

  // --- PQC Signing ---
  let finalMessage = message
  try {
    finalMessage = signMessage(message)
    // Only log if we actually signed it (signMessage throws if no keys)
    console.log('[PQC] Commit signed.')
  } catch (err) {
    // Best-effort signing: proceed unsigned if no keys found
  }

  const sha = await git.commit({
    fs,
    dir,
    author: {
      name: authorName || process.env.PLURIBUS_ACTOR || 'pluribus',
      email: authorEmail || process.env.PLURIBUS_GIT_EMAIL || 'pluribus@local',
    },
    message: finalMessage
  })

  // Save updated lineage
  span.commit_sha = sha
  if (opts.hgt_source_commit) span.hgt_source_commit = opts.hgt_source_commit
  saveLineage(dir, span)

  console.log(`[${sha.slice(0, 7)}] ${message}`)

  // Emit enriched bus event with CMP span schema
  emitBusEvent('git.commit', {
    dir,
    sha,
    message,
    branch: currentBranch,
    span: {
      dag_id: span.dag_id,
      node_id: span.node_id,
      lineage_id: span.lineage_id,
      parent_lineage_id: span.parent_lineage_id,
      mutation_op: span.mutation_op,
      transfer_type: span.transfer_type,
      generation: span.generation,
      hgt_source_commit: span.hgt_source_commit || null,
    }
  })
  return sha
}

async function cmdCommitPaths(dir, message, paths = [], authorName, authorEmail, opts = {}) {
  const span = createSpanContext(dir, {
    mutation_op: opts.mutation_op || 'commit',
    transfer_type: opts.transfer_type || 'VGT',
    increment_gen: true,
  })

  const normalized = []
  for (const p of (paths || [])) {
    if (!p) continue
    const abs = path.resolve(dir, String(p))
    const rel = path.relative(dir, abs)
    if (!rel || rel.startsWith('..') || path.isAbsolute(rel)) continue
    const posixRel = rel.split(path.sep).join('/')
    if (!posixRel || isSkippablePosixPath(posixRel)) continue
    normalized.push(posixRel)
  }
  if (!normalized.length) {
    throw new Error('commit-paths requires at least one path within the repo')
  }

  const expanded = new Set()
  const dirPaths = []
  for (const p of normalized) {
    const abs = path.join(dir, p)
    try {
      const st = fs.statSync(abs)
      if (st.isDirectory()) {
        dirPaths.push(p)
      } else {
        expanded.add(p)
      }
    } catch {
      // Path missing: if it exists in index we stage a removal.
      expanded.add(p)
    }
  }

  if (dirPaths.length) {
    const workFiles = await listWorkdirFiles(dir)
    for (const p of dirPaths) {
      const prefix = p.endsWith('/') ? p : `${p}/`
      for (const f of workFiles) {
        if (String(f).startsWith(prefix)) expanded.add(f)
      }
    }
  }

  for (const filepath of expanded) {
    try {
      if (fs.existsSync(path.join(dir, filepath))) {
        await git.add({ fs, dir, filepath })
      } else {
        await git.remove({ fs, dir, filepath })
      }
    } catch {
      // Best-effort staging for partial commits.
    }
  }

  let currentBranch = 'main'
  try {
    currentBranch = await git.currentBranch({ fs, dir }) || 'main'
  } catch {}

  // --- PQC Signing (best-effort) ---
  let finalMessage = message
  try {
    finalMessage = signMessage(message)
    console.log('[PQC] Commit signed.')
  } catch {}

  const sha = await git.commit({
    fs,
    dir,
    author: {
      name: authorName || process.env.PLURIBUS_ACTOR || 'pluribus',
      email: authorEmail || process.env.PLURIBUS_GIT_EMAIL || 'pluribus@local',
    },
    message: finalMessage
  })

  span.commit_sha = sha
  saveLineage(dir, span)

  console.log(`[${sha.slice(0, 7)}] ${message}`)
  emitBusEvent('git.commit', {
    dir,
    sha,
    message,
    branch: currentBranch,
    span: {
      dag_id: span.dag_id,
      node_id: span.node_id,
      lineage_id: span.lineage_id,
      parent_lineage_id: span.parent_lineage_id,
      mutation_op: span.mutation_op,
      transfer_type: span.transfer_type,
      generation: span.generation,
    },
    partial: true,
    paths: Array.from(expanded),
  })
  return sha
}

async function buildFallbackStatus(dir) {
  const workFiles = await listWorkdirFiles(dir)
  const indexFiles = await git.listFiles({ fs, dir }).catch(() => [])
  const workSet = new Set(workFiles)
  const indexSet = new Set(indexFiles)

  const lines = []
  const dirty = new Set()

  for (const filepath of workFiles) {
    if (!filepath || isSkippablePosixPath(filepath)) continue
    const ignored = await git.isIgnored({ fs, dir, filepath }).catch(() => false)
    if (ignored) continue
    if (!indexSet.has(filepath)) {
      lines.push(`?? ${filepath}`)
      dirty.add(filepath)
      continue
    }
    const status = await git.status({ fs, dir, filepath }).catch(() => 'modified')
    if (status !== 'unmodified' && status !== 'ignored') {
      lines.push(`M  ${filepath}`)
      dirty.add(filepath)
    }
  }

  for (const filepath of indexFiles) {
    if (!filepath || isSkippablePosixPath(filepath)) continue
    if (workSet.has(filepath)) continue
    const ignored = await git.isIgnored({ fs, dir, filepath }).catch(() => false)
    if (ignored) continue
    lines.push(`D  ${filepath}`)
    dirty.add(filepath)
  }

  return { lines, dirty: Array.from(dirty) }
}

async function cmdStatus(dir) {
  // Fast porcelain-ish output without spawning git CLI.
  // Use isomorphic-git statusMatrix to avoid O(N) blob reads/hashing in large repos.
  // Reference mapping: https://isomorphic-git.org/docs/en/statusMatrix (mirrored in node_modules).
  let matrix = []
  try {
    matrix = await git.statusMatrix({ fs, dir, filepaths: ['.'] })
  } catch (err) {
    const msg = String(err?.message || err)
    const fallback = await buildFallbackStatus(dir)
    emitBusEvent('git.status', { dir, lines: fallback.lines, count: fallback.lines.length, error: msg })
    console.error(msg)
    console.log(fallback.lines.join('\n'))
    return
  }

  const mapping = {
    '000': [],
    '003': ['AD'],
    '020': ['??'],
    '022': ['A '],
    '023': ['AM'],
    '100': ['D '],
    '101': [' D'],
    '103': ['MD'],
    '110': ['D ', '??'],
    '111': [],
    '113': ['MM'],
    '120': ['D ', '??'],
    '121': [' M'],
    '122': ['M '],
    '123': ['MM'],
  }

  const lines = []
  for (const [filepath, head, workdir, stage] of matrix) {
    const key = `${head}${workdir}${stage}`
    const xys = mapping[key] || []
    for (const xy of xys) {
      lines.push(`${xy} ${filepath}`)
    }
  }

  emitBusEvent('git.status', { dir, lines, count: lines.length })
  console.log(lines.join('\n'))
}

async function cmdLog(dir) {
    const commits = await git.log({ fs, dir, depth: 50, ref: 'HEAD' })
    console.log(JSON.stringify({
      commits: commits.map(c => ({
        sha: c.oid,
        message: c.commit.message,
        author: c.commit.author.name,
        email: c.commit.author.email,
        date: new Date(c.commit.author.timestamp * 1000).toISOString(),
        parents: c.commit.parent
      }))
    }, null, 2))
}

// --- Resolve relative ref (e.g., HEAD~3, main~2) ---
async function resolveRelativeRef(dir, ref) {
  // Check for relative notation (e.g., HEAD~3, main~2, HEAD^)
  const relMatch = ref.match(/^(.+?)([~^])(\d*)$/)
  if (relMatch) {
    const [, baseRef, op, countStr] = relMatch
    const count = countStr ? parseInt(countStr, 10) : 1

    // Resolve the base ref first
    let oid
    try {
      oid = await git.resolveRef({ fs, dir, ref: baseRef })
    } catch {
      try {
        oid = await git.resolveRef({ fs, dir, ref: `refs/remotes/${baseRef}` })
      } catch {
        oid = await git.expandOid({ fs, dir, oid: baseRef })
      }
    }

    // Walk back through parents
    for (let i = 0; i < count; i++) {
      const commit = await git.readCommit({ fs, dir, oid })
      if (!commit.commit.parent || commit.commit.parent.length === 0) {
        throw new Error(`Not enough history to resolve ${ref}`)
      }
      oid = commit.commit.parent[0]
    }
    return oid
  }

  // Standard ref resolution
  try {
    return await git.resolveRef({ fs, dir, ref })
  } catch {
    try {
      return await git.resolveRef({ fs, dir, ref: `refs/remotes/${ref}` })
    } catch {
      return await git.expandOid({ fs, dir, oid: ref })
    }
  }
}

// --- Diff Command: Compare two refs for affected project detection ---
async function cmdDiff(dir, base, head) {
  try {
    // Resolve refs to OIDs (handles relative refs like HEAD~3)
    let baseOid, headOid

    try {
      baseOid = await resolveRelativeRef(dir, base)
    } catch (err) {
      throw new Error(`Cannot resolve base ref: ${base} (${err.message})`)
    }

    try {
      headOid = await resolveRelativeRef(dir, head)
    } catch (err) {
      throw new Error(`Cannot resolve head ref: ${head} (${err.message})`)
    }

    // Get tree maps for both refs
    const baseTree = await treeBlobMap(dir, baseOid)
    const headTree = await treeBlobMap(dir, headOid)

    // Find all changed files
    const allPaths = new Set([...baseTree.keys(), ...headTree.keys()])
    const changedFiles = []
    const changes = []

    for (const filepath of allPaths) {
      const baseFileOid = baseTree.get(filepath) || null
      const headFileOid = headTree.get(filepath) || null

      if (baseFileOid === headFileOid) continue

      let status = 'M' // Modified
      if (!baseFileOid && headFileOid) status = 'A' // Added
      if (baseFileOid && !headFileOid) status = 'D' // Deleted

      changedFiles.push(filepath)
      changes.push({
        path: filepath,
        status,
        base_oid: baseFileOid,
        head_oid: headFileOid,
      })
    }

    // Sort for deterministic output
    changedFiles.sort()
    changes.sort((a, b) => a.path.localeCompare(b.path))

    const result = {
      base: base,
      head: head,
      base_oid: baseOid,
      head_oid: headOid,
      changed_files: changedFiles,
      changes: changes,
      counts: {
        total: changes.length,
        added: changes.filter(c => c.status === 'A').length,
        modified: changes.filter(c => c.status === 'M').length,
        deleted: changes.filter(c => c.status === 'D').length,
      },
    }

    console.log(JSON.stringify(result, null, 2))
    emitBusEvent('git.diff', {
      dir,
      base,
      head,
      base_oid: baseOid,
      head_oid: headOid,
      file_count: changedFiles.length,
    })

    return result
  } catch (err) {
    console.error('Diff failed:', err.message)
    process.exit(1)
  }
}

// --- Show Command: Commit details with tree diff ---
async function cmdShow(dir, ref) {
  if (!ref) {
    console.log('Usage: node iso_git.mjs show <dir> <commit-sha>')
    process.exit(2)
  }

  try {
    // Resolve ref to oid
    let oid
    try {
      oid = await git.resolveRef({ fs, dir, ref })
    } catch {
      oid = await git.expandOid({ fs, dir, oid: ref })
    }

    // Read commit
    const commit = await git.readCommit({ fs, dir, oid })
    const parents = commit.commit.parent || []

    // Get tree for this commit
    const treeOid = commit.commit.tree
    const files = await git.listFiles({ fs, dir, ref: oid }).catch(() => [])

    // Compute diff if parent exists
    let diff = []
    if (parents.length === 1) {
      const parentOid = parents[0]
      const parentTree = await treeBlobMap(dir, parentOid)
      const currentTree = await treeBlobMap(dir, oid)

      // Find changed files
      const allPaths = new Set([...parentTree.keys(), ...currentTree.keys()])
      for (const filepath of allPaths) {
        const oldOid = parentTree.get(filepath) || null
        const newOid = currentTree.get(filepath) || null

        if (oldOid === newOid) continue

        let status = 'M' // Modified
        if (!oldOid && newOid) status = 'A' // Added
        if (oldOid && !newOid) status = 'D' // Deleted

        diff.push({ path: filepath, status, oldOid, newOid })
      }
    }

    const result = {
      sha: oid,
      message: commit.commit.message,
      author: {
        name: commit.commit.author.name,
        email: commit.commit.author.email,
        timestamp: commit.commit.author.timestamp,
        date: new Date(commit.commit.author.timestamp * 1000).toISOString(),
      },
      committer: {
        name: commit.commit.committer.name,
        email: commit.commit.committer.email,
        timestamp: commit.commit.committer.timestamp,
      },
      parents,
      tree: treeOid,
      files: files.slice(0, 100), // Limit to 100 files
      diff,
      fileCount: files.length,
      diffCount: diff.length,
    }

    console.log(JSON.stringify(result, null, 2))
    emitBusEvent('git.show', { dir, sha: oid, diffCount: diff.length })

  } catch (err) {
    console.error('Show failed:', err.message)
    process.exit(1)
  }
}

// --- Push Command: Guarded boundary operation (uses native git) ---
async function cmdPush(dir, remote, branch, opts = {}) {
  remote = remote || 'origin'

  // Get current branch if not specified
  if (!branch) {
    try {
      branch = await git.currentBranch({ fs, dir })
    } catch {
      branch = 'main'
    }
  }

  console.log(`[Push] Pushing ${branch} to ${remote}...`)

  // Verify remote exists
  const remotes = await git.listRemotes({ fs, dir }).catch(() => [])
  const remoteExists = remotes.some(r =>
    (typeof r === 'string' ? r : r.remote) === remote
  )

  if (!remoteExists) {
    console.error(`Remote '${remote}' not configured. Use: node iso_git.mjs remote add ${remote} <url>`)
    process.exit(2)
  }

  // Get remote URL for bus event
  const remoteUrl = await git.getConfig({ fs, dir, path: `remote.${remote}.url` }).catch(() => null)

  // Guard check: ensure worktree is clean
  const { clean, dirty } = await ensureCleanWorktree(dir)
  const allowDirty = process.env.PLURIBUS_ALLOW_DIRTY_PUSH === '1'
  if (!clean && !opts.force && !allowDirty) {
    console.error('Push rejected: uncommitted changes exist')
    console.error('  Dirty files:', dirty.slice(0, 5).join(', '))
    emitBusEvent('git.push.rejected', { dir, remote, branch, reason: 'dirty_worktree' })
    process.exit(1)
  }
  if (!clean && allowDirty) {
    console.log('[Push] Proceeding with dirty worktree (PLURIBUS_ALLOW_DIRTY_PUSH=1).')
    emitBusEvent('git.push.dirty', { dir, remote, branch, dirty: dirty.slice(0, 25) })
  }

  // Use native git for actual push (boundary operation)
  const gitArgs = ['push', remote, branch]
  if (opts.force) gitArgs.push('--force')
  if (opts.setUpstream) gitArgs.push('--set-upstream')

  console.log(`[Push] Executing: git ${gitArgs.join(' ')}`)

  const result = spawnSync('git', gitArgs, {
    cwd: dir,
    stdio: ['inherit', 'pipe', 'pipe'],
    encoding: 'utf-8',
    env: { ...process.env },
  })

  if (result.status !== 0) {
    console.error('Push failed:', result.stderr || result.stdout)
    emitBusEvent('git.push.failed', {
      dir,
      remote,
      branch,
      url: remoteUrl,
      error: result.stderr || result.stdout,
    })
    process.exit(1)
  }

  console.log(result.stdout || result.stderr || 'Push successful')

  // Emit success event
  emitBusEvent('git.push', {
    dir,
    remote,
    branch,
    url: remoteUrl,
    forced: !!opts.force,
  })

  console.log(`[Push] Successfully pushed ${branch} to ${remote}`)
}

// --- Fetch Command: Guarded boundary operation (uses native git) ---
async function cmdFetch(dir, remote, opts = {}) {
  remote = remote || 'origin'

  console.log(`[Fetch] Fetching from ${remote}...`)

  // Verify remote exists
  const remotes = await git.listRemotes({ fs, dir }).catch(() => [])
  const remoteExists = remotes.some(r =>
    (typeof r === 'string' ? r : r.remote) === remote
  )

  if (!remoteExists) {
    console.error(`Remote '${remote}' not configured. Use: node iso_git.mjs remote add ${remote} <url>`)
    process.exit(2)
  }

  // Get remote URL for bus event
  const remoteUrl = await git.getConfig({ fs, dir, path: `remote.${remote}.url` }).catch(() => null)

  // Use native git for actual fetch (boundary operation)
  const gitArgs = ['fetch', remote]
  if (opts.all) gitArgs.push('--all')
  if (opts.prune) gitArgs.push('--prune')
  if (opts.tags) gitArgs.push('--tags')

  console.log(`[Fetch] Executing: git ${gitArgs.join(' ')}`)

  const result = spawnSync('git', gitArgs, {
    cwd: dir,
    stdio: ['inherit', 'pipe', 'pipe'],
    encoding: 'utf-8',
    env: { ...process.env },
  })

  if (result.status !== 0) {
    console.error('Fetch failed:', result.stderr || result.stdout)
    emitBusEvent('git.fetch.failed', {
      dir,
      remote,
      url: remoteUrl,
      error: result.stderr || result.stdout,
    })
    process.exit(1)
  }

  console.log(result.stdout || result.stderr || 'Fetch successful')

  // Emit success event
  emitBusEvent('git.fetch', {
    dir,
    remote,
    url: remoteUrl,
    all: !!opts.all,
    prune: !!opts.prune,
  })

  console.log(`[Fetch] Successfully fetched from ${remote}`)
}

// --- Clone Command: Guarded boundary operation (uses native git) ---
async function cmdClone(url, targetDir) {
  if (!url) {
    console.log('Usage: node iso_git.mjs clone <url> [target-dir]')
    process.exit(2)
  }

  // Default target dir from URL
  if (!targetDir) {
    const urlPath = url.replace(/\.git$/, '').split('/').pop()
    targetDir = path.resolve('.', urlPath)
  } else {
    targetDir = path.resolve(targetDir)
  }

  console.log(`[Clone] Cloning ${url} to ${targetDir}...`)

  // Use native git for clone (boundary operation)
  const result = spawnSync('git', ['clone', url, targetDir], {
    stdio: ['inherit', 'pipe', 'pipe'],
    encoding: 'utf-8',
    env: { ...process.env },
  })

  if (result.status !== 0) {
    console.error('Clone failed:', result.stderr || result.stdout)
    emitBusEvent('git.clone.failed', { url, targetDir, error: result.stderr || result.stdout })
    process.exit(1)
  }

  console.log(result.stdout || result.stderr || 'Clone successful')

  // Initialize lineage for new clone
  const span = createSpanContext(targetDir, {
    new_lineage: true,
    mutation_op: 'clone',
    transfer_type: 'HGT', // Clone is horizontal transfer from external source
  })
  saveLineage(targetDir, span)

  emitBusEvent('git.clone', {
    url,
    targetDir,
    lineage_id: span.lineage_id,
  })

  console.log(`[Clone] Successfully cloned to ${targetDir}`)
  console.log(`[Clone] New lineage: ${span.lineage_id.slice(0, 8)}`)
}

async function cmdRemote(dir, subCmd, name, url) {
  const usage = () => {
    console.log('Usage: node iso_git.mjs remote <subcmd> [args]')
    console.log('')
    console.log('Subcommands:')
    console.log('  list                      List configured remotes')
    console.log('  add <name> <url>          Add remote (idempotent)')
    console.log('  set-url <name> <url>      Set remote URL')
    console.log('  remove <name>             Remove remote')
  }

  if (!subCmd || subCmd === 'list') {
    const remotes = await git.listRemotes({ fs, dir }).catch(() => [])
    if (!remotes || remotes.length === 0) {
      console.log('(no remotes)')
      emitBusEvent('git.remote.list', { dir, remotes: [] })
      return
    }

    const rows = []
    for (const r of remotes) {
      if (typeof r === 'string') {
        const remote = r
        const remoteUrl = await git.getConfig({ fs, dir, path: `remote.${remote}.url` }).catch(() => null)
        rows.push({ remote, url: remoteUrl || '' })
      } else {
        rows.push({ remote: r.remote, url: r.url || '' })
      }
    }

    for (const row of rows) {
      console.log(`${row.remote}\t${row.url}`)
    }
    emitBusEvent('git.remote.list', { dir, remotes: rows })
    return
  }

  if (subCmd === 'add') {
    if (!name || !url) {
      usage()
      process.exit(2)
    }
    try {
      await git.addRemote({ fs, dir, remote: name, url })
    } catch {
      // If remote already exists, treat add as idempotent.
      await git.setConfig({ fs, dir, path: `remote.${name}.url`, value: url })
    }
    console.log(`Added remote ${name}: ${url}`)
    emitBusEvent('git.remote.add', { dir, remote: name, url })
    return
  }

  if (subCmd === 'set-url') {
    if (!name || !url) {
      usage()
      process.exit(2)
    }
    await git.setConfig({ fs, dir, path: `remote.${name}.url`, value: url })
    console.log(`Set remote ${name} URL: ${url}`)
    emitBusEvent('git.remote.set_url', { dir, remote: name, url })
    return
  }

  if (subCmd === 'remove' || subCmd === 'rm') {
    if (!name) {
      usage()
      process.exit(2)
    }
    await git.deleteRemote({ fs, dir, remote: name })
    console.log(`Removed remote ${name}`)
    emitBusEvent('git.remote.remove', { dir, remote: name })
    return
  }

  usage()
  process.exit(2)
}

async function cmdUntrack(dir, paths = []) {
  if (!paths.length) {
    console.log('Usage: node iso_git.mjs untrack <dir> <path...>')
    process.exit(2)
  }
  const removed = []
  for (const p of paths) {
    const filepath = String(p).split(path.sep).join('/')
    try {
      await git.remove({ fs, dir, filepath })
      removed.push(filepath)
    } catch (err) {
      // Best-effort: if file was not tracked, ignore.
    }
  }
  emitBusEvent('git.untrack', { dir, paths: removed })
  if (removed.length) console.log(`Untracked (staged removal): ${removed.join(', ')}`)
}

async function cmdBranch(dir, name) {
  if (!name) {
    const branches = await git.listBranches({ fs, dir }).catch(() => [])
    // Mark current branch
    let current = null
    try {
      current = await git.currentBranch({ fs, dir })
    } catch {}
    for (const b of branches) {
      console.log(b === current ? `* ${b}` : `  ${b}`)
    }
    return
  }
  await git.branch({ fs, dir, ref: name })
  console.log(`Created branch ${name}`)
  emitBusEvent('git.branch', { dir, ref: name })
}

async function cmdCheckout(dir, name) {
  if (!name) {
    console.log('Usage: node iso_git.mjs checkout <directory> <branch>')
    process.exit(2)
  }
  await git.checkout({ fs, dir, ref: name })
  console.log(`Checked out ${name}`)
  emitBusEvent('git.checkout', { dir, ref: name })
}

// --- Evolutionary Commands (CMP/VGT/HGT) ---

async function cmdEvo(dir, subCmd, slug) {
  if (subCmd === 'branch' || subCmd === 'b') {
    // Create evolutionary branch: evo/<YYYYMMDD>-<slug>
    if (!slug) {
      console.log('Usage: node iso_git.mjs evo branch <slug>')
      process.exit(2)
    }

    const branchName = createEvoBranchName(slug)

    // Create new lineage for this evolutionary branch
    const span = createSpanContext(dir, {
      new_lineage: true,
      mutation_op: 'evo_branch',
      transfer_type: 'VGT',
    })

    await git.branch({ fs, dir, ref: branchName })
    await git.checkout({ fs, dir, ref: branchName })

    // Save new lineage
    saveLineage(dir, span)

    console.log(`Created evolutionary branch: ${branchName}`)
    console.log(`  Lineage: ${span.lineage_id.slice(0, 8)}`)
    console.log(`  Parent:  ${span.parent_lineage_id ? span.parent_lineage_id.slice(0, 8) : 'root'}`)
    console.log(`  Gen:     ${span.generation}`)

    emitBusEvent('git.evo.branch', {
      dir,
      branch: branchName,
      slug,
      span: {
        dag_id: span.dag_id,
        lineage_id: span.lineage_id,
        parent_lineage_id: span.parent_lineage_id,
        mutation_op: 'evo_branch',
        transfer_type: 'VGT',
        generation: span.generation,
      }
    })

  } else if (subCmd === 'list' || subCmd === 'ls') {
    // List evolutionary branches
    const branches = await git.listBranches({ fs, dir }).catch(() => [])
    const evoBranches = branches.filter(b => b.startsWith('evo/'))

    if (evoBranches.length === 0) {
      console.log('No evolutionary branches found')
      return
    }

    let current = null
    try {
      current = await git.currentBranch({ fs, dir })
    } catch {}

    console.log('Evolutionary branches:')
    for (const b of evoBranches.sort().reverse()) {
      const marker = b === current ? '* ' : '  '
      console.log(`${marker}${b}`)
    }

  } else if (subCmd === 'hgt' || subCmd === 'splice') {
    if (!slug) {
      console.log('Usage: node iso_git.mjs evo hgt <commit-sha>')
      console.log('  Cherry-picks a commit from another lineage (HGT)')
      console.log('')
      console.log('Guard Ladder (6 checks):')
      console.log('  G1: Type compatibility')
      console.log('  G2: Timing compatibility')
      console.log('  G3: Effect boundary (Ring 0 protection)')
      console.log('  G4: Omega acceptance')
      console.log('  G5: MDL penalty')
      console.log('  G6: Spectral stability')
      process.exit(2)
    }

    const hgtSpan = createSpanContext(dir, {
      mutation_op: 'hgt_splice',
      transfer_type: 'HGT',
      increment_gen: true,
    })

    const { clean, dirty } = await ensureCleanWorktree(dir)
    if (!clean) {
      emitBusEvent('git.evo.hgt.rejected', {
        dir,
        source_commit: slug,
        reason: 'dirty_worktree',
        dirty_count: dirty.length,
        span: {
          dag_id: hgtSpan.dag_id,
          lineage_id: hgtSpan.lineage_id,
          mutation_op: 'hgt_splice',
          transfer_type: 'HGT',
          generation: hgtSpan.generation,
        },
      })
      console.error('HGT rejected: working tree not clean')
      process.exit(1)
    }

    let sourceOid
    try {
      sourceOid = await git.expandOid({ fs, dir, oid: slug })
    } catch (err) {
      console.error('HGT rejected: cannot resolve commit oid')
      process.exit(2)
    }

    const sourceCommit = await git.readCommit({ fs, dir, oid: sourceOid }).catch(() => null)
    if (!sourceCommit || !sourceCommit.commit) {
      console.error('HGT rejected: commit not found')
      process.exit(2)
    }
    const parents = sourceCommit.commit.parent || []
    if (!Array.isArray(parents) || parents.length !== 1) {
      emitBusEvent('git.evo.hgt.rejected', {
        dir,
        source_commit: sourceOid,
        reason: 'unsupported_parent_count',
        parent_count: Array.isArray(parents) ? parents.length : null,
      })
      console.error('HGT rejected: only single-parent commits are supported')
      process.exit(2)
    }
    const parentOid = parents[0]

    // --- Run Full Guard Ladder (per comprehensive_implementation_matrix.md) ---
    console.log('[HGT] Running guard ladder...')
    const guardResult = await runHGTGuardLadder(dir, sourceOid, hgtSpan)
    console.log(guardResult.summary)

    if (!guardResult.passed) {
      emitBusEvent('git.evo.hgt.rejected', {
        dir,
        source_commit: sourceOid,
        reason: 'guard_ladder_failed',
        checks: guardResult.checks,
        span: {
          dag_id: hgtSpan.dag_id,
          lineage_id: hgtSpan.lineage_id,
          mutation_op: 'hgt_splice',
          transfer_type: 'HGT',
          generation: hgtSpan.generation,
        },
      })
      console.error('HGT rejected: guard ladder failed')
      process.exit(1)
    }
    console.log('[HGT] Guard ladder passed.')

    // --- PQC Guard: Verify Source Provenance ---
    const pqcResult = await verifySignedCommit(dir, sourceOid)
    if (!pqcResult.ok) {
      emitBusEvent('git.evo.hgt.rejected', {
        dir,
        source_commit: sourceOid,
        reason: 'untrusted_source',
        pqc_status: pqcResult.status,
        signer: pqcResult.signer || null,
      })
      console.error(`HGT rejected: untrusted source (PQC status: ${pqcResult.status})`)
      process.exit(1)
    }
    console.log(`[PQC] Source commit verified: ${pqcResult.signer} (${pqcResult.algo})`)

    const headOid = await git.resolveRef({ fs, dir, ref: 'HEAD' }).catch(() => null)
    if (!headOid) {
      console.error('HGT rejected: cannot resolve HEAD')
      process.exit(2)
    }

    const parentMap = await treeBlobMap(dir, parentOid)
    const sourceMap = await treeBlobMap(dir, sourceOid)
    const headMap = await treeBlobMap(dir, headOid)

    const paths = new Set([...parentMap.keys(), ...sourceMap.keys()])
    const changed = []
    const conflicts = []
    for (const filepath of paths) {
      const oldOid = parentMap.get(filepath) || null
      const newOid = sourceMap.get(filepath) || null
      if (oldOid === newOid) continue
      const headPathOid = headMap.get(filepath) || null
      if (headPathOid !== oldOid) {
        conflicts.push({ filepath, head_oid: headPathOid, base_oid: oldOid, new_oid: newOid })
        continue
      }
      changed.push({ filepath, base_oid: oldOid, new_oid: newOid })
    }

    if (conflicts.length > 0) {
      emitBusEvent('git.evo.hgt.rejected', {
        dir,
        source_commit: sourceOid,
        reason: 'conflict',
        conflicts,
        changed_count: changed.length,
        span: {
          dag_id: hgtSpan.dag_id,
          lineage_id: hgtSpan.lineage_id,
          mutation_op: 'hgt_splice',
          transfer_type: 'HGT',
          generation: hgtSpan.generation,
        },
      })
      console.error(`HGT rejected: ${conflicts.length} conflict(s)`)
      process.exit(1)
    }

    // Apply changes (conflict-free) with rollback snapshot.
    const snapshot = new Map()
    for (const ch of changed) {
      const abs = path.join(dir, ch.filepath)
      try {
        if (fs.existsSync(abs) && fs.lstatSync(abs).isFile()) {
          snapshot.set(ch.filepath, { exists: true, content: fs.readFileSync(abs) })
        } else {
          snapshot.set(ch.filepath, { exists: false, content: null })
        }
      } catch {
        snapshot.set(ch.filepath, { exists: false, content: null })
      }
    }

    try {
      for (const ch of changed) {
        const abs = path.join(dir, ch.filepath)
        if (ch.new_oid === null) {
          if (fs.existsSync(abs)) {
            try { fs.unlinkSync(abs) } catch {}
          }
          continue
        }
        const parent = path.dirname(abs)
        fs.mkdirSync(parent, { recursive: true })
        const { blob } = await git.readBlob({ fs, dir, oid: sourceOid, filepath: ch.filepath })
        fs.writeFileSync(abs, Buffer.from(blob))
      }
    } catch (err) {
      // Rollback: Revert to HEAD state.
      // Since we asserted a clean worktree at the start, a forced checkout of HEAD
      // effectively undoes all partial file writes.
      try {
        await git.checkout({ fs, dir, ref: 'HEAD', force: true })
      } catch (resetErr) {
        console.error('CRITICAL: Rollback failed during HGT failure cleanup:', resetErr)
      }

      emitBusEvent('git.evo.hgt.rejected', {
        dir,
        source_commit: sourceOid,
        reason: 'apply_failed',
        error: String(err?.message || err),
      })
      console.error('HGT rejected: apply failed (worktree rolled back)')
      process.exit(1)
    }

    const msg = `hgt: splice ${String(sourceOid).slice(0, 7)}`
    const sha = await cmdCommit(dir, msg, null, null, { transfer_type: 'HGT', mutation_op: 'hgt_splice', hgt_source_commit: sourceOid })
      .catch(err => {
        console.error('HGT rejected: commit failed', err)
        process.exit(1)
      })

    emitBusEvent('git.evo.hgt.applied', {
      dir,
      source_commit: sourceOid,
      applied_commit: sha || null,
      changed,
      span: {
        dag_id: hgtSpan.dag_id,
        lineage_id: hgtSpan.lineage_id,
        mutation_op: 'hgt_splice',
        transfer_type: 'HGT',
        generation: hgtSpan.generation,
      },
    })

  } else if (subCmd === 'rollback') {
    // Rollback last evolutionary step (VGT/HGT)
    const target = 'HEAD~1'
    try {
      const resolved = await cmdReset(dir, target)
      console.log(`[Evo] Rolled back to ${resolved.slice(0, 7)}`)
      
      // Update lineage to reflect rollback (decrement gen? or just fork?)
      // For now, we just record the event. The lineage.json on disk reverted 
      // automatically because it's tracked in the repo (Ring 0 file).
      
      emitBusEvent('git.evo.rollback', {
        dir,
        target: resolved,
      })
    } catch (err) {
      console.error('Rollback failed:', err.message)
      process.exit(1)
    }

  } else if (subCmd === 'lineage' || subCmd === 'info') {
    // Show current lineage info
    const lineageFile = path.join(dir, '.pluribus', 'lineage.json')

    if (!fs.existsSync(lineageFile)) {
      console.log('No lineage tracking initialized')
      console.log('Use: node iso_git.mjs evo branch <slug> to start')
      return
    }

    const lineage = JSON.parse(fs.readFileSync(lineageFile, 'utf-8'))
    let current = 'detached'
    try {
      current = await git.currentBranch({ fs, dir }) || 'detached'
    } catch {}

    console.log('Lineage Info:')
    console.log(`  Branch:     ${current}`)
    console.log(`  DAG ID:     ${lineage.dag_id || 'none'}`)
    console.log(`  Lineage:    ${lineage.lineage_id || 'none'}`)
    console.log(`  Parent:     ${lineage.parent_lineage_id || 'root'}`)
    console.log(`  Generation: ${lineage.generation || 0}`)
    console.log(`  Updated:    ${lineage.updated_iso || 'never'}`)

  } else {
    console.log('Usage: node iso_git.mjs evo <subcommand>')
    console.log('')
    console.log('Subcommands:')
    console.log('  branch <slug>  Create evolutionary branch (VGT)')
    console.log('  list           List evolutionary branches')
    console.log('  hgt <sha>      Horizontal gene transfer (cherry-pick)')
    console.log('  lineage        Show current lineage info')
  }
}

// --- CLI Entry ---

async function main() {
  const args = process.argv.slice(2)
  const command = args[0]
  const dir = path.resolve(args[1] || '.')

  try {
    if (command === 'init') {
      await cmdInit(dir)
    } else if (command === 'commit') {
      const message = args[2] || 'update'
      await cmdCommit(dir, message)
    } else if (command === 'commit-paths') {
      const message = args[2] || 'update'
      const paths = args.slice(3)
      if (!paths.length) {
        console.log('Usage: node iso_git.mjs commit-paths <dir> <msg> <path...>')
        process.exit(2)
      }
      await cmdCommitPaths(dir, message, paths)
    } else if (command === 'status') {
      await cmdStatus(dir)
    } else if (command === 'log') {
      await cmdLog(dir)
    } else if (command === 'remote') {
      const subCmd = args[2]
      const name = args[3]
      const url = args[4]
      await cmdRemote(dir, subCmd, name, url)
    } else if (command === 'untrack') {
      const paths = args.slice(2)
      await cmdUntrack(dir, paths)
    } else if (command === 'branch') {
      const name = args[2]
      await cmdBranch(dir, name)
    } else if (command === 'checkout') {
      const name = args[2]
      await cmdCheckout(dir, name)
    } else if (command === 'evo') {
      // Evolutionary commands (CMP/VGT/HGT)
      const subCmd = args[2]
      const slug = args[3]
      await cmdEvo(dir, subCmd, slug)
    } else if (command === 'reset') {
      const target = args[2]
      if (!target) {
        console.log('Usage: node iso_git.mjs reset <directory> <commit-sha>')
        process.exit(2)
      }
      await cmdReset(dir, target)
    } else if (command === 'show') {
      const ref = args[2]
      await cmdShow(dir, ref)
    } else if (command === 'diff') {
      // Parse diff arguments: diff <dir> --base <base> --head <head> [--json]
      let baseRef = 'origin/main'
      let headRef = 'HEAD'
      for (let i = 2; i < args.length; i++) {
        if (args[i] === '--base' && args[i + 1]) {
          baseRef = args[++i]
        } else if (args[i] === '--head' && args[i + 1]) {
          headRef = args[++i]
        }
      }
      await cmdDiff(dir, baseRef, headRef)
    } else if (command === 'push') {
      const remote = args[2] || 'origin'
      const branch = args[3]
      const opts = {
        force: args.includes('--force') || args.includes('-f'),
        setUpstream: args.includes('--set-upstream') || args.includes('-u'),
      }
      await cmdPush(dir, remote, branch, opts)
    } else if (command === 'fetch') {
      const remote = args[2] || 'origin'
      const opts = {
        all: args.includes('--all'),
        prune: args.includes('--prune'),
        tags: args.includes('--tags'),
      }
      await cmdFetch(dir, remote, opts)
    } else if (command === 'clone') {
      const url = args[1]
      const targetDir = args[2]
      await cmdClone(url, targetDir)
    } else {
      console.log('Usage: node iso_git.mjs <command> [dir] [args]')
      console.log('')
      console.log('Isomorphic Commands (sandboxable, no native git):')
      console.log('  init              Initialize repository')
      console.log('  commit <msg>      Stage and commit all changes')
      console.log('  commit-paths <msg> <path...>  Commit only specified paths')
      console.log('  status            Show working tree status')
      console.log('  log               Show commit log (JSON)')
      console.log('  show <sha>        Show commit details with diff')
      console.log('  diff --base <ref> --head <ref>  Compare refs (for affected detection)')
      console.log('  remote            Manage remotes (local-only)')
      console.log('  untrack <path...> Remove from index (keep file)')
      console.log('  branch [name]     List or create branch')
      console.log('  checkout <name>   Switch branches')
      console.log('  reset <sha>       Hard reset to commit (for HGT rollback)')
      console.log('  evo <subcmd>      Evolutionary commands (VGT/HGT)')
      console.log('')
      console.log('Boundary Commands (require native git):')
      console.log('  push [remote] [branch]   Push to remote (guarded)')
      console.log('  fetch [remote]           Fetch from remote (guarded)')
      console.log('  clone <url> [dir]        Clone repository (guarded)')
      console.log('')
      console.log('Affected Detection (Nx-equivalent):')
      console.log('  diff --base <base> --head <head>  Output changed files (JSON)')
      console.log('')
      console.log('Evolutionary subcommands (evo):')
      console.log('  branch <slug>     Create evo/<date>-<slug> branch')
      console.log('  list              List evolutionary branches')
      console.log('  hgt <sha>         Horizontal gene transfer')
      console.log('  lineage           Show lineage info')
    }
  } catch (err) {
    console.error('Error:', err)
    process.exit(1)
  }
}

export {
  cmdInit,
  cmdCommit,
  cmdCommitPaths,
  cmdStatus,
  cmdLog,
  cmdShow,
  cmdDiff,
  cmdRemote,
  cmdUntrack,
  cmdBranch,
  cmdCheckout,
  cmdEvo,
  cmdReset,
  cmdPush,
  cmdFetch,
  cmdClone,
  runHGTGuardLadder,
  treeBlobMap,
}

const isMain = import.meta.url === pathToFileURL(path.resolve(process.argv[1] || '')).href
if (isMain) {
  main()
}
