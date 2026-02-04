#!/usr/bin/env node
/**
 * Mastra Bridge (MVP)
 * ==================
 *
 * Goal: provide a bus-evidenced workflow runner that is compatible with a future
 * Mastra integration, without making Mastra a hard dependency today.
 *
 * Behavior:
 * - Reads a JSON workflow spec (steps[] with {name, cmd[]}).
 * - Runs each step sequentially.
 * - Emits `mastra.workflow.*` + `mastra.step.*` events to the Pluribus bus.
 *
 * Usage:
 *   node mastra_bridge.mjs --workflow pluribus_next/tools/mastra_sample_workflow.json
 *
 * Env:
 *   PLURIBUS_BUS_DIR=/pluribus/.pluribus/bus
 *   PLURIBUS_ACTOR=codex
 */

import { spawnSync } from 'node:child_process'
import fs from 'node:fs'
import path from 'node:path'
import process from 'node:process'

function nowIsoUtc() {
  return new Date().toISOString()
}

function parseArgs(argv) {
  const out = { workflow: null, reqId: null }
  for (let i = 0; i < argv.length; i++) {
    const a = argv[i]
    if (a === '--workflow') out.workflow = argv[++i]
    else if (a === '--req-id') out.reqId = argv[++i]
  }
  return out
}

function emitBus(topic, kind, level, data) {
  const busDir = process.env.PLURIBUS_BUS_DIR || '/pluribus/.pluribus/bus'
  const actor = process.env.PLURIBUS_ACTOR || 'mastra-bridge'
  const tool = '/pluribus/pluribus_next/tools/agent_bus.py'
  const payload = JSON.stringify({ ...data, iso: nowIsoUtc() })

  spawnSync('python3', [tool, '--bus-dir', busDir, 'pub', '--topic', topic, '--kind', kind, '--level', level, '--actor', actor, '--data', payload], {
    stdio: 'ignore',
    env: { ...process.env, PYTHONDONTWRITEBYTECODE: '1' },
  })
}

function loadWorkflow(workflowPath) {
  const abs = path.resolve(process.cwd(), workflowPath)
  const raw = fs.readFileSync(abs, 'utf-8')
  const obj = JSON.parse(raw)
  if (!obj || !Array.isArray(obj.steps)) throw new Error('workflow JSON must contain steps[]')
  return { abs, obj }
}

function main() {
  const args = parseArgs(process.argv.slice(2))
  if (!args.workflow) {
    console.error('Usage: node mastra_bridge.mjs --workflow <path.json> [--req-id <id>]')
    process.exit(2)
  }

  const reqId = args.reqId || `mastra-${Date.now()}`
  const { abs, obj } = loadWorkflow(args.workflow)
  const steps = obj.steps

  emitBus('mastra.workflow.start', 'request', 'info', { req_id: reqId, workflow: abs, step_count: steps.length })

  let ok = true
  for (let i = 0; i < steps.length; i++) {
    const step = steps[i] || {}
    const name = String(step.name || `step-${i + 1}`)
    const cmd = Array.isArray(step.cmd) ? step.cmd.map(String) : null
    if (!cmd || cmd.length === 0) {
      emitBus('mastra.step.end', 'response', 'error', { req_id: reqId, step: name, index: i, status: 'error', error: 'missing cmd[]' })
      ok = false
      break
    }

    emitBus('mastra.step.start', 'log', 'info', { req_id: reqId, step: name, index: i, cmd })
    const res = spawnSync(cmd[0], cmd.slice(1), { encoding: 'utf-8' })
    const exitCode = typeof res.status === 'number' ? res.status : 1
    const level = exitCode === 0 ? 'info' : 'error'
    emitBus('mastra.step.end', 'response', level, {
      req_id: reqId,
      step: name,
      index: i,
      status: exitCode === 0 ? 'ok' : 'error',
      exit_code: exitCode,
      stdout: (res.stdout || '').slice(0, 8000),
      stderr: (res.stderr || '').slice(0, 8000),
    })

    if (exitCode !== 0) {
      ok = false
      break
    }
  }

  emitBus('mastra.workflow.end', 'response', ok ? 'info' : 'error', { req_id: reqId, status: ok ? 'ok' : 'error' })
  process.exit(ok ? 0 : 1)
}

main()

