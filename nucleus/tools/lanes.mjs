import fs from 'fs';
import path from 'path';
import crypto from 'crypto';
import { parseArgs } from 'util';

const PLURIBUS_ROOT = process.env.PLURIBUS_ROOT || '/pluribus';
const STATE_PATH = path.join(PLURIBUS_ROOT, 'nucleus', 'state', 'lanes.json');
const BUS_PATH = path.join(PLURIBUS_ROOT, '.pluribus', 'bus', 'events.ndjson');

function loadState() {
  if (!fs.existsSync(STATE_PATH)) {
    return {
      version: '1.0',
      generated: new Date().toISOString(),
      updated: new Date().toISOString(),
      lanes: [],
      agents: []
    };
  }
  return JSON.parse(fs.readFileSync(STATE_PATH, 'utf8'));
}

function saveState(state) {
  state.updated = new Date().toISOString();
  fs.mkdirSync(path.dirname(STATE_PATH), { recursive: true });
  fs.writeFileSync(STATE_PATH, JSON.stringify(state, null, 2), 'utf8');
}

function renderWipMeter(pct, width = 20) {
  const filled = Math.round((pct / 100) * width);
  const empty = width - filled;
  return '█'.repeat(filled) + '░'.repeat(empty);
}

function statusEmoji(status) {
  const map = { green: '🟢', yellow: '🟡', red: '🔴', blocked: '🔴' };
  return map[status] || '⚪';
}

function updateLane(id, options) {
  const state = loadState();
  const lane = state.lanes.find(l => l.id === id);
  if (!lane) throw new Error(`Lane not found: ${id}`);

  if (options.wip !== undefined) lane.wip_pct = Math.max(0, Math.min(100, parseInt(options.wip)));
  if (options.status) lane.status = options.status;
  if (options.commit) {
    lane.commits = lane.commits || [];
    lane.commits.push(options.commit);
  }
  if (options.tier) lane.tier = options.tier;
  if (options.host) lane.host = options.host;
  if (options.slot !== undefined) lane.slot = parseInt(options.slot);
  
  // History
  if (options.wip !== undefined || options.note) {
    lane.history = lane.history || [];
    lane.history.push({
      ts: new Date().toISOString(),
      wip_pct: lane.wip_pct,
      note: options.note || `Updated to ${lane.wip_pct}%`
    });
  }

  saveState(state);
  return lane;
}

function addLane(id, name, owner, options) {
  const state = loadState();
  if (state.lanes.find(l => l.id === id)) throw new Error(`Lane exists: ${id}`);

  const lane = {
    id, name, owner,
    status: 'green',
    wip_pct: 0,
    description: '',
    commits: [],
    blockers: [],
    next_actions: [],
    tier: options.tier,
    host: options.host,
    slot: options.slot ? parseInt(options.slot) : undefined,
    history: [{ ts: new Date().toISOString(), wip_pct: 0, note: 'Lane created' }]
  };
  state.lanes.push(lane);
  saveState(state);
  return lane;
}

function emitEvent(actor, data) {
  const event = {
    id: crypto.randomUUID(),
    ts: Date.now(),
    iso: new Date().toISOString(),
    topic: 'operator.lanes.state',
    kind: 'metric',
    actor: actor || 'lanes-tool',
    data
  };
  fs.mkdirSync(path.dirname(BUS_PATH), { recursive: true });
  fs.appendFileSync(BUS_PATH, JSON.stringify(event) + '\n');
  return event;
}

function renderMarkdown() {
  const state = loadState();
  const now = new Date().toISOString();
  const activeAgents = state.agents.filter(a => a.status === 'active').length;
  
  let lines = [
    '# LANES REPORT',
    `**Generated**: ${now}`,
    `**State**: `nucleus/state/lanes.json``,
    `**Agents**: ${activeAgents} active`,
    '',
    '---',
    '',
    '## Lane Status',
    '',
    '| Lane | Tier | Host | WIP | Meter |',
    '|------|------|------|-----|-------|'
  ];

  state.lanes.forEach(lane => {
    const emoji = statusEmoji(lane.status);
    const meter = renderWipMeter(lane.wip_pct);
    const tier = lane.tier || 'N/A';
    const host = lane.host || 'N/A';
    lines.push(`| ${emoji} ${lane.name} | ${tier} | ${host} | ${lane.wip_pct}% | `${meter}` |`);
  });
  
  return lines.join('\n');
}

// Main
try {
  const { values, positionals } = parseArgs({
    options: {
      status: { type: 'boolean' },
      update: { type: 'string' },
      wip: { type: 'string' },
      note: { type: 'string' },
      commit: { type: 'string' },
      'set-status': { type: 'string' },
      tier: { type: 'string' },
      host: { type: 'string' },
      slot: { type: 'string' },
      emit: { type: 'boolean' },
      render: { type: 'boolean' },
      json: { type: 'boolean' },
      'add-lane': { type: 'boolean' }, // Flag to trigger positional logic
      actor: { type: 'string' },
      help: { type: 'boolean' }
    },
    allowPositionals: true
  });

  if (values.help) {
    console.log(`Usage: node lanes.mjs [options]
    --status        Show current lanes
    --update ID     Update lane
    --wip N         Set WIP %
    --tier TIER     Set tier
    --host HOST     Set host
    --slot N        Set slot
    --render        Render Markdown
    --add-lane ID NAME OWNER   Add new lane (use positionals)`);
    process.exit(0);
  }

  if (values['add-lane'] && positionals.length >= 3) {
    const [id, name, owner] = positionals;
    const lane = addLane(id, name, owner, values);
    console.log(`Added lane: ${lane.id}`);
  }

  if (values.update) {
    const lane = updateLane(values.update, {
      wip: values.wip,
      status: values['set-status'],
      note: values.note,
      commit: values.commit,
      tier: values.tier,
      host: values.host,
      slot: values.slot
    });
    console.log(`Updated ${lane.id}: ${lane.wip_pct}%`);
  }

  if (values.emit) {
    const state = loadState();
    const event = emitEvent(values.actor, { overall: 0 }); // Todo: calc overall
    console.log(`Emitted event ${event.id}`);
  }

  if (values.render) {
    console.log(renderMarkdown());
  } else if (values.json) {
    console.log(JSON.stringify(loadState(), null, 2));
  } else if (!values.update && !values['add-lane'] && !values.emit) {
     console.log(renderMarkdown());
  }

} catch (e) {
  console.error(e.message);
  process.exit(1);
}
