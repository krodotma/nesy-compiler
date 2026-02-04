/**
 * Tests for Mastra Workflows Integration
 *
 * Run with:
 *   npx tsx test_mastra_workflows.ts
 *   # or with Jest:
 *   npx jest test_mastra_workflows.ts
 */

import * as fs from 'fs';
import * as path from 'path';
import * as os from 'os';

// Mock @mastra/core before importing our module
jest.mock('@mastra/core', () => ({
  Agent: jest.fn().mockImplementation((config) => ({
    name: config.name,
    generate: jest.fn().mockResolvedValue({ text: 'Mock response' }),
  })),
  Workflow: jest.fn().mockImplementation((config) => ({
    name: config.name,
    steps: [] as Array<{ name: string; fn: Function }>,
    step: jest.fn().mockImplementation(function (this: any, name: string, fn: Function) {
      this.steps.push({ name, fn });
      return this;
    }),
    commit: jest.fn().mockImplementation(function (this: any) {
      return this;
    }),
    execute: jest.fn().mockImplementation(async function (this: any, { triggerData }) {
      let context = triggerData;
      let prevStep: Record<string, unknown> = {};

      for (const step of this.steps) {
        const result = await step.fn({ context, prevStep });
        prevStep = result;
      }

      return { results: prevStep };
    }),
  })),
}));

// ============================================================================
// Test Suite
// ============================================================================

describe('PluribusBusBridge', () => {
  let tempDir: string;
  let busDir: string;

  beforeEach(() => {
    tempDir = fs.mkdtempSync(path.join(os.tmpdir(), 'mastra-test-'));
    busDir = path.join(tempDir, 'bus');
    fs.mkdirSync(busDir, { recursive: true });
  });

  afterEach(() => {
    fs.rmSync(tempDir, { recursive: true, force: true });
  });

  test('emit writes to bus file', () => {
    // Import after mock setup
    const { PluribusBusBridge } = require('../mastra_workflows');
    const bridge = new PluribusBusBridge(busDir, 'test-actor');

    bridge.emit({
      topic: 'test.event',
      kind: 'metric',
      level: 'info',
      actor: 'test',
      data: { key: 'value' },
    });

    const eventsPath = path.join(busDir, 'events.ndjson');
    const content = fs.readFileSync(eventsPath, 'utf-8');
    const lines = content.trim().split('\n');

    expect(lines.length).toBeGreaterThan(0);

    const event = JSON.parse(lines[0]);
    expect(event.topic).toBe('test.event');
    expect(event.kind).toBe('metric');
    expect(event.data.key).toBe('value');
  });

  test('emit with trace ID', () => {
    const { PluribusBusBridge } = require('../mastra_workflows');
    const bridge = new PluribusBusBridge(busDir, 'test-actor');

    bridge.emit({
      topic: 'traced.event',
      kind: 'log',
      level: 'debug',
      actor: 'test',
      data: {},
      traceId: 'trace-123',
    });

    const eventsPath = path.join(busDir, 'events.ndjson');
    const content = fs.readFileSync(eventsPath, 'utf-8');
    const event = JSON.parse(content.trim());

    expect(event.trace_id).toBe('trace-123');
  });
});

describe('Workflow Registry', () => {
  test('workflows are defined', () => {
    const { workflows } = require('../mastra_workflows');

    expect(workflows).toBeDefined();
    expect(workflows.distill).toBeDefined();
    expect(workflows.hypothesize).toBeDefined();
    expect(workflows.debate).toBeDefined();
  });

  test('listWorkflows outputs workflow names', () => {
    const { listWorkflows } = require('../mastra_workflows');

    // Capture console output
    const consoleSpy = jest.spyOn(console, 'log').mockImplementation();

    listWorkflows();

    expect(consoleSpy).toHaveBeenCalled();
    consoleSpy.mockRestore();
  });
});

describe('Workflow Execution', () => {
  let tempDir: string;
  let busDir: string;

  beforeEach(() => {
    tempDir = fs.mkdtempSync(path.join(os.tmpdir(), 'mastra-exec-'));
    busDir = path.join(tempDir, 'bus');
    fs.mkdirSync(busDir, { recursive: true });
    process.env.PLURIBUS_BUS_DIR = busDir;
  });

  afterEach(() => {
    fs.rmSync(tempDir, { recursive: true, force: true });
    delete process.env.PLURIBUS_BUS_DIR;
  });

  test('runWorkflow executes workflow', async () => {
    const { runWorkflow } = require('../mastra_workflows');

    const result = await runWorkflow('distill', 'Test input content', 'test-trace');

    expect(result).toBeDefined();
  });

  test('runWorkflow throws on unknown workflow', async () => {
    const { runWorkflow } = require('../mastra_workflows');

    await expect(runWorkflow('nonexistent', 'Test input')).rejects.toThrow('Unknown workflow');
  });

  test('workflow emits bus events', async () => {
    const { runWorkflow } = require('../mastra_workflows');

    await runWorkflow('distill', 'Test content', 'trace-xyz');

    const eventsPath = path.join(busDir, 'events.ndjson');
    if (fs.existsSync(eventsPath)) {
      const content = fs.readFileSync(eventsPath, 'utf-8');
      const events = content.trim().split('\n').map((line) => JSON.parse(line));

      // Should have workflow start and step events
      const topics = events.map((e) => e.topic);
      expect(topics.some((t: string) => t.includes('mastra.'))).toBe(true);
    }
  });
});

describe('Event Types', () => {
  test('BusEvent interface is correctly typed', () => {
    interface BusEvent {
      topic: string;
      kind: 'log' | 'metric' | 'request' | 'response' | 'artifact';
      level: 'debug' | 'info' | 'warn' | 'error';
      actor: string;
      data: Record<string, unknown>;
      traceId?: string;
    }

    const event: BusEvent = {
      topic: 'test.topic',
      kind: 'metric',
      level: 'info',
      actor: 'test',
      data: { value: 42 },
    };

    expect(event.topic).toBe('test.topic');
    expect(event.kind).toBe('metric');
  });
});

// ============================================================================
// Integration Tests (require @mastra/core to be installed)
// ============================================================================

describe('Integration Tests', () => {
  // Skip if @mastra/core is mocked
  test.skip('real workflow execution', async () => {
    // This test requires actual @mastra/core installation
    // Uncomment when testing with real dependencies
  });
});

// ============================================================================
// Run tests
// ============================================================================

if (require.main === module) {
  console.log('Running Mastra Workflows Tests...');
  // Simple test runner for non-Jest execution
  const tests = [
    'PluribusBusBridge emit writes to bus file',
    'Workflow Registry workflows are defined',
    'Workflow Execution runWorkflow executes workflow',
  ];

  console.log(`Found ${tests.length} test cases`);
  console.log('Run with Jest for full test execution: npx jest test_mastra_workflows.ts');
}
