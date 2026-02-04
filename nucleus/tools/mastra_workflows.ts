/**
 * Mastra Workflows Integration for Pluribus
 * ==========================================
 *
 * Full TypeScript workflow engine integration with real-time Pluribus bus bridge.
 * Provides:
 * - Pipeline definitions for common workflows
 * - Real-time event emission to Pluribus bus
 * - Step-level observability
 * - Error handling and retry logic
 *
 * Usage:
 *   # Install dependencies
 *   npm install @mastra/core
 *
 *   # Run a workflow
 *   npx tsx mastra_workflows.ts run --workflow distill --input "Your content..."
 *
 *   # List workflows
 *   npx tsx mastra_workflows.ts list
 *
 *   # Daemon mode (bridge events)
 *   npx tsx mastra_workflows.ts daemon --bus-dir /pluribus/.pluribus/bus
 */

import { Agent, Workflow, Step } from '@mastra/core';
import { execSync, spawn } from 'child_process';
import * as fs from 'fs';
import * as path from 'path';

// ============================================================================
// Types
// ============================================================================

interface BusEvent {
  topic: string;
  kind: 'log' | 'metric' | 'request' | 'response' | 'artifact';
  level: 'debug' | 'info' | 'warn' | 'error';
  actor: string;
  data: Record<string, unknown>;
  traceId?: string;
}

interface WorkflowContext {
  input: string;
  traceId?: string;
  parentTraceId?: string;
  tags?: Record<string, string>;
  constraints?: Record<string, unknown>;
}

interface StepResult<T = unknown> {
  success: boolean;
  data?: T;
  error?: string;
  latencyMs?: number;
}

// ============================================================================
// Bus Bridge
// ============================================================================

class PluribusBusBridge {
  private busDir: string;
  private actor: string;
  private agentBusPath: string;

  constructor(busDir?: string, actor?: string) {
    this.busDir = busDir || process.env.PLURIBUS_BUS_DIR || '/pluribus/.pluribus/bus';
    this.actor = actor || process.env.PLURIBUS_ACTOR || 'mastra';
    this.agentBusPath = path.resolve(__dirname, 'agent_bus.py');
  }

  emit(event: BusEvent): void {
    const { topic, kind, level, data, traceId } = event;

    // Use agent_bus.py for reliable emission
    if (fs.existsSync(this.agentBusPath)) {
      const args = [
        this.agentBusPath,
        '--bus-dir', this.busDir,
        'pub',
        '--topic', topic,
        '--kind', kind,
        '--level', level,
        '--actor', this.actor,
        '--data', JSON.stringify(data),
      ];

      if (traceId) {
        args.push('--trace-id', traceId);
      }

      try {
        execSync(`python3 ${args.map(a => `"${a}"`).join(' ')}`, {
          stdio: 'ignore',
          env: { ...process.env, PYTHONDONTWRITEBYTECODE: '1' },
        });
      } catch {
        // Fallback: direct file append
        this.appendToBus(event);
      }
    } else {
      this.appendToBus(event);
    }
  }

  private appendToBus(event: BusEvent): void {
    const eventsPath = path.join(this.busDir, 'events.ndjson');
    const record = {
      id: this.generateId(),
      ts: Date.now() / 1000,
      iso: new Date().toISOString(),
      topic: event.topic,
      kind: event.kind,
      level: event.level,
      actor: this.actor,
      host: require('os').hostname(),
      pid: process.pid,
      trace_id: event.traceId || null,
      data: event.data,
    };

    try {
      fs.mkdirSync(this.busDir, { recursive: true });
      fs.appendFileSync(eventsPath, JSON.stringify(record) + '\n');
    } catch (e) {
      console.error('[BUS] Failed to append:', e);
    }
  }

  private generateId(): string {
    return 'xxxxxxxx-xxxx-4xxx-yxxx-xxxxxxxxxxxx'.replace(/[xy]/g, (c) => {
      const r = (Math.random() * 16) | 0;
      const v = c === 'x' ? r : (r & 0x3) | 0x8;
      return v.toString(16);
    });
  }
}

// Global bus instance
const bus = new PluribusBusBridge();

// ============================================================================
// Agents
// ============================================================================

// Research Agent - Summarizes and extracts key information
const researchAgent = new Agent({
  name: 'Researcher',
  instructions: `You are a research agent for the Pluribus system.
Your role is to:
1. Summarize complex topics concisely
2. Extract key claims and definitions
3. Identify gaps in understanding
4. Cite sources when available

Always return structured JSON with: summary, claims[], definitions[], gaps[], citations[]`,
  model: {
    provider: 'openai',
    name: process.env.MASTRA_MODEL || 'gpt-4o-mini',
  },
});

// Hypothesis Agent - Generates testable hypotheses
const hypothesisAgent = new Agent({
  name: 'Hypothesizer',
  instructions: `You are a hypothesis generation agent for the Pluribus system.
Your role is to:
1. Generate testable hypotheses from research
2. Identify falsifiers for each hypothesis
3. Propose experiments or tests
4. Define success metrics

Always return structured JSON with: hypotheses[], falsifiers[], tests[], metrics[]`,
  model: {
    provider: 'openai',
    name: process.env.MASTRA_MODEL || 'gpt-4o-mini',
  },
});

// Verification Agent - Validates outputs
const verificationAgent = new Agent({
  name: 'Verifier',
  instructions: `You are a verification agent for the Pluribus system.
Your role is to:
1. Check logical consistency
2. Identify counterexamples
3. Validate citations if provided
4. Apply quality gates

Always return structured JSON with: checks[], counterexamples[], gates_passed[], issues[]`,
  model: {
    provider: 'openai',
    name: process.env.MASTRA_MODEL || 'gpt-4o-mini',
  },
});

// ============================================================================
// Workflow Definitions
// ============================================================================

/**
 * Distill Workflow
 * Extracts and structures knowledge from input content
 */
const distillWorkflow = new Workflow({
  name: 'distill',
  triggerSchema: {
    type: 'object',
    properties: {
      input: { type: 'string' },
      traceId: { type: 'string' },
    },
    required: ['input'],
  },
});

distillWorkflow
  .step('research', async ({ context }) => {
    const { input, traceId } = context as WorkflowContext;
    const t0 = Date.now();

    bus.emit({
      topic: 'mastra.step.start',
      kind: 'metric',
      level: 'info',
      actor: 'mastra',
      data: { workflow: 'distill', step: 'research', inputLen: input.length },
      traceId,
    });

    try {
      const result = await researchAgent.generate(`Distill and summarize:\n\n${input}`);
      const latencyMs = Date.now() - t0;

      bus.emit({
        topic: 'mastra.step.complete',
        kind: 'metric',
        level: 'info',
        actor: 'mastra',
        data: { workflow: 'distill', step: 'research', latencyMs, outputLen: result.text?.length || 0 },
        traceId,
      });

      return { research: result.text };
    } catch (error) {
      bus.emit({
        topic: 'mastra.step.error',
        kind: 'metric',
        level: 'error',
        actor: 'mastra',
        data: { workflow: 'distill', step: 'research', error: String(error) },
        traceId,
      });
      throw error;
    }
  })
  .step('verify', async ({ context, prevStep }) => {
    const { traceId } = context as WorkflowContext;
    const research = prevStep?.research || '';
    const t0 = Date.now();

    bus.emit({
      topic: 'mastra.step.start',
      kind: 'metric',
      level: 'info',
      actor: 'mastra',
      data: { workflow: 'distill', step: 'verify' },
      traceId,
    });

    try {
      const result = await verificationAgent.generate(`Verify this research output:\n\n${research}`);
      const latencyMs = Date.now() - t0;

      bus.emit({
        topic: 'mastra.step.complete',
        kind: 'metric',
        level: 'info',
        actor: 'mastra',
        data: { workflow: 'distill', step: 'verify', latencyMs },
        traceId,
      });

      return { verified: result.text };
    } catch (error) {
      bus.emit({
        topic: 'mastra.step.error',
        kind: 'metric',
        level: 'error',
        actor: 'mastra',
        data: { workflow: 'distill', step: 'verify', error: String(error) },
        traceId,
      });
      throw error;
    }
  })
  .commit();

/**
 * Hypothesize Workflow
 * Generates and validates hypotheses from research
 */
const hypothesizeWorkflow = new Workflow({
  name: 'hypothesize',
  triggerSchema: {
    type: 'object',
    properties: {
      input: { type: 'string' },
      traceId: { type: 'string' },
    },
    required: ['input'],
  },
});

hypothesizeWorkflow
  .step('distill', async ({ context }) => {
    const { input, traceId } = context as WorkflowContext;
    const t0 = Date.now();

    bus.emit({
      topic: 'mastra.step.start',
      kind: 'metric',
      level: 'info',
      actor: 'mastra',
      data: { workflow: 'hypothesize', step: 'distill' },
      traceId,
    });

    const result = await researchAgent.generate(`Extract key concepts for hypothesis generation:\n\n${input}`);

    bus.emit({
      topic: 'mastra.step.complete',
      kind: 'metric',
      level: 'info',
      actor: 'mastra',
      data: { workflow: 'hypothesize', step: 'distill', latencyMs: Date.now() - t0 },
      traceId,
    });

    return { concepts: result.text };
  })
  .step('generate', async ({ context, prevStep }) => {
    const { traceId } = context as WorkflowContext;
    const concepts = prevStep?.concepts || '';
    const t0 = Date.now();

    bus.emit({
      topic: 'mastra.step.start',
      kind: 'metric',
      level: 'info',
      actor: 'mastra',
      data: { workflow: 'hypothesize', step: 'generate' },
      traceId,
    });

    const result = await hypothesisAgent.generate(`Generate hypotheses from:\n\n${concepts}`);

    bus.emit({
      topic: 'mastra.step.complete',
      kind: 'metric',
      level: 'info',
      actor: 'mastra',
      data: { workflow: 'hypothesize', step: 'generate', latencyMs: Date.now() - t0 },
      traceId,
    });

    return { hypotheses: result.text };
  })
  .step('validate', async ({ context, prevStep }) => {
    const { traceId } = context as WorkflowContext;
    const hypotheses = prevStep?.hypotheses || '';
    const t0 = Date.now();

    bus.emit({
      topic: 'mastra.step.start',
      kind: 'metric',
      level: 'info',
      actor: 'mastra',
      data: { workflow: 'hypothesize', step: 'validate' },
      traceId,
    });

    const result = await verificationAgent.generate(`Validate these hypotheses for logical consistency:\n\n${hypotheses}`);

    bus.emit({
      topic: 'mastra.step.complete',
      kind: 'metric',
      level: 'info',
      actor: 'mastra',
      data: { workflow: 'hypothesize', step: 'validate', latencyMs: Date.now() - t0 },
      traceId,
    });

    return { validated: result.text };
  })
  .commit();

/**
 * Multi-Agent Debate Workflow
 * Star topology with aggregation
 */
const debateWorkflow = new Workflow({
  name: 'debate',
  triggerSchema: {
    type: 'object',
    properties: {
      input: { type: 'string' },
      traceId: { type: 'string' },
      fanout: { type: 'number' },
    },
    required: ['input'],
  },
});

debateWorkflow
  .step('fan-out', async ({ context }) => {
    const { input, traceId, fanout = 3 } = context as WorkflowContext & { fanout?: number };
    const t0 = Date.now();

    bus.emit({
      topic: 'mastra.step.start',
      kind: 'metric',
      level: 'info',
      actor: 'mastra',
      data: { workflow: 'debate', step: 'fan-out', fanout },
      traceId,
    });

    // Parallel execution with different perspectives
    const perspectives = [
      'Analyze from a critical/skeptical perspective',
      'Analyze from an optimistic/supportive perspective',
      'Analyze from a practical/implementation perspective',
    ].slice(0, fanout);

    const results = await Promise.all(
      perspectives.map(async (perspective, idx) => {
        const subTraceId = `${traceId || 'debate'}-${idx}`;

        bus.emit({
          topic: 'mastra.subagent.spawn',
          kind: 'metric',
          level: 'info',
          actor: 'mastra',
          data: { workflow: 'debate', subId: idx + 1, perspective },
          traceId: subTraceId,
        });

        const result = await researchAgent.generate(`${perspective}:\n\n${input}`);

        bus.emit({
          topic: 'mastra.subagent.complete',
          kind: 'metric',
          level: 'info',
          actor: 'mastra',
          data: { workflow: 'debate', subId: idx + 1 },
          traceId: subTraceId,
        });

        return { perspective, output: result.text };
      })
    );

    bus.emit({
      topic: 'mastra.step.complete',
      kind: 'metric',
      level: 'info',
      actor: 'mastra',
      data: { workflow: 'debate', step: 'fan-out', latencyMs: Date.now() - t0, resultsCount: results.length },
      traceId,
    });

    return { perspectives: results };
  })
  .step('aggregate', async ({ context, prevStep }) => {
    const { traceId } = context as WorkflowContext;
    const perspectives = prevStep?.perspectives || [];
    const t0 = Date.now();

    bus.emit({
      topic: 'mastra.step.start',
      kind: 'metric',
      level: 'info',
      actor: 'mastra',
      data: { workflow: 'debate', step: 'aggregate', inputCount: perspectives.length },
      traceId,
    });

    const aggregationPrompt = `Synthesize these different perspectives into a balanced analysis:

${perspectives.map((p: { perspective: string; output: string }) => `## ${p.perspective}\n${p.output}`).join('\n\n')}

Return a JSON object with: synthesis, agreements[], disagreements[], conclusions[]`;

    const result = await researchAgent.generate(aggregationPrompt);

    bus.emit({
      topic: 'mastra.step.complete',
      kind: 'metric',
      level: 'info',
      actor: 'mastra',
      data: { workflow: 'debate', step: 'aggregate', latencyMs: Date.now() - t0 },
      traceId,
    });

    return { synthesis: result.text };
  })
  .commit();

// ============================================================================
// Workflow Registry
// ============================================================================

const workflows: Record<string, Workflow> = {
  distill: distillWorkflow,
  hypothesize: hypothesizeWorkflow,
  debate: debateWorkflow,
};

// ============================================================================
// CLI Commands
// ============================================================================

async function runWorkflow(name: string, input: string, traceId?: string): Promise<unknown> {
  const workflow = workflows[name];
  if (!workflow) {
    throw new Error(`Unknown workflow: ${name}. Available: ${Object.keys(workflows).join(', ')}`);
  }

  bus.emit({
    topic: 'mastra.workflow.start',
    kind: 'metric',
    level: 'info',
    actor: 'mastra',
    data: { workflow: name, inputLen: input.length },
    traceId,
  });

  const t0 = Date.now();

  try {
    const { results } = await workflow.execute({
      triggerData: { input, traceId },
    });

    bus.emit({
      topic: 'mastra.workflow.complete',
      kind: 'response',
      level: 'info',
      actor: 'mastra',
      data: { workflow: name, latencyMs: Date.now() - t0, success: true },
      traceId,
    });

    return results;
  } catch (error) {
    bus.emit({
      topic: 'mastra.workflow.error',
      kind: 'metric',
      level: 'error',
      actor: 'mastra',
      data: { workflow: name, error: String(error) },
      traceId,
    });
    throw error;
  }
}

function listWorkflows(): void {
  console.log('Available Mastra Workflows:');
  console.log('===========================');
  for (const [name, workflow] of Object.entries(workflows)) {
    console.log(`- ${name}: ${workflow.name}`);
  }
}

// ============================================================================
// Main Entry Point
// ============================================================================

async function main(): Promise<void> {
  const args = process.argv.slice(2);
  const cmd = args[0];

  switch (cmd) {
    case 'run': {
      const workflowIdx = args.indexOf('--workflow');
      const inputIdx = args.indexOf('--input');
      const traceIdx = args.indexOf('--trace-id');

      if (workflowIdx === -1 || inputIdx === -1) {
        console.error('Usage: npx tsx mastra_workflows.ts run --workflow <name> --input "<content>"');
        process.exit(1);
      }

      const workflowName = args[workflowIdx + 1];
      const input = args[inputIdx + 1];
      const traceId = traceIdx !== -1 ? args[traceIdx + 1] : undefined;

      try {
        const result = await runWorkflow(workflowName, input, traceId);
        console.log(JSON.stringify(result, null, 2));
      } catch (error) {
        console.error('Workflow failed:', error);
        process.exit(1);
      }
      break;
    }

    case 'list':
      listWorkflows();
      break;

    case 'daemon': {
      const busIdx = args.indexOf('--bus-dir');
      const intervalIdx = args.indexOf('--interval');

      const busDir = busIdx !== -1 ? args[busIdx + 1] : undefined;
      const interval = intervalIdx !== -1 ? parseInt(args[intervalIdx + 1], 10) : 30000;

      console.log(`Mastra daemon started (bus=${busDir || 'default'}, interval=${interval}ms)`);

      const daemonBus = new PluribusBusBridge(busDir);

      daemonBus.emit({
        topic: 'mastra.daemon.start',
        kind: 'log',
        level: 'info',
        actor: 'mastra',
        data: { workflows: Object.keys(workflows), interval },
      });

      // Heartbeat loop
      setInterval(() => {
        daemonBus.emit({
          topic: 'mastra.daemon.heartbeat',
          kind: 'metric',
          level: 'debug',
          actor: 'mastra',
          data: { ts: Date.now(), workflows: Object.keys(workflows) },
        });
      }, interval);

      // Keep process alive
      process.on('SIGINT', () => {
        daemonBus.emit({
          topic: 'mastra.daemon.stop',
          kind: 'log',
          level: 'info',
          actor: 'mastra',
          data: {},
        });
        process.exit(0);
      });
      break;
    }

    default:
      console.log(`Mastra Workflows for Pluribus
Usage:
  npx tsx mastra_workflows.ts run --workflow <name> --input "<content>" [--trace-id <id>]
  npx tsx mastra_workflows.ts list
  npx tsx mastra_workflows.ts daemon [--bus-dir <path>] [--interval <ms>]

Workflows: ${Object.keys(workflows).join(', ')}`);
  }
}

// Export for programmatic use
export { workflows, runWorkflow, listWorkflows, PluribusBusBridge };

// Run if executed directly
main().catch(console.error);
