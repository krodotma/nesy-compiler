/**
 * Mastra Agent Workflow
 * Demonstrates a typed agent workflow using Mastra, bridging events to Pluribus Bus.
 * 
 * Usage:
 *   npm install @mastra/core
 *   tsx mastra_agent.ts
 */

import { Agent, Workflow } from '@mastra/core';
import { exec } from 'child_process';

// Pluribus Bus Bridge
function emitBus(topic: string, data: any) {
  const payload = JSON.stringify({
    topic,
    kind: 'metric',
    level: 'info',
    actor: 'mastra',
    data
  });
  // Use bus-run wrapper or direct file append if local
  // For demo, we just log
  console.log(`[BUS] ${topic}:`, data);
}

// Define Agent
const researcher = new Agent({
  name: 'Researcher',
  instructions: 'You are a research agent. Summarize topics concisely.',
  model: {
    provider: 'openai',
    name: 'gpt-4o',
  },
});

// Define Workflow
const researchFlow = new Workflow({
  name: 'topic-research',
});

researchFlow
  .step('research', async ({ context }) => {
    const topic = context.topic;
    emitBus('mastra.step.start', { step: 'research', topic });
    const result = await researcher.generate(`Research this: ${topic}`);
    emitBus('mastra.step.end', { step: 'research', result: result.text });
    return { summary: result.text };
  })
  .commit();

// Execute
async function run() {
  const { results } = await researchFlow.execute({ 
    triggerData: { topic: 'Isomorphic Architectures' } 
  });
  console.log('Final:', results);
}

run().catch(console.error);
