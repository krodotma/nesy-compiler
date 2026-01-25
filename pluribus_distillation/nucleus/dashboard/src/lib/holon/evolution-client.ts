/**
 * Evolution Client
 * [Ultrathink Agent 1: Architect]
 * 
 * The bridge between the CodeWarrior UI and the backend Evolution Engine.
 * Handles event publication for mutation requests.
 */

import { createBusClient, type BusClient } from '../bus/bus-client';

let clientInstance: BusClient | null = null;

export class EvolutionClient {
  private static async getClient(): Promise<BusClient> {
    if (!clientInstance) {
      clientInstance = createBusClient({ platform: 'browser' });
      await clientInstance.connect();
    }
    return clientInstance;
  }

  /**
   * Request a mutation for a specific lineage
   */
  static async requestMutation(
    lineageId: string, 
    strategyId: string,
    context?: Record<string, any>
  ): Promise<void> {
    const client = await this.getClient();
    
    await client.publish({
      topic: 'evolution.request.mutation',
      kind: 'command',
      level: 'info',
      actor: 'ui.codewarrior',
      data: {
        lineage_id: lineageId,
        strategy: strategyId,
        context: context || {},
        timestamp: Date.now()
      }
    });
    
    console.log(`[EvolutionClient] Requested mutation: ${strategyId} for ${lineageId}`);
  }

  /**
   * Request a CMP re-evaluation
   */
  static async requestEvaluation(lineageId: string): Promise<void> {
    const client = await this.getClient();
    
    await client.publish({
      topic: 'cmp.request.evaluation',
      kind: 'command',
      level: 'info',
      actor: 'ui.codewarrior',
      data: {
        lineage_id: lineageId
      }
    });
  }
}
