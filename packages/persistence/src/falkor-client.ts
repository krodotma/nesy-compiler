import { createClient } from 'redis';

export interface GraphConfig {
  host: string;
  port: number;
  graphName: string;
}

const DEFAULT_CONFIG: GraphConfig = {
  host: process.env.FALKORDB_HOST || 'localhost',
  port: parseInt(process.env.FALKORDB_PORT || '6380', 10),
  graphName: 'pluribus',
};

export class FalkorClient {
  private client: ReturnType<typeof createClient>;
  private config: GraphConfig;
  private connected: boolean = false;

  constructor(config?: Partial<GraphConfig>) {
    this.config = { ...DEFAULT_CONFIG, ...config };
    this.client = createClient({
      socket: {
        host: this.config.host,
        port: this.config.port,
      },
    });

    this.client.on('error', (err) => console.error('FalkorDB Client Error', err));
  }

  async connect(): Promise<void> {
    if (!this.connected) {
      await this.client.connect();
      this.connected = true;
    }
  }

  async query(cypher: string, params: Record<string, any> = {}): Promise<any> {
    if (!this.connected) await this.connect();
    // FalkorDB uses the GRAPH.QUERY command
    // Format: GRAPH.QUERY <key> <query> [params]
    // Note: This is a raw Redis command wrapper.
    // Ideally use 'falkordb' node package if available, but raw redis is lighter for now.
    
    // Construct params string if needed (simple string replacement for POC)
    // Real implementation should use parameterized queries if supported by the driver/module
    return await this.client.sendCommand(['GRAPH.QUERY', this.config.graphName, cypher]);
  }

  async disconnect(): Promise<void> {
    if (this.connected) {
      await this.client.disconnect();
      this.connected = false;
    }
  }
}