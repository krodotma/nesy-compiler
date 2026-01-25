/**
 * VLM API Clients - Real HTTP Implementations for Vision Language Models
 *
 * Provides actual API integrations for:
 *   - GLM-4.6V (Zhipu AI) - Best for Chinese OCR, 98.7% accuracy
 *   - GLM-4.6V-Flash (Zhipu AI) - FREE tier, good for high-volume
 *   - Qwen3-VL-32B (Alibaba) - Best multilingual, 32 languages
 *   - Claude Vision (Anthropic) - Best reasoning
 *   - GPT-4 Vision (OpenAI) - Most versatile
 *
 * All clients use golden-ratio optimization for:
 *   - Request retry timing (φ-scaled backoff)
 *   - Quality scoring of responses
 *   - Token budget management
 *
 * @module vision/vlm-clients
 */

import { PHI, FIBONACCI } from './screen-capture';

// =============================================================================
// CONSTANTS
// =============================================================================

/** Retry delays using Fibonacci scaling (milliseconds) */
const RETRY_DELAYS = [
  FIBONACCI[5] * 100,   // 500ms
  FIBONACCI[7] * 100,   // 1300ms
  FIBONACCI[9] * 100,   // 3400ms
  FIBONACCI[11] * 100,  // 8900ms
] as const;

/** Default timeout (φ * 10 seconds) */
const DEFAULT_TIMEOUT = Math.round(PHI * 10000);

/** Maximum image size for API calls (in bytes) */
const MAX_IMAGE_SIZE = 20 * 1024 * 1024;  // 20MB

// =============================================================================
// TYPES
// =============================================================================

export interface VLMClientConfig {
  apiKey: string;
  baseUrl?: string;
  timeout?: number;
  maxRetries?: number;
  /** Organization ID (for OpenAI) */
  orgId?: string;
  /** Custom headers */
  headers?: Record<string, string>;
}

export interface VLMRequest {
  /** Base64 image data (without data URL prefix) */
  imageBase64: string;
  /** MIME type of the image */
  mimeType: 'image/png' | 'image/jpeg' | 'image/webp' | 'image/gif';
  /** User prompt/question about the image */
  prompt: string;
  /** System prompt (if supported) */
  systemPrompt?: string;
  /** Maximum tokens in response */
  maxTokens?: number;
  /** Temperature (0-1) */
  temperature?: number;
  /** Additional model-specific parameters */
  extraParams?: Record<string, unknown>;
}

export interface VLMResponse {
  /** The model's response text */
  content: string;
  /** Tokens used in request */
  promptTokens: number;
  /** Tokens used in response */
  completionTokens: number;
  /** Total tokens */
  totalTokens: number;
  /** Model used */
  model: string;
  /** Response latency in ms */
  latencyMs: number;
  /** Golden score for response quality */
  goldenScore: number;
  /** Raw API response (for debugging) */
  raw?: unknown;
}

export interface VLMClient {
  /** Provider name */
  name: string;
  /** Send a vision request */
  analyze(request: VLMRequest): Promise<VLMResponse>;
  /** Check if client is configured */
  isConfigured(): boolean;
  /** Get estimated cost per 1M tokens (input/output) */
  getCost(): { input: number; output: number };
}

// =============================================================================
// BASE CLIENT CLASS
// =============================================================================

abstract class BaseVLMClient implements VLMClient {
  abstract name: string;
  protected config: VLMClientConfig;
  protected retryCount = 0;

  constructor(config: VLMClientConfig) {
    this.config = {
      timeout: DEFAULT_TIMEOUT,
      maxRetries: 3,
      ...config,
    };
  }

  abstract analyze(request: VLMRequest): Promise<VLMResponse>;
  abstract getCost(): { input: number; output: number };

  isConfigured(): boolean {
    return !!this.config.apiKey;
  }

  /**
   * Make HTTP request with retry logic using Fibonacci-scaled delays.
   */
  protected async fetchWithRetry(
    url: string,
    options: RequestInit,
    attempt = 0
  ): Promise<Response> {
    try {
      const controller = new AbortController();
      const timeoutId = setTimeout(
        () => controller.abort(),
        this.config.timeout
      );

      const response = await fetch(url, {
        ...options,
        signal: controller.signal,
      });

      clearTimeout(timeoutId);

      if (!response.ok && attempt < (this.config.maxRetries ?? 3)) {
        // Check for rate limiting
        if (response.status === 429 || response.status >= 500) {
          const delay = RETRY_DELAYS[Math.min(attempt, RETRY_DELAYS.length - 1)];
          await this.sleep(delay);
          return this.fetchWithRetry(url, options, attempt + 1);
        }
      }

      return response;
    } catch (err) {
      if (attempt < (this.config.maxRetries ?? 3)) {
        const delay = RETRY_DELAYS[Math.min(attempt, RETRY_DELAYS.length - 1)];
        await this.sleep(delay);
        return this.fetchWithRetry(url, options, attempt + 1);
      }
      throw err;
    }
  }

  protected sleep(ms: number): Promise<void> {
    return new Promise(resolve => setTimeout(resolve, ms));
  }

  /**
   * Calculate golden score for response quality.
   */
  protected calculateResponseScore(response: VLMResponse): number {
    // Factors: response length, latency, token efficiency
    const lengthScore = Math.min(response.content.length / 500, 1);
    const latencyScore = Math.max(0, 1 - response.latencyMs / 10000);
    const efficiencyScore = response.completionTokens > 0
      ? Math.min(response.content.length / (response.completionTokens * 4), 1)
      : 0.5;

    // Geometric mean with φ-weighting
    return Math.pow(
      Math.pow(lengthScore, PHI) *
      Math.pow(latencyScore, 1.0) *
      Math.pow(efficiencyScore, 1 / PHI),
      1 / 3
    );
  }
}

// =============================================================================
// GLM-4.6V CLIENT (Zhipu AI)
// =============================================================================

/**
 * Client for GLM-4.6V and GLM-4.6V-Flash models.
 *
 * GLM-4.6V: Best Chinese OCR (98.7% accuracy), $0.036/1M input
 * GLM-4.6V-Flash: FREE tier, slightly lower accuracy
 *
 * @see https://open.bigmodel.cn/dev/api
 */
export class GLM4VClient extends BaseVLMClient {
  name = 'glm-4.6v';
  private useFlash: boolean;

  constructor(config: VLMClientConfig & { useFlash?: boolean }) {
    super({
      ...config,
      baseUrl: config.baseUrl || 'https://open.bigmodel.cn/api/paas/v4',
    });
    this.useFlash = config.useFlash ?? false;
    if (this.useFlash) {
      this.name = 'glm-4.6v-flash';
    }
  }

  getCost(): { input: number; output: number } {
    if (this.useFlash) {
      return { input: 0, output: 0 };  // FREE!
    }
    return { input: 0.036, output: 0.036 };  // per 1M tokens
  }

  async analyze(request: VLMRequest): Promise<VLMResponse> {
    const startTime = performance.now();

    const model = this.useFlash ? 'glm-4v-flash' : 'glm-4v';

    const payload = {
      model,
      messages: [
        ...(request.systemPrompt ? [{
          role: 'system',
          content: request.systemPrompt,
        }] : []),
        {
          role: 'user',
          content: [
            {
              type: 'image_url',
              image_url: {
                url: `data:${request.mimeType};base64,${request.imageBase64}`,
              },
            },
            {
              type: 'text',
              text: request.prompt,
            },
          ],
        },
      ],
      max_tokens: request.maxTokens || 1024,
      temperature: request.temperature ?? 0.7,
      ...request.extraParams,
    };

    const response = await this.fetchWithRetry(
      `${this.config.baseUrl}/chat/completions`,
      {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
          'Authorization': `Bearer ${this.config.apiKey}`,
          ...this.config.headers,
        },
        body: JSON.stringify(payload),
      }
    );

    if (!response.ok) {
      const errorText = await response.text();
      throw new Error(`GLM-4V API error: ${response.status} - ${errorText}`);
    }

    const data = await response.json();
    const latencyMs = performance.now() - startTime;

    const result: VLMResponse = {
      content: data.choices?.[0]?.message?.content || '',
      promptTokens: data.usage?.prompt_tokens || 0,
      completionTokens: data.usage?.completion_tokens || 0,
      totalTokens: data.usage?.total_tokens || 0,
      model,
      latencyMs,
      goldenScore: 0,
      raw: data,
    };

    result.goldenScore = this.calculateResponseScore(result);

    return result;
  }
}

// =============================================================================
// QWEN3-VL CLIENT (Alibaba)
// =============================================================================

/**
 * Client for Qwen3-VL models.
 *
 * Qwen3-VL-32B: Best multilingual (32 languages), 99.3% OCR accuracy
 *
 * @see https://help.aliyun.com/zh/dashscope/
 */
export class Qwen3VLClient extends BaseVLMClient {
  name = 'qwen3-vl-32b';
  private modelSize: '7b' | '32b' | '72b';

  constructor(config: VLMClientConfig & { modelSize?: '7b' | '32b' | '72b' }) {
    super({
      ...config,
      baseUrl: config.baseUrl || 'https://dashscope.aliyuncs.com/api/v1',
    });
    this.modelSize = config.modelSize || '32b';
    this.name = `qwen3-vl-${this.modelSize}`;
  }

  getCost(): { input: number; output: number } {
    // Approximate pricing
    switch (this.modelSize) {
      case '7b': return { input: 0.02, output: 0.06 };
      case '32b': return { input: 0.08, output: 0.24 };
      case '72b': return { input: 0.20, output: 0.60 };
    }
  }

  async analyze(request: VLMRequest): Promise<VLMResponse> {
    const startTime = performance.now();

    const model = `qwen-vl-max`;  // Qwen3-VL alias

    const payload = {
      model,
      input: {
        messages: [
          ...(request.systemPrompt ? [{
            role: 'system',
            content: request.systemPrompt,
          }] : []),
          {
            role: 'user',
            content: [
              {
                image: `data:${request.mimeType};base64,${request.imageBase64}`,
              },
              {
                text: request.prompt,
              },
            ],
          },
        ],
      },
      parameters: {
        max_tokens: request.maxTokens || 1024,
        temperature: request.temperature ?? 0.7,
        ...request.extraParams,
      },
    };

    const response = await this.fetchWithRetry(
      `${this.config.baseUrl}/services/aigc/multimodal-generation/generation`,
      {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
          'Authorization': `Bearer ${this.config.apiKey}`,
          ...this.config.headers,
        },
        body: JSON.stringify(payload),
      }
    );

    if (!response.ok) {
      const errorText = await response.text();
      throw new Error(`Qwen-VL API error: ${response.status} - ${errorText}`);
    }

    const data = await response.json();
    const latencyMs = performance.now() - startTime;

    const result: VLMResponse = {
      content: data.output?.choices?.[0]?.message?.content || '',
      promptTokens: data.usage?.input_tokens || 0,
      completionTokens: data.usage?.output_tokens || 0,
      totalTokens: (data.usage?.input_tokens || 0) + (data.usage?.output_tokens || 0),
      model,
      latencyMs,
      goldenScore: 0,
      raw: data,
    };

    result.goldenScore = this.calculateResponseScore(result);

    return result;
  }
}

// =============================================================================
// CLAUDE VISION CLIENT (Anthropic)
// =============================================================================

/**
 * Client for Claude Vision (Claude 3.5 Sonnet, Claude 3 Opus).
 *
 * Best for complex reasoning about images.
 *
 * @see https://docs.anthropic.com/claude/docs/vision
 */
export class ClaudeVisionClient extends BaseVLMClient {
  name = 'claude-3.5-sonnet';
  private modelId: string;

  constructor(config: VLMClientConfig & { modelId?: string }) {
    super({
      ...config,
      baseUrl: config.baseUrl || 'https://api.anthropic.com/v1',
    });
    this.modelId = config.modelId || 'claude-3-5-sonnet-20241022';
    this.name = this.modelId;
  }

  getCost(): { input: number; output: number } {
    if (this.modelId.includes('opus')) {
      return { input: 15.0, output: 75.0 };
    }
    if (this.modelId.includes('sonnet')) {
      return { input: 3.0, output: 15.0 };
    }
    return { input: 0.25, output: 1.25 };  // Haiku
  }

  async analyze(request: VLMRequest): Promise<VLMResponse> {
    const startTime = performance.now();

    const payload = {
      model: this.modelId,
      max_tokens: request.maxTokens || 1024,
      system: request.systemPrompt,
      messages: [
        {
          role: 'user',
          content: [
            {
              type: 'image',
              source: {
                type: 'base64',
                media_type: request.mimeType,
                data: request.imageBase64,
              },
            },
            {
              type: 'text',
              text: request.prompt,
            },
          ],
        },
      ],
      ...request.extraParams,
    };

    const response = await this.fetchWithRetry(
      `${this.config.baseUrl}/messages`,
      {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
          'x-api-key': this.config.apiKey,
          'anthropic-version': '2023-06-01',
          ...this.config.headers,
        },
        body: JSON.stringify(payload),
      }
    );

    if (!response.ok) {
      const errorText = await response.text();
      throw new Error(`Claude API error: ${response.status} - ${errorText}`);
    }

    const data = await response.json();
    const latencyMs = performance.now() - startTime;

    const content = data.content
      ?.filter((block: any) => block.type === 'text')
      .map((block: any) => block.text)
      .join('\n') || '';

    const result: VLMResponse = {
      content,
      promptTokens: data.usage?.input_tokens || 0,
      completionTokens: data.usage?.output_tokens || 0,
      totalTokens: (data.usage?.input_tokens || 0) + (data.usage?.output_tokens || 0),
      model: this.modelId,
      latencyMs,
      goldenScore: 0,
      raw: data,
    };

    result.goldenScore = this.calculateResponseScore(result);

    return result;
  }
}

// =============================================================================
// GPT-4 VISION CLIENT (OpenAI)
// =============================================================================

/**
 * Client for GPT-4 Vision (GPT-4o, GPT-4 Turbo).
 *
 * Most versatile, great for general-purpose vision tasks.
 *
 * @see https://platform.openai.com/docs/guides/vision
 */
export class GPT4VisionClient extends BaseVLMClient {
  name = 'gpt-4o';
  private modelId: string;

  constructor(config: VLMClientConfig & { modelId?: string }) {
    super({
      ...config,
      baseUrl: config.baseUrl || 'https://api.openai.com/v1',
    });
    this.modelId = config.modelId || 'gpt-4o';
    this.name = this.modelId;
  }

  getCost(): { input: number; output: number } {
    if (this.modelId === 'gpt-4o') {
      return { input: 2.5, output: 10.0 };
    }
    if (this.modelId === 'gpt-4o-mini') {
      return { input: 0.15, output: 0.60 };
    }
    return { input: 10.0, output: 30.0 };  // GPT-4 Turbo
  }

  async analyze(request: VLMRequest): Promise<VLMResponse> {
    const startTime = performance.now();

    const payload = {
      model: this.modelId,
      max_tokens: request.maxTokens || 1024,
      temperature: request.temperature ?? 0.7,
      messages: [
        ...(request.systemPrompt ? [{
          role: 'system',
          content: request.systemPrompt,
        }] : []),
        {
          role: 'user',
          content: [
            {
              type: 'image_url',
              image_url: {
                url: `data:${request.mimeType};base64,${request.imageBase64}`,
                detail: 'high',
              },
            },
            {
              type: 'text',
              text: request.prompt,
            },
          ],
        },
      ],
      ...request.extraParams,
    };

    const response = await this.fetchWithRetry(
      `${this.config.baseUrl}/chat/completions`,
      {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
          'Authorization': `Bearer ${this.config.apiKey}`,
          ...(this.config.orgId ? { 'OpenAI-Organization': this.config.orgId } : {}),
          ...this.config.headers,
        },
        body: JSON.stringify(payload),
      }
    );

    if (!response.ok) {
      const errorText = await response.text();
      throw new Error(`OpenAI API error: ${response.status} - ${errorText}`);
    }

    const data = await response.json();
    const latencyMs = performance.now() - startTime;

    const result: VLMResponse = {
      content: data.choices?.[0]?.message?.content || '',
      promptTokens: data.usage?.prompt_tokens || 0,
      completionTokens: data.usage?.completion_tokens || 0,
      totalTokens: data.usage?.total_tokens || 0,
      model: this.modelId,
      latencyMs,
      goldenScore: 0,
      raw: data,
    };

    result.goldenScore = this.calculateResponseScore(result);

    return result;
  }
}

// =============================================================================
// CLIENT FACTORY
// =============================================================================

export type VLMProviderType = 'glm-4v' | 'glm-4v-flash' | 'qwen-vl' | 'claude' | 'gpt4';

/**
 * Create a VLM client for the specified provider.
 */
export function createVLMClient(
  provider: VLMProviderType,
  config: VLMClientConfig
): VLMClient {
  switch (provider) {
    case 'glm-4v':
      return new GLM4VClient(config);
    case 'glm-4v-flash':
      return new GLM4VClient({ ...config, useFlash: true });
    case 'qwen-vl':
      return new Qwen3VLClient(config);
    case 'claude':
      return new ClaudeVisionClient(config);
    case 'gpt4':
      return new GPT4VisionClient(config);
    default:
      throw new Error(`Unknown VLM provider: ${provider}`);
  }
}

// =============================================================================
// MULTI-PROVIDER ROUTER
// =============================================================================

/**
 * Route vision requests to optimal provider based on task and availability.
 * Uses golden-ratio weighted selection.
 */
export class VLMRouter {
  private clients: Map<string, VLMClient> = new Map();
  private providerWeights: Map<string, number>;

  constructor() {
    // Default weights using golden ratio
    this.providerWeights = new Map([
      ['glm-4v', PHI],            // Best OCR
      ['glm-4v-flash', 1.0],      // FREE, good for high-volume
      ['qwen-vl', 1 / PHI],       // Multilingual
      ['claude', 1 / (PHI * PHI)], // Expensive but smart
      ['gpt4', 1 / (PHI ** 3)],   // Most expensive
    ]);
  }

  /**
   * Register a client for routing.
   */
  registerClient(provider: VLMProviderType, client: VLMClient): void {
    this.clients.set(provider, client);
  }

  /**
   * Set custom weight for a provider.
   */
  setWeight(provider: string, weight: number): void {
    this.providerWeights.set(provider, weight);
  }

  /**
   * Select best available provider based on weights and configuration.
   */
  selectProvider(
    preferredProviders?: VLMProviderType[],
    excludeProviders?: VLMProviderType[]
  ): VLMClient | null {
    // Filter to configured providers
    const available = [...this.clients.entries()]
      .filter(([name, client]) => {
        if (!client.isConfigured()) return false;
        if (excludeProviders?.includes(name as VLMProviderType)) return false;
        return true;
      });

    if (available.length === 0) return null;

    // If preferred providers specified, try those first
    if (preferredProviders?.length) {
      for (const pref of preferredProviders) {
        const client = this.clients.get(pref);
        if (client?.isConfigured()) {
          return client;
        }
      }
    }

    // Otherwise, select by weight
    available.sort((a, b) => {
      const weightA = this.providerWeights.get(a[0]) || 0;
      const weightB = this.providerWeights.get(b[0]) || 0;
      return weightB - weightA;
    });

    return available[0]?.[1] || null;
  }

  /**
   * Analyze with automatic provider selection.
   */
  async analyze(
    request: VLMRequest,
    options?: {
      preferredProviders?: VLMProviderType[];
      excludeProviders?: VLMProviderType[];
      fallbackOnError?: boolean;
    }
  ): Promise<VLMResponse & { provider: string }> {
    const client = this.selectProvider(
      options?.preferredProviders,
      options?.excludeProviders
    );

    if (!client) {
      throw new Error('No VLM provider available');
    }

    try {
      const response = await client.analyze(request);
      return { ...response, provider: client.name };
    } catch (err) {
      if (options?.fallbackOnError) {
        // Try next available provider
        const nextClient = this.selectProvider(
          undefined,
          [...(options.excludeProviders || []), client.name as VLMProviderType]
        );

        if (nextClient) {
          const response = await nextClient.analyze(request);
          return { ...response, provider: nextClient.name };
        }
      }
      throw err;
    }
  }
}

// =============================================================================
// DEFAULT EXPORT
// =============================================================================

export default {
  GLM4VClient,
  Qwen3VLClient,
  ClaudeVisionClient,
  GPT4VisionClient,
  createVLMClient,
  VLMRouter,
  RETRY_DELAYS,
  DEFAULT_TIMEOUT,
  MAX_IMAGE_SIZE,
};
