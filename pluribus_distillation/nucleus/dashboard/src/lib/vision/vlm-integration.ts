/**
 * Vision Language Model Integration for SUPERWORKERS
 *
 * Routes vision tasks to optimal VLM providers (GLM-4.6V, Qwen3-VL).
 * Implements golden-ratio scoring for provider selection.
 *
 * @see https://www.marktechpost.com/2025/12/09/zhipu-ai-releases-glm-4-6v-a-128k-context-vision-language-model-with-native-tool-calling/
 * @see https://github.com/QwenLM/Qwen3-VL
 *
 * @module vision/vlm-integration
 */

import { captureFrameForVLM, CapturedFrame, PHI, FIBONACCI } from './screen-capture';

// =============================================================================
// TYPES
// =============================================================================

export type VLMProviderName =
  | 'glm-4.6v'
  | 'glm-4.6v-flash'
  | 'qwen3-vl-32b'
  | 'qwen3-vl-2b'
  | 'qwen2-vl'
  | 'local-vllm';

export type VisionTaskType =
  | 'ocr-chinese'
  | 'ocr-english'
  | 'ocr-multilingual'
  | 'gui-automation'
  | 'screenshot-to-code'
  | 'diagram-parse'
  | 'error-debug'
  | 'code-screenshot'
  | 'document-analysis'
  | 'visual-qa';

export interface VLMProvider {
  name: VLMProviderName;
  endpoint: string;
  model: string;
  apiKeyEnv?: string;
  capabilities: VisionTaskType[];
  /** Cost per 1M input tokens (USD) */
  costInput: number;
  /** Cost per 1M output tokens (USD) */
  costOutput: number;
  /** Max context length in tokens */
  contextLength: number;
  /** OCR accuracy (0-1) */
  ocrAccuracy: number;
  /** Supports native tool calling */
  toolCalling: boolean;
  /** Golden score weight (higher = prefer this provider) */
  goldenWeight: number;
}

export interface VisionRequest {
  /** Task type for routing */
  taskType: VisionTaskType;
  /** Image data (base64 or URL) */
  image?: string;
  /** Capture screen instead of using provided image */
  captureScreen?: boolean;
  /** Text prompt/question */
  prompt: string;
  /** Force specific provider */
  provider?: VLMProviderName;
  /** OCR mode: extract all text vs selective analysis */
  ocrMode?: 'full' | 'selective';
  /** Output format preference */
  outputFormat?: 'json' | 'markdown' | 'text';
  /** Max tokens for response */
  maxTokens?: number;
}

export interface VisionResponse {
  /** Provider used */
  provider: VLMProviderName;
  /** Model used */
  model: string;
  /** Response content */
  content: string;
  /** Parsed JSON if outputFormat was 'json' */
  parsed?: Record<string, unknown>;
  /** Token usage */
  usage: {
    promptTokens: number;
    completionTokens: number;
    totalTokens: number;
  };
  /** Cost in USD */
  cost: number;
  /** Latency in milliseconds */
  latencyMs: number;
  /** Golden quality score */
  goldenScore: number;
  /** Capture metadata if screen was captured */
  capture?: {
    width: number;
    height: number;
    displaySurface: string;
  };
}

export interface ChatMessage {
  role: 'system' | 'user' | 'assistant';
  content: string | ContentPart[];
}

export type ContentPart =
  | { type: 'text'; text: string }
  | { type: 'image_url'; image_url: { url: string; detail?: 'low' | 'high' | 'auto' } };

// =============================================================================
// VLM PROVIDER REGISTRY
// =============================================================================

/**
 * Registry of Vision Language Model providers with golden-ratio weighted scoring.
 */
export const VLM_PROVIDERS: Record<VLMProviderName, VLMProvider> = {
  'glm-4.6v': {
    name: 'glm-4.6v',
    endpoint: 'https://open.bigmodel.cn/api/paas/v4/chat/completions',
    model: 'glm-4.6v',
    apiKeyEnv: 'ZHIPU_API_KEY',
    capabilities: [
      'ocr-chinese', 'ocr-english', 'ocr-multilingual',
      'gui-automation', 'screenshot-to-code', 'diagram-parse',
      'error-debug', 'code-screenshot', 'document-analysis', 'visual-qa',
    ],
    costInput: 0.30,
    costOutput: 0.90,
    contextLength: 128000,
    ocrAccuracy: 0.987,
    toolCalling: true,
    goldenWeight: PHI,  // Highest priority
  },
  'glm-4.6v-flash': {
    name: 'glm-4.6v-flash',
    endpoint: 'https://open.bigmodel.cn/api/paas/v4/chat/completions',
    model: 'glm-4v-flash',
    apiKeyEnv: 'ZHIPU_API_KEY',
    capabilities: [
      'ocr-chinese', 'ocr-english', 'gui-automation',
      'screenshot-to-code', 'error-debug', 'visual-qa',
    ],
    costInput: 0,  // Free!
    costOutput: 0,
    contextLength: 128000,
    ocrAccuracy: 0.95,
    toolCalling: true,
    goldenWeight: 1.0,  // Good default
  },
  'qwen3-vl-32b': {
    name: 'qwen3-vl-32b',
    endpoint: 'http://localhost:8000/v1/chat/completions',  // Local vLLM
    model: 'Qwen/Qwen3-VL-32B-Instruct',
    capabilities: [
      'ocr-chinese', 'ocr-english', 'ocr-multilingual',
      'gui-automation', 'diagram-parse', 'code-screenshot',
      'document-analysis', 'visual-qa',
    ],
    costInput: 0,  // Local
    costOutput: 0,
    contextLength: 128000,
    ocrAccuracy: 0.993,  // Best for Chinese
    toolCalling: false,
    goldenWeight: 1 / PHI,  // Medium priority (local = slower)
  },
  'qwen3-vl-2b': {
    name: 'qwen3-vl-2b',
    endpoint: 'http://localhost:8000/v1/chat/completions',
    model: 'Qwen/Qwen3-VL-2B-Instruct',
    capabilities: [
      'ocr-english', 'visual-qa', 'error-debug',
    ],
    costInput: 0,
    costOutput: 0,
    contextLength: 32000,
    ocrAccuracy: 0.88,
    toolCalling: false,
    goldenWeight: 1 / (PHI * PHI),  // Low priority (small model)
  },
  'qwen2-vl': {
    name: 'qwen2-vl',
    endpoint: 'http://localhost:8000/v1/chat/completions',
    model: 'Qwen/Qwen2-VL-72B-Instruct',
    capabilities: [
      'ocr-chinese', 'ocr-english', 'ocr-multilingual',
      'diagram-parse', 'document-analysis', 'visual-qa',
    ],
    costInput: 0,
    costOutput: 0,
    contextLength: 32000,
    ocrAccuracy: 0.96,
    toolCalling: false,
    goldenWeight: 1.0,
  },
  'local-vllm': {
    name: 'local-vllm',
    endpoint: 'http://localhost:8000/v1/chat/completions',
    model: 'auto',  // Use whatever is loaded
    capabilities: ['visual-qa'],
    costInput: 0,
    costOutput: 0,
    contextLength: 32000,
    ocrAccuracy: 0.85,
    toolCalling: false,
    goldenWeight: 1 / (PHI * PHI * PHI),  // Fallback
  },
};

// =============================================================================
// GOLDEN RATIO PROVIDER SELECTION
// =============================================================================

/**
 * Select optimal VLM provider using golden-ratio weighted scoring.
 *
 * Scoring factors (phi-weighted):
 * 1. Task capability match (must have)
 * 2. OCR accuracy (if OCR task)
 * 3. Cost (lower = better)
 * 4. Tool calling support (if needed)
 * 5. Provider golden weight
 */
export function selectVLMProvider(
  taskType: VisionTaskType,
  options: {
    preferFree?: boolean;
    requireToolCalling?: boolean;
    preferLocal?: boolean;
  } = {}
): VLMProvider {
  const candidates = Object.values(VLM_PROVIDERS).filter(
    (p) => p.capabilities.includes(taskType)
  );

  if (candidates.length === 0) {
    // Fallback to most capable
    return VLM_PROVIDERS['glm-4.6v'];
  }

  // Score each candidate
  const scored = candidates.map((provider) => {
    let score = provider.goldenWeight;

    // OCR accuracy bonus (phi-weighted)
    if (taskType.startsWith('ocr')) {
      score *= Math.pow(provider.ocrAccuracy, PHI);
    }

    // Cost preference
    if (options.preferFree && provider.costInput === 0) {
      score *= PHI;
    }

    // Tool calling requirement
    if (options.requireToolCalling) {
      score *= provider.toolCalling ? PHI : (1 / PHI);
    }

    // Local preference
    if (options.preferLocal && provider.endpoint.includes('localhost')) {
      score *= PHI;
    }

    return { provider, score };
  });

  // Sort by score (descending) and return best
  scored.sort((a, b) => b.score - a.score);
  return scored[0].provider;
}

// =============================================================================
// SYSTEM PROMPTS
// =============================================================================

const VISION_SYSTEM_PROMPTS: Record<VisionTaskType, string> = {
  'ocr-chinese': `You are a vision-enabled SUPERWORKER specialized in Chinese OCR.
Extract ALL text from the image with maximum accuracy.
Output format: JSON with "text" (full extracted text) and "blocks" (array of {text, confidence, bbox}).
Preserve original formatting, line breaks, and structure.`,

  'ocr-english': `You are a vision-enabled SUPERWORKER specialized in English OCR.
Extract ALL text from the image with maximum accuracy.
Output format: JSON with "text" (full extracted text) and "blocks" (array of {text, confidence, bbox}).
Preserve original formatting, line breaks, and structure.`,

  'ocr-multilingual': `You are a vision-enabled SUPERWORKER specialized in multilingual OCR.
Extract ALL text from the image in any language detected.
Output format: JSON with "text" (full extracted text), "language" (detected), and "blocks" (array of {text, confidence, bbox, lang}).
Preserve original formatting and structure.`,

  'gui-automation': `You are a vision-enabled SUPERWORKER for GUI automation.
Analyze the screenshot and identify:
1. All clickable elements (buttons, links, inputs)
2. Their bounding boxes (x, y, width, height)
3. Their likely actions/purposes
4. Current state (enabled/disabled, selected, etc.)
Output format: JSON with "elements" array of {type, text, bbox, state, action}.`,

  'screenshot-to-code': `You are a vision-enabled SUPERWORKER for screenshot-to-code conversion.
Analyze the UI screenshot and generate pixel-accurate HTML/CSS/JS to recreate it.
Use modern CSS (flexbox, grid) and semantic HTML.
Output format: JSON with "html", "css", "js" fields.`,

  'diagram-parse': `You are a vision-enabled SUPERWORKER for diagram analysis.
Parse the architecture/flow diagram and extract:
1. All nodes/entities and their labels
2. All connections/arrows and their direction
3. Any annotations or labels on connections
Output format: JSON with "nodes" (array of {id, label, type}) and "edges" (array of {from, to, label}).`,

  'error-debug': `You are a vision-enabled SUPERWORKER for error debugging.
Analyze the screenshot containing error messages/stack traces:
1. Extract the error message and type
2. Identify the file/line number if visible
3. Parse the stack trace
4. Suggest potential causes and fixes
Output format: JSON with "error", "stackTrace", "suggestions" fields.`,

  'code-screenshot': `You are a vision-enabled SUPERWORKER for code analysis.
Extract and analyze the code from the screenshot:
1. Extract the code text exactly
2. Identify the programming language
3. Detect any syntax errors or issues
4. Provide brief analysis
Output format: JSON with "code", "language", "issues", "analysis" fields.`,

  'document-analysis': `You are a vision-enabled SUPERWORKER for document analysis.
Analyze the document image (PDF page, report, form):
1. Extract all text content
2. Identify document structure (headers, tables, lists)
3. Extract key-value pairs from forms
4. Summarize the document
Output format: JSON with "text", "structure", "keyValues", "summary" fields.`,

  'visual-qa': `You are a vision-enabled SUPERWORKER for visual question answering.
Answer the user's question about the image accurately and concisely.
If the question requires counting, measuring, or identifying specific elements, be precise.
Output in natural language unless JSON is specifically requested.`,
};

// =============================================================================
// CORE FUNCTIONS
// =============================================================================

/**
 * Process a vision request through the optimal VLM provider.
 */
export async function processVisionRequest(
  request: VisionRequest
): Promise<VisionResponse> {
  const startTime = performance.now();

  // Select provider
  const provider = request.provider
    ? VLM_PROVIDERS[request.provider]
    : selectVLMProvider(request.taskType, {
        preferFree: true,
        requireToolCalling: request.taskType === 'gui-automation',
      });

  // Get image data
  let imageData: string;
  let captureMetadata: VisionResponse['capture'];

  if (request.captureScreen) {
    const frame = await captureFrameForVLM();
    imageData = frame.dataUrl;
    captureMetadata = {
      width: frame.width,
      height: frame.height,
      displaySurface: frame.displaySurface,
    };
  } else if (request.image) {
    imageData = request.image;
  } else {
    throw new Error('No image provided and captureScreen not enabled');
  }

  // Build messages
  const systemPrompt = VISION_SYSTEM_PROMPTS[request.taskType];
  const messages: ChatMessage[] = [
    { role: 'system', content: systemPrompt },
    {
      role: 'user',
      content: [
        { type: 'text', text: request.prompt },
        {
          type: 'image_url',
          image_url: {
            url: imageData,
            detail: request.ocrMode === 'full' ? 'high' : 'auto',
          },
        },
      ],
    },
  ];

  // Make API request
  const apiKey = provider.apiKeyEnv
    ? (typeof process !== 'undefined' ? process.env[provider.apiKeyEnv] : undefined)
    : undefined;

  const response = await fetch(provider.endpoint, {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json',
      ...(apiKey && { Authorization: `Bearer ${apiKey}` }),
    },
    body: JSON.stringify({
      model: provider.model,
      messages,
      max_tokens: request.maxTokens || 4096,
      temperature: 0.1,  // Low for accuracy
    }),
  });

  if (!response.ok) {
    const errorText = await response.text();
    throw new Error(`VLM request failed: ${response.status} ${errorText}`);
  }

  const data = await response.json();
  const latencyMs = performance.now() - startTime;

  // Extract content
  const content = data.choices?.[0]?.message?.content || '';
  const usage = data.usage || { prompt_tokens: 0, completion_tokens: 0, total_tokens: 0 };

  // Calculate cost
  const cost =
    (usage.prompt_tokens / 1_000_000) * provider.costInput +
    (usage.completion_tokens / 1_000_000) * provider.costOutput;

  // Try to parse JSON if requested
  let parsed: Record<string, unknown> | undefined;
  if (request.outputFormat === 'json') {
    try {
      // Extract JSON from markdown code blocks if present
      const jsonMatch = content.match(/```(?:json)?\s*([\s\S]*?)```/);
      const jsonStr = jsonMatch ? jsonMatch[1] : content;
      parsed = JSON.parse(jsonStr.trim());
    } catch {
      // Parsing failed, return raw content
    }
  }

  // Calculate golden score based on response quality
  const goldenScore = calculateResponseGoldenScore({
    latencyMs,
    tokenEfficiency: content.length / usage.total_tokens,
    hasStructuredOutput: !!parsed,
    providerAccuracy: provider.ocrAccuracy,
  });

  return {
    provider: provider.name,
    model: provider.model,
    content,
    parsed,
    usage: {
      promptTokens: usage.prompt_tokens,
      completionTokens: usage.completion_tokens,
      totalTokens: usage.total_tokens,
    },
    cost,
    latencyMs,
    goldenScore,
    capture: captureMetadata,
  };
}

/**
 * Inject vision context into a SUPERWORKER chat request.
 */
export async function injectVisionContext(
  messages: ChatMessage[],
  options: {
    captureScreen?: boolean;
    imageUrl?: string;
    taskType?: VisionTaskType;
    ocrMode?: 'full' | 'selective';
  } = {}
): Promise<ChatMessage[]> {
  let imageData: string;

  if (options.captureScreen) {
    const frame = await captureFrameForVLM();
    imageData = frame.dataUrl;
  } else if (options.imageUrl) {
    imageData = options.imageUrl;
  } else {
    return messages;  // No vision context to add
  }

  // Get appropriate system prompt
  const taskType = options.taskType || 'visual-qa';
  const visionSystemPrompt = VISION_SYSTEM_PROMPTS[taskType];

  // Build vision message
  const visionMessage: ChatMessage = {
    role: 'user',
    content: [
      {
        type: 'text',
        text: options.ocrMode === 'full'
          ? 'Extract ALL text from this screenshot using OCR. Return structured JSON.'
          : 'Analyze this screenshot for context. Identify key UI elements, text, and any errors.',
      },
      {
        type: 'image_url',
        image_url: {
          url: imageData,
          detail: options.ocrMode === 'full' ? 'high' : 'auto',
        },
      },
    ],
  };

  // Prepend vision system prompt and append vision message
  return [
    { role: 'system', content: visionSystemPrompt },
    ...messages,
    visionMessage,
  ];
}

/**
 * Calculate golden score for VLM response quality.
 */
function calculateResponseGoldenScore(metrics: {
  latencyMs: number;
  tokenEfficiency: number;
  hasStructuredOutput: boolean;
  providerAccuracy: number;
}): number {
  // Normalize factors to 0-1 range
  const latencyScore = Math.max(0, 1 - metrics.latencyMs / 10000);  // 10s = 0
  const efficiencyScore = Math.min(metrics.tokenEfficiency / 10, 1);  // 10 chars/token = 1
  const structureBonus = metrics.hasStructuredOutput ? 1.0 : 0.5;
  const accuracyScore = metrics.providerAccuracy;

  // Phi-weighted geometric mean
  const factors = [
    Math.pow(latencyScore, 1 / PHI),
    Math.pow(efficiencyScore, 1 / (PHI * PHI)),
    Math.pow(structureBonus, 1 / (PHI * PHI * PHI)),
    Math.pow(accuracyScore, PHI),  // Accuracy weighted highest
  ];

  const product = factors.reduce((a, b) => a * b, 1);
  return Math.pow(product, 1 / factors.length);
}

// =============================================================================
// CONVENIENCE FUNCTIONS
// =============================================================================

/**
 * Quick OCR extraction from screen capture.
 */
export async function ocrFromScreen(
  language: 'chinese' | 'english' | 'multilingual' = 'english'
): Promise<string> {
  const taskType: VisionTaskType =
    language === 'chinese' ? 'ocr-chinese' :
    language === 'multilingual' ? 'ocr-multilingual' : 'ocr-english';

  const response = await processVisionRequest({
    taskType,
    captureScreen: true,
    prompt: 'Extract all text from this screenshot.',
    ocrMode: 'full',
    outputFormat: 'json',
  });

  if (response.parsed && typeof response.parsed.text === 'string') {
    return response.parsed.text;
  }
  return response.content;
}

/**
 * Quick GUI element detection from screen capture.
 */
export async function detectGUIElements(): Promise<Array<{
  type: string;
  text: string;
  bbox: { x: number; y: number; width: number; height: number };
  state: string;
  action: string;
}>> {
  const response = await processVisionRequest({
    taskType: 'gui-automation',
    captureScreen: true,
    prompt: 'Identify all interactive GUI elements in this screenshot.',
    outputFormat: 'json',
  });

  if (response.parsed && Array.isArray(response.parsed.elements)) {
    return response.parsed.elements;
  }
  return [];
}

/**
 * Quick error analysis from screen capture.
 */
export async function analyzeErrorScreenshot(): Promise<{
  error: string;
  stackTrace: string[];
  suggestions: string[];
}> {
  const response = await processVisionRequest({
    taskType: 'error-debug',
    captureScreen: true,
    prompt: 'Analyze this error screenshot and provide debugging insights.',
    outputFormat: 'json',
  });

  if (response.parsed) {
    return {
      error: (response.parsed.error as string) || '',
      stackTrace: (response.parsed.stackTrace as string[]) || [],
      suggestions: (response.parsed.suggestions as string[]) || [],
    };
  }
  return { error: response.content, stackTrace: [], suggestions: [] };
}

// =============================================================================
// EXPORTS
// =============================================================================

export {
  PHI,
  FIBONACCI,
  VISION_SYSTEM_PROMPTS,
};

export default {
  processVisionRequest,
  injectVisionContext,
  selectVLMProvider,
  ocrFromScreen,
  detectGUIElements,
  analyzeErrorScreenshot,
  VLM_PROVIDERS,
};
