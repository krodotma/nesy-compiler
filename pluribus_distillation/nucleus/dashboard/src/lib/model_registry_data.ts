// Generated from nucleus/specs/model_registry.json. Do not edit by hand.
import type { ModelDef, EdgeModelDef } from './model_registry_types';

export const MODEL_REGISTRY: ModelDef[] = [
  {
    "baseId": "Nemotron-Mini-4B-Instruct",
    "name": "Nemotron Mini 4B",
    "vramBase": "2.5GB",
    "vramMB": 2500,
    "color": "lime",
    "quants": [
      "q4f16_1",
      "q4f32_1"
    ],
    "defaultQuant": "q4f16_1",
    "speedRating": 4,
    "category": "fast",
    "backend": "webllm",
    "availability": "custom",
    "capabilities": [
      "text"
    ],
    "systemPrompt": "You are Nemotron, a helpful and efficient AI assistant by NVIDIA."
  },
  {
    "baseId": "SmolLM2-135M-Instruct",
    "name": "SmolLM2 135M",
    "vramBase": "150MB",
    "vramMB": 150,
    "color": "cyan",
    "quants": [
      "q0f16",
      "q0f32",
      "q4f16_1",
      "q4f32_1"
    ],
    "defaultQuant": "q4f16_1",
    "speedRating": 5,
    "category": "ultrafast",
    "backend": "webllm",
    "availability": "prebuilt",
    "capabilities": [
      "text"
    ],
    "systemPrompt": "You are SmolLM2, a tiny but capable assistant. Be concise."
  },
  {
    "baseId": "SmolLM2-360M-Instruct",
    "name": "SmolLM2 360M",
    "vramBase": "400MB",
    "vramMB": 400,
    "color": "cyan",
    "quants": [
      "q0f16",
      "q0f32",
      "q4f16_1",
      "q4f32_1"
    ],
    "defaultQuant": "q4f16_1",
    "speedRating": 5,
    "category": "ultrafast",
    "backend": "webllm",
    "availability": "prebuilt",
    "capabilities": [
      "text"
    ],
    "systemPrompt": "You are SmolLM2, a quick and helpful assistant."
  },
  {
    "baseId": "Qwen3-0.6B",
    "name": "Qwen3 0.6B",
    "vramBase": "500MB",
    "vramMB": 500,
    "color": "green",
    "quants": [
      "q0f16",
      "q4f16_1",
      "q4f32_1"
    ],
    "defaultQuant": "q4f16_1",
    "speedRating": 5,
    "category": "ultrafast",
    "backend": "webllm",
    "availability": "prebuilt",
    "capabilities": [
      "text"
    ],
    "systemPrompt": "You are Qwen, a smart and logical assistant."
  },
  {
    "baseId": "Llama-3.2-1B-Instruct",
    "name": "Llama 3.2 1B",
    "vramBase": "800MB",
    "vramMB": 800,
    "color": "orange",
    "quants": [
      "q0f16",
      "q0f32",
      "q4f16_1",
      "q4f32_1"
    ],
    "defaultQuant": "q4f16_1",
    "speedRating": 4,
    "category": "fast",
    "backend": "webllm",
    "availability": "prebuilt",
    "capabilities": [
      "text"
    ],
    "systemPrompt": "You are Llama, a helpful and harmless assistant."
  },
  {
    "baseId": "gemma-2-2b-it",
    "name": "Gemma 2 2B",
    "vramBase": "1.5GB",
    "vramMB": 1500,
    "color": "purple",
    "quants": [
      "q4f16_1",
      "q4f32_1"
    ],
    "defaultQuant": "q4f16_1",
    "speedRating": 4,
    "category": "fast",
    "backend": "webllm",
    "availability": "prebuilt",
    "capabilities": [
      "text"
    ],
    "systemPrompt": "You are Gemma, a creative and knowledgeable assistant."
  },
  {
    "baseId": "Phi-3.5-mini-instruct",
    "name": "Phi 3.5 Mini",
    "vramBase": "2.5GB",
    "vramMB": 2500,
    "color": "blue",
    "quants": [
      "q4f16_1",
      "q4f32_1"
    ],
    "defaultQuant": "q4f16_1",
    "speedRating": 3,
    "category": "standard",
    "backend": "webllm",
    "availability": "prebuilt",
    "capabilities": [
      "text"
    ],
    "systemPrompt": "You are Phi, a reasoning-focused assistant."
  },
  {
    "baseId": "Qwen2.5-3B-Instruct",
    "name": "Qwen 2.5 3B",
    "vramBase": "2.5GB",
    "vramMB": 2500,
    "color": "green",
    "quants": [
      "q4f16_1",
      "q4f32_1"
    ],
    "defaultQuant": "q4f16_1",
    "speedRating": 3,
    "category": "standard",
    "backend": "webllm",
    "availability": "prebuilt",
    "capabilities": [
      "text"
    ],
    "systemPrompt": "You are Qwen, an intelligent assistant."
  },
  {
    "baseId": "Llama-3.2-3B-Instruct",
    "name": "Llama 3.2 3B",
    "vramBase": "3GB",
    "vramMB": 3000,
    "color": "orange",
    "quants": [
      "q4f16_1",
      "q4f32_1"
    ],
    "defaultQuant": "q4f16_1",
    "speedRating": 3,
    "category": "standard",
    "backend": "webllm",
    "availability": "prebuilt",
    "capabilities": [
      "text"
    ],
    "systemPrompt": "You are Llama, a helpful assistant."
  },
  {
    "baseId": "Phi-3.5-vision-instruct",
    "name": "Phi 3.5 Vision",
    "vramBase": "4GB",
    "vramMB": 4000,
    "color": "purple",
    "quants": [
      "q4f16_1"
    ],
    "defaultQuant": "q4f16_1",
    "speedRating": 2,
    "category": "vision",
    "backend": "webllm",
    "availability": "prebuilt",
    "capabilities": [
      "text",
      "vision"
    ],
    "systemPrompt": "You are Phi Vision, capable of analyzing images and text."
  },
  {
    "baseId": "gemma-2-9b-it",
    "name": "Gemma 2 9B",
    "vramBase": "6GB",
    "vramMB": 6000,
    "color": "purple",
    "quants": [
      "q4f16_1",
      "q4f32_1"
    ],
    "defaultQuant": "q4f16_1",
    "speedRating": 2,
    "category": "heavy",
    "backend": "webllm",
    "availability": "prebuilt",
    "capabilities": [
      "text"
    ],
    "systemPrompt": "You are Gemma 9B, a highly capable assistant."
  },
  {
    "baseId": "GLM-4-9B-Chat",
    "name": "GLM-4 9B",
    "vramBase": "6GB",
    "vramMB": 6000,
    "color": "blue",
    "quants": [
      "q4f16_1",
      "q4f32_1"
    ],
    "defaultQuant": "q4f16_1",
    "speedRating": 2,
    "category": "heavy",
    "backend": "webllm",
    "availability": "prebuilt",
    "capabilities": [
      "text"
    ],
    "systemPrompt": "You are GLM-4, a powerful open model."
  },
  {
    "baseId": "Llama-3.2-11B-Vision-Instruct",
    "name": "Llama 3.2 11B Vision",
    "vramBase": "8GB",
    "vramMB": 8000,
    "color": "orange",
    "quants": [
      "q4f16_1"
    ],
    "defaultQuant": "q4f16_1",
    "speedRating": 1,
    "category": "heavy",
    "backend": "webllm",
    "availability": "prebuilt",
    "capabilities": [
      "text",
      "vision"
    ],
    "systemPrompt": "You are Llama Vision, an advanced multimodal assistant."
  }
];

export const EDGE_MODEL_CATALOG: EdgeModelDef[] = [
  {
    "id": "silero-vad-v5",
    "name": "Silero VAD v5",
    "backend": "onnx",
    "availability": "custom",
    "sizeMB": 2.2,
    "capabilities": [
      "speech",
      "audio"
    ],
    "notes": "Auralux VAD (Silero) ONNX runtime"
  },
  {
    "id": "hubert-soft-quantized",
    "name": "HuBERT Soft (Quantized)",
    "backend": "onnx",
    "availability": "custom",
    "sizeMB": 92,
    "capabilities": [
      "speech",
      "audio"
    ],
    "notes": "Auralux SSL encoder"
  },
  {
    "id": "vocos-q8",
    "name": "Vocos Q8 Vocoder",
    "backend": "onnx",
    "availability": "custom",
    "sizeMB": 52,
    "capabilities": [
      "audio"
    ],
    "notes": "Auralux neural vocoder"
  },
  {
    "id": "vllm-llama-3.1-8b-instruct",
    "name": "Llama 3.1 8B (vLLM)",
    "backend": "vllm",
    "availability": "planned",
    "capabilities": [
      "text",
      "reasoning"
    ],
    "notes": "Server-side vLLM baseline"
  },
  {
    "id": "vllm-qwen2.5-7b-instruct",
    "name": "Qwen 2.5 7B (vLLM)",
    "backend": "vllm",
    "availability": "planned",
    "capabilities": [
      "text",
      "reasoning"
    ],
    "notes": "Server-side vLLM multilingual"
  },
  {
    "id": "vllm-mistral-7b-instruct",
    "name": "Mistral 7B (vLLM)",
    "backend": "vllm",
    "availability": "planned",
    "capabilities": [
      "text",
      "code"
    ],
    "notes": "Server-side vLLM rapid chat"
  },
  {
    "id": "vllm-deepseek-coder-6.7b-instruct",
    "name": "DeepSeek Coder 6.7B (vLLM)",
    "backend": "vllm",
    "availability": "planned",
    "capabilities": [
      "code",
      "reasoning"
    ],
    "notes": "Server-side vLLM code focus"
  },
  {
    "id": "vllm-gemma-2-9b-it",
    "name": "Gemma 2 9B (vLLM)",
    "backend": "vllm",
    "availability": "planned",
    "capabilities": [
      "text",
      "reasoning"
    ],
    "notes": "Server-side vLLM open model"
  },
  {
    "id": "vllm-llama-3.1-70b-instruct",
    "name": "Llama 3.1 70B (vLLM)",
    "backend": "vllm",
    "availability": "planned",
    "capabilities": [
      "text",
      "reasoning"
    ],
    "notes": "Server-side vLLM high-capacity"
  },
  {
    "id": "use-lite",
    "name": "Universal Sentence Encoder Lite",
    "backend": "planned",
    "availability": "planned",
    "sizeMB": 20,
    "capabilities": [
      "text"
    ],
    "notes": "Planned TFJS embeddings"
  },
  {
    "id": "mobilenet-v2",
    "name": "MobileNet V2",
    "backend": "planned",
    "availability": "planned",
    "sizeMB": 14,
    "capabilities": [
      "vision",
      "image"
    ],
    "notes": "Planned TFJS vision baseline"
  }
];
