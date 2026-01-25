// Shared model registry types (WebLLM + edge catalogs)
export type QuantType = 'q0f16' | 'q0f32' | 'q4f16_1' | 'q4f32_1' | 'q3f16_1';
export type InferenceBackend = 'webllm' | 'tfjs' | 'onnx';
export type ModelBackend = 'webllm' | 'tfjs' | 'onnx' | 'planned';
export type ModelAvailability = 'prebuilt' | 'custom' | 'planned';
export type ModelCapability = 'text' | 'vision' | 'code' | 'reasoning' | 'speech' | 'audio' | 'image' | 'video' | 'spatial';
export type EdgeBackend = 'onnx' | 'tfjs' | 'vllm' | 'planned';

export interface BackendStatus {
  available: boolean;
  version: string | null;
  error: string | null;
}

export interface ModelDef {
  baseId: string;
  name: string;
  vramBase: string;
  vramMB: number;
  color: string;
  quants: QuantType[];
  defaultQuant: QuantType;
  speedRating: number; // 1-5, higher = faster
  category: 'ultrafast' | 'fast' | 'standard' | 'vision' | 'heavy';
  backend: ModelBackend;
  availability: ModelAvailability;
  capabilities: ModelCapability[];
  systemPrompt?: string;
}

export interface EdgeModelDef {
  id: string;
  name: string;
  backend: EdgeBackend;
  availability: ModelAvailability;
  sizeMB?: number;
  capabilities: ModelCapability[];
  notes?: string;
}
