import protobuf from 'protobufjs';

// Vite raw import of canonical SKY schema (outside dashboard root).
// Note: requires `vite.config.ts` fs allow-list to include `../proto`.
// eslint-disable-next-line @typescript-eslint/ban-ts-comment
// @ts-ignore
import skyProtoRaw from '../../../../proto/sky/v1/sky.proto?raw';

let cachedRoot: protobuf.Root | null = null;
let cachedSkyEnvelope: protobuf.Type | null = null;

export function getSkyRoot(): protobuf.Root {
  if (cachedRoot) return cachedRoot;
  const parsed = protobuf.parse(String(skyProtoRaw), { keepCase: true });
  cachedRoot = parsed.root;
  return cachedRoot;
}

export function getSkyEnvelopeType(): protobuf.Type {
  if (cachedSkyEnvelope) return cachedSkyEnvelope;
  const root = getSkyRoot();
  const t = root.lookupType('pluribus.sky.v1.SkyEnvelope');
  cachedSkyEnvelope = t;
  return t;
}

