import protobuf from 'protobufjs';
import fs from 'node:fs';
import { fileURLToPath } from 'node:url';
import { dirname, resolve } from 'node:path';

let cachedRoot: protobuf.Root | null = null;
let cachedSkyEnvelope: protobuf.Type | null = null;

function readCanonicalSkyProto(): string {
  const here = dirname(fileURLToPath(import.meta.url));
  const protoPath = resolve(here, '../../../../proto/sky/v1/sky.proto');
  return fs.readFileSync(protoPath, 'utf-8');
}

export function getSkyRoot(): protobuf.Root {
  if (cachedRoot) return cachedRoot;
  const parsed = protobuf.parse(readCanonicalSkyProto(), { keepCase: true });
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
