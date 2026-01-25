import type protobuf from 'protobufjs';

import { SKY_MAGIC_V1, SKY_VERSION_V1 } from './constants';
import { getSkyEnvelopeType } from './proto.browser';

export function encodeSkyEnvelope(message: Record<string, unknown>): Uint8Array {
  const SkyEnvelope = getSkyEnvelopeType();
  const errMsg = SkyEnvelope.verify(message);
  if (errMsg) throw new Error(`SkyEnvelope.verify: ${errMsg}`);
  const created = SkyEnvelope.create(message);
  return SkyEnvelope.encode(created).finish();
}

export function decodeSkyEnvelope(bytes: Uint8Array): protobuf.Message<{}> & Record<string, unknown> {
  const SkyEnvelope = getSkyEnvelopeType();
  return SkyEnvelope.decode(bytes) as protobuf.Message<{}> & Record<string, unknown>;
}

export function makeHelloEnvelope(params: {
  swarmId?: string;
  sourcePeerId?: string;
  targetPeerId?: string;
  traceId?: string;
  label?: string;
  capabilities?: string[];
  preferredTransport?: string;
  tsMs?: number;
}): Record<string, unknown> {
  const tsMs = params.tsMs ?? Date.now();
  return {
    magic: SKY_MAGIC_V1,
    version: SKY_VERSION_V1,
    ts_ms: tsMs,
    trace_id: params.traceId ?? '',
    swarm_id: params.swarmId ?? '',
    source_peer_id: params.sourcePeerId ?? '',
    target_peer_id: params.targetPeerId ?? '',
    hello: {
      capabilities: params.capabilities ?? ['ws+pb'],
      preferred_transport: params.preferredTransport ?? 'ws',
      label: params.label ?? '',
    },
  };
}

