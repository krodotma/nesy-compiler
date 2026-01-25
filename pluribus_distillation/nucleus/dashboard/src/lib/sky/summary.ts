import type protobuf from 'protobufjs';

export type SkyBusSummary = {
  topic: string;
  level: 'info' | 'error';
  data: Record<string, unknown>;
};

export function skyEnvelopeToBusSummary(env: protobuf.Message<{}> & Record<string, unknown>): SkyBusSummary {
  const bodyKey =
    (env['signal'] ? 'signal' :
      env['ice_config'] ? 'ice_config' :
      env['health'] ? 'health' :
      env['topology'] ? 'topology' :
      env['qr_join'] ? 'qr_join' :
      env['hello'] ? 'hello' :
      env['error'] ? 'error' :
      'unknown');

  const swarmId = String(env['swarm_id'] ?? '');
  const sourcePeerId = String(env['source_peer_id'] ?? '');
  const targetPeerId = String(env['target_peer_id'] ?? '');
  const traceId = String(env['trace_id'] ?? '');

  const summary: Record<string, unknown> = {
    swarm_id: swarmId,
    source_peer_id: sourcePeerId,
    target_peer_id: targetPeerId,
    trace_id: traceId,
    body: bodyKey,
  };

  let topic = 'sky.unknown';
  let level: 'info' | 'error' = 'info';

  if (bodyKey === 'signal') {
    const signal = (env['signal'] ?? {}) as Record<string, unknown>;
    const t = Number(signal['type'] ?? 0);
    summary['signal_type'] = t;
    topic =
      t === 0 ? 'sky.signal.offer'
        : t === 1 ? 'sky.signal.answer'
          : t === 2 ? 'sky.signal.ice_candidate'
            : t === 3 ? 'sky.signal.ice_restart'
              : 'sky.signal.unknown';
  } else if (bodyKey === 'ice_config') {
    const iceConfig = (env['ice_config'] ?? {}) as Record<string, unknown>;
    summary['stun_url_count'] = Array.isArray(iceConfig['stun_urls']) ? (iceConfig['stun_urls'] as unknown[]).length : 0;
    summary['turn_count'] = Array.isArray(iceConfig['turn']) ? (iceConfig['turn'] as unknown[]).length : 0;
    summary['force_relay'] = Boolean(iceConfig['force_relay']);
    topic = 'sky.ice.config';
  } else if (bodyKey === 'health') {
    topic = 'sky.health';
  } else if (bodyKey === 'topology') {
    topic = 'sky.topology';
  } else if (bodyKey === 'qr_join') {
    topic = 'sky.qr.join';
  } else if (bodyKey === 'hello') {
    topic = 'sky.hello';
  } else if (bodyKey === 'error') {
    topic = 'sky.error';
    level = 'error';
  }

  return { topic, level, data: summary };
}

