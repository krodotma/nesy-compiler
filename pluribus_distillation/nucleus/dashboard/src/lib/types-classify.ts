import { TYPES_LAYOUT, TYPES_SCHEMA, type SextetSemantic } from './types-schema';

export interface TopicTypesTag {
  nodeId: string;
  label: string;
  semantics: SextetSemantic;
}

const DEFAULT_NODE_ID = 'types-process';

const RULES: Array<{ prefixes: string[]; nodeId: string }> = [
  { prefixes: ['agent.', 'providers.', 'provider.'], nodeId: 'types-object' },
  { prefixes: ['dialogos.', 'chat.', 'prompt.', 'response.'], nodeId: 'types-symbol' },
  { prefixes: ['omega.', 'telemetry.', 'qa.', 'health.', 'monitor.'], nodeId: 'types-observer' },
  { prefixes: ['lens.', 'collimator.', 'topology.'], nodeId: 'types-shape' },
  { prefixes: ['schema.', 'spec.', 'config.', 'types.'], nodeId: 'types-type' },
  { prefixes: ['operator.', 'task.', 'rd.tasks.', 'infer_sync.', 'pb'], nodeId: 'types-process' },
];

function buildTag(nodeId: string): TopicTypesTag {
  const node = TYPES_LAYOUT.indexById[nodeId] ?? TYPES_LAYOUT.indexById[TYPES_SCHEMA.id];
  const semantics = (node?.semantics?.[0] ?? 'Type') as SextetSemantic;
  return {
    nodeId,
    label: node?.label ?? nodeId,
    semantics,
  };
}

export function classifyTopicTypes(topic: string): TopicTypesTag {
  const normalized = (topic || '').toLowerCase();
  for (const rule of RULES) {
    if (rule.prefixes.some((prefix) => normalized.startsWith(prefix))) {
      return buildTag(rule.nodeId);
    }
  }
  return buildTag(DEFAULT_NODE_ID);
}
