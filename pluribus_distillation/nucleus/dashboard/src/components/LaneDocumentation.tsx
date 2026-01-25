/**
 * LaneDocumentation - Component documentation and usage examples
 *
 * Phase 6, Iteration 50 of OITERATE lanes-widget-enhancement
 *
 * Features:
 * - Component API documentation
 * - Interactive examples
 * - Props reference tables
 * - Usage patterns
 * - Best practices guide
 */

import { component$, useSignal, $ } from '@builder.io/qwik';

// ============================================================================
// Types
// ============================================================================

export interface ComponentDoc {
  name: string;
  description: string;
  category: string;
  iteration: number;
  props: PropDoc[];
  examples: ExampleDoc[];
  notes?: string[];
}

export interface PropDoc {
  name: string;
  type: string;
  required: boolean;
  default?: string;
  description: string;
}

export interface ExampleDoc {
  title: string;
  description: string;
  code: string;
}

// ============================================================================
// Component Documentation Data
// ============================================================================

export const componentDocs: ComponentDoc[] = [
  // Phase 1: Immediate Fix (1-5)
  {
    name: 'LanesWidget',
    description: 'Core lanes display widget with real-time updates and filtering',
    category: 'Core',
    iteration: 1,
    props: [
      { name: 'lanes', type: 'Lane[]', required: true, description: 'Array of lane objects to display' },
      { name: 'showAll', type: 'boolean', required: false, default: 'false', description: 'Show all lanes vs collapsed view' },
      { name: 'onLaneClick$', type: 'QRL<(lane: Lane) => void>', required: false, description: 'Callback when a lane is clicked' },
    ],
    examples: [
      {
        title: 'Basic Usage',
        description: 'Display lanes with default settings',
        code: `<LanesWidget lanes={lanes} />`,
      },
      {
        title: 'With Click Handler',
        description: 'Handle lane selection',
        code: `<LanesWidget
  lanes={lanes}
  onLaneClick$={$((lane) => console.log('Selected:', lane.name))}
/>`,
      },
    ],
    notes: [
      'Automatically polls for updates every 30 seconds',
      'Supports status-based filtering (green, yellow, red)',
    ],
  },

  // Phase 2: Infinite Scroll & History (6-15)
  {
    name: 'LaneInfiniteScroll',
    description: 'Infinite scroll implementation for large lane lists',
    category: 'Performance',
    iteration: 6,
    props: [
      { name: 'lanes', type: 'Lane[]', required: true, description: 'Array of all lane objects' },
      { name: 'pageSize', type: 'number', required: false, default: '20', description: 'Number of items per page' },
      { name: 'threshold', type: 'number', required: false, default: '0.8', description: 'Scroll threshold for loading more' },
    ],
    examples: [
      {
        title: 'Basic Infinite Scroll',
        description: 'Load lanes as user scrolls',
        code: `<LaneInfiniteScroll lanes={allLanes} pageSize={25} />`,
      },
    ],
  },
  {
    name: 'LaneHistoryView',
    description: 'Historical view of lane changes over time',
    category: 'History',
    iteration: 7,
    props: [
      { name: 'laneId', type: 'string', required: true, description: 'Lane ID to show history for' },
      { name: 'range', type: "'24h' | '7d' | '30d'", required: false, default: "'7d'", description: 'Time range to display' },
    ],
    examples: [
      {
        title: 'Lane History',
        description: 'Show 7-day history for a lane',
        code: `<LaneHistoryView laneId="lane_123" range="7d" />`,
      },
    ],
  },
  {
    name: 'LaneTimeline',
    description: 'Timeline visualization of lane events',
    category: 'History',
    iteration: 8,
    props: [
      { name: 'events', type: 'TimelineEvent[]', required: true, description: 'Array of timeline events' },
      { name: 'zoom', type: 'number', required: false, default: '1', description: 'Zoom level (0.5 to 2)' },
    ],
    examples: [
      {
        title: 'Event Timeline',
        description: 'Display lane events on a timeline',
        code: `<LaneTimeline events={laneEvents} zoom={1.5} />`,
      },
    ],
  },
  {
    name: 'LaneHistorySearch',
    description: 'Search through lane history with filters',
    category: 'History',
    iteration: 9,
    props: [
      { name: 'onSearch$', type: 'QRL<(query: SearchQuery) => void>', required: true, description: 'Search callback' },
      { name: 'filters', type: 'FilterOption[]', required: false, description: 'Available filter options' },
    ],
    examples: [
      {
        title: 'History Search',
        description: 'Search lane history',
        code: `<LaneHistorySearch onSearch$={$((q) => searchHistory(q))} />`,
      },
    ],
  },
  {
    name: 'LaneHistoryChart',
    description: 'Chart visualization of lane metrics over time',
    category: 'History',
    iteration: 10,
    props: [
      { name: 'data', type: 'ChartData[]', required: true, description: 'Data points to chart' },
      { name: 'type', type: "'line' | 'bar' | 'area'", required: false, default: "'line'", description: 'Chart type' },
    ],
    examples: [
      {
        title: 'WIP Progress Chart',
        description: 'Show WIP progress over time',
        code: `<LaneHistoryChart data={wipData} type="area" />`,
      },
    ],
  },

  // Phase 3: Human-in-the-Loop Controls (16-25)
  {
    name: 'LaneApprovalGate',
    description: 'Approval gates for lane state changes',
    category: 'Controls',
    iteration: 16,
    props: [
      { name: 'lane', type: 'Lane', required: true, description: 'Lane requiring approval' },
      { name: 'onApprove$', type: 'QRL<(comment?: string) => void>', required: true, description: 'Approval callback' },
      { name: 'onReject$', type: 'QRL<(reason: string) => void>', required: true, description: 'Rejection callback' },
    ],
    examples: [
      {
        title: 'Approval Gate',
        description: 'Gate lane state changes',
        code: `<LaneApprovalGate
  lane={lane}
  onApprove$={$((c) => approveLane(lane.id, c))}
  onReject$={$((r) => rejectLane(lane.id, r))}
/>`,
      },
    ],
  },
  {
    name: 'LaneManualOverride',
    description: 'Manual override controls for lane properties',
    category: 'Controls',
    iteration: 17,
    props: [
      { name: 'lane', type: 'Lane', required: true, description: 'Lane to override' },
      { name: 'onOverride$', type: 'QRL<(changes: Partial<Lane>) => void>', required: true, description: 'Override callback' },
    ],
    examples: [
      {
        title: 'Manual Override',
        description: 'Override lane status',
        code: `<LaneManualOverride
  lane={lane}
  onOverride$={$((changes) => updateLane(lane.id, changes))}
/>`,
      },
    ],
  },
  {
    name: 'LaneNotifications',
    description: 'Notification management for lane events',
    category: 'Controls',
    iteration: 18,
    props: [
      { name: 'laneId', type: 'string', required: true, description: 'Lane ID for notifications' },
      { name: 'channels', type: "'email' | 'slack' | 'webhook'[]", required: false, description: 'Notification channels' },
    ],
    examples: [
      {
        title: 'Lane Notifications',
        description: 'Configure lane notifications',
        code: `<LaneNotifications laneId="lane_123" channels={['slack', 'email']} />`,
      },
    ],
  },
  {
    name: 'LaneAssignment',
    description: 'Assign lanes to agents or users',
    category: 'Controls',
    iteration: 19,
    props: [
      { name: 'lane', type: 'Lane', required: true, description: 'Lane to assign' },
      { name: 'agents', type: 'Agent[]', required: true, description: 'Available agents' },
      { name: 'onAssign$', type: 'QRL<(agentId: string) => void>', required: true, description: 'Assignment callback' },
    ],
    examples: [
      {
        title: 'Assign Lane',
        description: 'Assign lane to an agent',
        code: `<LaneAssignment
  lane={lane}
  agents={availableAgents}
  onAssign$={$((id) => assignLane(lane.id, id))}
/>`,
      },
    ],
  },

  // Phase 4: Advanced Visualization (26-35)
  {
    name: 'LaneNetworkGraph',
    description: 'Force-directed graph showing lane relationships',
    category: 'Visualization',
    iteration: 28,
    props: [
      { name: 'lanes', type: 'Lane[]', required: true, description: 'Lanes to visualize' },
      { name: 'dependencies', type: 'Dependency[]', required: false, description: 'Lane dependencies' },
      { name: 'width', type: 'number', required: false, default: '600', description: 'Graph width' },
      { name: 'height', type: 'number', required: false, default: '400', description: 'Graph height' },
    ],
    examples: [
      {
        title: 'Dependency Graph',
        description: 'Visualize lane dependencies',
        code: `<LaneNetworkGraph lanes={lanes} dependencies={deps} />`,
      },
    ],
  },
  {
    name: 'LaneHeatmap',
    description: 'Heatmap visualization of lane activity',
    category: 'Visualization',
    iteration: 29,
    props: [
      { name: 'data', type: 'HeatmapData[]', required: true, description: 'Heatmap data points' },
      { name: 'metric', type: "'wip' | 'activity' | 'blockers'", required: false, default: "'activity'", description: 'Metric to display' },
    ],
    examples: [
      {
        title: 'Activity Heatmap',
        description: 'Show lane activity over time',
        code: `<LaneHeatmap data={activityData} metric="activity" />`,
      },
    ],
  },
  {
    name: 'LaneMiniWidget',
    description: 'Compact lane status widget for dashboards',
    category: 'Visualization',
    iteration: 30,
    props: [
      { name: 'lane', type: 'Lane', required: true, description: 'Lane to display' },
      { name: 'compact', type: 'boolean', required: false, default: 'false', description: 'Extra compact mode' },
    ],
    examples: [
      {
        title: 'Mini Widget',
        description: 'Compact lane display',
        code: `<LaneMiniWidget lane={lane} compact />`,
      },
    ],
  },
  {
    name: 'LaneRealtime',
    description: 'Real-time lane updates via WebSocket',
    category: 'Visualization',
    iteration: 31,
    props: [
      { name: 'laneIds', type: 'string[]', required: true, description: 'Lane IDs to monitor' },
      { name: 'onUpdate$', type: 'QRL<(update: LaneUpdate) => void>', required: false, description: 'Update callback' },
    ],
    examples: [
      {
        title: 'Real-time Updates',
        description: 'Subscribe to lane updates',
        code: `<LaneRealtime
  laneIds={['lane_1', 'lane_2']}
  onUpdate$={$((u) => handleUpdate(u))}
/>`,
      },
    ],
  },
  {
    name: 'LaneForecasting',
    description: 'ML-based lane completion forecasting',
    category: 'Visualization',
    iteration: 33,
    props: [
      { name: 'lane', type: 'Lane', required: true, description: 'Lane to forecast' },
      { name: 'history', type: 'HistoryPoint[]', required: true, description: 'Historical data' },
    ],
    examples: [
      {
        title: 'Completion Forecast',
        description: 'Predict lane completion',
        code: `<LaneForecasting lane={lane} history={laneHistory} />`,
      },
    ],
  },

  // Phase 5: Multi-Agent Orchestration (36-45)
  {
    name: 'AgentCoordinationDashboard',
    description: 'Dashboard for coordinating multiple agents',
    category: 'Orchestration',
    iteration: 36,
    props: [
      { name: 'agents', type: 'Agent[]', required: true, description: 'Agents to display' },
      { name: 'onAssignTask$', type: 'QRL<(agentId: string, task: Task) => void>', required: false, description: 'Task assignment callback' },
    ],
    examples: [
      {
        title: 'Agent Dashboard',
        description: 'Coordinate agents',
        code: `<AgentCoordinationDashboard agents={agents} />`,
      },
    ],
  },
  {
    name: 'CrossLaneDependencies',
    description: 'Visualize and manage cross-lane dependencies',
    category: 'Orchestration',
    iteration: 37,
    props: [
      { name: 'lanes', type: 'Lane[]', required: true, description: 'All lanes' },
      { name: 'dependencies', type: 'Dependency[]', required: true, description: 'Dependency relationships' },
    ],
    examples: [
      {
        title: 'Dependencies View',
        description: 'Show cross-lane dependencies',
        code: `<CrossLaneDependencies lanes={lanes} dependencies={deps} />`,
      },
    ],
  },
  {
    name: 'BusEventMonitor',
    description: 'Monitor Pluribus bus events in real-time',
    category: 'Orchestration',
    iteration: 39,
    props: [
      { name: 'topics', type: 'string[]', required: false, description: 'Topics to filter' },
      { name: 'maxEvents', type: 'number', required: false, default: '100', description: 'Max events to display' },
    ],
    examples: [
      {
        title: 'Bus Monitor',
        description: 'Monitor bus events',
        code: `<BusEventMonitor topics={['lane.*', 'agent.*']} />`,
      },
    ],
  },
  {
    name: 'MultiAgentChat',
    description: 'Chat interface for agent-to-agent communication',
    category: 'Orchestration',
    iteration: 44,
    props: [
      { name: 'agents', type: 'Agent[]', required: true, description: 'Participating agents' },
      { name: 'onMessage$', type: 'QRL<(msg: Message) => void>', required: false, description: 'Message callback' },
    ],
    examples: [
      {
        title: 'Agent Chat',
        description: 'Inter-agent communication',
        code: `<MultiAgentChat agents={agents} />`,
      },
    ],
  },
  {
    name: 'SystemHealthMonitor',
    description: 'System-wide health monitoring dashboard',
    category: 'Orchestration',
    iteration: 45,
    props: [
      { name: 'services', type: 'Service[]', required: true, description: 'Services to monitor' },
      { name: 'refreshInterval', type: 'number', required: false, default: '5000', description: 'Refresh interval (ms)' },
    ],
    examples: [
      {
        title: 'Health Monitor',
        description: 'Monitor system health',
        code: `<SystemHealthMonitor services={services} />`,
      },
    ],
  },

  // Phase 6: Production Hardening (46-50)
  {
    name: 'LaneErrorBoundary',
    description: 'Error boundary with recovery options',
    category: 'Production',
    iteration: 46,
    props: [
      { name: 'componentName', type: 'string', required: false, default: "'Unknown'", description: 'Component name for errors' },
      { name: 'maxRetries', type: 'number', required: false, default: '3', description: 'Max retry attempts' },
      { name: 'showDetails', type: 'boolean', required: false, default: 'false', description: 'Show error details (dev mode)' },
    ],
    examples: [
      {
        title: 'Error Boundary',
        description: 'Wrap components with error handling',
        code: `<LaneErrorBoundary componentName="LanesWidget" maxRetries={3}>
  <LanesWidget lanes={lanes} />
</LaneErrorBoundary>`,
      },
    ],
  },
  {
    name: 'VirtualizedList',
    description: 'Virtualized list for efficient rendering of large datasets',
    category: 'Production',
    iteration: 47,
    props: [
      { name: 'items', type: 'T[]', required: true, description: 'Items to render' },
      { name: 'itemHeight', type: 'number', required: true, description: 'Fixed item height' },
      { name: 'containerHeight', type: 'number', required: true, description: 'Container height' },
      { name: 'overscan', type: 'number', required: false, default: '3', description: 'Extra items to render' },
      { name: 'renderItem$', type: 'QRL<(item: T, index: number) => JSX.Element>', required: true, description: 'Item render function' },
    ],
    examples: [
      {
        title: 'Virtualized Lanes',
        description: 'Efficiently render many lanes',
        code: `<VirtualizedList
  items={lanes}
  itemHeight={60}
  containerHeight={400}
  renderItem$={$((lane) => <LaneCard lane={lane} />)}
/>`,
      },
    ],
  },
  {
    name: 'AccessibleLaneCard',
    description: 'Accessible lane card with ARIA support',
    category: 'Production',
    iteration: 48,
    props: [
      { name: 'lane', type: 'Lane', required: true, description: 'Lane to display' },
      { name: 'isSelected', type: 'boolean', required: false, default: 'false', description: 'Whether selected' },
      { name: 'onClick$', type: 'QRL<() => void>', required: false, description: 'Click callback' },
    ],
    examples: [
      {
        title: 'Accessible Card',
        description: 'Lane card with full accessibility',
        code: `<AccessibleLaneCard lane={lane} isSelected={selected} />`,
      },
    ],
  },
  {
    name: 'LaneTestUtils',
    description: 'Testing utilities for lane components',
    category: 'Production',
    iteration: 49,
    props: [],
    examples: [
      {
        title: 'Mock Data',
        description: 'Create mock lanes for testing',
        code: `import { createMockLanes, createTestContext } from './LaneTestUtils';

const lanes = createMockLanes(10);
const context = createTestContext({ laneCount: 5, agentCount: 3 });`,
      },
      {
        title: 'Run Tests',
        description: 'Run a test suite',
        code: `import { runTestSuite, formatSuiteResult } from './LaneTestUtils';

const result = await runTestSuite({
  name: 'Lane Tests',
  tests: [
    { name: 'renders correctly', fn: () => { /* test */ } },
  ],
});
console.log(formatSuiteResult(result));`,
      },
    ],
  },
];

// ============================================================================
// Documentation Component
// ============================================================================

export const LaneDocumentation = component$(() => {
  const selectedCategory = useSignal<string>('all');
  const searchQuery = useSignal('');
  const expandedComponent = useSignal<string | null>(null);

  // Get unique categories
  const categories = ['all', ...new Set(componentDocs.map(d => d.category))];

  // Filter components
  const filteredDocs = componentDocs.filter(doc => {
    const matchesCategory = selectedCategory.value === 'all' || doc.category === selectedCategory.value;
    const matchesSearch = searchQuery.value === '' ||
      doc.name.toLowerCase().includes(searchQuery.value.toLowerCase()) ||
      doc.description.toLowerCase().includes(searchQuery.value.toLowerCase());
    return matchesCategory && matchesSearch;
  });

  const toggleExpand = $((name: string) => {
    expandedComponent.value = expandedComponent.value === name ? null : name;
  });

  return (
    <div class="p-4 max-w-6xl mx-auto">
      {/* Header */}
      <div class="mb-8">
        <h1 class="text-2xl font-bold text-foreground mb-2">
          Lane Widget Components
        </h1>
        <p class="text-sm text-muted-foreground">
          Complete documentation for the 50-iteration OITERATE lanes-widget-enhancement
        </p>
      </div>

      {/* Search and Filters */}
      <div class="flex flex-wrap gap-4 mb-6">
        <input
          type="text"
          placeholder="Search components..."
          value={searchQuery.value}
          onInput$={(e) => searchQuery.value = (e.target as HTMLInputElement).value}
          class="flex-1 min-w-[200px] px-4 py-2 text-sm rounded-lg border border-border bg-card text-foreground placeholder:text-muted-foreground focus:outline-none focus:ring-2 focus:ring-primary"
        />
        <div class="flex gap-2 flex-wrap">
          {categories.map(cat => (
            <button
              key={cat}
              onClick$={() => selectedCategory.value = cat}
              class={`px-3 py-1.5 text-xs rounded-lg transition-colors ${
                selectedCategory.value === cat
                  ? 'bg-primary text-primary-foreground'
                  : 'bg-muted/30 text-muted-foreground hover:bg-muted/50'
              }`}
            >
              {cat.charAt(0).toUpperCase() + cat.slice(1)}
            </button>
          ))}
        </div>
      </div>

      {/* Stats */}
      <div class="grid grid-cols-2 md:grid-cols-4 gap-4 mb-8">
        <div class="p-4 rounded-lg border border-border bg-card">
          <div class="text-2xl font-bold text-foreground">{componentDocs.length}</div>
          <div class="text-xs text-muted-foreground">Total Components</div>
        </div>
        <div class="p-4 rounded-lg border border-border bg-card">
          <div class="text-2xl font-bold text-emerald-400">50</div>
          <div class="text-xs text-muted-foreground">Iterations</div>
        </div>
        <div class="p-4 rounded-lg border border-border bg-card">
          <div class="text-2xl font-bold text-cyan-400">6</div>
          <div class="text-xs text-muted-foreground">Phases</div>
        </div>
        <div class="p-4 rounded-lg border border-border bg-card">
          <div class="text-2xl font-bold text-amber-400">{categories.length - 1}</div>
          <div class="text-xs text-muted-foreground">Categories</div>
        </div>
      </div>

      {/* Component List */}
      <div class="space-y-4">
        {filteredDocs.map(doc => (
          <div
            key={doc.name}
            class="rounded-lg border border-border bg-card overflow-hidden"
          >
            {/* Header */}
            <button
              onClick$={() => toggleExpand(doc.name)}
              class="w-full p-4 text-left hover:bg-muted/5 transition-colors"
            >
              <div class="flex items-center justify-between">
                <div class="flex items-center gap-3">
                  <span class="text-xs px-2 py-0.5 rounded bg-primary/20 text-primary">
                    #{doc.iteration}
                  </span>
                  <span class="font-medium text-foreground">{doc.name}</span>
                  <span class="text-xs px-2 py-0.5 rounded bg-muted/30 text-muted-foreground">
                    {doc.category}
                  </span>
                </div>
                <span class="text-muted-foreground">
                  {expandedComponent.value === doc.name ? '▼' : '▶'}
                </span>
              </div>
              <p class="text-sm text-muted-foreground mt-1">{doc.description}</p>
            </button>

            {/* Expanded Content */}
            {expandedComponent.value === doc.name && (
              <div class="px-4 pb-4 border-t border-border/50">
                {/* Props Table */}
                {doc.props.length > 0 && (
                  <div class="mt-4">
                    <h3 class="text-xs font-semibold text-muted-foreground mb-2">PROPS</h3>
                    <div class="overflow-x-auto">
                      <table class="w-full text-xs">
                        <thead>
                          <tr class="border-b border-border/50">
                            <th class="text-left p-2 text-muted-foreground font-medium">Name</th>
                            <th class="text-left p-2 text-muted-foreground font-medium">Type</th>
                            <th class="text-left p-2 text-muted-foreground font-medium">Required</th>
                            <th class="text-left p-2 text-muted-foreground font-medium">Default</th>
                            <th class="text-left p-2 text-muted-foreground font-medium">Description</th>
                          </tr>
                        </thead>
                        <tbody>
                          {doc.props.map(prop => (
                            <tr key={prop.name} class="border-b border-border/30">
                              <td class="p-2 font-mono text-cyan-400">{prop.name}</td>
                              <td class="p-2 font-mono text-amber-400">{prop.type}</td>
                              <td class="p-2">
                                <span class={prop.required ? 'text-red-400' : 'text-muted-foreground'}>
                                  {prop.required ? 'Yes' : 'No'}
                                </span>
                              </td>
                              <td class="p-2 font-mono text-muted-foreground">{prop.default || '-'}</td>
                              <td class="p-2 text-foreground">{prop.description}</td>
                            </tr>
                          ))}
                        </tbody>
                      </table>
                    </div>
                  </div>
                )}

                {/* Examples */}
                {doc.examples.length > 0 && (
                  <div class="mt-4">
                    <h3 class="text-xs font-semibold text-muted-foreground mb-2">EXAMPLES</h3>
                    <div class="space-y-3">
                      {doc.examples.map((example, i) => (
                        <div key={i} class="rounded-lg bg-muted/10 p-3">
                          <div class="flex items-center justify-between mb-2">
                            <span class="text-xs font-medium text-foreground">{example.title}</span>
                          </div>
                          <p class="text-[10px] text-muted-foreground mb-2">{example.description}</p>
                          <pre class="text-[10px] font-mono text-cyan-300 bg-black/30 p-2 rounded overflow-x-auto">
                            {example.code}
                          </pre>
                        </div>
                      ))}
                    </div>
                  </div>
                )}

                {/* Notes */}
                {doc.notes && doc.notes.length > 0 && (
                  <div class="mt-4">
                    <h3 class="text-xs font-semibold text-muted-foreground mb-2">NOTES</h3>
                    <ul class="text-xs text-muted-foreground space-y-1">
                      {doc.notes.map((note, i) => (
                        <li key={i} class="flex items-start gap-2">
                          <span class="text-primary">•</span>
                          <span>{note}</span>
                        </li>
                      ))}
                    </ul>
                  </div>
                )}
              </div>
            )}
          </div>
        ))}
      </div>

      {/* Footer */}
      <div class="mt-8 p-4 rounded-lg border border-border/30 bg-muted/5 text-center">
        <p class="text-xs text-muted-foreground">
          OITERATE lanes-widget-enhancement • 50 iterations • 6 phases • Complete
        </p>
      </div>
    </div>
  );
});

// ============================================================================
// Quick Reference Component
// ============================================================================

export const QuickReference = component$(() => {
  return (
    <div class="p-4 rounded-lg border border-border bg-card">
      <h2 class="text-sm font-semibold text-foreground mb-4">Quick Reference</h2>

      {/* Phase Overview */}
      <div class="space-y-3">
        <div class="p-3 rounded bg-muted/10">
          <div class="flex items-center gap-2 mb-1">
            <span class="text-xs font-bold text-emerald-400">Phase 1</span>
            <span class="text-xs text-muted-foreground">Immediate Fix (1-5)</span>
          </div>
          <p class="text-[10px] text-muted-foreground">Core lane display, filtering, collapsible sections</p>
        </div>

        <div class="p-3 rounded bg-muted/10">
          <div class="flex items-center gap-2 mb-1">
            <span class="text-xs font-bold text-cyan-400">Phase 2</span>
            <span class="text-xs text-muted-foreground">Infinite Scroll & History (6-15)</span>
          </div>
          <p class="text-[10px] text-muted-foreground">Pagination, history views, timeline, search</p>
        </div>

        <div class="p-3 rounded bg-muted/10">
          <div class="flex items-center gap-2 mb-1">
            <span class="text-xs font-bold text-amber-400">Phase 3</span>
            <span class="text-xs text-muted-foreground">Human-in-the-Loop (16-25)</span>
          </div>
          <p class="text-[10px] text-muted-foreground">Approvals, overrides, notifications, assignments</p>
        </div>

        <div class="p-3 rounded bg-muted/10">
          <div class="flex items-center gap-2 mb-1">
            <span class="text-xs font-bold text-purple-400">Phase 4</span>
            <span class="text-xs text-muted-foreground">Advanced Visualization (26-35)</span>
          </div>
          <p class="text-[10px] text-muted-foreground">Network graphs, heatmaps, real-time, forecasting</p>
        </div>

        <div class="p-3 rounded bg-muted/10">
          <div class="flex items-center gap-2 mb-1">
            <span class="text-xs font-bold text-pink-400">Phase 5</span>
            <span class="text-xs text-muted-foreground">Multi-Agent Orchestration (36-45)</span>
          </div>
          <p class="text-[10px] text-muted-foreground">Agent coordination, dependencies, bus monitoring</p>
        </div>

        <div class="p-3 rounded bg-muted/10">
          <div class="flex items-center gap-2 mb-1">
            <span class="text-xs font-bold text-red-400">Phase 6</span>
            <span class="text-xs text-muted-foreground">Production Hardening (46-50)</span>
          </div>
          <p class="text-[10px] text-muted-foreground">Error handling, performance, accessibility, testing, docs</p>
        </div>
      </div>
    </div>
  );
});

// ============================================================================
// Best Practices Component
// ============================================================================

export const BestPractices = component$(() => {
  const practices = [
    {
      title: 'Use Error Boundaries',
      description: 'Wrap lane components with LaneErrorBoundary for graceful error handling',
      code: '<LaneErrorBoundary><LanesWidget ... /></LaneErrorBoundary>',
    },
    {
      title: 'Virtualize Large Lists',
      description: 'Use VirtualizedList for lists with >100 items',
      code: '<VirtualizedList items={lanes} itemHeight={60} ... />',
    },
    {
      title: 'Enable Accessibility',
      description: 'Use AccessibleLaneCard for proper ARIA support',
      code: '<AccessibleLaneCard lane={lane} isSelected={selected} />',
    },
    {
      title: 'Debounce Updates',
      description: 'Use useDebounce for search inputs and frequent updates',
      code: 'const debouncedSearch = useDebounce(searchQuery, 300);',
    },
    {
      title: 'Handle WebSocket Reconnection',
      description: 'LaneRealtime handles reconnection automatically, but handle disconnected state in UI',
      code: '<LaneRealtime laneIds={ids} onDisconnect$={showOfflineBanner} />',
    },
  ];

  return (
    <div class="p-4 rounded-lg border border-border bg-card">
      <h2 class="text-sm font-semibold text-foreground mb-4">Best Practices</h2>
      <div class="space-y-4">
        {practices.map((practice, i) => (
          <div key={i} class="p-3 rounded bg-muted/10">
            <h3 class="text-xs font-medium text-foreground mb-1">{practice.title}</h3>
            <p class="text-[10px] text-muted-foreground mb-2">{practice.description}</p>
            <pre class="text-[9px] font-mono text-cyan-300 bg-black/30 p-2 rounded overflow-x-auto">
              {practice.code}
            </pre>
          </div>
        ))}
      </div>
    </div>
  );
});

export default LaneDocumentation;
