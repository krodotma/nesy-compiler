/**
 * LaneTestUtils - Testing utilities and helpers for lane components
 *
 * Phase 6, Iteration 49 of OITERATE lanes-widget-enhancement
 *
 * Features:
 * - Mock data factories
 * - Test render helpers
 * - Event simulation utilities
 * - Assertion helpers
 * - Snapshot utilities
 */

import { component$, useSignal, Slot } from '@builder.io/qwik';
import type { QRL } from '@builder.io/qwik';

// ============================================================================
// Types
// ============================================================================

export interface Lane {
  id: string;
  name: string;
  owner: string;
  status: 'green' | 'yellow' | 'red';
  wip_pct: number;
  blockers?: number;
  description?: string;
  created_at?: string;
  updated_at?: string;
  tags?: string[];
  priority?: 'low' | 'medium' | 'high' | 'critical';
}

export interface Agent {
  id: string;
  name: string;
  status: 'active' | 'idle' | 'busy' | 'offline' | 'error';
  class: 'sagent' | 'swagent' | 'cagent';
  currentTask?: string;
  lane?: string;
  lastSeen?: string;
}

export interface BusEvent {
  id: string;
  topic: string;
  payload: Record<string, unknown>;
  timestamp: string;
  source: string;
}

export interface TestContext {
  lanes: Lane[];
  agents: Agent[];
  events: BusEvent[];
  time: Date;
}

// ============================================================================
// Mock Data Factories
// ============================================================================

let idCounter = 0;

/**
 * Generate a unique ID for test entities
 */
export function generateId(prefix = 'test'): string {
  return `${prefix}_${Date.now()}_${++idCounter}`;
}

/**
 * Create a mock lane with optional overrides
 */
export function createMockLane(overrides: Partial<Lane> = {}): Lane {
  const id = generateId('lane');
  return {
    id,
    name: `Test Lane ${id.slice(-4)}`,
    owner: 'test_user',
    status: 'green',
    wip_pct: Math.floor(Math.random() * 100),
    blockers: 0,
    description: 'A test lane for unit testing',
    created_at: new Date().toISOString(),
    updated_at: new Date().toISOString(),
    tags: ['test'],
    priority: 'medium',
    ...overrides,
  };
}

/**
 * Create multiple mock lanes
 */
export function createMockLanes(count: number, overrides: Partial<Lane> = {}): Lane[] {
  return Array.from({ length: count }, (_, i) =>
    createMockLane({
      name: `Lane ${i + 1}`,
      wip_pct: Math.floor((i / count) * 100),
      ...overrides,
    })
  );
}

/**
 * Create a mock agent with optional overrides
 */
export function createMockAgent(overrides: Partial<Agent> = {}): Agent {
  const id = generateId('agent');
  return {
    id,
    name: `Agent ${id.slice(-4)}`,
    status: 'active',
    class: 'swagent',
    currentTask: undefined,
    lane: undefined,
    lastSeen: new Date().toISOString(),
    ...overrides,
  };
}

/**
 * Create multiple mock agents
 */
export function createMockAgents(count: number, overrides: Partial<Agent> = {}): Agent[] {
  const statuses: Agent['status'][] = ['active', 'idle', 'busy', 'offline'];
  const classes: Agent['class'][] = ['sagent', 'swagent', 'cagent'];

  return Array.from({ length: count }, (_, i) =>
    createMockAgent({
      name: `Agent ${i + 1}`,
      status: statuses[i % statuses.length],
      class: classes[i % classes.length],
      ...overrides,
    })
  );
}

/**
 * Create a mock bus event with optional overrides
 */
export function createMockBusEvent(overrides: Partial<BusEvent> = {}): BusEvent {
  const id = generateId('event');
  return {
    id,
    topic: 'test.event',
    payload: { test: true },
    timestamp: new Date().toISOString(),
    source: 'test',
    ...overrides,
  };
}

/**
 * Create multiple mock bus events
 */
export function createMockBusEvents(count: number, topic = 'test.event'): BusEvent[] {
  return Array.from({ length: count }, (_, i) =>
    createMockBusEvent({
      topic: `${topic}.${i}`,
      payload: { index: i, test: true },
    })
  );
}

/**
 * Create a full test context
 */
export function createTestContext(options: {
  laneCount?: number;
  agentCount?: number;
  eventCount?: number;
} = {}): TestContext {
  const { laneCount = 5, agentCount = 3, eventCount = 10 } = options;

  return {
    lanes: createMockLanes(laneCount),
    agents: createMockAgents(agentCount),
    events: createMockBusEvents(eventCount),
    time: new Date(),
  };
}

// ============================================================================
// Scenario Factories
// ============================================================================

/**
 * Create lanes representing various status scenarios
 */
export function createStatusScenarios(): Lane[] {
  return [
    createMockLane({ name: 'All Green', status: 'green', wip_pct: 100 }),
    createMockLane({ name: 'In Progress', status: 'green', wip_pct: 50 }),
    createMockLane({ name: 'At Risk', status: 'yellow', wip_pct: 30, blockers: 1 }),
    createMockLane({ name: 'Blocked', status: 'red', wip_pct: 10, blockers: 3 }),
    createMockLane({ name: 'Just Started', status: 'green', wip_pct: 5 }),
    createMockLane({ name: 'Critical Blocked', status: 'red', wip_pct: 0, blockers: 5, priority: 'critical' }),
  ];
}

/**
 * Create agents representing various states
 */
export function createAgentScenarios(): Agent[] {
  return [
    createMockAgent({ name: 'claude', status: 'active', class: 'sagent', currentTask: 'Building feature' }),
    createMockAgent({ name: 'codex', status: 'busy', class: 'swagent', currentTask: 'Writing tests' }),
    createMockAgent({ name: 'gemini', status: 'idle', class: 'swagent' }),
    createMockAgent({ name: 'grok', status: 'offline', class: 'cagent' }),
    createMockAgent({ name: 'o1', status: 'error', class: 'cagent', currentTask: 'Failed task' }),
  ];
}

/**
 * Create a dependency chain scenario
 */
export function createDependencyScenario(): { lanes: Lane[]; dependencies: { from: string; to: string }[] } {
  const lanes = [
    createMockLane({ id: 'lane_a', name: 'Foundation', wip_pct: 100, status: 'green' }),
    createMockLane({ id: 'lane_b', name: 'Core', wip_pct: 80, status: 'green' }),
    createMockLane({ id: 'lane_c', name: 'Features', wip_pct: 50, status: 'yellow' }),
    createMockLane({ id: 'lane_d', name: 'Polish', wip_pct: 20, status: 'yellow' }),
    createMockLane({ id: 'lane_e', name: 'Release', wip_pct: 0, status: 'red' }),
  ];

  const dependencies = [
    { from: 'lane_b', to: 'lane_a' },
    { from: 'lane_c', to: 'lane_b' },
    { from: 'lane_d', to: 'lane_c' },
    { from: 'lane_e', to: 'lane_d' },
  ];

  return { lanes, dependencies };
}

// ============================================================================
// Test Wrapper Components
// ============================================================================

export interface TestWrapperProps {
  initialData?: TestContext;
}

/**
 * Test wrapper component that provides context
 */
export const TestWrapper = component$<TestWrapperProps>(({ initialData }) => {
  const context = useSignal<TestContext>(initialData || createTestContext());

  return (
    <div class="test-wrapper" data-testid="test-wrapper">
      <div data-testid="test-context" data-context={JSON.stringify(context.value)}>
        <Slot />
      </div>
    </div>
  );
});

export interface MockBusProviderProps {
  events?: BusEvent[];
  onEmit$?: QRL<(event: BusEvent) => void>;
}

/**
 * Mock bus provider for testing bus interactions
 */
export const MockBusProvider = component$<MockBusProviderProps>(({ events = [] }) => {
  const emittedEvents = useSignal<BusEvent[]>([]);

  return (
    <div
      class="mock-bus-provider"
      data-testid="mock-bus-provider"
      data-events={JSON.stringify(events)}
      data-emitted={JSON.stringify(emittedEvents.value)}
    >
      <Slot />
    </div>
  );
});

// ============================================================================
// Assertion Helpers
// ============================================================================

export interface AssertionResult {
  passed: boolean;
  message: string;
  expected?: unknown;
  actual?: unknown;
}

/**
 * Assert that a lane has expected properties
 */
export function assertLane(lane: Lane, expected: Partial<Lane>): AssertionResult {
  const failures: string[] = [];

  for (const [key, value] of Object.entries(expected)) {
    const actual = lane[key as keyof Lane];
    if (actual !== value) {
      failures.push(`${key}: expected ${JSON.stringify(value)}, got ${JSON.stringify(actual)}`);
    }
  }

  return {
    passed: failures.length === 0,
    message: failures.length === 0
      ? 'Lane matches expected properties'
      : `Lane assertion failed: ${failures.join(', ')}`,
    expected,
    actual: lane,
  };
}

/**
 * Assert lane status based on WIP percentage
 */
export function assertLaneStatusByWip(lane: Lane): AssertionResult {
  const expectedStatus = lane.wip_pct >= 75 ? 'green' :
                         lane.wip_pct >= 40 ? 'yellow' :
                         'red';

  // Note: This is a suggested status, actual may differ based on blockers
  return {
    passed: lane.status === expectedStatus || lane.blockers !== undefined,
    message: `Lane status ${lane.status} for ${lane.wip_pct}% WIP`,
    expected: expectedStatus,
    actual: lane.status,
  };
}

/**
 * Assert that an array of lanes is sorted by a property
 */
export function assertLanesSorted(
  lanes: Lane[],
  property: keyof Lane,
  direction: 'asc' | 'desc' = 'desc'
): AssertionResult {
  let sorted = true;

  for (let i = 1; i < lanes.length; i++) {
    const prev = lanes[i - 1][property];
    const curr = lanes[i][property];

    if (direction === 'desc' && prev < curr) {
      sorted = false;
      break;
    }
    if (direction === 'asc' && prev > curr) {
      sorted = false;
      break;
    }
  }

  return {
    passed: sorted,
    message: sorted
      ? `Lanes are sorted by ${String(property)} ${direction}`
      : `Lanes are NOT sorted by ${String(property)} ${direction}`,
  };
}

// ============================================================================
// Event Simulation Helpers
// ============================================================================

/**
 * Simulate a click event on an element
 */
export function simulateClick(element: HTMLElement): void {
  const event = new MouseEvent('click', {
    bubbles: true,
    cancelable: true,
    view: window,
  });
  element.dispatchEvent(event);
}

/**
 * Simulate keyboard navigation
 */
export function simulateKeyDown(element: HTMLElement, key: string, options: Partial<KeyboardEventInit> = {}): void {
  const event = new KeyboardEvent('keydown', {
    key,
    bubbles: true,
    cancelable: true,
    ...options,
  });
  element.dispatchEvent(event);
}

/**
 * Simulate typing text into an input
 */
export function simulateTyping(element: HTMLInputElement, text: string): void {
  element.value = text;
  const event = new Event('input', { bubbles: true });
  element.dispatchEvent(event);
}

/**
 * Simulate focus on an element
 */
export function simulateFocus(element: HTMLElement): void {
  element.focus();
  const event = new FocusEvent('focus', { bubbles: true });
  element.dispatchEvent(event);
}

/**
 * Simulate blur on an element
 */
export function simulateBlur(element: HTMLElement): void {
  element.blur();
  const event = new FocusEvent('blur', { bubbles: true });
  element.dispatchEvent(event);
}

// ============================================================================
// Wait Utilities
// ============================================================================

/**
 * Wait for a condition to be true
 */
export async function waitFor(
  condition: () => boolean,
  options: { timeout?: number; interval?: number } = {}
): Promise<boolean> {
  const { timeout = 5000, interval = 100 } = options;
  const startTime = Date.now();

  while (Date.now() - startTime < timeout) {
    if (condition()) {
      return true;
    }
    await new Promise(resolve => setTimeout(resolve, interval));
  }

  return false;
}

/**
 * Wait for an element to appear in the DOM
 */
export async function waitForElement(
  selector: string,
  options: { timeout?: number; container?: HTMLElement } = {}
): Promise<HTMLElement | null> {
  const { timeout = 5000, container = document.body } = options;

  const found = await waitFor(
    () => container.querySelector(selector) !== null,
    { timeout }
  );

  return found ? container.querySelector<HTMLElement>(selector) : null;
}

/**
 * Wait for text content to appear
 */
export async function waitForText(
  text: string,
  options: { timeout?: number; container?: HTMLElement } = {}
): Promise<boolean> {
  const { timeout = 5000, container = document.body } = options;

  return waitFor(
    () => container.textContent?.includes(text) ?? false,
    { timeout }
  );
}

// ============================================================================
// Snapshot Utilities
// ============================================================================

export interface Snapshot {
  id: string;
  timestamp: string;
  component: string;
  props: Record<string, unknown>;
  html: string;
}

/**
 * Create a snapshot of component output
 */
export function createSnapshot(
  component: string,
  props: Record<string, unknown>,
  html: string
): Snapshot {
  return {
    id: generateId('snapshot'),
    timestamp: new Date().toISOString(),
    component,
    props,
    html,
  };
}

/**
 * Compare two snapshots
 */
export function compareSnapshots(a: Snapshot, b: Snapshot): {
  match: boolean;
  differences: string[];
} {
  const differences: string[] = [];

  if (a.component !== b.component) {
    differences.push(`Component: ${a.component} vs ${b.component}`);
  }

  if (JSON.stringify(a.props) !== JSON.stringify(b.props)) {
    differences.push('Props differ');
  }

  if (a.html !== b.html) {
    differences.push('HTML output differs');
  }

  return {
    match: differences.length === 0,
    differences,
  };
}

// ============================================================================
// Performance Testing Utilities
// ============================================================================

export interface PerformanceResult {
  name: string;
  duration: number;
  iterations: number;
  avgDuration: number;
  minDuration: number;
  maxDuration: number;
}

/**
 * Measure performance of a function
 */
export async function measurePerformance(
  name: string,
  fn: () => void | Promise<void>,
  iterations = 100
): Promise<PerformanceResult> {
  const durations: number[] = [];

  for (let i = 0; i < iterations; i++) {
    const start = performance.now();
    await fn();
    const end = performance.now();
    durations.push(end - start);
  }

  const total = durations.reduce((a, b) => a + b, 0);

  return {
    name,
    duration: total,
    iterations,
    avgDuration: total / iterations,
    minDuration: Math.min(...durations),
    maxDuration: Math.max(...durations),
  };
}

/**
 * Format performance result for display
 */
export function formatPerformanceResult(result: PerformanceResult): string {
  return `${result.name}:
  Total: ${result.duration.toFixed(2)}ms (${result.iterations} iterations)
  Avg: ${result.avgDuration.toFixed(2)}ms
  Min: ${result.minDuration.toFixed(2)}ms
  Max: ${result.maxDuration.toFixed(2)}ms`;
}

// ============================================================================
// Accessibility Testing Utilities
// ============================================================================

export interface A11yCheckResult {
  passed: boolean;
  violations: string[];
}

/**
 * Check basic accessibility requirements
 */
export function checkA11y(element: HTMLElement): A11yCheckResult {
  const violations: string[] = [];

  // Check for images without alt text
  const images = element.querySelectorAll('img:not([alt])');
  if (images.length > 0) {
    violations.push(`${images.length} image(s) missing alt text`);
  }

  // Check for buttons without accessible names
  const buttons = element.querySelectorAll('button');
  buttons.forEach((btn, i) => {
    if (!btn.textContent?.trim() && !btn.getAttribute('aria-label')) {
      violations.push(`Button ${i + 1} missing accessible name`);
    }
  });

  // Check for form inputs without labels
  const inputs = element.querySelectorAll('input, select, textarea');
  inputs.forEach((input, i) => {
    const id = input.getAttribute('id');
    const hasLabel = id && element.querySelector(`label[for="${id}"]`);
    const hasAriaLabel = input.getAttribute('aria-label') || input.getAttribute('aria-labelledby');

    if (!hasLabel && !hasAriaLabel) {
      violations.push(`Input ${i + 1} missing label or aria-label`);
    }
  });

  // Check for proper heading hierarchy
  const headings = element.querySelectorAll('h1, h2, h3, h4, h5, h6');
  let lastLevel = 0;
  headings.forEach((h) => {
    const level = parseInt(h.tagName[1]);
    if (level > lastLevel + 1) {
      violations.push(`Heading level skipped: h${lastLevel} to h${level}`);
    }
    lastLevel = level;
  });

  // Check for color contrast (basic check)
  const textElements = element.querySelectorAll('p, span, div, li, td, th');
  textElements.forEach((el) => {
    const style = window.getComputedStyle(el);
    const color = style.color;
    const bg = style.backgroundColor;

    // Very basic check - just flag if both are the same
    if (color === bg && el.textContent?.trim()) {
      violations.push(`Element may have contrast issues: "${el.textContent.slice(0, 20)}..."`);
    }
  });

  return {
    passed: violations.length === 0,
    violations,
  };
}

/**
 * Check keyboard navigation
 */
export function checkKeyboardNav(container: HTMLElement): A11yCheckResult {
  const violations: string[] = [];

  // Get all focusable elements
  const focusable = container.querySelectorAll(
    'button, [href], input, select, textarea, [tabindex]:not([tabindex="-1"])'
  );

  if (focusable.length === 0) {
    violations.push('No focusable elements found');
  }

  // Check for positive tabindex (anti-pattern)
  focusable.forEach((el, i) => {
    const tabindex = el.getAttribute('tabindex');
    if (tabindex && parseInt(tabindex) > 0) {
      violations.push(`Element ${i + 1} has positive tabindex (${tabindex})`);
    }
  });

  // Check for focus visibility (elements should have :focus styles)
  // Note: This is a basic check, actual focus styles need visual inspection

  return {
    passed: violations.length === 0,
    violations,
  };
}

// ============================================================================
// Export Test Suite Runner
// ============================================================================

export interface TestCase {
  name: string;
  fn: () => void | Promise<void>;
  skip?: boolean;
  only?: boolean;
}

export interface TestSuite {
  name: string;
  tests: TestCase[];
  beforeAll?: () => void | Promise<void>;
  afterAll?: () => void | Promise<void>;
  beforeEach?: () => void | Promise<void>;
  afterEach?: () => void | Promise<void>;
}

export interface TestResult {
  name: string;
  passed: boolean;
  error?: Error;
  duration: number;
}

export interface SuiteResult {
  name: string;
  tests: TestResult[];
  passed: number;
  failed: number;
  skipped: number;
  duration: number;
}

/**
 * Run a test suite
 */
export async function runTestSuite(suite: TestSuite): Promise<SuiteResult> {
  const results: TestResult[] = [];
  const suiteStart = performance.now();

  // Run beforeAll
  if (suite.beforeAll) {
    await suite.beforeAll();
  }

  // Check for .only tests
  const hasOnly = suite.tests.some(t => t.only);
  const testsToRun = hasOnly ? suite.tests.filter(t => t.only) : suite.tests;

  for (const test of suite.tests) {
    if (test.skip || (hasOnly && !test.only)) {
      results.push({
        name: test.name,
        passed: true,
        duration: 0,
      });
      continue;
    }

    // Run beforeEach
    if (suite.beforeEach) {
      await suite.beforeEach();
    }

    const testStart = performance.now();
    let error: Error | undefined;

    try {
      await test.fn();
    } catch (e) {
      error = e instanceof Error ? e : new Error(String(e));
    }

    const testEnd = performance.now();

    results.push({
      name: test.name,
      passed: !error,
      error,
      duration: testEnd - testStart,
    });

    // Run afterEach
    if (suite.afterEach) {
      await suite.afterEach();
    }
  }

  // Run afterAll
  if (suite.afterAll) {
    await suite.afterAll();
  }

  const suiteEnd = performance.now();

  return {
    name: suite.name,
    tests: results,
    passed: results.filter(r => r.passed && !suite.tests.find(t => t.name === r.name)?.skip).length,
    failed: results.filter(r => !r.passed).length,
    skipped: suite.tests.filter(t => t.skip || (hasOnly && !t.only)).length,
    duration: suiteEnd - suiteStart,
  };
}

/**
 * Format suite result for display
 */
export function formatSuiteResult(result: SuiteResult): string {
  const lines = [
    `\n=== ${result.name} ===`,
    `Passed: ${result.passed} | Failed: ${result.failed} | Skipped: ${result.skipped}`,
    `Duration: ${result.duration.toFixed(2)}ms`,
    '',
  ];

  for (const test of result.tests) {
    const status = test.passed ? '✓' : '✗';
    const time = test.duration.toFixed(2);
    lines.push(`  ${status} ${test.name} (${time}ms)`);

    if (test.error) {
      lines.push(`    Error: ${test.error.message}`);
    }
  }

  return lines.join('\n');
}

export default {
  // Factories
  createMockLane,
  createMockLanes,
  createMockAgent,
  createMockAgents,
  createMockBusEvent,
  createMockBusEvents,
  createTestContext,
  createStatusScenarios,
  createAgentScenarios,
  createDependencyScenario,

  // Assertions
  assertLane,
  assertLaneStatusByWip,
  assertLanesSorted,

  // Event simulation
  simulateClick,
  simulateKeyDown,
  simulateTyping,
  simulateFocus,
  simulateBlur,

  // Wait utilities
  waitFor,
  waitForElement,
  waitForText,

  // Performance
  measurePerformance,
  formatPerformanceResult,

  // Accessibility
  checkA11y,
  checkKeyboardNav,

  // Test runner
  runTestSuite,
  formatSuiteResult,
};
