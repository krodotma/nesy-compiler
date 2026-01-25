/**
 * ChromaticVisualizer - Main Chromatic Agents Visualization Component
 *
 * Prism metaphor visualization where main (WHITE) refracts into
 * chromatic agent branches (PAIP clones), merging back on completion.
 *
 * This is a Qwik component wrapping vanilla Three.js for compatibility.
 *
 * Steps covered:
 * - Step 5: Scene Architecture
 * - Step 8: Camera & Controls
 * - Step 16: Dashboard Component Integration (Bus Subscription)
 * - Step 17: Performance Optimization (Mutation Queue)
 */

import { component$, useSignal, useVisibleTask$, type Signal, $ } from '@builder.io/qwik';
import type { AgentVisualEvent, AgentId, ChromaticState, AgentVisualData, VisualMutation } from './types';
import { AgentVisualState } from './types';
import { AGENT_COLORS, getAllAgentIds, getAgentOrbitalPosition } from './utils/colorMap';
import { BusSubscription, type BusSubscriptionState } from './BusSubscription';

// M3 Components - ChromaticVisualizer
import '@material/web/elevation/elevation.js';
import '@material/web/ripple/ripple.js';
import '@material/web/button/filled-button.js';
import '@material/web/button/text-button.js';

// Note: Three.js and its addons are dynamically imported at runtime
// to avoid SSR issues. The types are inferred from the imported modules.

// =============================================================================
// Props
// =============================================================================

interface ChromaticVisualizerProps {
  /** Bus events signal from parent */
  events?: Signal<AgentVisualEvent[]>;
  /** Width in pixels (default: container width) */
  width?: number;
  /** Height in pixels (default: 600) */
  height?: number;
  /** Enable bloom post-processing (default: true) */
  enableBloom?: boolean;
  /** Enable orbit controls (default: true) */
  enableControls?: boolean;
  /** Auto-focus on active agents (default: true) */
  autoFocus?: boolean;
  /** Enable live bus subscription (default: true) */
  enableBus?: boolean;
  /** WebSocket URL for bus bridge (default: auto-detect) */
  wsUrl?: string;
}

// =============================================================================
// Chromatic Visualizer Component
// =============================================================================

export const ChromaticVisualizer = component$<ChromaticVisualizerProps>((props) => {
  const containerRef = useSignal<HTMLDivElement>();
  const canvasRef = useSignal<HTMLCanvasElement>();
  const isInitialized = useSignal(false);
  const error = useSignal<string | null>(null);
  const fps = useSignal(0);

  // Bus connection state
  const busConnected = useSignal(false);
  const busReconnecting = useSignal(false);
  const eventsPerMinute = useSignal(0);

  // Mutation queue for frame-rate independent updates
  const mutationQueue = useSignal<VisualMutation[]>([]);

  // State for HUD display
  const agentStates = useSignal<Record<AgentId, { state: AgentVisualState; intensity: number }>>({
    claude: { state: AgentVisualState.IDLE, intensity: 0 },
    qwen: { state: AgentVisualState.IDLE, intensity: 0 },
    gemini: { state: AgentVisualState.IDLE, intensity: 0 },
    codex: { state: AgentVisualState.IDLE, intensity: 0 },
    main: { state: AgentVisualState.WORKING, intensity: 1 },
  });

  useVisibleTask$(async ({ cleanup }) => {
    const canvas = canvasRef.value;
    const container = containerRef.value;
    if (!canvas || !container) return;

    try {
      // Dynamic imports for Three.js (avoids SSR issues)
      const THREE = await import('three');
      const { OrbitControls } = await import('three/examples/jsm/controls/OrbitControls.js');
      const { EffectComposer } = await import('three/examples/jsm/postprocessing/EffectComposer.js');
      const { RenderPass } = await import('three/examples/jsm/postprocessing/RenderPass.js');
      const { UnrealBloomPass } = await import('three/examples/jsm/postprocessing/UnrealBloomPass.js');

      // Dynamic imports for our modules
      const { PrismCore } = await import('./PrismCore');
      const { AgentTree } = await import('./AgentTree');

      // =============================================================================
      // Scene Setup (Step 5)
      // =============================================================================

      const width = props.width ?? container.clientWidth;
      const height = props.height ?? 600;

      // Renderer
      const renderer = new THREE.WebGLRenderer({
        canvas,
        antialias: true,
        alpha: true,
      });
      renderer.setSize(width, height);
      renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
      renderer.toneMapping = THREE.ACESFilmicToneMapping;
      renderer.toneMappingExposure = 1.2;

      // Scene
      const scene = new THREE.Scene();
      scene.background = new THREE.Color(0x0a0a0f);
      scene.fog = new THREE.Fog(0x0a0a0f, 10, 50);

      // Camera (Step 8)
      const camera = new THREE.PerspectiveCamera(60, width / height, 0.1, 100);
      camera.position.set(0, 5, 12);
      camera.lookAt(0, 0, 0);

      // Orbit Controls (Step 8)
      type OrbitControlsType = InstanceType<typeof OrbitControls>;
      let controls: OrbitControlsType | null = null;
      if (props.enableControls !== false) {
        controls = new OrbitControls(camera, canvas) as OrbitControlsType;
        controls.enableDamping = true;
        controls.dampingFactor = 0.05;
        controls.minDistance = 5;
        controls.maxDistance = 30;
        controls.maxPolarAngle = Math.PI * 0.85;
        controls.target.set(0, 0, 0);
      }

      // =============================================================================
      // Lighting
      // =============================================================================

      // Ambient light
      const ambientLight = new THREE.AmbientLight(0x404040, 0.5);
      scene.add(ambientLight);

      // Main directional light
      const directionalLight = new THREE.DirectionalLight(0xffffff, 1);
      directionalLight.position.set(5, 10, 5);
      scene.add(directionalLight);

      // Colored point lights for each agent direction
      const agentIds: Exclude<AgentId, 'main'>[] = ['claude', 'qwen', 'gemini', 'codex'];
      type PointLightType = InstanceType<typeof THREE.PointLight>;
      const pointLights = new Map<AgentId, PointLightType>();

      for (const agentId of agentIds) {
        const pos = getAgentOrbitalPosition(agentId, 3);
        const color = new THREE.Color(AGENT_COLORS[agentId].hex);
        const light = new THREE.PointLight(color, 0.3, 10);
        light.position.set(...pos);
        scene.add(light);
        pointLights.set(agentId, light);
      }

      // =============================================================================
      // Grid Helper
      // =============================================================================

      const gridHelper = new THREE.GridHelper(20, 20, 0x222233, 0x111122);
      gridHelper.position.y = -2;
      scene.add(gridHelper);

      // =============================================================================
      // Core Objects
      // =============================================================================

      // Central Prism (Step 6)
      const prismCore = new PrismCore(scene);

      // Agent Trees (Step 7)
      type AgentTreeType = InstanceType<typeof AgentTree>;
      const agentTrees = new Map<AgentId, AgentTreeType>();
      for (const agentId of agentIds) {
        const tree = new AgentTree(agentId, scene);
        agentTrees.set(agentId, tree);
      }

      // Main tree (white, always visible)
      const mainTree = new AgentTree('main', scene);
      mainTree.show();
      mainTree.setIntensity(0.5);
      agentTrees.set('main', mainTree);

      // =============================================================================
      // Post-processing (Bloom for neon glow - Step 7)
      // =============================================================================

      type EffectComposerType = InstanceType<typeof EffectComposer>;
      let composer: EffectComposerType | null = null;
      if (props.enableBloom !== false) {
        composer = new EffectComposer(renderer);

        const renderPass = new RenderPass(scene, camera);
        composer.addPass(renderPass);

        const bloomPass = new UnrealBloomPass(
          new THREE.Vector2(width, height),
          1.5, // Strength
          0.4, // Radius
          0.85 // Threshold
        );
        composer.addPass(bloomPass);
      }

      // =============================================================================
      // State Management
      // =============================================================================

      const state: ChromaticState = {
        agents: new Map(),
        prismIntensity: 0.5,
        focusedAgent: null,
        busConnected: false,
        eventsPerMinute: 0,
        mainAhead: 0,
      };

      // Initialize agent visual data
      for (const agentId of getAllAgentIds()) {
        const agentData: AgentVisualData = {
          id: agentId,
          state: AgentVisualState.IDLE,
          hue: AGENT_COLORS[agentId].hue,
          color: AGENT_COLORS[agentId].hex,
          intensity: agentId === 'main' ? 0.5 : 0,
          codeGraph: null,
          branch: null,
          position: getAgentOrbitalPosition(agentId, 5),
          opacity: agentId === 'main' ? 1 : 0,
          lastUpdate: Date.now(),
        };
        state.agents.set(agentId, agentData);
      }

      // =============================================================================
      // Event Processing
      // =============================================================================

      function processEvent(event: AgentVisualEvent): void {
        const agentData = state.agents.get(event.agent_id);
        if (!agentData) return;

        // Update agent data
        agentData.state = event.state;
        agentData.intensity = event.activity_intensity;
        agentData.codeGraph = event.code_graph;
        agentData.branch = event.branch;
        agentData.lastUpdate = Date.now();

        // Update tree
        const tree = agentTrees.get(event.agent_id);
        if (tree) {
          tree.setState(event.state);
          tree.setIntensity(event.activity_intensity);

          if (event.code_graph) {
            tree.updateCodeGraph(event.code_graph);
          }

          if (event.state !== 'idle' && event.state !== 'cleanup') {
            tree.show();
          }
        }

        // Update prism beams
        if (event.agent_id !== 'main') {
          if (event.state === AgentVisualState.WORKING ||
              event.state === AgentVisualState.COMMITTING ||
              event.state === AgentVisualState.PUSHING) {
            prismCore.activateBeam(event.agent_id);
          } else if (event.state === AgentVisualState.IDLE ||
                     event.state === AgentVisualState.CLEANUP) {
            prismCore.deactivateBeam(event.agent_id);
          }
        }

        // Update point light intensity
        const light = pointLights.get(event.agent_id);
        if (light) {
          light.intensity = 0.3 + event.activity_intensity * 0.7;
        }

        // Update HUD state
        agentStates.value = {
          ...agentStates.value,
          [event.agent_id]: {
            state: event.state,
            intensity: event.activity_intensity,
          },
        };

        // Auto-focus on most active agent
        if (props.autoFocus !== false && event.activity_intensity > 0.7) {
          state.focusedAgent = event.agent_id;
        }
      }

      // =============================================================================
      // Animation Loop
      // =============================================================================

      let lastTime = performance.now();
      let frameCount = 0;
      let fpsTime = 0;
      let animationId = 0;

      function animate(): void {
        animationId = requestAnimationFrame(animate);

        const currentTime = performance.now();
        const deltaTime = (currentTime - lastTime) / 1000;
        lastTime = currentTime;

        // FPS calculation
        frameCount++;
        fpsTime += deltaTime;
        if (fpsTime >= 1) {
          fps.value = Math.round(frameCount / fpsTime);
          frameCount = 0;
          fpsTime = 0;
        }

        // Process any new events from props
        if (props.events?.value) {
          for (const event of props.events.value) {
            processEvent(event);
          }
        }

        // Update orbit controls
        if (controls) {
          controls.update();
        }

        // Update prism core
        prismCore.update(deltaTime);

        // Update all agent trees
        for (const tree of agentTrees.values()) {
          tree.update(deltaTime);
        }

        // Camera auto-focus (smooth transition to focused agent)
        if (state.focusedAgent && controls && props.autoFocus !== false) {
          const tree = agentTrees.get(state.focusedAgent);
          if (tree) {
            const center = tree.getCenterOfMass();
            controls.target.lerp(new THREE.Vector3(...center), deltaTime * 2);
          }
        }

        // Render
        if (composer) {
          composer.render();
        } else {
          renderer.render(scene, camera);
        }
      }

      // Start animation
      animate();
      isInitialized.value = true;

      // =============================================================================
      // Resize Handler
      // =============================================================================

      function handleResize(): void {
        if (!container) return;
        const newWidth = props.width ?? container.clientWidth;
        const newHeight = props.height ?? 600;

        camera.aspect = newWidth / newHeight;
        camera.updateProjectionMatrix();
        renderer.setSize(newWidth, newHeight);

        if (composer) {
          composer.setSize(newWidth, newHeight);
        }
      }

      window.addEventListener('resize', handleResize);

      // =============================================================================
      // Bus Subscription (Step 16: Dashboard Component Integration)
      // =============================================================================

      let busSubscription: BusSubscription | null = null;

      if (props.enableBus !== false && !props.events) {
        busSubscription = new BusSubscription({
          wsUrl: props.wsUrl,
          autoReconnect: true,
          maxReconnectAttempts: 10,
        });

        // Subscribe to state changes
        busSubscription.onStateChange((subState: BusSubscriptionState) => {
          busConnected.value = subState.connected;
          busReconnecting.value = subState.reconnecting;
          eventsPerMinute.value = subState.eventsPerMinute;

          if (subState.error) {
            console.warn('[ChromaticVisualizer] Bus error:', subState.error);
          }
        });

        // Subscribe to visualization events
        busSubscription.onVisualizationEvent((vizEvent: AgentVisualEvent) => {
          // Add to mutation queue for frame-rate independent processing
          const mutation: VisualMutation = {
            type: 'state_change',
            agent_id: vizEvent.agent_id,
            timestamp: Date.now(),
            payload: vizEvent,
            apply: (state: ChromaticState, deltaTime: number) => {
              processEvent(vizEvent);
            },
          };
          mutationQueue.value = [...mutationQueue.value, mutation];

          // Direct processing for immediate feedback
          processEvent(vizEvent);
        });

        // Connect to bus
        busSubscription.connect().catch((err) => {
          console.warn('[ChromaticVisualizer] Failed to connect to bus:', err);
          // Fall back to demo mode if bus connection fails
        });
      }

      // =============================================================================
      // Demo Mode: Simulate events if no bus connection
      // =============================================================================

      let demoInterval: ReturnType<typeof setInterval> | null = null;

      if (!props.events && props.enableBus === false) {
        // Run demo mode
        let demoPhase = 0;

        demoInterval = setInterval(() => {
          demoPhase = (demoPhase + 1) % 20;

          // Cycle through agents with activity
          const activeAgent = agentIds[Math.floor(demoPhase / 5) % 4];
          const phase = demoPhase % 5;

          let demoState: AgentVisualState;
          let intensity: number;

          switch (phase) {
            case 0:
              demoState = AgentVisualState.CLONING;
              intensity = 0.3;
              break;
            case 1:
            case 2:
              demoState = AgentVisualState.WORKING;
              intensity = 0.6 + Math.random() * 0.4;
              break;
            case 3:
              demoState = AgentVisualState.COMMITTING;
              intensity = 1;
              break;
            case 4:
              demoState = AgentVisualState.PUSHING;
              intensity = 0.8;
              break;
            default:
              demoState = AgentVisualState.IDLE;
              intensity = 0;
          }

          const demoEvent: AgentVisualEvent = {
            agent_id: activeAgent,
            branch: `${activeAgent.toUpperCase()}_DEMO_BRANCH`,
            clone_path: `/tmp/pluribus_${activeAgent}_demo`,
            state: demoState,
            color_hue: AGENT_COLORS[activeAgent].hue,
            code_graph: {
              root: '/pluribus',
              nodes: [
                { path: 'nucleus/tools/agent_bus.py', lines_changed: 50, node_type: 'file', dependencies: [], last_modified_iso: new Date().toISOString(), visual_weight: 7 },
                { path: 'nucleus/dashboard/src/app.tsx', lines_changed: 25, node_type: 'file', dependencies: ['agent_bus.py'], last_modified_iso: new Date().toISOString(), visual_weight: 5 },
                { path: 'nucleus/mcp/host.py', lines_changed: 10, node_type: 'file', dependencies: [], last_modified_iso: new Date().toISOString(), visual_weight: 3 },
              ],
              edges: [['nucleus/dashboard/src/app.tsx', 'nucleus/tools/agent_bus.py']],
              timestamp_iso: new Date().toISOString(),
            },
            activity_intensity: intensity,
            timestamp_iso: new Date().toISOString(),
          };

          processEvent(demoEvent);
        }, 1000);
      }

      // =============================================================================
      // Cleanup
      // =============================================================================

      cleanup(() => {
        cancelAnimationFrame(animationId);
        window.removeEventListener('resize', handleResize);

        if (demoInterval) {
          clearInterval(demoInterval);
        }

        // Disconnect bus subscription
        if (busSubscription) {
          busSubscription.disconnect();
          busSubscription = null;
        }

        // Dispose Three.js resources
        prismCore.dispose();
        for (const tree of agentTrees.values()) {
          tree.dispose();
        }

        if (composer) {
          composer.dispose();
        }

        renderer.dispose();

        // Clear scene
        scene.clear();
      });

    } catch (err) {
      error.value = err instanceof Error ? err.message : 'Failed to initialize 3D scene';
      console.error('[ChromaticVisualizer] Error:', err);
    }
  });

  return (
    <div
      ref={containerRef}
      class="glass-surface-elevated glass-animate-enter overflow-hidden relative"
    >
      {/* Header */}
      <div class="p-3 border-b border-glass-border-subtle flex items-center justify-between glass-surface-subtle">
        <div class="flex items-center gap-2">
          <span class="text-lg">&#x1F48E;</span>
          <span class="font-semibold text-glass-text-primary">Chromatic Agents</span>
          <span class="text-xs px-2 py-0.5 rounded glass-chip text-purple-400 shadow-[0_0_8px_rgba(168,85,247,0.3)]">
            PRISM
          </span>
        </div>
        <div class="flex items-center gap-4 text-xs text-muted-foreground">
          <span>{fps.value} FPS</span>
          {eventsPerMinute.value > 0 && (
            <span class="text-cyan-400">{eventsPerMinute.value} evt/min</span>
          )}
          <span class={
            busConnected.value
              ? 'text-green-400'
              : busReconnecting.value
                ? 'text-yellow-400 animate-pulse'
                : isInitialized.value
                  ? 'text-orange-400'
                  : 'text-yellow-400'
          }>
            {busConnected.value
              ? 'BUS LIVE'
              : busReconnecting.value
                ? 'RECONNECTING...'
                : isInitialized.value
                  ? 'DEMO'
                  : 'INIT...'}
          </span>
        </div>
      </div>

      {/* Error Display */}
      {error.value && (
        <div class="p-4 glass-status-critical text-sm">
          {error.value}
        </div>
      )}

      {/* Canvas Container */}
      <div class="relative" style={{ height: `${props.height ?? 600}px` }}>
        <canvas
          ref={canvasRef}
          class="w-full h-full"
          style={{ display: 'block' }}
        />

        {/* HUD Overlay */}
        <div class="absolute bottom-4 left-4 glass-surface-overlay rounded-lg p-3 text-xs font-mono">
          <div class="text-glass-text-muted mb-2">AGENT STATUS</div>
          {(['claude', 'qwen', 'gemini', 'codex'] as const).map((agentId) => {
            const agentState = agentStates.value[agentId];
            const color = AGENT_COLORS[agentId].hex;
            const barWidth = Math.round(agentState.intensity * 100);

            return (
              <div key={agentId} class="flex items-center gap-2 mb-1">
                <span
                  class="w-2 h-2 rounded-full"
                  style={{ backgroundColor: color }}
                />
                <span class="w-12 uppercase">{agentId}</span>
                <div class="w-24 h-2 bg-muted/30 rounded overflow-hidden">
                  <div
                    class="h-full transition-all duration-300"
                    style={{
                      width: `${barWidth}%`,
                      backgroundColor: color,
                    }}
                  />
                </div>
                <span class="w-20 text-muted-foreground uppercase">
                  {agentState.state}
                </span>
              </div>
            );
          })}
        </div>

        {/* Controls hint */}
        <div class="absolute bottom-4 right-4 text-xs text-muted-foreground bg-black/50 rounded px-2 py-1">
          Drag to rotate | Scroll to zoom
        </div>
      </div>
    </div>
  );
});

export default ChromaticVisualizer;
