import { component$, useSignal, useTask$, useVisibleTask$, noSerialize, type NoSerialize } from '@builder.io/qwik';

// M3 Components - PBDeepGraph
import '@material/web/elevation/elevation.js';
import '@material/web/ripple/ripple.js';

export interface PBDeepGraphNode {
  id: string;
  label: string;
  type: string;
  weight?: number;
}

export interface PBDeepGraphEdge {
  source: string;
  target: string;
  kind?: string;
}

export interface PBDeepGraphData {
  nodes: PBDeepGraphNode[];
  edges: PBDeepGraphEdge[];
}

interface PBDeepGraphProps {
  graph?: PBDeepGraphData | null;
  height?: number;
}

type GraphState = {
  renderer: any;
  scene: any;
  camera: any;
  group: any;
  observer: ResizeObserver;
  updateGraph: (graph: PBDeepGraphData | null | undefined) => void;
  stop: () => void;
};

const TYPE_COLORS: Record<string, number> = {
  root: 0xf8fafc,
  branch: 0x38bdf8,
  lost: 0xf59e0b,
  untracked: 0xf43f5e,
  drift: 0xa3e635,
};

const TYPE_BIAS: Record<string, number> = {
  root: 0,
  branch: 1.2,
  lost: 0.95,
  untracked: 1.05,
  drift: 1.1,
};

export const PBDeepGraph = component$<PBDeepGraphProps>((props) => {
  const containerRef = useSignal<HTMLDivElement>();
  const canvasRef = useSignal<HTMLCanvasElement>();
  const graphState = useSignal<NoSerialize<GraphState> | null>(null);

  useVisibleTask$(({ cleanup }) => {
    const init = async () => {
      const container = containerRef.value;
      const canvas = canvasRef.value;
      if (!container || !canvas) return;

      const THREE = await import('three');

      const renderer = new THREE.WebGLRenderer({
        canvas,
        antialias: true,
        alpha: true,
      });
      renderer.setPixelRatio(Math.min(window.devicePixelRatio || 1, 2));

      const scene = new THREE.Scene();
      scene.background = new THREE.Color(0x0b0c12);

      const camera = new THREE.PerspectiveCamera(58, 1, 0.1, 100);
      camera.position.set(0, 2.2, 8);
      camera.lookAt(0, 0, 0);

      const ambient = new THREE.AmbientLight(0xffffff, 0.7);
      scene.add(ambient);
      const point = new THREE.PointLight(0xffffff, 0.9);
      point.position.set(5, 6, 4);
      scene.add(point);

      const group = new THREE.Group();
      scene.add(group);

      let raf = 0;
      const renderLoop = () => {
        group.rotation.y += 0.0022;
        renderer.render(scene, camera);
        raf = window.requestAnimationFrame(renderLoop);
      };

      const resize = () => {
        const width = Math.max(1, container.clientWidth);
        const height = props.height ?? 360;
        renderer.setSize(width, height, false);
        camera.aspect = width / height;
        camera.updateProjectionMatrix();
      };
      resize();
      renderLoop();

      const observer = new ResizeObserver(resize);
      observer.observe(container);

      const disposeObject = (obj: any) => {
        obj.traverse?.((child: any) => {
          if (child.geometry?.dispose) child.geometry.dispose();
          const material = child.material;
          if (Array.isArray(material)) {
            material.forEach((mat) => mat?.dispose?.());
          } else if (material?.dispose) {
            material.dispose();
          }
        });
      };

      const updateGraph = (graph: PBDeepGraphData | null | undefined) => {
        while (group.children.length > 0) {
          const child = group.children.pop();
          if (child) {
            group.remove(child);
            disposeObject(child);
          }
        }

        if (!graph || graph.nodes.length === 0) return;

        const positions = new Map<string, any>();
        const golden = Math.PI * (3 - Math.sqrt(5));

        graph.nodes.forEach((node, idx) => {
          if (node.type === 'root') {
            positions.set(node.id, new THREE.Vector3(0, 0, 0));
            return;
          }
          const bias = TYPE_BIAS[node.type] ?? 1;
          const angle = idx * golden;
          const radius = (1.2 + Math.sqrt(idx + 1) * 0.55) * bias;
          const height = ((idx % 9) - 4) * 0.18;
          const x = Math.cos(angle) * radius;
          const z = Math.sin(angle) * radius;
          positions.set(node.id, new THREE.Vector3(x, height, z));
        });

        const edgePoints: any[] = [];
        for (const edge of graph.edges) {
          const a = positions.get(edge.source);
          const b = positions.get(edge.target);
          if (a && b) {
            edgePoints.push(a, b);
          }
        }
        if (edgePoints.length > 0) {
          const edgeGeo = new THREE.BufferGeometry().setFromPoints(edgePoints);
          const edgeMat = new THREE.LineBasicMaterial({ color: 0x334155, transparent: true, opacity: 0.55 });
          const lines = new THREE.LineSegments(edgeGeo, edgeMat);
          group.add(lines);
        }

        for (const node of graph.nodes) {
          const pos = positions.get(node.id);
          if (!pos) continue;
          const color = TYPE_COLORS[node.type] ?? 0x94a3b8;
          const radius = Math.max(0.08, 0.08 + Math.sqrt(node.weight ?? 1) * 0.05);
          const geo = new THREE.SphereGeometry(1, 18, 14);
          const mat = new THREE.MeshStandardMaterial({
            color,
            emissive: color,
            emissiveIntensity: 0.55,
            roughness: 0.35,
            metalness: 0.35,
          });
          const mesh = new THREE.Mesh(geo, mat);
          mesh.position.copy(pos);
          mesh.scale.setScalar(radius);
          group.add(mesh);
        }
      };

      const state: GraphState = {
        renderer,
        scene,
        camera,
        group,
        observer,
        updateGraph,
        stop: () => {
          if (raf) {
            window.cancelAnimationFrame(raf);
            raf = 0;
          }
        },
      };
      graphState.value = noSerialize(state);

      cleanup(() => {
        observer.disconnect();
        state.stop();
        renderer.dispose();
        graphState.value = null;
      });
    };

    init();
  });

  useTask$(({ track }) => {
    const graph = track(() => props.graph);
    const state = track(() => graphState.value) as GraphState | null;
    if (state) {
      state.updateGraph(graph);
    }
  });

  const nodeCount = props.graph?.nodes?.length ?? 0;
  const edgeCount = props.graph?.edges?.length ?? 0;
  const height = props.height ?? 360;

  return (
    <div class="glass-surface glass-surface-2 glass-gradient-border p-3">
      <div class="flex items-center justify-between pb-2">
        <div class="text-xs uppercase tracking-[0.2em] text-muted-foreground glass-chromatic-subtle">PBDEEP Spatial Graph</div>
        <div class="text-[10px] text-muted-foreground">nodes {nodeCount} | edges {edgeCount}</div>
      </div>
      <div ref={containerRef} class="relative w-full" style={{ height: `${height}px` }}>
        <canvas ref={canvasRef} class="w-full h-full block rounded-md bg-black/60" />
      </div>
    </div>
  );
});
