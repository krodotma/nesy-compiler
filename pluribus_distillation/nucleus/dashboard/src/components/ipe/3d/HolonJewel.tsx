/**
 * HolonJewel.tsx
 * [Ultrathink Agent 2: Artist]
 * 
 * A Three.js driven visualization of a Code Holon.
 * 
 * Aesthetic Goals:
 * - "Iridescent Glass" material representing code purity.
 * - Procedural geometry complexity based on LOC (complexity prop).
 * - "God Rays" bloom effect for high-CMP states.
 */

import {
  component$,
  useSignal,
  useVisibleTask$,
  noSerialize,
} from '@builder.io/qwik';
import * as THREE from 'three';
import type { HolonNode } from '../../../lib/holon/types';

interface HolonJewelProps {
  node: HolonNode;
  width?: number;
  height?: number;
}

export const HolonJewel = component$<HolonJewelProps>(({ node, width = 300, height = 200 }) => {
  const containerRef = useSignal<HTMLDivElement>();

  useVisibleTask$(({ cleanup }) => {
    if (!containerRef.value) return;

    // --- SCENE SETUP ---
    const scene = new THREE.Scene();
    // Transparent background to blend with the dashboard
    // scene.background = null; 

    // Camera
    const camera = new THREE.PerspectiveCamera(45, width / height, 0.1, 1000);
    camera.position.z = 5;

    // Renderer
    const renderer = new THREE.WebGLRenderer({ 
      alpha: true, 
      antialias: true,
      powerPreference: "high-performance"
    });
    renderer.setSize(width, height);
    renderer.setPixelRatio(window.devicePixelRatio);
    containerRef.value.appendChild(renderer.domElement);

    // --- GEOMETRY (Procedural based on complexity) ---
    // Higher complexity = more detail (Icosahedron detail level)
    const detail = Math.max(0, Math.min(4, Math.floor(node.complexity / 20)));
    const geometry = new THREE.IcosahedronGeometry(1.8, detail);

    // --- MATERIAL (Iridescent Glass) ---
    // Using MeshPhysicalMaterial for that "premium" look
    const material = new THREE.MeshPhysicalMaterial({
      color: node.cmp > 80 ? 0x00ffff : 0x4488ff, // Cyan for high CMP, Blue for normal
      roughness: 0.1,
      metalness: 0.1,
      transmission: 0.6, // Glass-like
      thickness: 1.5,
      ior: 1.5, // Index of refraction
      reflectivity: 0.5,
      iridescence: 1.0,
      iridescenceIOR: 1.3,
      clearcoat: 1.0,
      clearcoatRoughness: 0.1,
      wireframe: false,
    });

    // Wireframe overlay for "Tech" feel
    const wireGeo = new THREE.WireframeGeometry(geometry);
    const wireMat = new THREE.LineBasicMaterial({ 
      color: 0xffffff, 
      transparent: true, 
      opacity: 0.15 
    });
    const wireframe = new THREE.LineSegments(wireGeo, wireMat);

    const jewel = new THREE.Mesh(geometry, material);
    jewel.add(wireframe);
    scene.add(jewel);

    // --- LIGHTING ---
    const ambientLight = new THREE.AmbientLight(0xffffff, 0.5);
    scene.add(ambientLight);

    const pointLight = new THREE.PointLight(0xffffff, 1);
    pointLight.position.set(5, 5, 5);
    scene.add(pointLight);

    const blueLight = new THREE.PointLight(0x0088ff, 2);
    blueLight.position.set(-5, -5, 0);
    scene.add(blueLight);

    // --- ANIMATION LOOP ---
    let running = true;
    const animate = () => {
      if (!running) return;
      
      const time = Date.now() * 0.001;

      // Rotate based on stability (higher stability = smoother/slower rotation)
      const rotSpeed = 0.5 / (Math.max(1, node.stability) * 0.1);
      
      jewel.rotation.x = Math.sin(time * rotSpeed) * 0.2;
      jewel.rotation.y += 0.01 * rotSpeed;

      // Pulse scaling based on CMP (Heartbeat)
      // "Entelexis Pulse"
      const pulse = 1 + Math.sin(time * 2) * 0.02 * (node.cmp / 100);
      jewel.scale.set(pulse, pulse, pulse);

      renderer.render(scene, camera);
      requestAnimationFrame(animate);
    };
    animate();

    // --- INTERACTION ---
    const onMouseMove = (e: MouseEvent) => {
      const rect = containerRef.value!.getBoundingClientRect();
      const x = ((e.clientX - rect.left) / rect.width) * 2 - 1;
      const y = -((e.clientY - rect.top) / rect.height) * 2 + 1;
      
      // Look at mouse
      jewel.rotation.x += y * 0.1;
      jewel.rotation.y += x * 0.1;
    };
    containerRef.value.addEventListener('mousemove', onMouseMove);

    cleanup(() => {
      running = false;
      containerRef.value?.removeEventListener('mousemove', onMouseMove);
      renderer.dispose();
      geometry.dispose();
      material.dispose();
    });
  });

  return (
    <div 
      ref={containerRef} 
      class="relative flex items-center justify-center overflow-hidden rounded-xl bg-gradient-to-b from-gray-900/50 to-black/50 border border-[var(--glass-border-subtle)]"
      style={{ width: `${width}px`, height: `${height}px` }}
    >
      {/* [Ultrathink Agent 3: Ontologist] - Semantic Overlay */}
      <div class="absolute bottom-3 left-4 pointer-events-none">
        <div class="text-[10px] text-gray-500 font-mono tracking-widest uppercase">ETYMON</div>
        <div class="text-xs text-blue-200 font-bold font-mono">{node.etymon}</div>
      </div>
      
      <div class="absolute top-3 right-4 pointer-events-none text-right">
        <div class="text-[10px] text-gray-500 font-mono tracking-widest uppercase">COMPLEXITY</div>
        <div class="text-xs text-cyan-200 font-bold font-mono">V:{node.complexity}</div>
      </div>
    </div>
  );
});
