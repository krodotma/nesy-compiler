import { component$, useVisibleTask$, useSignal, noSerialize } from '@builder.io/qwik';
import * as THREE from 'three';
// @ts-ignore
import fragmentShader from './shaders/smoked-glass.frag.glsl?raw';

// M3 Components - GenerativeBackground
import '@material/web/elevation/elevation.js';

/**
 * GenerativeBackground
 * ====================
 * Visual Style Model 2.0 - Core Renderer
 * 
 * Renders the "Smoked Glass" volumetric fog using a WebGL fragment shader.
 * Consumes uniforms from the VisualEngine via CSS variables or direct injection.
 * 
 * Architecture:
 * - Raw Three.js (no React-Three-Fiber) for maximum performance in Qwik.
 * - Single full-screen PlaneBufferGeometry.
 * - ShaderMaterial with uniforms bound to the MaterialDefinition.
 */

export const GenerativeBackground = component$(() => {
  const canvasRef = useSignal<HTMLCanvasElement>();
  const isReducedMotion = useSignal(false);

  useVisibleTask$(({ cleanup }) => {
    if (!canvasRef.value) return;

    // Phase 2 Step 12: Mobile Optimization
    const mq = window.matchMedia('(prefers-reduced-motion: reduce)');
    isReducedMotion.value = mq.matches;
    const mqListener = (e: MediaQueryListEvent) => isReducedMotion.value = e.matches;
    mq.addEventListener('change', mqListener);

    if (isReducedMotion.value) {
        // Just clear the canvas or don't initialize Three.js
        return;
    }

    // 1. Setup Three.js Scene
    const scene = new THREE.Scene();
    const camera = new THREE.OrthographicCamera(-1, 1, 1, -1, 0, 1);
    
    const renderer = new THREE.WebGLRenderer({ 
      canvas: canvasRef.value, 
      alpha: false, // We render the full background
      antialias: false, // Not needed for noise clouds, saves GPU
      powerPreference: "high-performance"
    });
    
    renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2)); // Cap at 2x

    // 2. Initial Uniforms (Will be updated by VisualEngine)
    // We parse the CSS variables from :root to get the initial state
    const getCssColor = (varName: string) => {
      const style = getComputedStyle(document.documentElement);
      const val = style.getPropertyValue(varName).trim();
      // Simple parse for now - in production, VisualEngine should pass values directly via event
      // Fallback colors if parsing fails (CSS variables might be oklch which Three.js doesn't support deeply yet)
      // So we rely on the ArtInjector to emit an event with the RGB values
      return new THREE.Color(0x111111); 
    };

    const uniforms = {
      uTime: { value: 0.0 },
      uResolution: { value: new THREE.Vector2() },
      uMouse: { value: new THREE.Vector2() },
      uScrollY: { value: 0.0 },
      // Colors - placeholders, updated via event
      uColorStart: { value: new THREE.Color(0.1, 0.1, 0.12) }, 
      uColorEnd: { value: new THREE.Color(0.05, 0.05, 0.08) },
      uColorPrimary: { value: new THREE.Color(0.0, 0.8, 0.9) },
      uNoiseScale: { value: 1.0 },
      uNoiseSpeed: { value: 0.2 }
    };

    // 3. Geometry & Material
    const geometry = new THREE.PlaneGeometry(2, 2);
    const material = new THREE.ShaderMaterial({
      vertexShader: `
        varying vec2 vUv;
        void main() {
          vUv = uv;
          gl_Position = vec4(position, 1.0);
        }
      `,
      fragmentShader: fragmentShader,
      uniforms: uniforms,
      depthWrite: false,
      depthTest: false
    });

    const mesh = new THREE.Mesh(geometry, material);
    scene.add(mesh);

    // 4. Resize Handler
    const handleResize = () => {
      if (!canvasRef.value) return;
      const { innerWidth, innerHeight } = window;
      renderer.setSize(innerWidth, innerHeight);
      uniforms.uResolution.value.set(innerWidth, innerHeight);
    };
    
    window.addEventListener('resize', handleResize);
    handleResize(); // Init

    // 5. Input Handlers
    const handleMouseMove = (e: MouseEvent) => {
      uniforms.uMouse.value.set(e.clientX, window.innerHeight - e.clientY);
    };
    window.addEventListener('mousemove', handleMouseMove);

    const handleScroll = () => {
      uniforms.uScrollY.value = window.scrollY;
    };
    window.addEventListener('scroll', handleScroll);

    // 6. VisualEngine Bridge (The "Intelligence" Link)
    const handleMaterialUpdate = (e: CustomEvent) => {
        // Expected detail: { colors: { bgStart: string, ... }, shader: { ... } }
        // Note: For now, we assume the shader handles the color transitions or we parse them here.
        // Ideally, VisualEngine sends raw RGB [0-1] arrays to avoid parsing overhead.
        
        // This is a placeholder for the full wire-up. 
        // In Step 13, we'll make ArtInjector emit standardized Three.js compatible data.
        console.log("Shader received material update", e.detail);
    };
    window.addEventListener('pluribus:art:update', handleMaterialUpdate as EventListener);

    // 7. Render Loop
    let animationFrameId: number;
    const clock = new THREE.Clock();

    const animate = () => {
      animationFrameId = requestAnimationFrame(animate);
      uniforms.uTime.value = clock.getElapsedTime();
      renderer.render(scene, camera);
    };
    
    animate();

    // Cleanup
    cleanup(() => {
      window.removeEventListener('resize', handleResize);
      window.removeEventListener('mousemove', handleMouseMove);
      window.removeEventListener('pluribus:art:update', handleMaterialUpdate as EventListener);
      mq.removeEventListener('change', mqListener);
      cancelAnimationFrame(animationFrameId);
      renderer.dispose();
      geometry.dispose();
      material.dispose();
    });
  });

  return (
    <canvas
      ref={canvasRef}
      class="generative-canvas"
      style={{
        position: 'fixed',
        top: 0,
        left: 0,
        width: '100vw',
        height: '100vh',
        zIndex: -1, // Behind everything (Step 9 Stratigraphy)
        pointerEvents: 'none', // Allow clicks to pass through
        opacity: 1 // Managed by CSS/Theme
      }}
    />
  );
});
