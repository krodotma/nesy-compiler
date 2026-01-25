/**
 * GenerativeBackground.tsx
 *
 * The Canvas of the Cell.
 * Renders deterministic, semi-functional generative art behind the dashboard.
 *
 * Defaults to "Creation" shader by Danilo Guanabara.
 */

import { component$, useVisibleTask$, useSignal } from '@builder.io/qwik';
import { createBusClient } from '../../lib/bus/bus-client';

interface Props {
  entropy?: number;
  mood?: 'calm' | 'anxious' | 'focused';
  enabled?: boolean;
  wsUrl?: string;
  requestScene?: boolean;
}

const VERT = `
attribute vec2 aPos;
void main() {
  gl_Position = vec4(aPos, 0.0, 1.0);
}
`;

// "Creation" by Danilo Guanabara
// http://www.pouet.net/prod.php?which=57245
const FRAG = `
precision highp float;
uniform vec2 iResolution;
uniform float iTime;
uniform float iEntropy;
uniform float iRate;
uniform vec3 iMood;

void main() {
  vec3 c;
  float l, z = iTime;
  for(int i=0; i<3; i++) {
    vec2 uv, p = gl_FragCoord.xy / iResolution.xy;
    uv = p;
    p -= 0.5;
    p.x *= iResolution.x / iResolution.y;
    z += 0.07;
    l = length(p);
    uv += p / l * (sin(z) + 1.0) * abs(sin(l * 9.0 - z - z));
    c[i] = 0.01 / length(mod(uv, 1.0) - 0.5);
  }
  gl_FragColor = vec4(c / l, 1.0); // Use time in alpha? Custom adaptation for opacity control handled by React
}
`;

function moodToVec3(mood: 'calm' | 'anxious' | 'focused'): [number, number, number] {
  if (mood === 'anxious') return [1.0, 0.15, 0.15];
  if (mood === 'focused') return [0.25, 0.65, 1.0];
  return [0.2, 1.0, 0.55];
}

function shannonEntropyNormalized(counts: Map<string, number>): number {
  let total = 0;
  for (const v of counts.values()) total += v;
  if (total <= 1) return 0;
  const k = counts.size;
  if (k <= 1) return 0;
  let h = 0;
  for (const v of counts.values()) {
    const p = v / total;
    h += -p * Math.log2(p);
  }
  return Math.max(0, Math.min(1, h / Math.log2(k)));
}

export const GenerativeBackground = component$<Props>(({ entropy, mood, enabled = true, wsUrl, requestScene = false }) => {
  const canvasRef = useSignal<HTMLCanvasElement>();
  const entropySig = useSignal<number>(typeof entropy === 'number' ? entropy : 0.08);
  const rateSig = useSignal<number>(0);
  const moodSig = useSignal<'calm' | 'anxious' | 'focused'>(mood ?? 'calm');
  const shaderSig = useSignal<string>(FRAG); // Defaults to Creation

  useVisibleTask$(({ track }) => {
    track(() => enabled);
    track(() => entropy);
    track(() => mood);
    track(() => wsUrl);
    const activeFrag = track(() => shaderSig.value);

    if (!enabled) return;
    if (typeof window !== 'undefined' && window.matchMedia?.('(prefers-reduced-motion: reduce)')?.matches) return;

    const canvas = canvasRef.value;
    if (!canvas) return;

    const targetEntropy = typeof entropy === 'number' ? entropy : null;
    const targetMood = mood ?? null;

    // --- Bus connection (Preserved) ---
    const timestampsMs: number[] = [];
    const topicWindow: string[] = [];
    const topicCounts = new Map<string, number>();
    const levelCounts = new Map<string, number>();
    const TOPIC_MAX = 160;
    const WINDOW_MS = 15_000;
    let busDisconnect: null | (() => void) = null;
    let statsTimer: ReturnType<typeof setInterval> | null = null;

    const useBusWeather = targetEntropy === null || targetMood === null;
    if (!useBusWeather) {
      entropySig.value = targetEntropy ?? 0.3;
      moodSig.value = targetMood ?? 'calm';
    }

    try {
      const last = (window as any).__PLURIBUS_LAST_ART_SCENE__;
      if (last?.topic === 'art.scene.change') {
        const data = last?.data;
        if (data?.scene?.glsl) shaderSig.value = String(data.scene.glsl);
      }
    } catch { }

    const url = wsUrl || `${location.protocol === 'https:' ? 'wss:' : 'ws:'}//${location.host}/ws/bus`;
    const client = createBusClient({ platform: 'browser', wsUrl: url });

    const onWindowArt = (e: Event) => {
      const detail = (e as CustomEvent).detail;
      if (detail?.topic === 'art.scene.change' && detail?.data?.scene?.glsl) {
        shaderSig.value = String(detail.data.scene.glsl);
      }
    };
    window.addEventListener('pluribus:art', onWindowArt);

    client
      .connect()
      .then(() => {
        busDisconnect = client.subscribe('*', (evt) => {
          const topic = String((evt as any).topic ?? '');
          if (topic === 'art.scene.change') {
            try {
              (window as any).__PLURIBUS_LAST_ART_SCENE__ = { topic, data: (evt as any).data };
              window.dispatchEvent(new CustomEvent('pluribus:art', { detail: { topic, data: (evt as any).data } }));
            } catch { }

            const data = (evt as any).data;
            if (data?.scene?.glsl) {
              shaderSig.value = String(data.scene.glsl);
            }
            return;
          }
          if (!useBusWeather) return;
          const now = Date.now();
          timestampsMs.push(now);
          while (timestampsMs.length && now - timestampsMs[0] > WINDOW_MS) timestampsMs.shift();

          if (topic) {
            topicWindow.push(topic);
            topicCounts.set(topic, (topicCounts.get(topic) || 0) + 1);
            if (topicWindow.length > TOPIC_MAX) {
              const old = topicWindow.shift()!;
              const next = (topicCounts.get(old) || 1) - 1;
              if (next <= 0) topicCounts.delete(old);
              else topicCounts.set(old, next);
            }
          }
          const level = String((evt as any).level ?? '');
          if (level) levelCounts.set(level, (levelCounts.get(level) || 0) + 1);
          if (levelCounts.size > 16) levelCounts.clear();
        });

        if (useBusWeather) {
          statsTimer = setInterval(() => {
            const now = Date.now();
            while (timestampsMs.length && now - timestampsMs[0] > WINDOW_MS) timestampsMs.shift();
            rateSig.value = timestampsMs.length / (WINDOW_MS / 1000);

            if (targetEntropy === null) entropySig.value = shannonEntropyNormalized(topicCounts);
            if (targetMood === null) {
              const err = levelCounts.get('error') || 0;
              const warn = levelCounts.get('warn') || 0;
              moodSig.value = err > 0 ? 'anxious' : warn > 0 ? 'focused' : 'calm';
            }
          }, 1000);
        }

        const onExplicitRequest = (e: Event) => {
          if (__E2E__) return;
          const detail = (e as CustomEvent).detail || {};
          const seed = typeof crypto !== 'undefined' && typeof crypto.getRandomValues === 'function'
            ? crypto.getRandomValues(new Uint32Array(1))[0]
            : Math.floor(Math.random() * 2 ** 32);

          fetch('/api/emit', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
              topic: 'art.scene.request',
              kind: 'request',
              level: 'info',
              actor: 'dashboard',
              data: {
                source: 'GenerativeBackground',
                seed,
                reason: detail.reason || 'manual_user_request',
                at: new Date().toISOString()
              },
            }),
          }).catch(() => { });
        };
        window.addEventListener('pluribus:art:request', onExplicitRequest);
        return () => window.removeEventListener('pluribus:art:request', onExplicitRequest);
      })
      .catch(() => { });

    // --- WebGL shader ---
    const gl = canvas.getContext('webgl2') || canvas.getContext('webgl');
    let frameId: number | null = null;
    let resizeTimer: number | null = null;

    const resize = () => {
      const dpr = Math.max(1, Math.min(2, window.devicePixelRatio || 1));
      const w = Math.floor(window.innerWidth * dpr);
      const h = Math.floor(window.innerHeight * dpr);
      if (canvas.width !== w) canvas.width = w;
      if (canvas.height !== h) canvas.height = h;
      if (gl) (gl as WebGLRenderingContext).viewport(0, 0, w, h);
    };
    const onResize = () => {
      if (resizeTimer) window.clearTimeout(resizeTimer);
      resizeTimer = window.setTimeout(resize, 50);
    };
    window.addEventListener('resize', onResize, { passive: true });
    resize();

    if (!gl) return;

    const glc = gl as WebGLRenderingContext;
    const compile = (type: number, src: string) => {
      const sh = glc.createShader(type);
      if (!sh) throw new Error('shader alloc failed');
      glc.shaderSource(sh, src);
      glc.compileShader(sh);
      if (!glc.getShaderParameter(sh, glc.COMPILE_STATUS)) {
        const info = glc.getShaderInfoLog(sh) || 'shader compile failed';
        console.error('Art Dept: Shader Error', info);
        glc.deleteShader(sh);
        throw new Error(info);
      }
      return sh;
    };

    const prog = glc.createProgram();
    if (!prog) return;
    try {
      const vs = compile(glc.VERTEX_SHADER, VERT);
      const fs = compile(glc.FRAGMENT_SHADER, activeFrag);
      glc.attachShader(prog, vs);
      glc.attachShader(prog, fs);
      glc.linkProgram(prog);
      glc.deleteShader(vs);
      glc.deleteShader(fs);
      if (!glc.getProgramParameter(prog, glc.LINK_STATUS)) {
        console.error('Art Dept: Link Error', glc.getProgramInfoLog(prog));
        throw new Error('program link failed');
      }
    } catch (e) {
      console.error('Art Dept: Fallback to Void', e);
      return;
    }

    glc.useProgram(prog);
    const posLoc = glc.getAttribLocation(prog, 'aPos');
    const resLoc = glc.getUniformLocation(prog, 'iResolution');
    const timeLoc = glc.getUniformLocation(prog, 'iTime');
    const entLoc = glc.getUniformLocation(prog, 'iEntropy');
    const rateLoc = glc.getUniformLocation(prog, 'iRate');
    const moodLoc = glc.getUniformLocation(prog, 'iMood');

    const buf = glc.createBuffer();
    glc.bindBuffer(glc.ARRAY_BUFFER, buf);
    glc.bufferData(
      glc.ARRAY_BUFFER,
      new Float32Array([-1, -1, 1, -1, -1, 1, -1, 1, 1, -1, 1, 1]),
      glc.STATIC_DRAW
    );
    glc.enableVertexAttribArray(posLoc);
    glc.vertexAttribPointer(posLoc, 2, glc.FLOAT, false, 0, 0);

    const t0 = performance.now();
    const render = () => {
      const t = (performance.now() - t0) / 1000;
      const w = canvas.width;
      const h = canvas.height;
      const e = entropySig.value;
      const r = rateSig.value;
      const m = moodSig.value;
      const mv = moodToVec3(m);

      if (resLoc) glc.uniform2f(resLoc, w, h);
      if (timeLoc) glc.uniform1f(timeLoc, t);
      if (entLoc) glc.uniform1f(entLoc, e);
      if (rateLoc) glc.uniform1f(rateLoc, r);
      if (moodLoc) glc.uniform3f(moodLoc, mv[0], mv[1], mv[2]);

      glc.drawArrays(glc.TRIANGLES, 0, 6);
      frameId = requestAnimationFrame(render);
    };
    render();

    return () => {
      if (frameId) cancelAnimationFrame(frameId);
      window.removeEventListener('pluribus:art', onWindowArt);
      window.removeEventListener('resize', onResize);
      if (resizeTimer) window.clearTimeout(resizeTimer);
      if (statsTimer) clearInterval(statsTimer);
      if (busDisconnect) busDisconnect();
      client.disconnect();
      try {
        if (buf) glc.deleteBuffer(buf);
        glc.deleteProgram(prog);
      } catch { }
    };
  });

  return (
    <canvas
      ref={canvasRef}
      class="fixed inset-0 pointer-events-none z-0 transition-opacity duration-700"
      style={{ opacity: 0.7 + entropySig.value * 0.3 }}
    />
  );
});
