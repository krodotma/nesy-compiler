import { component$, useVisibleTask$ } from '@builder.io/qwik';
import { P5Canvas } from '../creative';

// M3 Components - PBDeepPulse
import '@material/web/elevation/elevation.js';
import '@material/web/progress/circular-progress.js';

interface PBDeepPulseProps {
  percent: number;
  stage: string;
  status: string;
  eventRate: number;
  pulseCount: number;
  activeReqId?: string | null;
  width?: number;
  height?: number;
}

const PBDEEP_SKETCH = `
let pulses = [];
let lastPulse = -1;

p.setup = () => {
  const w = p._pluribusData?.width || 520;
  const h = p._pluribusData?.height || 220;
  p.createCanvas(w, h);
  p.frameRate(60);
};

p.draw = () => {
  const state = window.__PBDEEP_STATE || {};
  const pct = Number(state.percent || 0);
  const pulse = Number(state.pulse || 0);
  const stage = String(state.stage || "idle");
  const status = String(state.status || "idle");
  const rate = Number(state.eventRate || 0);

  if (pulse !== lastPulse) {
    pulses.push({ r: 0, a: 180 });
    lastPulse = pulse;
  }

  p.background(10, 12, 18);

  const cx = p.width * 0.5;
  const cy = p.height * 0.5;
  const radius = Math.min(p.width, p.height) * 0.28;

  p.noFill();
  p.stroke(80, 90, 120, 160);
  p.strokeWeight(3);
  p.ellipse(cx, cy, radius * 2, radius * 2);

  p.stroke(56, 189, 248, 220);
  p.strokeWeight(6);
  const endAngle = p.map(pct, 0, 100, -p.HALF_PI, p.TWO_PI - p.HALF_PI);
  p.beginShape();
  for (let a = -p.HALF_PI; a <= endAngle; a += 0.06) {
    p.vertex(cx + Math.cos(a) * radius, cy + Math.sin(a) * radius);
  }
  p.endShape();

  pulses = pulses.filter((pulse) => pulse.a > 0);
  pulses.forEach((pulse) => {
    pulse.r += 2.2;
    pulse.a -= 2.0;
    p.stroke(255, 255, 255, pulse.a);
    p.strokeWeight(2);
    p.ellipse(cx, cy, radius * 2 + pulse.r, radius * 2 + pulse.r);
  });

  p.noStroke();
  p.fill(200, 220, 255, 220);
  p.textAlign(p.CENTER, p.CENTER);
  p.textSize(16);
  p.text(Math.round(pct) + "%", cx, cy - 6);
  p.textSize(10);
  p.fill(140, 160, 190, 200);
  p.text(stage + " / " + status, cx, cy + 12);

  p.textAlign(p.LEFT, p.TOP);
  p.textSize(10);
  p.fill(120, 140, 170, 180);
  p.text("events/min: " + rate.toFixed(1), 12, 10);
};
`;

export const PBDeepPulse = component$<PBDeepPulseProps>((props) => {
  useVisibleTask$(({ track }) => {
    track(() => props.percent);
    track(() => props.stage);
    track(() => props.status);
    track(() => props.eventRate);
    track(() => props.pulseCount);
    track(() => props.activeReqId);

    if (typeof window === 'undefined') return;
    (window as any).__PBDEEP_STATE = {
      percent: props.percent,
      stage: props.stage,
      status: props.status,
      eventRate: props.eventRate,
      pulse: props.pulseCount,
      req_id: props.activeReqId,
      ts: Date.now(),
    };
  });

  const width = props.width ?? 520;
  const height = props.height ?? 220;

  return (
    <div class="glass-surface glass-gradient-border-hero p-3">
      <div class="flex items-center justify-between pb-2">
        <div class="text-xs uppercase tracking-[0.2em] text-muted-foreground glass-chromatic-subtle">PBDEEP Pulse</div>
        <div class="text-[10px] text-muted-foreground">req {props.activeReqId?.slice(0, 8) || '-'}</div>
      </div>
      <P5Canvas
        sketchId="pbdeep-pulse"
        sketchCode={PBDEEP_SKETCH}
        data={{ width, height }}
        width={width}
        height={height}
        autoPlay={true}
        showControls={false}
      />
    </div>
  );
});
