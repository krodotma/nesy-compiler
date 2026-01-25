/**
 * HeartbeatOmega: Stability Loom (Composite Visualizer)
 * =====================================================
 * Multi-thread footer visualizer combining pulse, throughput,
 * pressure, and capacity into a unified realtime weave.
 */

import { component$, useVisibleTask$, useSignal, useStore, noSerialize, useContext, useTask$ } from '@builder.io/qwik';
import { DashboardLayoutContext } from '../../lib/state/dashboard_layout_context';
import { NeonSlider, NeonCheckbox, NeonButton, NeonStatusBadge } from '../ui/NeonControls';

type PressureState = 'nominal' | 'elevated' | 'critical' | 'unknown';
type LaneStatus = 'ok' | 'warn' | 'crit';

interface MemoryTelemetry {
  usedPct: number;
  swapUsedPct: number;
  psiSome: number;
  psiFull: number;
  pressure: PressureState;
}

interface OmegaState {
  beat: number;
  entropy: number;
  rings: [number, number, number, number];
  metrics: {
    eventRate: number;
    workerCount: number;
    lastError: string | null;
    memory: MemoryTelemetry;
  };
}

interface Point3D { x: number; y: number; z: number; }

const HISTORY_MAX = 120;

export const HeartbeatOmega = component$(() => {
  const layoutCtx = useContext(DashboardLayoutContext);
  const canvasRef = useSignal<HTMLCanvasElement>();
  const workerRef = useSignal<Worker>();
  const expanded = useSignal(false);

  const state = useSignal<OmegaState>({
    beat: 0,
    entropy: 0.1,
    rings: [1, 1, 1, 1],
    metrics: {
      eventRate: 0,
      workerCount: 0,
      lastError: null,
      memory: { usedPct: 0, swapUsedPct: 0, psiSome: 0, psiFull: 0, pressure: 'unknown' }
    }
  });

  const history = useStore({
    eventRate: new Array(HISTORY_MAX).fill(0),
    pressure: new Array(HISTORY_MAX).fill(0),
    capacity: new Array(HISTORY_MAX).fill(0),
    dynamics: new Array(HISTORY_MAX).fill(0),
    leverage: new Array(HISTORY_MAX).fill(0),
    pulse: new Array(HISTORY_MAX).fill(0)
  });

  const config = useStore({
    auto: true,
    eventMax: 18,
    workerScale: 8,
    pressureScale: 1.0,
    leverageScale: 6.0,
    memWarn: 85,
    memCrit: 92,
    dynamicsGain: 1.0,
    toleranceBias: 1.0,
    showTolerances: true,
    showPulse: true,
    showPerf: true,
    showPressure: true,
    showCapacity: true,
    showDynamics: true,
    showLeverage: true,
    focusLane: 'all' as 'all' | 'perf' | 'press' | 'cap' | 'dyn'
  });

  const readout = useStore({
    perf: { value: 0, delta: 0, status: 'ok' as LaneStatus },
    pressure: { value: 0, delta: 0, status: 'ok' as LaneStatus },
    capacity: { value: 0, delta: 0, status: 'ok' as LaneStatus },
    dynamics: { value: 0, delta: 0, status: 'ok' as LaneStatus },
    leverage: { value: 0, delta: 0, status: 'ok' as LaneStatus },
    pulse: { value: 0, delta: 0, status: 'ok' as LaneStatus }
  });

  useTask$(({ track }) => {
    const mode = track(() => layoutCtx.flowMode.value);
    if (mode === 'A' && !config.auto) {
      config.auto = true;
    }
  });

  useVisibleTask$(({ cleanup }) => {
    const w = new Worker(new URL('../../workers/omega.worker.ts', import.meta.url), { type: 'module' });
    workerRef.value = noSerialize(w);

    let rotation = 0;
    const particles: Point3D[] = [];
    for (let i = 0; i < 200; i++) {
      const theta = Math.random() * Math.PI * 2;
      const phi = Math.acos((Math.random() * 2) - 1);
      const r = 100;
      particles.push({
        x: r * Math.sin(phi) * Math.cos(theta),
        y: r * Math.sin(phi) * Math.sin(theta),
        z: r * Math.cos(phi)
      });
    }

    const clamp01 = (value: number) => Math.max(0, Math.min(1, value));
    const rgbaFromHex = (hex: string, alpha: number) => {
      const normalized = hex.replace('#', '');
      if (normalized.length !== 6) {
        return `rgba(148, 163, 184, ${alpha})`;
      }
      const r = parseInt(normalized.slice(0, 2), 16);
      const g = parseInt(normalized.slice(2, 4), 16);
      const b = parseInt(normalized.slice(4, 6), 16);
      return `rgba(${r}, ${g}, ${b}, ${alpha})`;
    };

    const computePressure = (memory: MemoryTelemetry, scale: number) => {
      const memRatio = clamp01(memory.usedPct / 100);
      const swapRatio = clamp01(memory.swapUsedPct / 100);
      const psiRatio = clamp01((memory.psiFull * 2) + memory.psiSome);
      const composite = (memRatio * 0.6) + (swapRatio * 0.2) + (psiRatio * 0.2);
      return clamp01(composite * scale);
    };

    const computeThresholds = () => {
      const toleranceBias = Math.max(0.75, Math.min(1.35, config.toleranceBias));
      return {
        perfWarn: clamp01(0.6 * toleranceBias),
        perfCrit: clamp01(0.85 * toleranceBias),
        pressureWarn: clamp01(0.5 * toleranceBias),
        pressureCrit: clamp01(0.75 * toleranceBias),
        capacityWarn: clamp01(0.4 * toleranceBias),
        capacityCrit: clamp01(0.25 * toleranceBias),
        dynamicsWarn: clamp01(0.65 * toleranceBias),
        dynamicsCrit: clamp01(0.85 * toleranceBias)
      };
    };

    const statusFromValue = (value: number, warn: number, crit: number, mode: 'high' | 'low'): LaneStatus => {
      if (mode === 'high') {
        if (value >= crit) return 'crit';
        if (value >= warn) return 'warn';
        return 'ok';
      }
      if (value <= crit) return 'crit';
      if (value <= warn) return 'warn';
      return 'ok';
    };

    const computeScales = (metrics: OmegaState['metrics']) => {
      if (!config.auto) {
        return {
          eventMax: Math.max(4, config.eventMax),
          workerScale: Math.max(1, config.workerScale),
          pressureScale: Math.max(0.3, config.pressureScale),
          leverageScale: Math.max(1, config.leverageScale)
        };
      }
      const eventMax = Math.max(10, Math.min(60, metrics.eventRate * 2.5));
      const workerScale = Math.max(4, Math.min(24, metrics.workerCount + 4));
      const leverageRaw = metrics.workerCount > 0 ? metrics.eventRate / metrics.workerCount : metrics.eventRate;
      const leverageScale = Math.max(2, Math.min(12, leverageRaw * 3));
      const pressureScale = metrics.memory.pressure === 'critical'
        ? 1.25
        : metrics.memory.pressure === 'elevated'
          ? 1.1
          : 1.0;
      return { eventMax, workerScale, pressureScale, leverageScale };
    };

    const updateHistory = (s: OmegaState) => {
      const scales = computeScales(s.metrics);
      const thresholds = computeThresholds();
      const memory = s.metrics.memory;
      const eventRatio = clamp01(s.metrics.eventRate / scales.eventMax);
      const pressureRatio = computePressure(memory, scales.pressureScale);
      const warn = Math.min(config.memWarn, config.memCrit - 1);
      const crit = Math.max(config.memCrit, warn + 1);
      const memRatio = clamp01(memory.usedPct / 100);
      const memPenalty = clamp01((memory.usedPct - warn) / (crit - warn));
      const entropy = clamp01(s.entropy * config.dynamicsGain);
      const capacityRatio = clamp01(
        (s.metrics.workerCount / scales.workerScale) * 0.6
        + (1 - memPenalty) * 0.3
        + (1 - entropy) * 0.1
      );
      const leverageRaw = s.metrics.workerCount > 0 ? s.metrics.eventRate / s.metrics.workerCount : s.metrics.eventRate;
      const leverageRatio = clamp01(leverageRaw / scales.leverageScale);
      const dynamicsRatio = entropy;
      const pulseRatio = clamp01((s.beat * 0.85) + 0.1);

      history.eventRate.push(eventRatio);
      history.pressure.push(pressureRatio);
      history.capacity.push(capacityRatio);
      history.dynamics.push(dynamicsRatio);
      history.leverage.push(leverageRatio);
      history.pulse.push(pulseRatio);

      if (history.eventRate.length > HISTORY_MAX) {
        history.eventRate.shift();
        history.pressure.shift();
        history.capacity.shift();
        history.dynamics.shift();
        history.leverage.shift();
        history.pulse.shift();
      }
      const perfValue = history.eventRate[history.eventRate.length - 1] ?? eventRatio;
      const perfPrev = history.eventRate[history.eventRate.length - 2] ?? perfValue;
      const pressureValue = history.pressure[history.pressure.length - 1] ?? pressureRatio;
      const pressurePrev = history.pressure[history.pressure.length - 2] ?? pressureValue;
      const capacityValue = history.capacity[history.capacity.length - 1] ?? capacityRatio;
      const capacityPrev = history.capacity[history.capacity.length - 2] ?? capacityValue;
      const dynamicsValue = history.dynamics[history.dynamics.length - 1] ?? dynamicsRatio;
      const dynamicsPrev = history.dynamics[history.dynamics.length - 2] ?? dynamicsValue;
      const leverageValue = history.leverage[history.leverage.length - 1] ?? leverageRatio;
      const leveragePrev = history.leverage[history.leverage.length - 2] ?? leverageValue;
      const pulseValue = history.pulse[history.pulse.length - 1] ?? pulseRatio;
      const pulsePrev = history.pulse[history.pulse.length - 2] ?? pulseValue;

      readout.perf.value = perfValue;
      readout.perf.delta = perfValue - perfPrev;
      readout.perf.status = statusFromValue(perfValue, thresholds.perfWarn, thresholds.perfCrit, 'high');
      readout.pressure.value = pressureValue;
      readout.pressure.delta = pressureValue - pressurePrev;
      readout.pressure.status = statusFromValue(pressureValue, thresholds.pressureWarn, thresholds.pressureCrit, 'high');
      readout.capacity.value = capacityValue;
      readout.capacity.delta = capacityValue - capacityPrev;
      readout.capacity.status = statusFromValue(capacityValue, thresholds.capacityWarn, thresholds.capacityCrit, 'low');
      readout.dynamics.value = dynamicsValue;
      readout.dynamics.delta = dynamicsValue - dynamicsPrev;
      readout.dynamics.status = statusFromValue(dynamicsValue, thresholds.dynamicsWarn, thresholds.dynamicsCrit, 'high');
      readout.leverage.value = leverageValue;
      readout.leverage.delta = leverageValue - leveragePrev;
      readout.leverage.status = 'ok';
      readout.pulse.value = pulseValue;
      readout.pulse.delta = pulseValue - pulsePrev;
      readout.pulse.status = 'ok';

      return { scales, eventRatio, pressureRatio, capacityRatio, leverageRatio, dynamicsRatio, entropy, pulseRatio };
    };

    const project = (p: Point3D, w: number, h: number, beat: number, scale: number) => {
      const cos = Math.cos(rotation);
      const sin = Math.sin(rotation);
      const x1 = p.x * cos - p.z * sin;
      const z1 = p.x * sin + p.z * cos;

      const fov = 300;
      const dist = 400 + z1;
      const factor = fov / (dist || 1);
      const pulse = 1.0 + (beat * 0.2);

      return {
        x: w / 2 + x1 * factor * pulse * scale,
        y: h / 2 + p.y * factor * pulse * scale,
        r: Math.max(0.5, (3 * factor)),
        alpha: Math.min(1, (z1 + 100) / 200)
      };
    };

    const drawSparkline = (
      ctx: CanvasRenderingContext2D,
      values: number[],
      x: number,
      y: number,
      width: number,
      height: number,
      color: string,
      lineWidth: number,
      alpha: number
    ) => {
      if (values.length < 2) return;
      ctx.save();
      ctx.beginPath();
      ctx.strokeStyle = color;
      ctx.lineWidth = lineWidth;
      ctx.globalAlpha = alpha;
      const step = width / (values.length - 1);
      for (let i = 0; i < values.length; i++) {
        const v = clamp01(values[i]);
        const px = x + (i * step);
        const py = y + height - (v * height);
        if (i === 0) ctx.moveTo(px, py);
        else ctx.lineTo(px, py);
      }
      ctx.stroke();
      ctx.restore();
    };

    const drawToleranceBand = (
      ctx: CanvasRenderingContext2D,
      x: number,
      y: number,
      width: number,
      height: number,
      warn: number,
      crit: number,
      color: string,
      mode: 'high' | 'low',
      alphaScale: number
    ) => {
      const warnY = y + height - (clamp01(warn) * height);
      const critY = y + height - (clamp01(crit) * height);
      const warnAlpha = 0.08 * alphaScale;
      const critAlpha = 0.16 * alphaScale;
      ctx.save();
      if (mode === 'high') {
        ctx.fillStyle = rgbaFromHex(color, warnAlpha);
        ctx.fillRect(x, y, width, Math.max(0, warnY - y));
        ctx.fillStyle = rgbaFromHex(color, critAlpha);
        ctx.fillRect(x, y, width, Math.max(0, critY - y));
      } else {
        ctx.fillStyle = rgbaFromHex(color, warnAlpha);
        ctx.fillRect(x, warnY, width, Math.max(0, y + height - warnY));
        ctx.fillStyle = rgbaFromHex(color, critAlpha);
        ctx.fillRect(x, critY, width, Math.max(0, y + height - critY));
      }
      ctx.restore();
    };

    const drawMarker = (
      ctx: CanvasRenderingContext2D,
      x: number,
      y: number,
      width: number,
      height: number,
      value: number,
      color: string,
      size: number,
      alpha: number
    ) => {
      const px = x + width - (size + 2);
      const py = y + height - (clamp01(value) * height);
      ctx.save();
      ctx.beginPath();
      ctx.arc(px, py, size, 0, Math.PI * 2);
      ctx.fillStyle = rgbaFromHex(color, 0.9 * alpha);
      ctx.fill();
      ctx.restore();
    };

    const drawLoom = (
      ctx: CanvasRenderingContext2D,
      x: number,
      y: number,
      width: number,
      height: number,
      density: 'compact' | 'expanded',
      memory: MemoryTelemetry,
      transparency: number
    ) => {
      const laneCount = 4;
      const laneGap = density === 'compact' ? 2 : 4;
      const laneHeight = (height - (laneGap * (laneCount - 1))) / laneCount;
      const pulseRailWidth = density === 'compact' ? 12 : 16;
      const minLabelPad = density === 'compact' ? 28 : 34;
      const leftPad = Math.min(72, Math.max(pulseRailWidth + minLabelPad, width * 0.26));
      const gaugeWidth = density === 'compact' ? 4 : 6;
      const rightPad = gaugeWidth + 10;
      const laneWidth = Math.max(10, width - leftPad - rightPad);
      const focusLane = config.focusLane;
      const focusActive = focusLane !== 'all';

      const gradient = ctx.createLinearGradient(x, y, x + width, y);
      gradient.addColorStop(0, `rgba(15, 23, 42, ${density === 'compact' ? 0.6 : 0.4})`);
      gradient.addColorStop(1, `rgba(8, 47, 73, ${density === 'compact' ? 0.7 : 0.5})`);
      ctx.fillStyle = gradient;
      ctx.fillRect(x, y, width, height);

      const sheenAlpha = 0.08 + (transparency * 0.25);
      ctx.save();
      ctx.globalAlpha = sheenAlpha;
      const sheen = ctx.createLinearGradient(x, y, x + width, y + height);
      sheen.addColorStop(0, 'rgba(148, 163, 184, 0)');
      sheen.addColorStop(0.45, 'rgba(148, 163, 184, 0.35)');
      sheen.addColorStop(0.6, 'rgba(148, 163, 184, 0.12)');
      sheen.addColorStop(1, 'rgba(148, 163, 184, 0)');
      ctx.fillStyle = sheen;
      ctx.fillRect(x, y, width, height);
      ctx.restore();

      ctx.save();
      ctx.strokeStyle = 'rgba(148, 163, 184, 0.08)';
      ctx.lineWidth = 1;
      for (let i = 0; i < width; i += 14) {
        ctx.beginPath();
        ctx.moveTo(x + i, y);
        ctx.lineTo(x + i - 10, y + height);
        ctx.stroke();
      }
      ctx.restore();

      const pulseRailX = x + 4;
      const pulseRailY = y + 4;
      const pulseRailHeight = height - 8;
      const pulseRailInnerWidth = Math.max(6, pulseRailWidth - 4);
      const pulseAlpha = config.showPulse ? (focusActive ? 0.35 : 0.85) : 0;
      if (pulseAlpha > 0.01) {
        drawSparkline(
          ctx,
          history.pulse,
          pulseRailX,
          pulseRailY,
          pulseRailInnerWidth,
          pulseRailHeight,
          '#F97316',
          density === 'compact' ? 1.2 : 1.6,
          0.85 * pulseAlpha
        );
        ctx.save();
        ctx.strokeStyle = `rgba(248, 113, 113, ${0.25 * pulseAlpha})`;
        ctx.lineWidth = 1;
        ctx.strokeRect(pulseRailX - 1, pulseRailY - 1, pulseRailInnerWidth + 2, pulseRailHeight + 2);
        ctx.restore();
      }

      const laneY = (index: number) => y + (index * (laneHeight + laneGap));

      const throughputY = laneY(0);
      const pressureY = laneY(1);
      const capacityY = laneY(2);
      const dynamicsY = laneY(3);

      const manualState = memory.usedPct >= config.memCrit
        ? 'critical'
        : memory.usedPct >= config.memWarn
          ? 'elevated'
          : 'nominal';
      const pressureState = config.auto ? memory.pressure : manualState;
      const pressureColor = pressureState === 'critical'
        ? '#EF4444'
        : pressureState === 'elevated'
          ? '#F59E0B'
          : '#06B6D4';
      const dynamicsColor = '#F472B6';
      const {
        perfWarn,
        perfCrit,
        pressureWarn,
        pressureCrit,
        capacityWarn,
        capacityCrit,
        dynamicsWarn,
        dynamicsCrit
      } = computeThresholds();
      const perfAlpha = config.showPerf ? (focusActive && focusLane !== 'perf' ? 0.2 : 1) : 0;
      const pressureAlpha = config.showPressure ? (focusActive && focusLane !== 'press' ? 0.2 : 1) : 0;
      const capacityAlpha = config.showCapacity ? (focusActive && focusLane !== 'cap' ? 0.2 : 1) : 0;
      const dynamicsAlpha = config.showDynamics ? (focusActive && focusLane !== 'dyn' ? 0.2 : 1) : 0;
      const leverageAlpha = config.showLeverage ? (perfAlpha * 0.35) : 0;

      const laneX = x + leftPad;
      if (config.showTolerances) {
        drawToleranceBand(ctx, laneX, throughputY, laneWidth, laneHeight, perfWarn, perfCrit, '#06B6D4', 'high', perfAlpha);
        drawToleranceBand(ctx, laneX, pressureY, laneWidth, laneHeight, pressureWarn, pressureCrit, pressureColor, 'high', pressureAlpha);
        drawToleranceBand(ctx, laneX, capacityY, laneWidth, laneHeight, capacityWarn, capacityCrit, '#10B981', 'low', capacityAlpha);
        drawToleranceBand(ctx, laneX, dynamicsY, laneWidth, laneHeight, dynamicsWarn, dynamicsCrit, dynamicsColor, 'high', dynamicsAlpha);
      }
      if (perfAlpha > 0.01) {
        drawSparkline(
          ctx,
          history.eventRate,
          laneX,
          throughputY,
          laneWidth,
          laneHeight,
          '#06B6D4',
          density === 'compact' ? 1.4 : 2.0,
          0.85 * perfAlpha
        );
      }
      if (pressureAlpha > 0.01) {
        drawSparkline(
          ctx,
          history.pressure,
          laneX,
          pressureY,
          laneWidth,
          laneHeight,
          pressureColor,
          density === 'compact' ? 1.4 : 2.0,
          0.9 * pressureAlpha
        );
      }
      if (capacityAlpha > 0.01) {
        drawSparkline(
          ctx,
          history.capacity,
          laneX,
          capacityY,
          laneWidth,
          laneHeight,
          '#10B981',
          density === 'compact' ? 1.4 : 2.0,
          0.8 * capacityAlpha
        );
      }
      if (dynamicsAlpha > 0.01) {
        drawSparkline(
          ctx,
          history.dynamics,
          laneX,
          dynamicsY,
          laneWidth,
          laneHeight,
          dynamicsColor,
          density === 'compact' ? 1.2 : 1.8,
          0.75 * dynamicsAlpha
        );
      }
      const markerSize = density === 'compact' ? 2 : 3;
      if (perfAlpha > 0.01) {
        drawMarker(ctx, laneX, throughputY, laneWidth, laneHeight, history.eventRate[history.eventRate.length - 1] ?? 0, '#06B6D4', markerSize, perfAlpha);
      }
      if (pressureAlpha > 0.01) {
        drawMarker(ctx, laneX, pressureY, laneWidth, laneHeight, history.pressure[history.pressure.length - 1] ?? 0, pressureColor, markerSize, pressureAlpha);
      }
      if (capacityAlpha > 0.01) {
        drawMarker(ctx, laneX, capacityY, laneWidth, laneHeight, history.capacity[history.capacity.length - 1] ?? 0, '#10B981', markerSize, capacityAlpha);
      }
      if (dynamicsAlpha > 0.01) {
        drawMarker(ctx, laneX, dynamicsY, laneWidth, laneHeight, history.dynamics[history.dynamics.length - 1] ?? 0, dynamicsColor, markerSize, dynamicsAlpha);
      }

      ctx.save();
      ctx.setLineDash([4, 4]);
      if (leverageAlpha > 0.01) {
        drawSparkline(
          ctx,
          history.leverage,
          laneX,
          throughputY,
          laneWidth,
          laneHeight,
          '#E2E8F0',
          1,
          leverageAlpha
        );
      }
      ctx.restore();

      ctx.save();
      ctx.font = density === 'compact'
        ? '9px ui-monospace, SFMono-Regular, Menlo, monospace'
        : '10px ui-monospace, SFMono-Regular, Menlo, monospace';
      ctx.textBaseline = 'middle';
      const labelX = x + pulseRailWidth + 6;
      const labels = [
        { text: 'PERF', color: '#06B6D4', alpha: perfAlpha },
        { text: 'PRESS', color: pressureColor, alpha: pressureAlpha },
        { text: 'CAP', color: '#10B981', alpha: capacityAlpha },
        { text: 'DYN', color: dynamicsColor, alpha: dynamicsAlpha },
      ];
      labels.forEach((label, idx) => {
        const ly = laneY(idx) + (laneHeight / 2);
        ctx.fillStyle = label.color;
        ctx.globalAlpha = label.alpha > 0 ? (0.4 + (0.4 * label.alpha)) : 0.12;
        ctx.fillText(label.text, labelX, ly);
      });
      ctx.restore();

      ctx.save();
      ctx.strokeStyle = 'rgba(148, 163, 184, 0.15)';
      ctx.lineWidth = 1;
      ctx.beginPath();
      ctx.moveTo(x + leftPad - 6, y + 4);
      ctx.lineTo(x + leftPad - 6, y + height - 4);
      ctx.stroke();
      ctx.restore();

      const gaugeHeight = height - 10;
      const gaugeX = x + width - gaugeWidth - 4;
      const gaugeY = y + 5;
      ctx.save();
      ctx.fillStyle = 'rgba(148, 163, 184, 0.12)';
      ctx.fillRect(gaugeX, gaugeY, gaugeWidth, gaugeHeight);
      const gaugeFill = Math.max(0.08, Math.min(1, transparency));
      ctx.fillStyle = `rgba(148, 163, 184, ${0.35 + (gaugeFill * 0.4)})`;
      ctx.fillRect(gaugeX, gaugeY + ((1 - gaugeFill) * gaugeHeight), gaugeWidth, gaugeFill * gaugeHeight);
      ctx.restore();

      const workerBars = Math.min(12, state.value.metrics.workerCount);
      if (workerBars > 0) {
        const barWidth = 3;
        const barGap = 2;
        const startX = x + width - rightPad - (workerBars * (barWidth + barGap));
        for (let i = 0; i < workerBars; i++) {
          const bx = startX + i * (barWidth + barGap);
          const bh = laneHeight * 0.8;
          ctx.fillStyle = 'rgba(16, 185, 129, 0.6)';
          ctx.fillRect(bx, capacityY + laneHeight - bh, barWidth, bh);
        }
      }
    };

    const draw = (meta: ReturnType<typeof updateHistory>) => {
      const cvs = canvasRef.value;
      if (!cvs) return;
      const ctx = cvs.getContext('2d');
      if (!ctx) return;

      const rect = cvs.getBoundingClientRect();
      if (cvs.width !== rect.width || cvs.height !== rect.height) {
        cvs.width = rect.width;
        cvs.height = rect.height;
      }
      const w = cvs.width;
      const h = cvs.height;
      ctx.clearRect(0, 0, w, h);

      if (expanded.value) {
        rotation += 0.01 + (state.value.entropy * 0.05);

        ctx.beginPath();
        ctx.strokeStyle = `rgba(6, 182, 212, ${0.1})`;
        particles.forEach((p, i) => {
          if (i % 10 === 0) {
            const proj = project(p, w, h, state.value.beat, 1.0);
            if (i === 0) ctx.moveTo(proj.x, proj.y);
            else ctx.lineTo(proj.x, proj.y);
          }
        });
        ctx.stroke();

        particles.forEach((p, i) => {
          const isCore = i < 50;
          const color = isCore
            ? `rgba(239, 68, 68, ${state.value.rings[0]})`
            : `rgba(16, 185, 129, ${state.value.rings[2]})`;
          const proj = project(p, w, h, state.value.beat, isCore ? 0.5 : 1.2);
          ctx.beginPath();
          ctx.arc(proj.x, proj.y, proj.r, 0, Math.PI * 2);
          ctx.fillStyle = color;
          ctx.globalAlpha = proj.alpha;
          ctx.fill();
        });
        ctx.globalAlpha = 1.0;

        const center = project({ x: 0, y: 0, z: 0 }, w, h, state.value.beat, 1);
        const grd = ctx.createRadialGradient(center.x, center.y, 5, center.x, center.y, 60);
        grd.addColorStop(0, `rgba(234, 179, 8, ${0.8 + state.value.beat * 0.2})`);
        grd.addColorStop(1, 'rgba(234, 179, 8, 0)');
        ctx.fillStyle = grd;
        ctx.globalCompositeOperation = 'lighter';
        ctx.beginPath();
        ctx.arc(center.x, center.y, 60, 0, Math.PI * 2);
        ctx.fill();
        ctx.globalCompositeOperation = 'source-over';

        const loomHeight = Math.max(80, h * 0.3);
        const transparency = Math.min(1, (state.value.rings[3] * 0.7) + (state.value.metrics.eventRate > 0 ? 0.3 : 0));
        drawLoom(ctx, 0, h - loomHeight, w, loomHeight, 'expanded', state.value.metrics.memory, transparency);
      } else {
        const transparency = Math.min(1, (state.value.rings[3] * 0.7) + (state.value.metrics.eventRate > 0 ? 0.3 : 0));
        drawLoom(ctx, 0, 0, w, h, 'compact', state.value.metrics.memory, transparency);
      }
    };

    w.onmessage = (ev) => {
      if (ev.data.type === 'OMEGA_TICK') {
        const s = ev.data.state as OmegaState;
        const memory = s.metrics?.memory ?? state.value.metrics.memory;
        s.metrics = { ...s.metrics, memory };
        state.value = s;
        const meta = updateHistory(s);
        draw(meta);
      }
    };

    cleanup(() => w.terminate());
  });

  const memory = state.value.metrics.memory;
  const memUsed = Math.round(memory.usedPct);
  const pressure = memory.pressure === 'unknown' ? 'idle' : memory.pressure;
  const leverageRaw = state.value.metrics.workerCount > 0
    ? state.value.metrics.eventRate / state.value.metrics.workerCount
    : state.value.metrics.eventRate;
  const transparencyScore = Math.min(1, (state.value.rings[3] * 0.7) + (state.value.metrics.eventRate > 0 ? 0.3 : 0));
  const flowMode = layoutCtx.flowMode.value;
  const autoLocked = flowMode === 'A';
  const autoLabel = autoLocked ? 'AUTO (FLOW)' : (config.auto ? 'AUTO' : 'MANUAL');
  const manualDisabled = config.auto || autoLocked;
  const formatPercent = (value: number) => `${Math.round(value * 100)}%`;
  const formatRatio = (value: number) => `${value.toFixed(2)}x`;
  const formatDelta = (value: number, mode: 'percent' | 'ratio') => {
    const sign = value >= 0 ? '+' : '-';
    const magnitude = Math.abs(value);
    if (mode === 'ratio') {
      return `${sign}${magnitude.toFixed(2)}x`;
    }
    return `${sign}${Math.round(magnitude * 100)}%`;
  };
  const statusClass = (status: LaneStatus) => {
    if (status === 'crit') return 'border-red-500/40 text-red-300 bg-red-950/30';
    if (status === 'warn') return 'border-amber-500/40 text-amber-300 bg-amber-950/30';
    return 'border-emerald-500/30 text-emerald-300 bg-emerald-950/20';
  };
  const laneCards = [
    { key: 'pulse', label: 'Pulse', value: readout.pulse.value, delta: readout.pulse.delta, status: readout.pulse.status, visible: config.showPulse, focusKey: 'all' as const, mode: 'percent' as const },
    { key: 'perf', label: 'Perf', value: readout.perf.value, delta: readout.perf.delta, status: readout.perf.status, visible: config.showPerf, focusKey: 'perf' as const, mode: 'percent' as const },
    { key: 'press', label: 'Press', value: readout.pressure.value, delta: readout.pressure.delta, status: readout.pressure.status, visible: config.showPressure, focusKey: 'press' as const, mode: 'percent' as const },
    { key: 'cap', label: 'Cap', value: readout.capacity.value, delta: readout.capacity.delta, status: readout.capacity.status, visible: config.showCapacity, focusKey: 'cap' as const, mode: 'percent' as const },
    { key: 'dyn', label: 'Dyn', value: readout.dynamics.value, delta: readout.dynamics.delta, status: readout.dynamics.status, visible: config.showDynamics, focusKey: 'dyn' as const, mode: 'percent' as const },
    { key: 'lev', label: 'Lvg', value: readout.leverage.value, delta: readout.leverage.delta, status: readout.leverage.status, visible: config.showLeverage, focusKey: 'perf' as const, mode: 'ratio' as const },
  ];

  return (
    <div
      class={`relative overflow-hidden transition-all duration-500 ease-in-out ${expanded.value ? 'h-80' : 'h-14'} cursor-pointer bg-card border-t border-[var(--glass-border)] group`}
      onClick$={() => expanded.value = !expanded.value}
    >
      <canvas ref={canvasRef} class="w-full h-full block" />

      <div class="absolute top-0 left-0 p-3 text-xs font-mono text-slate-300/80 pointer-events-none flex flex-col gap-1.5">
        <div class="uppercase tracking-[0.15em] text-[11px] text-cyan-400/80 font-medium">Stability Loom</div>
        <div class="flex items-center gap-3 text-[11px]">
          <span>Ω {state.value.beat.toFixed(2)}</span>
          <span>Ent {state.value.entropy.toFixed(2)}</span>
          <span class="text-emerald-400/90">W {state.value.metrics.workerCount}</span>
        </div>
        <div class="text-amber-400/90 text-[11px]">
          Mem {memUsed}% / Psi {memory.psiFull.toFixed(2)} / {pressure}
        </div>
      </div>

      <div class="absolute top-0 right-0 p-3 text-[11px] font-mono text-slate-300/70 pointer-events-none flex flex-col items-end gap-1.5">
        <div>{autoLabel} / ER {state.value.metrics.eventRate.toFixed(1)}</div>
        <div>Lvg {leverageRaw.toFixed(2)} / T {transparencyScore.toFixed(2)}</div>
      </div>

      {expanded.value && (
        <div
          class="absolute bottom-2 left-2 right-2 rounded-lg bg-slate-950/85 backdrop-blur-sm border border-cyan-500/20 shadow-[0_0_20px_rgba(0,255,255,0.1)] p-3 text-[10px] font-mono text-slate-200 pointer-events-auto"
          onClick$={(ev) => ev.stopPropagation()}
        >
          {/* Header with mode toggle */}
          <div class="flex items-center justify-between mb-3 pb-2 border-b border-cyan-500/10">
            <span class="uppercase tracking-[0.25em] text-[10px] text-cyan-400/80 font-medium">Stability Loom Controls</span>
            <NeonButton
              color={config.auto ? 'emerald' : 'amber'}
              active={config.auto}
              compact
              disabled={autoLocked}
              onClick$={() => { if (!autoLocked) config.auto = !config.auto; }}
            >
              {autoLocked ? 'AUTO (FLOW)' : (config.auto ? 'AUTO OBSERVE' : 'MANUAL TUNE')}
            </NeonButton>
          </div>

          {/* Slider controls grid */}
          <div class="grid grid-cols-2 md:grid-cols-4 gap-3 mb-3">
            <NeonSlider
              label="Perf max"
              value={config.eventMax}
              min={6}
              max={60}
              color="cyan"
              compact
              disabled={manualDisabled}
              formatValue={(v) => `${v}/s`}
              onChange$={(v) => config.eventMax = v}
            />
            <NeonSlider
              label="Capacity"
              value={config.workerScale}
              min={2}
              max={24}
              color="emerald"
              compact
              disabled={manualDisabled}
              formatValue={(v) => `${v}w`}
              onChange$={(v) => config.workerScale = v}
            />
            <NeonSlider
              label="Leverage"
              value={config.leverageScale}
              min={1}
              max={12}
              step={0.5}
              color="purple"
              compact
              disabled={manualDisabled}
              formatValue={(v) => `${v.toFixed(1)}x`}
              onChange$={(v) => config.leverageScale = v}
            />
            <NeonSlider
              label="Pressure"
              value={config.pressureScale}
              min={0.6}
              max={1.6}
              step={0.05}
              color="amber"
              compact
              disabled={manualDisabled}
              formatValue={(v) => `${v.toFixed(2)}x`}
              onChange$={(v) => config.pressureScale = v}
            />
            <NeonSlider
              label="Mem warn"
              value={config.memWarn}
              min={70}
              max={95}
              color="amber"
              compact
              disabled={manualDisabled}
              formatValue={(v) => `${v}%`}
              onChange$={(v) => config.memWarn = v}
            />
            <NeonSlider
              label="Mem crit"
              value={config.memCrit}
              min={80}
              max={98}
              color="rose"
              compact
              disabled={manualDisabled}
              formatValue={(v) => `${v}%`}
              onChange$={(v) => config.memCrit = v}
            />
            <NeonSlider
              label="Dynamics"
              value={config.dynamicsGain}
              min={0.5}
              max={2.0}
              step={0.05}
              color="magenta"
              compact
              disabled={manualDisabled}
              formatValue={(v) => `${v.toFixed(2)}x`}
              onChange$={(v) => config.dynamicsGain = v}
            />
            <NeonSlider
              label="Tolerance"
              value={config.toleranceBias}
              min={0.75}
              max={1.35}
              step={0.05}
              color="cyan"
              compact
              disabled={manualDisabled}
              formatValue={(v) => `${v.toFixed(2)}x`}
              onChange$={(v) => config.toleranceBias = v}
            />
          </div>

          {/* Lane metric cards */}
          <div class="grid grid-cols-3 md:grid-cols-6 gap-2 mb-3">
            {laneCards.map((card) => {
              const valueText = card.mode === 'ratio' ? formatRatio(card.value) : formatPercent(card.value);
              const deltaText = formatDelta(card.delta, card.mode);
              const focusActive = config.focusLane === card.focusKey;
              const visibleClass = card.visible ? '' : 'opacity-30';
              const statusMap = { ok: 'ok' as const, warn: 'warn' as const, crit: 'crit' as const };
              return (
                <button
                  key={card.key}
                  class={`
                    flex flex-col gap-0.5 rounded-md px-2 py-1.5 text-left
                    bg-slate-900/60 border transition-all
                    ${card.status === 'crit' ? 'border-red-500/40 shadow-[0_0_8px_rgba(239,68,68,0.2)]' :
                      card.status === 'warn' ? 'border-amber-500/40 shadow-[0_0_8px_rgba(251,191,36,0.2)]' :
                      'border-emerald-500/30 shadow-[0_0_6px_rgba(16,185,129,0.15)]'}
                    ${visibleClass}
                    ${focusActive ? 'ring-1 ring-cyan-400/60 shadow-[0_0_12px_rgba(0,255,255,0.3)]' : ''}
                    hover:bg-slate-800/60
                  `}
                  onClick$={() => config.focusLane = card.focusKey}
                >
                  <div class="flex items-center justify-between text-[7px] uppercase tracking-[0.15em]">
                    <span class={card.status === 'crit' ? 'text-red-300' : card.status === 'warn' ? 'text-amber-300' : 'text-emerald-300'}>
                      {card.label}
                    </span>
                    <span class="text-slate-400">{deltaText}</span>
                  </div>
                  <div class="flex items-baseline justify-between">
                    <span class="text-[11px] font-semibold text-slate-100">{valueText}</span>
                    <NeonStatusBadge status={statusMap[card.status]} compact>{card.status}</NeonStatusBadge>
                  </div>
                </button>
              );
            })}
          </div>

          {/* Lane visibility toggles */}
          <div class="flex flex-wrap items-center gap-3 mb-2 pb-2 border-b border-slate-700/30">
            <span class="uppercase tracking-[0.2em] text-[8px] text-cyan-500/60">Lanes</span>
            <NeonCheckbox
              label="Pulse"
              color="orange"
              compact
              checked={config.showPulse}
              onChange$={(v) => config.showPulse = v}
            />
            <NeonCheckbox
              label="Perf"
              color="cyan"
              compact
              checked={config.showPerf}
              onChange$={(v) => config.showPerf = v}
            />
            <NeonCheckbox
              label="Press"
              color="amber"
              compact
              checked={config.showPressure}
              onChange$={(v) => config.showPressure = v}
            />
            <NeonCheckbox
              label="Cap"
              color="emerald"
              compact
              checked={config.showCapacity}
              onChange$={(v) => config.showCapacity = v}
            />
            <NeonCheckbox
              label="Dyn"
              color="magenta"
              compact
              checked={config.showDynamics}
              onChange$={(v) => config.showDynamics = v}
            />
            <NeonCheckbox
              label="Lvg"
              color="purple"
              compact
              checked={config.showLeverage}
              onChange$={(v) => config.showLeverage = v}
            />
            <NeonCheckbox
              label="Tol"
              color="cyan"
              compact
              checked={config.showTolerances}
              onChange$={(v) => config.showTolerances = v}
            />
          </div>

          {/* Focus mode buttons */}
          <div class="flex flex-wrap items-center gap-2">
            <span class="uppercase tracking-[0.2em] text-[8px] text-cyan-500/60">Focus</span>
            {(['all', 'perf', 'press', 'cap', 'dyn'] as const).map((lane) => (
              <NeonButton
                key={lane}
                color={lane === 'all' ? 'cyan' : lane === 'perf' ? 'cyan' : lane === 'press' ? 'amber' : lane === 'cap' ? 'emerald' : 'magenta'}
                active={config.focusLane === lane}
                compact
                onClick$={() => config.focusLane = lane}
              >
                {lane.toUpperCase()}
              </NeonButton>
            ))}
          </div>
        </div>
      )}

      <div class="absolute bottom-0 left-0 p-2 px-3 text-[11px] text-gray-500 opacity-0 group-hover:opacity-100 transition-opacity">
        {expanded.value ? 'CLICK TO CONDENSE' : 'CLICK TO EXPAND'}
      </div>
    </div>
  );
});
