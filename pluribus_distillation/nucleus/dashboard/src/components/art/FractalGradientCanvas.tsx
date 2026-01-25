/**
 * FractalGradientCanvas - Algorithmic Layered Fractal Gradient Background
 * ========================================================================
 * Based on: https://codepen.io/RectangleWorld/pen/krNYbJ by RectangleWorld
 *
 * Creates beautiful, randomly generated fractal gradient backgrounds
 * using canvas subdivision algorithms. Perfect for glass surfaces.
 *
 * Features:
 * - Randomized color generation with glass palette integration
 * - Fractal subdivision for smooth gradient transitions
 * - Click to regenerate
 * - Dark mode optimized colors
 * - Reduced motion support
 */

import { component$, useSignal, useVisibleTask$, $ } from '@builder.io/qwik';

interface FractalGradientCanvasProps {
  /** Width of canvas (default: 100%) */
  width?: number | string;
  /** Height of canvas (default: 320px) */
  height?: number | string;
  /** Number of gradient layers (default: 20) */
  numLayers?: number;
  /** Base background color (default: dark glass) */
  bgColor?: string;
  /** Color palette mode */
  palette?: 'neon' | 'cool' | 'warm' | 'monochrome' | 'random';
  /** Allow click to regenerate */
  interactive?: boolean;
  /** CSS class for container */
  class?: string;
  /** Opacity of the canvas (for layering) */
  opacity?: number;
}

// Glass palette colors for neon mode
const NEON_COLORS = [
  { r: 0, g: 255, b: 255 },    // Cyan
  { r: 255, g: 0, b: 255 },    // Magenta
  { r: 138, g: 43, b: 226 },   // Purple
  { r: 0, g: 200, b: 255 },    // Electric blue
  { r: 255, g: 100, b: 255 },  // Pink
  { r: 100, g: 255, b: 200 },  // Mint
];

const COOL_COLORS = [
  { r: 0, g: 150, b: 255 },    // Blue
  { r: 0, g: 200, b: 200 },    // Teal
  { r: 100, g: 100, b: 255 },  // Periwinkle
  { r: 0, g: 255, b: 200 },    // Aqua
  { r: 50, g: 100, b: 200 },   // Steel blue
];

const WARM_COLORS = [
  { r: 255, g: 100, b: 50 },   // Orange
  { r: 255, g: 50, b: 100 },   // Rose
  { r: 255, g: 200, b: 0 },    // Gold
  { r: 255, g: 150, b: 100 },  // Peach
  { r: 200, g: 50, b: 100 },   // Burgundy
];

const MONOCHROME_COLORS = [
  { r: 100, g: 100, b: 120 },
  { r: 80, g: 80, b: 100 },
  { r: 120, g: 120, b: 140 },
  { r: 60, g: 60, b: 80 },
  { r: 140, g: 140, b: 160 },
];

export const FractalGradientCanvas = component$<FractalGradientCanvasProps>((props) => {
  const {
    width = '100%',
    height = 320,
    numLayers = 20,
    bgColor = 'rgba(17, 17, 27, 1)',
    palette = 'neon',
    interactive = true,
    opacity = 1,
  } = props;

  const canvasRef = useSignal<HTMLCanvasElement>();
  const generationKey = useSignal(0);

  const getColorPalette = $((paletteType: string) => {
    switch (paletteType) {
      case 'neon': return NEON_COLORS;
      case 'cool': return COOL_COLORS;
      case 'warm': return WARM_COLORS;
      case 'monochrome': return MONOCHROME_COLORS;
      default: return null; // random
    }
  });

  const getRandomColor = $((paletteColors: typeof NEON_COLORS | null) => {
    if (paletteColors) {
      const color = paletteColors[Math.floor(Math.random() * paletteColors.length)];
      // Add some variation
      return {
        r: Math.min(255, Math.max(0, color.r + Math.floor(Math.random() * 40 - 20))),
        g: Math.min(255, Math.max(0, color.g + Math.floor(Math.random() * 40 - 20))),
        b: Math.min(255, Math.max(0, color.b + Math.floor(Math.random() * 40 - 20))),
      };
    }
    return {
      r: Math.floor(Math.random() * 255),
      g: Math.floor(Math.random() * 255),
      b: Math.floor(Math.random() * 255),
    };
  });

  // Fractal subdivision algorithm for smooth random data
  const createRandomData = $((iterations: number) => {
    interface DataPoint {
      x: number;
      y: number;
      next: DataPoint | null;
    }

    const pointList: { first: DataPoint } = {
      first: { x: 0, y: 1, next: null }
    };
    const lastPoint: DataPoint = { x: 1, y: 1, next: null };
    let minY = 1;
    let maxY = 1;
    const minRatio = 0.33;

    pointList.first.next = lastPoint;

    for (let i = 0; i < iterations; i++) {
      let point: DataPoint | null = pointList.first;
      while (point && point.next !== null) {
        const nextPoint = point.next;
        const ratio = minRatio + Math.random() * (1 - 2 * minRatio);
        const newX = point.x + ratio * (nextPoint.x - point.x);

        // Find the smaller interval
        const dx = ratio < 0.5
          ? newX - point.x
          : nextPoint.x - newX;

        let newY = point.y + ratio * (nextPoint.y - point.y);
        newY += dx * (Math.random() * 2 - 1);

        const newPoint: DataPoint = { x: newX, y: newY, next: null };

        if (newY < minY) minY = newY;
        else if (newY > maxY) maxY = newY;

        newPoint.next = nextPoint;
        point.next = newPoint;
        point = nextPoint;
      }
    }

    // Normalize to values between 0 and 1
    if (maxY !== minY) {
      const normalizeRate = 1 / (maxY - minY);
      let point: DataPoint | null = pointList.first;
      while (point !== null) {
        point.y = normalizeRate * (point.y - minY);
        point = point.next;
      }
    } else {
      let point: DataPoint | null = pointList.first;
      while (point !== null) {
        point.y = 1;
        point = point.next;
      }
    }

    return pointList;
  });

  // Create linear fractal gradient
  const createLinearFractalGradient = $(async (
    context: CanvasRenderingContext2D,
    x0: number, y0: number, x1: number, y1: number,
    angleVariation: number,
    r: number, g: number, b: number,
    a: number, alphaVariation: number,
    gradIterates: number
  ) => {
    const numGradSteps = Math.pow(2, gradIterates);
    let stopNumber = 0;
    const gradRGB = `rgba(${r},${g},${b},`;
    const zeroAlpha = 0.5 / 255;

    const angle = (1 - 2 * Math.random()) * angleVariation;
    const xm = 0.5 * (x0 + x1);
    const ym = 0.5 * (y0 + y1);
    const ux = x0 - xm;
    const uy = y0 - ym;
    const sinAngle = Math.sin(angle);
    const cosAngle = Math.cos(angle);
    const vx = cosAngle * ux - sinAngle * uy;
    const vy = sinAngle * ux + cosAngle * uy;
    const driftX0 = xm + vx;
    const driftY0 = ym + vy;
    const driftX1 = xm - vx;
    const driftY1 = ym - vy;

    const grad = context.createLinearGradient(driftX0, driftY0, driftX1, driftY1);
    const gradPoints = await createRandomData(gradIterates);

    let gradFunctionPoint = gradPoints.first;
    while (gradFunctionPoint !== null) {
      let alpha = a + gradFunctionPoint.y * alphaVariation;

      if (alpha < zeroAlpha) alpha = 0;
      else if (alpha > 1) alpha = 1;

      grad.addColorStop(stopNumber / numGradSteps, gradRGB + alpha + ')');
      stopNumber++;
      gradFunctionPoint = gradFunctionPoint.next;
    }

    return grad;
  });

  // Main generation function
  const generate = $(async () => {
    const canvas = canvasRef.value;
    if (!canvas) return;

    const context = canvas.getContext('2d');
    if (!context) return;

    const displayWidth = canvas.width;
    const displayHeight = canvas.height;

    // Clear canvas
    context.clearRect(0, 0, displayWidth, displayHeight);

    const angleVariation = Math.PI / 40;
    const x0 = 0;
    const y0 = 0;
    const w = displayWidth;
    const h = displayHeight;
    const xMid = x0 + w / 2;
    const yMid = y0 + h / 2;

    const paletteColors = await getColorPalette(palette);

    for (let i = 0; i < numLayers; i++) {
      const color = await getRandomColor(paletteColors);
      const baseAlpha = 0;
      const alphaVariation = 32 / 255;

      context.globalCompositeOperation = 'lighter';

      let gradRad: number;
      let gradIterates: number;

      if (Math.random() < 0.5) {
        gradRad = 1.1 * h / 2;
        gradIterates = 7;
        context.fillStyle = await createLinearFractalGradient(
          context, xMid, yMid - gradRad, xMid, yMid + gradRad,
          angleVariation, color.r, color.g, color.b, baseAlpha, alphaVariation, gradIterates
        );
      } else {
        gradRad = 1.1 * w / 2;
        gradIterates = 9;
        context.fillStyle = await createLinearFractalGradient(
          context, xMid - gradRad, yMid, xMid + gradRad, yMid,
          angleVariation, color.r, color.g, color.b, baseAlpha, alphaVariation, gradIterates
        );
      }

      context.fillRect(x0, y0, w, h);
    }

    // Background color
    context.globalCompositeOperation = 'destination-over';
    context.fillStyle = bgColor;
    context.fillRect(x0, y0, w, h);
    context.globalCompositeOperation = 'source-over';
  });

  const handleClick = $(() => {
    if (interactive) {
      generationKey.value++;
    }
  });

  // Generate on mount and when key changes
  useVisibleTask$(({ track }) => {
    track(() => generationKey.value);

    // Check for reduced motion preference
    const prefersReducedMotion = window.matchMedia('(prefers-reduced-motion: reduce)').matches;

    // Set canvas dimensions
    const canvas = canvasRef.value;
    if (canvas) {
      const rect = canvas.getBoundingClientRect();
      canvas.width = rect.width || 900;
      canvas.height = typeof height === 'number' ? height : rect.height || 320;
    }

    // Generate with slight delay for smooth experience
    if (!prefersReducedMotion) {
      generate();
    } else {
      // For reduced motion, just use a solid gradient
      const canvas = canvasRef.value;
      if (canvas) {
        const ctx = canvas.getContext('2d');
        if (ctx) {
          const grad = ctx.createLinearGradient(0, 0, canvas.width, canvas.height);
          grad.addColorStop(0, 'rgba(0, 100, 150, 0.1)');
          grad.addColorStop(1, 'rgba(100, 0, 150, 0.1)');
          ctx.fillStyle = grad;
          ctx.fillRect(0, 0, canvas.width, canvas.height);
        }
      }
    }
  });

  return (
    <canvas
      ref={canvasRef}
      onClick$={handleClick}
      class={[
        'absolute inset-0 w-full h-full',
        interactive && 'cursor-pointer',
        props.class,
      ].filter(Boolean).join(' ')}
      style={{
        width: typeof width === 'number' ? `${width}px` : width,
        height: typeof height === 'number' ? `${height}px` : height,
        opacity,
        zIndex: -1,
      }}
      title={interactive ? 'Click to regenerate' : undefined}
    />
  );
});

/**
 * FractalGradientSection - A section wrapper with fractal gradient background
 */
export const FractalGradientSection = component$<{
  palette?: FractalGradientCanvasProps['palette'];
  numLayers?: number;
  class?: string;
  children?: any;
}>((props) => {
  return (
    <div class={['relative overflow-hidden', props.class].filter(Boolean).join(' ')}>
      <FractalGradientCanvas
        palette={props.palette || 'neon'}
        numLayers={props.numLayers || 15}
        opacity={0.3}
        interactive={false}
      />
      <div class="relative z-10">
        {props.children}
      </div>
    </div>
  );
});

/**
 * Presets for common use cases
 */
export const FractalGradientPresets = {
  /** Hero section - large, vibrant */
  hero: {
    numLayers: 25,
    palette: 'neon' as const,
    opacity: 0.4,
  },
  /** Card background - subtle */
  card: {
    numLayers: 12,
    palette: 'cool' as const,
    opacity: 0.2,
  },
  /** Panel background - medium */
  panel: {
    numLayers: 15,
    palette: 'neon' as const,
    opacity: 0.25,
  },
  /** Modal background - dramatic */
  modal: {
    numLayers: 20,
    palette: 'neon' as const,
    opacity: 0.35,
  },
  /** Sidebar - monochrome */
  sidebar: {
    numLayers: 10,
    palette: 'monochrome' as const,
    opacity: 0.15,
  },
};

export default FractalGradientCanvas;
