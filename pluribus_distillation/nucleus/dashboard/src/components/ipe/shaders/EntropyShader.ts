/**
 * EntropyShader.ts
 *
 * A WebGL fragment shader simulating thermodynamic entropy vs negentropy.
 * Uses a reaction-diffusion style effect to visualize the battle between
 * Order (Blue/Cyan) and Chaos (Red/Orange).
 *
 * [Ultrathink Agent 2: Artist]
 * "We ditch the static div. We model the physics of code decay."
 */

export const ENTROPY_VERTEX_SHADER = `
  attribute vec2 position;
  void main() {
    gl_Position = vec4(position, 0.0, 1.0);
  }
`;

export const ENTROPY_FRAGMENT_SHADER = `
  precision mediump float;

  uniform vec2 u_resolution;
  uniform float u_time;
  uniform float u_entropy;    // 0.0 to 1.0 (Chaos level)
  uniform float u_negentropy; // 0.0 to 1.0 (Order level)
  uniform vec2 u_mouse;       // Interaction point

  // Pseudo-random
  float random(vec2 st) {
    return fract(sin(dot(st.xy, vec2(12.9898,78.233))) * 43758.5453123);
  }

  // Noise
  float noise(vec2 st) {
    vec2 i = floor(st);
    vec2 f = fract(st);
    float a = random(i);
    float b = random(i + vec2(1.0, 0.0));
    float c = random(i + vec2(0.0, 1.0));
    float d = random(i + vec2(1.0, 1.0));
    vec2 u = f * f * (3.0 - 2.0 * f);
    return mix(a, b, u.x) + (c - a)* u.y * (1.0 - u.x) + (d - b) * u.x * u.y;
  }

  // Fractal Brownian Motion
  float fbm(vec2 st) {
    float v = 0.0;
    float a = 0.5;
    vec2 shift = vec2(100.0);
    mat2 rot = mat2(cos(0.5), sin(0.5), -sin(0.5), cos(0.50));
    for (int i = 0; i < 5; i++) {
      v += a * noise(st);
      st = rot * st * 2.0 + shift;
      a *= 0.5;
    }
    return v;
  }

  void main() {
    vec2 st = gl_FragCoord.xy / u_resolution.xy;
    float aspect = u_resolution.x / u_resolution.y;
    st.x *= aspect;

    // Time scaling
    float t = u_time * 0.5;

    // 1. Base Chaos Field (Entropy) - Red/Orange Fire
    // Becomes more turbulent as u_entropy increases
    vec2 q = vec2(0.);
    q.x = fbm(st + 0.00 * t);
    q.y = fbm(st + vec2(1.0));

    vec2 r = vec2(0.);
    r.x = fbm(st + 1.0 * q + vec2(1.7, 9.2) + 0.15 * t);
    r.y = fbm(st + 1.0 * q + vec2(8.3, 2.8) + 0.126 * t);

    float f = fbm(st + r);

    // Color mixing for Chaos
    vec3 colorEntropy = mix(
      vec3(0.1, 0.0, 0.0), // Dark void
      vec3(0.9, 0.2, 0.1), // Fire red
      clamp((f * f) * 4.0, 0.0, 1.0)
    );
    
    // Add "Glitch" artifacts to chaos if entropy is high
    if (u_entropy > 0.7 && random(vec2(u_time)) > 0.9) {
        colorEntropy += vec3(0.2, 0.8, 0.0); // Digital artifact
    }

    // 2. Order Field (Negentropy) - Blue/Cyan Crystalline Structure
    // Structured grid-like noise
    vec2 grid = fract(st * 10.0);
    float gridLine = step(0.95, grid.x) + step(0.95, grid.y);
    float crystal = fbm(st * 3.0 - t * 0.2);
    
    vec3 colorOrder = mix(
      vec3(0.0, 0.1, 0.2), // Deep blue
      vec3(0.0, 0.8, 1.0), // Cyan glow
      crystal
    );
    colorOrder += vec3(gridLine * 0.2); // Add structure

    // 3. Battle of Forces
    // Mix based on relative dominance
    float battleFront = u_negentropy; // 0.0 = all chaos, 1.0 = all order
    
    // Add noise to the boundary so it's not a straight line
    float boundaryNoise = fbm(st * 5.0 + t) * 0.2; 
    float boundary = step(st.x / aspect, battleFront + boundaryNoise - 0.1);

    // Smooth boundary
    float smoothB = smoothstep(battleFront + boundaryNoise - 0.2, battleFront + boundaryNoise, st.x / aspect);

    // Interaction (Mouse/Focus) pushes Order
    float dist = distance(st, u_mouse * vec2(aspect, 1.0));
    float mousePush = 1.0 - smoothstep(0.0, 0.3, dist);
    smoothB += mousePush * 0.2;

    // Final Mix
    vec3 color = mix(colorEntropy, colorOrder, clamp(smoothB, 0.0, 1.0));

    // Vignette
    float vignette = 1.0 - smoothstep(0.5, 1.5, length(gl_FragCoord.xy / u_resolution.xy - 0.5));
    color *= vignette;

    gl_FragColor = vec4(color, 1.0);
  }
`;
