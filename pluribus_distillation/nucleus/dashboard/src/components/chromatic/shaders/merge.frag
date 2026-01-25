/**
 * Chromatic Agents Visualizer - Merge Animation Fragment Shader
 *
 * Step 20: Optimized GLSL shaders for performance
 *
 * This shader creates the visual effect when an agent's colored branch
 * merges back into the white main branch. The chromatic color drains
 * toward white in a beam-like convergence effect.
 *
 * Used for:
 * - Agent push success visualization
 * - Color drain to white transition
 * - Merge beam particle convergence
 */

precision highp float;

// =============================================================================
// Uniforms
// =============================================================================

uniform sampler2D uSceneTexture;      // Current scene render
uniform sampler2D uAgentTexture;      // Agent tree render (colored)
uniform sampler2D uNoiseTexture;      // Noise for organic transitions
uniform vec2 uResolution;             // Screen resolution
uniform float uTime;                  // Animation time
uniform float uProgress;              // Merge progress (0-1)
uniform vec3 uAgentColor;             // Agent's chromatic color (RGB)
uniform vec3 uMainColor;              // Main branch color (white)
uniform vec2 uMergeTarget;            // Screen-space target point (main branch)
uniform vec2 uAgentCenter;            // Screen-space agent center
uniform float uBeamWidth;             // Width of merge beam
uniform int uMergeStyle;              // 0=drain, 1=beam, 2=spiral, 3=dissolve

// =============================================================================
// Varyings
// =============================================================================

varying vec2 vUv;

// =============================================================================
// Constants
// =============================================================================

const float PI = 3.14159265359;
const float TWO_PI = 6.28318530718;

// Neon colors for agents
const vec3 MAGENTA = vec3(1.0, 0.0, 1.0);
const vec3 CYAN = vec3(0.0, 1.0, 1.0);
const vec3 YELLOW = vec3(1.0, 1.0, 0.0);
const vec3 GREEN = vec3(0.0, 1.0, 0.0);
const vec3 WHITE = vec3(1.0, 1.0, 1.0);

// =============================================================================
// Utility Functions
// =============================================================================

/**
 * Smooth minimum for organic blending
 */
float smin(float a, float b, float k) {
    float h = clamp(0.5 + 0.5 * (b - a) / k, 0.0, 1.0);
    return mix(b, a, h) - k * h * (1.0 - h);
}

/**
 * 2D rotation matrix
 */
mat2 rotate2D(float angle) {
    float s = sin(angle);
    float c = cos(angle);
    return mat2(c, -s, s, c);
}

/**
 * Value noise sampling
 */
float noise(vec2 p) {
    return texture2D(uNoiseTexture, p * 0.1).r;
}

/**
 * Fractal Brownian Motion for organic patterns
 */
float fbm(vec2 p) {
    float value = 0.0;
    float amplitude = 0.5;
    float frequency = 1.0;

    for (int i = 0; i < 4; i++) {
        value += amplitude * noise(p * frequency);
        amplitude *= 0.5;
        frequency *= 2.0;
    }

    return value;
}

/**
 * HSL to RGB conversion for color transitions
 */
vec3 hsl2rgb(vec3 hsl) {
    vec3 rgb = clamp(abs(mod(hsl.x * 6.0 + vec3(0.0, 4.0, 2.0), 6.0) - 3.0) - 1.0, 0.0, 1.0);
    return hsl.z + hsl.y * (rgb - 0.5) * (1.0 - abs(2.0 * hsl.z - 1.0));
}

/**
 * RGB to HSL conversion
 */
vec3 rgb2hsl(vec3 rgb) {
    float maxC = max(max(rgb.r, rgb.g), rgb.b);
    float minC = min(min(rgb.r, rgb.g), rgb.b);
    float delta = maxC - minC;

    float h = 0.0;
    float s = 0.0;
    float l = (maxC + minC) * 0.5;

    if (delta > 0.0) {
        s = delta / (1.0 - abs(2.0 * l - 1.0));

        if (maxC == rgb.r) {
            h = mod((rgb.g - rgb.b) / delta, 6.0);
        } else if (maxC == rgb.g) {
            h = (rgb.b - rgb.r) / delta + 2.0;
        } else {
            h = (rgb.r - rgb.g) / delta + 4.0;
        }
        h /= 6.0;
    }

    return vec3(h, s, l);
}

// =============================================================================
// Merge Effect Styles
// =============================================================================

/**
 * Style 0: Color Drain
 * Color drains from edges toward center, then flows to main
 */
vec4 colorDrain(vec2 uv, vec4 sceneColor, vec4 agentColor, float progress) {
    // Distance from agent center
    float distFromAgent = length(uv - uAgentCenter);
    float distToMain = length(uv - uMergeTarget);

    // Drain starts from edges
    float drainFront = progress * 2.0;
    float drainMask = smoothstep(drainFront, drainFront + 0.1, distFromAgent * 2.0);

    // Add noise for organic feel
    float noiseVal = fbm(uv * 5.0 + uTime * 0.5);
    drainMask += noiseVal * 0.1;

    // Interpolate color from agent to white based on drain
    vec3 drainedColor = mix(agentColor.rgb, WHITE, 1.0 - drainMask);

    // Final blend with scene
    float alpha = agentColor.a * (1.0 - progress * 0.5);
    return vec4(mix(sceneColor.rgb, drainedColor, alpha), sceneColor.a);
}

/**
 * Style 1: Beam Merge
 * Colored beam flows from agent to main branch
 */
vec4 beamMerge(vec2 uv, vec4 sceneColor, vec4 agentColor, float progress) {
    // Direction from agent to main
    vec2 dir = normalize(uMergeTarget - uAgentCenter);
    vec2 toUv = uv - uAgentCenter;

    // Project point onto beam line
    float alongBeam = dot(toUv, dir);
    float perpDist = length(toUv - dir * alongBeam);

    // Beam parameters
    float beamLength = length(uMergeTarget - uAgentCenter);
    float beamProgress = alongBeam / beamLength;

    // Beam mask (only show where beam is active)
    float beamFront = progress;
    float beamMask = 0.0;

    if (alongBeam > 0.0 && alongBeam < beamLength * beamFront) {
        // Width tapers toward target
        float taperWidth = uBeamWidth * (1.0 - beamProgress * 0.5);
        beamMask = smoothstep(taperWidth, taperWidth * 0.8, perpDist);

        // Add glow falloff
        beamMask *= 1.0 - smoothstep(0.0, beamLength * beamFront, alongBeam) * 0.3;
    }

    // Color transition along beam (agent -> white)
    vec3 beamColor = mix(uAgentColor, WHITE, beamProgress);

    // Add energy particles along beam
    float particles = sin(alongBeam * 50.0 - uTime * 20.0) * 0.5 + 0.5;
    particles *= beamMask;
    beamColor += uAgentColor * particles * 0.3;

    // Additive blend for glow effect
    vec3 result = sceneColor.rgb + beamColor * beamMask * (1.0 - progress * 0.3);

    return vec4(result, sceneColor.a);
}

/**
 * Style 2: Spiral Merge
 * Color spirals inward then shoots to main
 */
vec4 spiralMerge(vec2 uv, vec4 sceneColor, vec4 agentColor, float progress) {
    vec2 centered = uv - uAgentCenter;
    float dist = length(centered);
    float angle = atan(centered.y, centered.x);

    // Spiral parameters
    float spiralSpeed = 3.0;
    float spiralTightness = 5.0;

    // Rotating spiral mask
    float spiral = sin(angle * spiralTightness - uTime * spiralSpeed + dist * 20.0);
    spiral = smoothstep(0.0, 0.5, spiral);

    // Collapse toward center based on progress
    float collapseRadius = 0.5 * (1.0 - progress);
    float collapseMask = smoothstep(collapseRadius, collapseRadius + 0.1, dist);

    // Combine effects
    float effectMask = spiral * (1.0 - collapseMask);

    // After collapse, shoot beam to main
    vec4 beamEffect = vec4(0.0);
    if (progress > 0.5) {
        float beamProgress = (progress - 0.5) * 2.0;
        beamEffect = beamMerge(uv, vec4(0.0), agentColor, beamProgress);
    }

    // Color desaturation as it spirals in
    vec3 hsl = rgb2hsl(agentColor.rgb);
    hsl.y *= 1.0 - progress; // Reduce saturation
    hsl.z = mix(hsl.z, 1.0, progress); // Increase lightness
    vec3 fadedColor = hsl2rgb(hsl);

    vec3 result = sceneColor.rgb + fadedColor * effectMask * (1.0 - progress);
    result += beamEffect.rgb;

    return vec4(result, sceneColor.a);
}

/**
 * Style 3: Dissolve Merge
 * Particles dissolve and reform at main
 */
vec4 dissolveMerge(vec2 uv, vec4 sceneColor, vec4 agentColor, float progress) {
    // Noise-based dissolve pattern
    float dissolveNoise = fbm(uv * 10.0 + vec2(uTime * 0.2));

    // Dissolve threshold moves with progress
    float dissolveThreshold = progress;
    float dissolveMask = smoothstep(dissolveThreshold - 0.1, dissolveThreshold, dissolveNoise);

    // Particles that have dissolved reappear at main
    float reformNoise = fbm((uv - uMergeTarget) * 10.0 - vec2(uTime * 0.3));
    float reformThreshold = 1.0 - progress;
    float reformMask = smoothstep(reformThreshold - 0.1, reformThreshold, reformNoise);

    // Distance falloff for reform (only near main)
    float distToMain = length(uv - uMergeTarget);
    reformMask *= smoothstep(0.3, 0.0, distToMain);

    // Edge glow during dissolve
    float edgeGlow = abs(dissolveNoise - dissolveThreshold) < 0.05 ? 1.0 : 0.0;
    edgeGlow *= 1.0 - progress;

    // Combine effects
    vec3 dissolvedColor = agentColor.rgb * dissolveMask;
    vec3 reformedColor = WHITE * reformMask;
    vec3 glowColor = uAgentColor * edgeGlow * 2.0;

    vec3 result = sceneColor.rgb + dissolvedColor + reformedColor + glowColor;

    return vec4(result, sceneColor.a);
}

// =============================================================================
// Main Fragment Shader
// =============================================================================

void main() {
    // Sample textures
    vec4 sceneColor = texture2D(uSceneTexture, vUv);
    vec4 agentColor = texture2D(uAgentTexture, vUv);

    // Apply agent color uniform if agent texture is mask-only
    if (agentColor.a > 0.0 && length(agentColor.rgb) < 0.1) {
        agentColor.rgb = uAgentColor;
    }

    vec4 result;

    // Select merge style
    if (uMergeStyle == 0) {
        result = colorDrain(vUv, sceneColor, agentColor, uProgress);
    } else if (uMergeStyle == 1) {
        result = beamMerge(vUv, sceneColor, agentColor, uProgress);
    } else if (uMergeStyle == 2) {
        result = spiralMerge(vUv, sceneColor, agentColor, uProgress);
    } else {
        result = dissolveMerge(vUv, sceneColor, agentColor, uProgress);
    }

    // Final flash at completion
    if (uProgress > 0.95) {
        float flash = (uProgress - 0.95) * 20.0; // 0-1 in last 5%
        flash = flash * flash * (3.0 - 2.0 * flash); // Smoothstep
        result.rgb = mix(result.rgb, WHITE, flash * 0.5);
    }

    gl_FragColor = result;
}
