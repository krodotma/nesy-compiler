/**
 * Chromatic Agents Visualizer - GPU Particle Vertex Shader
 *
 * Step 20: Optimized GLSL shaders for performance
 *
 * This shader handles GPU-accelerated particle rendering for:
 * - Bus event sparks (color = source agent)
 * - Commit data bursts
 * - Push/pull directional streams
 * - Merge convergent spirals
 *
 * Designed for instanced rendering with up to 100k particles.
 */

precision highp float;

// =============================================================================
// Attributes (per-vertex)
// =============================================================================

attribute vec3 position;          // Base quad vertex position
attribute vec2 uv;                // Texture coordinates

// =============================================================================
// Attributes (per-instance, for instanced rendering)
// =============================================================================

attribute vec3 instancePosition;  // Particle world position
attribute vec3 instanceVelocity;  // Particle velocity
attribute vec4 instanceColor;     // RGBA color (agent hue encoded)
attribute float instanceSize;     // Particle size
attribute float instanceLife;     // Normalized lifetime (0-1, 1=dead)
attribute float instanceSeed;     // Random seed for variation

// =============================================================================
// Uniforms
// =============================================================================

uniform mat4 modelViewMatrix;
uniform mat4 projectionMatrix;
uniform float uTime;              // Global time for animation
uniform float uDeltaTime;         // Frame delta time
uniform vec3 uCameraPosition;     // Camera world position
uniform float uParticleScale;     // Global particle scale multiplier
uniform int uParticleMode;        // 0=spark, 1=burst, 2=stream, 3=spiral

// Attractor uniforms (for spiral/merge animations)
uniform vec3 uAttractorPosition;
uniform float uAttractorStrength;

// =============================================================================
// Varyings (to fragment shader)
// =============================================================================

varying vec2 vUv;
varying vec4 vColor;
varying float vLife;
varying float vGlow;

// =============================================================================
// Constants
// =============================================================================

const float PI = 3.14159265359;
const float TWO_PI = 6.28318530718;

// =============================================================================
// Noise Functions
// =============================================================================

/**
 * Simple 1D hash function
 */
float hash(float n) {
    return fract(sin(n) * 43758.5453123);
}

/**
 * 2D value noise
 */
float noise2D(vec2 p) {
    vec2 i = floor(p);
    vec2 f = fract(p);
    f = f * f * (3.0 - 2.0 * f); // Smoothstep

    float a = hash(i.x + i.y * 57.0);
    float b = hash(i.x + 1.0 + i.y * 57.0);
    float c = hash(i.x + (i.y + 1.0) * 57.0);
    float d = hash(i.x + 1.0 + (i.y + 1.0) * 57.0);

    return mix(mix(a, b, f.x), mix(c, d, f.x), f.y);
}

// =============================================================================
// Particle Motion Functions
// =============================================================================

/**
 * Calculate spark motion (random jitter, fade out)
 */
vec3 calculateSparkMotion(vec3 pos, vec3 vel, float life, float seed) {
    float jitter = (noise2D(vec2(seed * 100.0, uTime * 10.0)) - 0.5) * 0.1;
    pos += vel * uDeltaTime;
    pos.x += jitter;
    pos.y += jitter;
    return pos;
}

/**
 * Calculate burst motion (radial expansion with gravity)
 */
vec3 calculateBurstMotion(vec3 pos, vec3 vel, float life, float seed) {
    float gravity = -9.8 * life * life; // Accelerating fall
    pos += vel * uDeltaTime;
    pos.y += gravity * uDeltaTime * 0.01;
    return pos;
}

/**
 * Calculate stream motion (directional flow with turbulence)
 */
vec3 calculateStreamMotion(vec3 pos, vec3 vel, float life, float seed) {
    float turbulence = noise2D(vec2(pos.x * 2.0 + uTime, pos.z * 2.0)) - 0.5;
    pos += vel * uDeltaTime;
    pos.x += turbulence * 0.05;
    pos.z += turbulence * 0.05;
    return pos;
}

/**
 * Calculate spiral motion (toward attractor)
 */
vec3 calculateSpiralMotion(vec3 pos, vec3 vel, float life, float seed) {
    // Direction to attractor
    vec3 toAttractor = uAttractorPosition - pos;
    float dist = length(toAttractor);

    // Spiral offset
    float angle = uTime * 3.0 + seed * TWO_PI;
    float spiralRadius = dist * 0.3 * (1.0 - life);
    vec3 spiralOffset = vec3(
        cos(angle) * spiralRadius,
        sin(angle * 0.7) * spiralRadius * 0.5,
        sin(angle) * spiralRadius
    );

    // Move toward attractor with spiral
    vec3 attractForce = normalize(toAttractor) * uAttractorStrength * uDeltaTime;
    pos += attractForce + vel * uDeltaTime * (1.0 - life);
    pos += spiralOffset * 0.1;

    return pos;
}

// =============================================================================
// Size Calculation
// =============================================================================

/**
 * Calculate particle size with life-based scaling
 */
float calculateSize(float baseSize, float life, int mode) {
    float lifeScale = 1.0;

    if (mode == 0) {
        // Spark: quick fade
        lifeScale = 1.0 - life * life;
    } else if (mode == 1) {
        // Burst: pop then fade
        lifeScale = sin(life * PI);
    } else if (mode == 2) {
        // Stream: consistent size
        lifeScale = 1.0 - life * 0.5;
    } else if (mode == 3) {
        // Spiral: grow toward center
        lifeScale = 0.5 + life * 0.5;
    }

    return baseSize * lifeScale * uParticleScale;
}

// =============================================================================
// Billboard Rotation
// =============================================================================

/**
 * Create billboard rotation matrix to face camera
 */
mat3 createBillboardMatrix(vec3 particlePos) {
    vec3 look = normalize(uCameraPosition - particlePos);
    vec3 right = normalize(cross(vec3(0.0, 1.0, 0.0), look));
    vec3 up = cross(look, right);
    return mat3(right, up, look);
}

// =============================================================================
// Main Vertex Shader
// =============================================================================

void main() {
    // Skip dead particles
    if (instanceLife >= 1.0) {
        gl_Position = vec4(0.0);
        vColor = vec4(0.0);
        return;
    }

    // Calculate particle position based on mode
    vec3 particlePos = instancePosition;

    if (uParticleMode == 0) {
        particlePos = calculateSparkMotion(particlePos, instanceVelocity, instanceLife, instanceSeed);
    } else if (uParticleMode == 1) {
        particlePos = calculateBurstMotion(particlePos, instanceVelocity, instanceLife, instanceSeed);
    } else if (uParticleMode == 2) {
        particlePos = calculateStreamMotion(particlePos, instanceVelocity, instanceLife, instanceSeed);
    } else if (uParticleMode == 3) {
        particlePos = calculateSpiralMotion(particlePos, instanceVelocity, instanceLife, instanceSeed);
    }

    // Calculate size
    float size = calculateSize(instanceSize, instanceLife, uParticleMode);

    // Create billboard matrix
    mat3 billboard = createBillboardMatrix(particlePos);

    // Transform quad vertex to billboard
    vec3 vertexPos = billboard * (position * size);
    vec3 worldPos = particlePos + vertexPos;

    // Final position
    gl_Position = projectionMatrix * modelViewMatrix * vec4(worldPos, 1.0);

    // Calculate glow intensity (brighter at center of life)
    float glow = 1.0 - abs(instanceLife * 2.0 - 1.0);
    glow = pow(glow, 0.5); // Softer falloff

    // Pass to fragment shader
    vUv = uv;
    vLife = instanceLife;
    vGlow = glow;

    // Fade color alpha based on life
    vColor = instanceColor;
    vColor.a *= 1.0 - instanceLife;

    // Add seed-based color variation
    float hueShift = (instanceSeed - 0.5) * 0.1;
    vColor.rgb = mix(vColor.rgb, vColor.rgb * (1.0 + hueShift), 0.5);
}

// =============================================================================
// Companion Fragment Shader (inline for reference)
// =============================================================================

/*
precision highp float;

varying vec2 vUv;
varying vec4 vColor;
varying float vLife;
varying float vGlow;

uniform sampler2D uParticleTexture;

void main() {
    // Sample particle texture (soft circle)
    vec4 texColor = texture2D(uParticleTexture, vUv);

    // Radial gradient for soft particles
    float dist = length(vUv - 0.5) * 2.0;
    float alpha = 1.0 - smoothstep(0.0, 1.0, dist);
    alpha *= texColor.a;

    // Apply color and glow
    vec3 color = vColor.rgb * (1.0 + vGlow * 0.5);

    // Additive blending preparation
    gl_FragColor = vec4(color * alpha, alpha * vColor.a);
}
*/
