/**
 * Chromatic Agents Visualizer - Neon Bloom/Glow Fragment Shader
 *
 * Step 20: Optimized GLSL shaders for performance
 *
 * This shader implements a high-performance bloom effect for the neon aesthetic.
 * Uses a two-pass Gaussian blur with configurable intensity.
 */

precision highp float;

// Uniforms
uniform sampler2D uTexture;        // Source texture (scene render)
uniform sampler2D uBloomTexture;   // Blurred bloom texture (from blur pass)
uniform vec2 uResolution;          // Screen resolution
uniform float uBloomIntensity;     // Bloom strength (0-2, default 1)
uniform float uBloomThreshold;     // Luminance threshold for bloom (0-1)
uniform float uTime;               // Animation time for optional pulsing
uniform vec3 uTintColor;           // Optional color tint for the glow

// Varyings
varying vec2 vUv;

// =============================================================================
// Constants
// =============================================================================

const float GAMMA = 2.2;
const float INV_GAMMA = 1.0 / GAMMA;

// Bloom color temperature adjustments per agent (indexed by uniform)
const vec3 MAGENTA_TINT = vec3(1.0, 0.0, 1.0);   // Claude
const vec3 CYAN_TINT = vec3(0.0, 1.0, 1.0);      // Qwen
const vec3 YELLOW_TINT = vec3(1.0, 1.0, 0.0);    // Gemini
const vec3 GREEN_TINT = vec3(0.0, 1.0, 0.0);     // Codex
const vec3 WHITE_TINT = vec3(1.0, 1.0, 1.0);     // Main

// =============================================================================
// Utility Functions
// =============================================================================

/**
 * Convert sRGB to linear color space for physically correct blending
 */
vec3 sRGBToLinear(vec3 srgb) {
    return pow(srgb, vec3(GAMMA));
}

/**
 * Convert linear to sRGB for display
 */
vec3 linearToSRGB(vec3 linear) {
    return pow(linear, vec3(INV_GAMMA));
}

/**
 * Calculate perceived luminance
 */
float luminance(vec3 color) {
    return dot(color, vec3(0.2126, 0.7152, 0.0722));
}

/**
 * Soft threshold for bloom extraction
 * Uses a smooth curve instead of hard cutoff
 */
vec3 softThreshold(vec3 color, float threshold) {
    float lum = luminance(color);
    float soft = clamp((lum - threshold + 0.5) * 2.0, 0.0, 1.0);
    return color * soft * soft; // Quadratic falloff
}

/**
 * Apply chromatic aberration for extra neon feel
 */
vec3 chromaticAberration(sampler2D tex, vec2 uv, float amount) {
    vec2 offset = (uv - 0.5) * amount;
    float r = texture2D(tex, uv + offset).r;
    float g = texture2D(tex, uv).g;
    float b = texture2D(tex, uv - offset).b;
    return vec3(r, g, b);
}

// =============================================================================
// Main Glow Composition
// =============================================================================

void main() {
    // Sample original scene
    vec4 sceneColor = texture2D(uTexture, vUv);
    vec3 scene = sRGBToLinear(sceneColor.rgb);

    // Sample bloom texture (pre-blurred)
    vec4 bloomSample = texture2D(uBloomTexture, vUv);
    vec3 bloom = sRGBToLinear(bloomSample.rgb);

    // Apply threshold to bloom
    bloom = softThreshold(bloom, uBloomThreshold);

    // Apply tint color to bloom
    bloom *= uTintColor;

    // Optional: Add subtle chromatic aberration to bloom
    #ifdef CHROMATIC_ABERRATION
        vec3 aberrated = chromaticAberration(uBloomTexture, vUv, 0.002);
        bloom = mix(bloom, sRGBToLinear(aberrated), 0.3);
    #endif

    // Optional: Pulse the bloom with time
    #ifdef ANIMATED_GLOW
        float pulse = 1.0 + 0.1 * sin(uTime * 2.0);
        bloom *= pulse;
    #endif

    // Additive blend: scene + bloom
    vec3 result = scene + bloom * uBloomIntensity;

    // Tonemap to prevent oversaturation (Reinhard)
    result = result / (result + vec3(1.0));

    // Convert back to sRGB
    result = linearToSRGB(result);

    // Apply subtle vignette for focus
    #ifdef VIGNETTE
        float dist = length(vUv - 0.5) * 1.4;
        result *= 1.0 - dist * dist * 0.3;
    #endif

    gl_FragColor = vec4(result, sceneColor.a);
}

// =============================================================================
// Blur Pass Shader (for pre-computing bloom texture)
// =============================================================================

#ifdef BLUR_PASS

/**
 * 9-tap Gaussian blur, optimized for separable passes
 * Call this twice: once horizontal, once vertical
 */

uniform vec2 uBlurDirection; // (1,0) for horizontal, (0,1) for vertical

// Gaussian weights for 9 samples (sigma ~2.0)
const float weights[5] = float[5](
    0.227027,
    0.1945946,
    0.1216216,
    0.054054,
    0.016216
);

void mainBlur() {
    vec2 texelSize = 1.0 / uResolution;
    vec3 result = texture2D(uTexture, vUv).rgb * weights[0];

    for (int i = 1; i < 5; i++) {
        vec2 offset = uBlurDirection * texelSize * float(i) * 2.0;
        result += texture2D(uTexture, vUv + offset).rgb * weights[i];
        result += texture2D(uTexture, vUv - offset).rgb * weights[i];
    }

    gl_FragColor = vec4(result, 1.0);
}

#endif

// =============================================================================
// Bright Pass Shader (extract bright pixels for bloom)
// =============================================================================

#ifdef BRIGHT_PASS

void mainBright() {
    vec4 color = texture2D(uTexture, vUv);
    vec3 bright = softThreshold(color.rgb, uBloomThreshold);
    gl_FragColor = vec4(bright, color.a);
}

#endif
