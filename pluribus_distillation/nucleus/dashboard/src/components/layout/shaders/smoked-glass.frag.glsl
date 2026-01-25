
// Smoked Glass Fragment Shader
// Part of Visual Style Model 2.0
// Renders volumetric fog, grain, and gradient interpolation

uniform float uTime;
uniform vec2 uResolution;
uniform vec2 uMouse;
uniform float uScrollY;

// Visual Engine Tokens (injected via uniforms)
uniform vec3 uColorStart; // OKLCH converted to RGB by CPU
uniform vec3 uColorEnd;
uniform vec3 uColorPrimary;
uniform float uNoiseScale;
uniform float uNoiseSpeed;

varying vec2 vUv;

// -----------------------------------------------------------------------------
// Noise Functions (Simplex + Grain)
// -----------------------------------------------------------------------------

// Simplex 3D Noise 
// Source: https://github.com/stegu/webgl-noise/blob/master/src/noise3D.glsl
vec3 mod289(vec3 x) { return x - floor(x * (1.0 / 289.0)) * 289.0; }
vec4 mod289(vec4 x) { return x - floor(x * (1.0 / 289.0)) * 289.0; }
vec4 permute(vec4 x) { return mod289(((x*34.0)+1.0)*x); }
vec4 taylorInvSqrt(vec4 r) { return 1.79284291400159 - 0.85373472095314 * r; }

float snoise(vec3 v) {
  const vec2  C = vec2(1.0/6.0, 1.0/3.0) ;
  const vec4  D = vec4(0.0, 0.5, 1.0, 2.0);

  // First corner
  vec3 i  = floor(v + dot(v, C.yyy) );
  vec3 x0 = v - i + dot(i, C.xxx) ;

  // Other corners
  vec3 g = step(x0.yzx, x0.xyz);
  vec3 l = 1.0 - g;
  vec3 i1 = min( g.xyz, l.zxy );
  vec3 i2 = max( g.xyz, l.zxy );

  //   x0 = x0 - 0.0 + 0.0 * C.xxx;
  //   x1 = x0 - i1  + 1.0 * C.xxx;
  //   x2 = x0 - i2  + 2.0 * C.xxx;
  //   x3 = x0 - 1.0 + 3.0 * C.xxx;
  vec3 x1 = x0 - i1 + C.xxx;
  vec3 x2 = x0 - i2 + C.yyy; // 2.0*C.x = 1/3 = C.y
  vec3 x3 = x0 - D.yyy;      // -1.0+3.0*C.x = -0.5 = -D.y

  // Permutations
  i = mod289(i);
  vec4 p = permute( permute( permute(
             i.z + vec4(0.0, i1.z, i2.z, 1.0 ))
           + i.y + vec4(0.0, i1.y, i2.y, 1.0 ))
           + i.x + vec4(0.0, i1.x, i2.x, 1.0 ));

  // Gradients: 7x7 points over a square, mapped onto an octahedron.
  // The ring size 17*17 = 289 is close to a multiple of 49 (49*6 = 294)
  float n_ = 0.142857142857; // 1.0/7.0
  vec3  ns = n_ * D.wyz - D.xzx;

  vec4 j = p - 49.0 * floor(p * ns.z * ns.z);  //  mod(p,7*7)

  vec4 x_ = floor(j * ns.z);
  vec4 y_ = floor(j - 7.0 * x_ );    // mod(j,N)

  vec4 x = x_ *ns.x + ns.yyyy;
  vec4 y = y_ *ns.x + ns.yyyy;
  vec4 h = 1.0 - abs(x) - abs(y);

  vec4 b0 = vec4( x.xy, y.xy );
  vec4 b1 = vec4( x.zw, y.zw );

  //vec4 s0 = vec4(lessThan(b0,0.0))*2.0 - 1.0;
  //vec4 s1 = vec4(lessThan(b1,0.0))*2.0 - 1.0;
  vec4 s0 = floor(b0)*2.0 + 1.0;
  vec4 s1 = floor(b1)*2.0 + 1.0;
  vec4 sh = -step(h, vec4(0.0));

  vec4 a0 = b0.xzyw + s0.xzyw*sh.xxyy ;
  vec4 a1 = b1.xzyw + s1.xzyw*sh.zzww ;

  vec3 p0 = vec3(a0.xy,h.x);
  vec3 p1 = vec3(a0.zw,h.y);
  vec3 p2 = vec3(a1.xy,h.z);
  vec3 p3 = vec3(a1.zw,h.w);

  //Normalise gradients
  vec4 norm = taylorInvSqrt(vec4(dot(p0,p0), dot(p1,p1), dot(p2, p2), dot(p3,p3)));
  p0 *= norm.x;
  p1 *= norm.y;
  p2 *= norm.z;
  p3 *= norm.w;

  // Mix final noise value
  vec4 m = max(0.5 - vec4(dot(x0,x0), dot(x1,x1), dot(x2,x2), dot(x3,x3)), 0.0);
  m = m * m;
  return 105.0 * dot( m*m, vec4( dot(p0,x0), dot(p1,x1),
                                dot(p2,x2), dot(p3,x3) ) );
}

// High frequency film grain
float random(vec2 p) {
    return fract(sin(dot(p.xy, vec2(12.9898, 78.233))) * 43758.5453123);
}

// -----------------------------------------------------------------------------
// Main Render
// -----------------------------------------------------------------------------

void main() {
    // 1. Normalized Coordinates (0 to 1)
    vec2 st = gl_FragCoord.xy / uResolution.xy;
    
    // 2. Aspect Ratio Correction for circular shapes
    vec2 aspectSt = st;
    aspectSt.x *= uResolution.x / uResolution.y;
    
    // 3. Dynamic Fog / Smoke with Chromatic Aberration (Phase 2 Step 8)
    // and Scroll Parallax (Phase 2 Step 14)
    float t = uTime * uNoiseSpeed * 0.1;
    float scrollOffset = uScrollY * 0.0005;
    
    // Sample noise with slight channel offsets and scroll parallax
    float offset = 0.005; 
    float nr = snoise(vec3((aspectSt + vec2(offset, scrollOffset)) * uNoiseScale, t));
    float ng = snoise(vec3((aspectSt + vec2(0.0, scrollOffset)) * uNoiseScale, t));
    float nb = snoise(vec3((aspectSt + vec2(-offset, scrollOffset)) * uNoiseScale, t));
    
    // Combine noise layers
    float fog = (nr + ng + nb) / 3.0;
    fog = fog * 0.5 + 0.5;
    
    // 4. Gradient Composition
    // Interpolate between Start and End colors based on screen diagonal + fog
    float diagonal = (st.x + st.y) * 0.5;
    vec3 baseColor = mix(uColorStart, uColorEnd, diagonal + (fog * 0.2 - 0.1));
    
    // Mix in the chromatic shift
    baseColor.r += nr * 0.02 * uIntensity;
    baseColor.b += nb * 0.02 * uIntensity;
    
    // 5. Mouse Interaction "Spotlight"
    vec2 mouseNorm = uMouse / uResolution;
    // Fix Y-axis if needed (Three.js coords are usually bottom-left 0,0)
    
    float dist = distance(st, mouseNorm);
    float glow = 1.0 - smoothstep(0.0, 0.4, dist);
    // Add primary color accent at mouse position, blended with fog
    baseColor = mix(baseColor, uColorPrimary, glow * 0.15 * fog);
    
    // 6. Film Grain (Dithering)
    // Helps prevent banding in dark gradients
    float grain = random(st * uTime) * 0.05;
    vec3 finalColor = baseColor + grain;
    
    // 7. Vignette
    float vignette = 1.0 - smoothstep(0.5, 1.5, length(st - 0.5));
    finalColor *= vignette;

    gl_FragColor = vec4(finalColor, 1.0);
}
