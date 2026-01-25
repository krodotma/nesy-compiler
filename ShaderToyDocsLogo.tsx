/**
 * ShaderToyDocsLogo.tsx
 *
 * EXACT implementation of https://www.shadertoy.com/view/4t2XzD
 * From /pluribus/asset_shaders/orbits.md
 * Documentation section header logo - sphere with 64 orbiting rainbow lights
 * 
 * New filename to force cache invalidation.
 */

import { component$, useSignal, useVisibleTask$ } from "@builder.io/qwik";

const VERT = `
attribute vec2 aPos;
void main() {
  gl_Position = vec4(aPos, 0.0, 1.0);
}
`;

// EXACT shader from orbits.md (Shadertoy 4t2XzD)
const FRAG = `
precision highp float;

uniform vec2 iResolution;
uniform float iTime;

#define PI 3.14159265359
#define PI2 (PI/2.)

// tuneables
#define SCALE 15.0
#define ROUGH 0.25
#define REFL 0.02
#define LIGHTWRAP (sin(iTime/2.0)*5.0)
#define NUM_LIGHTS 64
#define SPHERE_RAD 4.5
#define ORBIT_DIST 4.0
#define ALBEDO vec3(0.25)
#define HUE_SHIFT_RATE 0.25
#define HUE_BAND_SCALE 0.25
#define VERTICAL_ACCUM_SIN_SCALE 0.5
#define LIGHT_INTENSITY 0.5

#define saturate(x) clamp(x, 0.0, 1.0)

float GGX(vec3 N, vec3 V, vec3 L, float roughness, float F0) {
    float alpha = roughness*roughness;
    vec3 H = normalize(V+L);
    float dotNL = saturate(dot(N,L));
    float dotLH = saturate(dot(L,H));
    float dotNH = saturate(dot(N,H));
    float alphaSqr = alpha*alpha;
    float denom = dotNH * dotNH *(alphaSqr-1.0) + 1.0;
    float D = alphaSqr/(PI * denom * denom);
    float dotLH5 = pow(1.0-dotLH,5.);
    float F = F0 + (1.-F0)*(dotLH5);
    float k = alpha/2.;
    float k2 = k*k;
    float invK2 = 1.-k2;
    float vis = 1./(dotLH*dotLH*invK2 + k2);
    return dotNL * D * F * vis;
}

vec3 hsv_to_rgb(float h, float s, float v) {
    float c = v * s;
    h = mod((h * 6.0), 6.0);
    float x = c * (1.0 - abs(mod(h, 2.0) - 1.0));
    vec3 color;
    if (0.0 <= h && h < 1.0) { color = vec3(c, x, 0.0); }
    else if (1.0 <= h && h < 2.0) { color = vec3(x, c, 0.0); }
    else if (2.0 <= h && h < 3.0) { color = vec3(0.0, c, x); }
    else if (3.0 <= h && h < 4.0) { color = vec3(0.0, x, c); }
    else if (4.0 <= h && h < 5.0) { color = vec3(x, 0.0, c); }
    else if (5.0 <= h && h < 6.0) { color = vec3(c, 0.0, x); }
    else { color = vec3(0.0, 0.0, 0.0); }
    color.rgb += v - c;
    return color;
}

struct PointLight { vec3 pos; vec3 color; };
struct LightAnim { vec3 period; vec3 shift; vec3 orbit; vec3 offset; };

vec3 sphereNorm(vec2 ws, vec3 c, float r) {
    vec3 pt = vec3((ws-c.xy)/r, 0.0);
    pt.z = -cos(length(pt.xy)*PI2);
    return normalize(pt);
}

bool sphereTest(vec2 ws, vec3 c, float r) {
    vec3 pt = vec3(ws-c.xy, c.z);
    return (dot(pt.xy, pt.xy) < r*r);
}

vec3 spherePos(vec2 ws, vec3 c, float r) {
    vec3 pt = vec3(ws, c.z);
    pt.z -= cos(length((ws-c.xy)/r)*PI2)*r;
    return pt;
}

vec4 sphere(vec3 pt, vec3 N, PointLight pl, float rough, float refl) {
    vec3 V = vec3(0.0, 0.0, -1.0);
    vec3 pToL = pl.pos - pt;
    vec3 L = normalize(pToL);
    float decay = length(pToL);
    decay = 1.0/(decay*decay);
    float diffuse = dot(N,L) / PI;
    float spec = GGX(N, V, L, rough, refl);
    if (dot(N,L) >= 0.0) {
        return vec4(decay * pl.color * (spec + diffuse * ALBEDO), pt.z);
    }
    return vec4(0.0, 0.0, 0.0, pt.z);
}

PointLight getLight(vec3 color, LightAnim anim) {
    vec3 pos = sin(iTime * anim.period + anim.shift) * anim.orbit + anim.offset;
    return PointLight(pos, color);
}

vec4 renderLight(vec2 cs, PointLight pt) {
    return vec4(pt.color * saturate(0.1 - abs(length(cs-pt.pos.xy)))*100.0, pt.pos.z);
}

void drawWriteZ(vec4 result, inout vec4 fragColor) {
    fragColor.xyz += result.xyz;
    fragColor.w = result.w;
}

void drawTestZ(vec4 result, inout vec4 fragColor) {
    if (result.w <= fragColor.w || fragColor.w < 0.0) {
        fragColor.xyz += result.xyz;
    }
}

void planet(vec2 csUv, inout vec4 fragColor, LightAnim anim, bool isGeo, vec3 norm, vec3 pos, vec3 color) {
    PointLight ptL = getLight(color, anim);
    if (isGeo) {
        drawWriteZ(sphere(pos, norm, ptL, ROUGH, REFL), fragColor);
    }
    drawTestZ(renderLight(csUv, ptL), fragColor);
}

void main() {
    vec2 fragCoord = gl_FragCoord.xy;
    float aspect = iResolution.x / iResolution.y;
    vec2 uv = fragCoord.xy / iResolution.xy;
    vec4 csUv = vec4(uv-vec2(0.5, 0.5), 0.0, 0.0);
    csUv.x *= aspect;
    csUv.xy *= SCALE;
    
    float sphereRad = SPHERE_RAD;
    float orbitDelta = ORBIT_DIST;
    float orbit = sphereRad+orbitDelta;
    
    LightAnim anim = LightAnim(vec3(1.0, 0.0, 1.0), vec3(0.0, PI2, PI2), vec3(orbit, 0.0, -orbit), vec3(0.0, 0.0, 10.0));
    vec3 sphereCenter = vec3(0.0, 0.0, 10.0);
    
    vec3 sPos = spherePos(csUv.xy, sphereCenter, sphereRad);
    vec3 sNorm = sphereNorm(csUv.xy, sphereCenter, sphereRad);
    bool isSphere = sphereTest(csUv.xy, sphereCenter, sphereRad);
    
    vec4 fragColor = vec4(0.0, 0.0, 0.0, -1.0);
    
    for (int i = 0; i < NUM_LIGHTS; ++i) {
        float rat = 1.0-float(i)/float(NUM_LIGHTS);
        float hue = mod(HUE_SHIFT_RATE*-iTime+rat*HUE_BAND_SCALE, 1.0);
        vec3 color = hsv_to_rgb(hue, 1.0, LIGHT_INTENSITY*rat);
        planet(csUv.xy, fragColor, anim, isSphere, sNorm, sPos, color);
        anim.orbit.y += sin(iTime)*VERTICAL_ACCUM_SIN_SCALE;
        anim.shift += LIGHTWRAP*2.0*PI/float(NUM_LIGHTS);
    }
    
    fragColor.xyz = pow(fragColor.xyz, 1.0/vec3(2.2));
    
    // Transparent background for non-lit areas
    float alpha = length(fragColor.xyz) > 0.01 ? 1.0 : 0.0;
    gl_FragColor = vec4(fragColor.xyz, alpha);
}
`;

export const ShaderToyDocsLogo = component$(() => {
    const canvasRef = useSignal<HTMLCanvasElement>();

    useVisibleTask$(({ cleanup }) => {
        console.log("[ShaderToyDocsLogo] visible task start");
        const canvas = canvasRef.value;
        if (!canvas) {
            console.error("[ShaderToyDocsLogo] No canvas ref");
            return;
        }

        const gl = canvas.getContext("webgl", { alpha: true, premultipliedAlpha: false });
        if (!gl) {
            console.error("[ShaderToyDocsLogo] WebGL not available");
            return;
        }
        console.log("[ShaderToyDocsLogo] WebGL context acquired");

        const compile = (src: string, type: number) => {
            const sh = gl.createShader(type);
            if (!sh) return null;
            gl.shaderSource(sh, src);
            gl.compileShader(sh);
            if (!gl.getShaderParameter(sh, gl.COMPILE_STATUS)) {
                console.error("[ShaderToyDocsLogo] Shader error:", gl.getShaderInfoLog(sh));
                return null;
            }
            return sh;
        };

        const vs = compile(VERT, gl.VERTEX_SHADER);
        const fs = compile(FRAG, gl.FRAGMENT_SHADER);
        if (!vs || !fs) return;

        const prog = gl.createProgram();
        if (!prog) return;
        gl.attachShader(prog, vs);
        gl.attachShader(prog, fs);
        gl.linkProgram(prog);
        if (!gl.getProgramParameter(prog, gl.LINK_STATUS)) {
            console.error("[ShaderToyDocsLogo] Link error:", gl.getProgramInfoLog(prog));
            return;
        }
        gl.useProgram(prog);
        console.log("[ShaderToyDocsLogo] Shader compiled and linked successfully");

        const buf = gl.createBuffer();
        gl.bindBuffer(gl.ARRAY_BUFFER, buf);
        gl.bufferData(gl.ARRAY_BUFFER, new Float32Array([-1, -1, 1, -1, -1, 1, 1, 1]), gl.STATIC_DRAW);

        const posLoc = gl.getAttribLocation(prog, "aPos");
        gl.enableVertexAttribArray(posLoc);
        gl.vertexAttribPointer(posLoc, 2, gl.FLOAT, false, 0, 0);

        const resLoc = gl.getUniformLocation(prog, "iResolution");
        const timeLoc = gl.getUniformLocation(prog, "iTime");

        let frameId: number;
        let dw = 0, dh = 0;
        const t0 = performance.now();

        const render = () => {
            const dpr = window.devicePixelRatio || 1;
            const rect = canvas.getBoundingClientRect();
            const w = Math.floor(rect.width * dpr);
            const h = Math.floor(rect.height * dpr);

            if (w !== dw || h !== dh) {
                dw = w; dh = h;
                canvas.width = w; canvas.height = h;
                gl.viewport(0, 0, w, h);
            }

            gl.clearColor(0, 0, 0, 0);
            gl.clear(gl.COLOR_BUFFER_BIT);
            gl.uniform2f(resLoc, dw, dh);
            gl.uniform1f(timeLoc, (performance.now() - t0) / 1000);
            gl.drawArrays(gl.TRIANGLE_STRIP, 0, 4);

            frameId = requestAnimationFrame(render);
        };

        console.log("[ShaderToyDocsLogo] Starting render loop");
        render();
        cleanup(() => cancelAnimationFrame(frameId));
    });

    return (
        <div class="header-logo-orb" aria-hidden="true" style={{ width: "64px", height: "64px", flexShrink: 0, borderRadius: "50%", overflow: "hidden" }}>
            <canvas ref={canvasRef} style={{ width: "100%", height: "100%" }} />
        </div>
    );
});
