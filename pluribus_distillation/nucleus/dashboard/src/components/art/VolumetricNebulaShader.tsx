/**
 * VolumetricNebulaShader.tsx
 * ==========================
 * Volumetric raymarching shader for the loading screen.
 * Based on provided GLSL snippet.
 */

import { component$, useSignal, useVisibleTask$ } from "@builder.io/qwik";

const FRAGMENT_SHADER = `
precision highp float;
uniform vec3 iResolution;
uniform float iTime;

// --- Helpers ---

// IQ's Palette
vec3 spectrum(float t) {
    return vec3(0.5) + vec3(0.5)*cos(6.28318*(vec3(1.0)*t + vec3(0.0, 0.33, 0.67)));
}

// 3D Noise
float hash(float n) { return fract(sin(n)*43758.5453); }
float noise(vec3 x) {
    vec3 p = floor(x);
    vec3 f = fract(x);
    f = f*f*(3.0-2.0*f);
    float n = p.x + p.y*57.0 + 113.0*p.z;
    return mix(mix(mix( hash(n+0.0), hash(n+1.0),f.x),
                   mix( hash(n+57.0), hash(n+58.0),f.x),f.y),
               mix(mix( hash(n+113.0), hash(n+114.0),f.x),
                   mix( hash(n+170.0), hash(n+171.0),f.x),f.y),f.z);
}

// SDF Map - Nebula Sphere
float map(vec3 p) {
    vec3 q = p + vec3(0.0, iTime * 0.1, 0.0);
    float f = 0.0;
    float amp = 0.5;
    for(int i=0; i<4; i++) {
        f += amp * noise(q);
        q = q * 2.03;
        amp *= 0.5;
    }
    // Deformed sphere
    return length(p) - 2.0 + f * 0.8;
}

// --- Provided Code ---

mat3 calcLookAtMatrix(vec3 ro, vec3 ta, vec3 up) {
    vec3 ww = normalize(ta - ro);
    vec3 uu = normalize(cross(ww,up));
    vec3 vv = normalize(cross(uu,ww));
    return mat3(uu, vv, ww);
}

void mainImage(out vec4 fragColor, in vec2 fragCoord) {
    float time = mod(iTime / 2., 1.); // Local time var

    vec3 camPos = vec3(1.8, 5.5, -5.5) * 1.75;
    vec3 camTar = vec3(.0,0,.0);
    vec3 camUp = vec3(-1,0,-1.5);
    mat3 camMat = calcLookAtMatrix(camPos, camTar, camUp);
    float focalLength = 5.;
    vec2 p = (-iResolution.xy + 2. * fragCoord.xy) / iResolution.y;

    vec3 rayDirection = normalize(camMat * vec3(p, focalLength));
    vec3 rayPosition = camPos;
    float rayLength = 0.;

    float distance = 0.;
    vec3 color = vec3(0);

    vec3 c;

    // Keep iteration count too low to pass through entire model,
    // giving the effect of fogged glass
    const float ITER = 82.;
    const float FUDGE_FACTORR = .8;
    const float INTERSECTION_PRECISION = .001;
    const float MAX_DIST = 20.;

    for (float i = 0.; i < ITER; i++) {
        // Step a little slower so we can accumilate glow
        rayLength += max(INTERSECTION_PRECISION, abs(distance) * FUDGE_FACTORR);
        rayPosition = camPos + rayDirection * rayLength;
        distance = map(rayPosition);

        // Add a lot of light when we're really close to the surface
        c = vec3(max(0., .01 - abs(distance)) * .5);
        c *= vec3(1.4,2.1,1.7); // blue green tint

        // Accumilate some purple glow for every step
        c += vec3(.6,.25,.7) * FUDGE_FACTORR / 160.;
        c *= smoothstep(20., 7., length(rayPosition));

        // Fade out further away from the camera
        float rl = smoothstep(MAX_DIST, .1, rayLength);
        c *= rl;

        // Vary colour as we move through space
        c *= spectrum(rl * 6. - .6);

        color += c;

        if (rayLength > MAX_DIST) {
            break;
        }
    }

    // Tonemapping and gamma
    color = pow(color, vec3(1. / 1.8)) * 2.;
    color = pow(color, vec3(2.)) * 3.;
    color = pow(color, vec3(1. / 2.2));

    fragColor = vec4(color, 1);
}

void main() {
    mainImage(gl_FragColor, gl_FragCoord.xy);
}
`;

export const VolumetricNebulaShader = component$(() => {
    const canvasRef = useSignal<HTMLCanvasElement>();

    useVisibleTask$(({ cleanup }) => {
        const canvas = canvasRef.value;
        if (!canvas) return;

        const gl = canvas.getContext('webgl');
        if (!gl) return;

        const vs = gl.createShader(gl.VERTEX_SHADER)!;
        gl.shaderSource(vs, 'attribute vec2 p;void main(){gl_Position=vec4(p,0,1);}');
        gl.compileShader(vs);

        const fs = gl.createShader(gl.FRAGMENT_SHADER)!;
        gl.shaderSource(fs, FRAGMENT_SHADER);
        gl.compileShader(fs);
        
        if (!gl.getShaderParameter(fs, gl.COMPILE_STATUS)) {
            console.error(gl.getShaderInfoLog(fs));
            return;
        }

        const program = gl.createProgram()!;
        gl.attachShader(program, vs);
        gl.attachShader(program, fs);
        gl.linkProgram(program);
        gl.useProgram(program);

        const buf = gl.createBuffer();
        gl.bindBuffer(gl.ARRAY_BUFFER, buf);
        gl.bufferData(gl.ARRAY_BUFFER, new Float32Array([-1,-1, 1,-1, -1,1, 1,1]), gl.STATIC_DRAW);
        
        const pLoc = gl.getAttribLocation(program, 'p');
        gl.enableVertexAttribArray(pLoc);
        gl.vertexAttribPointer(pLoc, 2, gl.FLOAT, false, 0, 0);

        const resLoc = gl.getUniformLocation(program, 'iResolution');
        const timeLoc = gl.getUniformLocation(program, 'iTime');

        let frameId: number;
        const start = Date.now();

        const render = () => {
            if (canvas.width !== canvas.clientWidth || canvas.height !== canvas.clientHeight) {
                canvas.width = canvas.clientWidth;
                canvas.height = canvas.clientHeight;
                gl.viewport(0, 0, canvas.width, canvas.height);
            }
            
            gl.uniform3f(resLoc, canvas.width, canvas.height, 1);
            gl.uniform1f(timeLoc, (Date.now() - start) * 0.001);
            
            gl.drawArrays(gl.TRIANGLE_STRIP, 0, 4);
            frameId = requestAnimationFrame(render);
        };

        render();
        cleanup(() => cancelAnimationFrame(frameId));
    });

    return <canvas ref={canvasRef} class="w-full h-full block" />;
});
