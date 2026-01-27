import { component$, useVisibleTask$, useSignal } from '@builder.io/qwik';

const FRAGMENT_SHADER = `
precision highp float;

uniform vec2 iResolution;
uniform float iTime;

#define PI 3.14159265359

void pR(inout vec2 p, float a) {
    p = cos(a)*p + sin(a)*vec2(p.y, -p.x);
}

float smax(float a, float b, float r) {
    vec2 u = max(vec2(r + a,r + b), vec2(0));
    return min(-r, max (a, b)) + length(u);
}

vec3 pal( in float t, in vec3 a, in vec3 b, in vec3 c, in vec3 d ) {
    return a + b*cos( 6.28318*(c*t+d) );
}

vec3 spectrum(float n) {
    return pal( n, vec3(0.5,0.5,0.5),vec3(0.5,0.5,0.5),vec3(1.0,1.0,1.0),vec3(0.0,0.33,0.67) );
}

vec4 inverseStereographic(vec3 p, out float k) {
    k = 2.0/(1.0+dot(p,p));
    return vec4(k*p,k-1.0);
}

float fTorus(vec4 p4) {
    float d1 = length(p4.xy) / length(p4.zw) - 1.;
    float d2 = length(p4.zw) / length(p4.xy) - 1.;
    float d = d1 < 0. ? -d1 : d2;
    d /= PI;
    return d;
}

float fixDistance(float d, float k) {
    float sn = sign(d);
    d = abs(d);
    d = d / k * 1.82;
    d += 1.;
    d = pow(d, .5);
    d -= 1.;
    d *= 5./3.;
    d *= sn;
    return d;
}

float time;

float map(vec3 p) {
    float k;
    vec4 p4 = inverseStereographic(p,k);
    pR(p4.zy, time * -PI / 2.);
    pR(p4.xw, time * -PI / 2.);
    float d = fTorus(p4);
    d = abs(d);
    d -= .2;
    d = fixDistance(d, k);
    d = smax(d, length(p) - 1.85, .2);
    return d;
}

mat3 calcLookAtMatrix(vec3 ro, vec3 ta, vec3 up) {
    vec3 ww = normalize(ta - ro);
    vec3 uu = normalize(cross(ww,up));
    vec3 vv = normalize(cross(uu,ww));
    return mat3(uu, vv, ww);
}

void main() {
    time = mod(iTime / 2., 1.);
    vec3 camPos = vec3(1.8, 5.5, -5.5) * 1.75;
    vec3 camTar = vec3(.0,0,.0);
    vec3 camUp = vec3(-1,0,-1.5);
    mat3 camMat = calcLookAtMatrix(camPos, camTar, camUp);
    float focalLength = 5.;

    vec2 p = (-iResolution.xy + 2. * gl_FragCoord.xy) / iResolution.y;
    vec3 rayDirection = normalize(camMat * vec3(p, focalLength));
    vec3 rayPosition = camPos;
    float rayLength = 0.;
    float distance = 0.;
    vec3 color = vec3(0);
    vec3 c;

    const float ITER = 82.;
    const float FUDGE_FACTORR = .8;
    const float INTERSECTION_PRECISION = .001;
    const float MAX_DIST = 20.;

    for (float i = 0.; i < ITER; i++) {
        rayLength += max(INTERSECTION_PRECISION, abs(distance) * FUDGE_FACTORR);
        rayPosition = camPos + rayDirection * rayLength;
        distance = map(rayPosition);
        c = vec3(max(0., .01 - abs(distance)) * .5);
        c *= vec3(1.4,2.1,1.7);
        c += vec3(.6,.25,.7) * FUDGE_FACTORR / 160.;
        c *= smoothstep(20., 7., length(rayPosition));
        float rl = smoothstep(MAX_DIST, .1, rayLength);
        c *= rl;
        c *= spectrum(rl * 6. - .6);
        color += c;
        if (rayLength > MAX_DIST) {
            break;
        }
    }

    color = pow(color, vec3(1. / 1.8)) * 2.;
    color = pow(color, vec3(2.)) * 3.;
    color = pow(color, vec3(1. / 2.2));
    
    // Transparent background - only show the torus glow
    float alpha = clamp(length(color) * 1.5, 0.0, 1.0);
    gl_FragColor = vec4(color, alpha);
}
`;

const VERTEX_SHADER = `
attribute vec2 position;
void main() {
    gl_Position = vec4(position, 0.0, 1.0);
}
`;

export const CliffordTorusShader = component$(() => {
    // Support ?noshade URL param to disable for performance testing
    const noShade = typeof window !== 'undefined' && new URLSearchParams(window.location.search).has('noshade');
    const canvasRef = useSignal<HTMLCanvasElement>();

    useVisibleTask$(({ cleanup }) => {
        if (noShade) return;
        const canvas = canvasRef.value;
        if (!canvas) return;

        const gl = canvas.getContext('webgl', { alpha: true, premultipliedAlpha: false });
        if (!gl) return;

        const vs = gl.createShader(gl.VERTEX_SHADER)!;
        gl.shaderSource(vs, VERTEX_SHADER);
        gl.compileShader(vs);

        const fs = gl.createShader(gl.FRAGMENT_SHADER)!;
        gl.shaderSource(fs, FRAGMENT_SHADER);
        gl.compileShader(fs);

        if (!gl.getShaderParameter(fs, gl.COMPILE_STATUS)) {
            console.error('Shader compile error:', gl.getShaderInfoLog(fs));
            return;
        }

        const program = gl.createProgram()!;
        gl.attachShader(program, vs);
        gl.attachShader(program, fs);
        gl.linkProgram(program);
        gl.useProgram(program);

        const vertices = new Float32Array([-1,-1, 1,-1, -1,1, 1,1]);
        const buffer = gl.createBuffer();
        gl.bindBuffer(gl.ARRAY_BUFFER, buffer);
        gl.bufferData(gl.ARRAY_BUFFER, vertices, gl.STATIC_DRAW);

        const posLoc = gl.getAttribLocation(program, 'position');
        gl.enableVertexAttribArray(posLoc);
        gl.vertexAttribPointer(posLoc, 2, gl.FLOAT, false, 0, 0);

        const iResolution = gl.getUniformLocation(program, 'iResolution');
        const iTime = gl.getUniformLocation(program, 'iTime');

        let animId: number;
        const startTime = performance.now();

        const render = () => {
            const w = canvas.clientWidth * window.devicePixelRatio;
            const h = canvas.clientHeight * window.devicePixelRatio;
            if (canvas.width !== w || canvas.height !== h) {
                canvas.width = w;
                canvas.height = h;
            }
            gl.viewport(0, 0, canvas.width, canvas.height);
            gl.uniform2f(iResolution, canvas.width, canvas.height);
            gl.uniform1f(iTime, (performance.now() - startTime) / 1000);
            gl.clearColor(0, 0, 0, 0);
            gl.clear(gl.COLOR_BUFFER_BIT);
            gl.drawArrays(gl.TRIANGLE_STRIP, 0, 4);
            animId = requestAnimationFrame(render);
        };
        render();

        cleanup(() => cancelAnimationFrame(animId));
    });

    // Skip canvas entirely when ?noshade is in URL (performance testing)
    if (noShade) return null;

    return (
        <canvas
            ref={canvasRef}
            style={{
                width: '100%',
                height: '100%',
                position: 'absolute',
                top: 0,
                left: 0,
                zIndex: 1,
                background: 'transparent',
            }}
        />
    );
});
