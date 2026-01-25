import { component$, useVisibleTask$, useSignal } from '@builder.io/qwik';

/**
 * CliffordTorusShader - 4D Stereographic Projection Torus
 * Based on ShaderToy WdB3Dw by IQ
 * 
 * A thick-walled Clifford torus intersected with a sphere,
 * rendered via inverse stereographic projection with
 * spectrum color palette.
 */
export const CliffordTorusShader = component$(() => {
    const canvasRef = useSignal<HTMLCanvasElement>();

    useVisibleTask$(({ cleanup }) => {
        const canvas = canvasRef.value;
        if (!canvas) return;

        // GLSL Shader Source - Clifford Torus (WdB3Dw)
        const SHADER = `
#define PI 3.14159265359

void pR(inout vec2 p, float a) {
    p = cos(a)*p + sin(a)*vec2(p.y, -p.x);
}

float smax(float a, float b, float r) {
    vec2 u = max(vec2(r + a, r + b), vec2(0));
    return min(-r, max(a, b)) + length(u);
}

// Spectrum colour palette - IQ
vec3 pal(in float t, in vec3 a, in vec3 b, in vec3 c, in vec3 d) {
    return a + b*cos(6.28318*(c*t+d));
}

vec3 spectrum(float n) {
    return pal(n, vec3(0.5,0.5,0.5), vec3(0.5,0.5,0.5), vec3(1.0,1.0,1.0), vec3(0.0,0.33,0.67));
}

vec4 inverseStereographic(vec3 p, out float k) {
    k = 2.0/(1.0+dot(p,p));
    return vec4(k*p, k-1.0);
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
    vec4 p4 = inverseStereographic(p, k);
    
    pR(p4.zy, time * -PI / 2.);
    pR(p4.xw, time * -PI / 2.);
    
    float d = fTorus(p4);
    d = abs(d);
    d -= .2;
    
    d = smax(d, length(p) - 1.8, .1);
    d = fixDistance(d, k);
    
    return d;
}

vec3 calcNormal(vec3 p) {
    vec2 e = vec2(.0001, 0);
    return normalize(vec3(
        map(p + e.xyy) - map(p - e.xyy),
        map(p + e.yxy) - map(p - e.yxy),
        map(p + e.yyx) - map(p - e.yyx)
    ));
}

void mainImage(out vec4 fragColor, in vec2 fragCoord) {
    vec2 uv = (fragCoord - .5 * iResolution.xy) / iResolution.y;
    
    time = iTime * .3;
    
    vec3 ro = vec3(0, 0, -4.5);
    vec3 rd = normalize(vec3(uv, 1.5));
    
    float t = 0.;
    float d;
    vec3 p;
    
    for (int i = 0; i < 80; i++) {
        p = ro + rd * t;
        d = map(p);
        if (abs(d) < .001 || t > 10.) break;
        t += d * .9;
    }
    
    vec3 col = vec3(0);
    float alpha = 0.;
    
    if (t < 10.) {
        vec3 n = calcNormal(p);
        float dif = max(dot(n, normalize(vec3(1, 1, -1))), 0.);
        float rim = pow(1. - max(dot(-rd, n), 0.), 3.);
        
        // Color based on position
        float k;
        vec4 p4 = inverseStereographic(p, k);
        float hue = atan(p4.x, p4.z) / PI * .5 + .5;
        hue += atan(p4.y, p4.w) / PI * .25;
        hue = fract(hue + time * .2);
        
        col = spectrum(hue);
        col *= dif * .8 + .2;
        col += rim * spectrum(hue + .3) * .5;
        col = pow(col, vec3(.4545));
        
        alpha = 1.;
    }
    
    fragColor = vec4(col, alpha);
}
`;

        // WebGL Setup
        let gl = canvas.getContext('webgl2', { alpha: true, premultipliedAlpha: false }) as WebGL2RenderingContext | null;
        let isWebGL2 = !!gl;
        if (!gl) {
            gl = canvas.getContext('webgl', { alpha: true, premultipliedAlpha: false }) as WebGLRenderingContext | null;
            isWebGL2 = false;
        }
        if (!gl) return;

        const vsSrc = isWebGL2
            ? `#version 300 es
               in vec2 position;
               void main() { gl_Position = vec4(position, 0., 1.); }`
            : `attribute vec2 position;
               void main() { gl_Position = vec4(position, 0., 1.); }`;

        const fsSrc = isWebGL2
            ? `#version 300 es
               precision highp float;
               uniform vec3 iResolution;
               uniform float iTime;
               out vec4 fragColor;
               ${SHADER}
               void main() { mainImage(fragColor, gl_FragCoord.xy); }`
            : `precision highp float;
               uniform vec3 iResolution;
               uniform float iTime;
               ${SHADER}
               void main() { mainImage(gl_FragColor, gl_FragCoord.xy); }`;

        function compileShader(src: string, type: number) {
            const s = gl!.createShader(type);
            if (!s) return null;
            gl!.shaderSource(s, src);
            gl!.compileShader(s);
            if (!gl!.getShaderParameter(s, gl!.COMPILE_STATUS)) {
                console.error('Shader compile error:', gl!.getShaderInfoLog(s));
                return null;
            }
            return s;
        }

        const vs = compileShader(vsSrc, gl.VERTEX_SHADER);
        const fs = compileShader(fsSrc, gl.FRAGMENT_SHADER);
        if (!vs || !fs) return;

        const program = gl.createProgram();
        if (!program) return;
        gl.attachShader(program, vs);
        gl.attachShader(program, fs);
        gl.linkProgram(program);

        if (!gl.getProgramParameter(program, gl.LINK_STATUS)) {
            console.error('Program link error:', gl.getProgramInfoLog(program));
            return;
        }

        const buf = gl.createBuffer();
        gl.bindBuffer(gl.ARRAY_BUFFER, buf);
        gl.bufferData(gl.ARRAY_BUFFER, new Float32Array([-1, -1, 1, -1, -1, 1, 1, 1]), gl.STATIC_DRAW);

        const posLoc = gl.getAttribLocation(program, 'position');
        const resLoc = gl.getUniformLocation(program, 'iResolution');
        const timeLoc = gl.getUniformLocation(program, 'iTime');

        gl.enable(gl.BLEND);
        gl.blendFunc(gl.SRC_ALPHA, gl.ONE_MINUS_SRC_ALPHA);

        let animId: number;
        const startTime = Date.now();

        function render() {
            if (!gl || !canvas) return;

            const w = canvas.clientWidth;
            const h = canvas.clientHeight;
            if (canvas.width !== w || canvas.height !== h) {
                canvas.width = w;
                canvas.height = h;
            }

            gl.viewport(0, 0, w, h);
            gl.clearColor(0, 0, 0, 0);
            gl.clear(gl.COLOR_BUFFER_BIT);

            gl.useProgram(program);
            gl.uniform3f(resLoc, w, h, 1);
            gl.uniform1f(timeLoc, (Date.now() - startTime) * 0.001);

            gl.bindBuffer(gl.ARRAY_BUFFER, buf);
            gl.enableVertexAttribArray(posLoc);
            gl.vertexAttribPointer(posLoc, 2, gl.FLOAT, false, 0, 0);
            gl.drawArrays(gl.TRIANGLE_STRIP, 0, 4);

            animId = requestAnimationFrame(render);
        }

        render();

        cleanup(() => {
            cancelAnimationFrame(animId);
            gl?.deleteProgram(program);
            gl?.deleteShader(vs);
            gl?.deleteShader(fs);
            gl?.deleteBuffer(buf);
        });
    });

    return (
        <canvas
            ref={canvasRef}
            style={{
                width: '100%',
                height: '100%',
                display: 'block',
                background: 'transparent'
            }}
        />
    );
});
