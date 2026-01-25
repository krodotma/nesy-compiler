/**
 * LoadingOrbShader - Fresh implementation from shader-test-clean.html
 * =====================================================================
 * Multi-buffer reaction-diffusion sphere with transparent background.
 * Source: https://www.shadertoy.com/view/MstXWS (emnh + Flexi + iq)
 * 
 * This is a FRESH implementation - no inherited broken code.
 */

import { component$, useSignal, useVisibleTask$ } from "@builder.io/qwik";

// Buffer A: Reaction-diffusion simulation
const BUFFER_A = `
vec3 hash33(in vec2 p){ 
    float n = sin(dot(p, vec2(41, 289)));    
    return fract(vec3(2097152, 262144, 32768)*n); 
}

vec4 tx(in vec2 p){ return texture(iChannel0, p); }

float blur(in vec2 p){
    vec3 e = vec3(1, 0, -1);
    vec2 px = 1./iResolution.xy;
    float res = 0.0;
    res += tx(p + e.xx*px ).x + tx(p + e.xz*px ).x + tx(p + e.zx*px ).x + tx(p + e.zz*px ).x;
    res += (tx(p + e.xy*px ).x + tx(p + e.yx*px ).x + tx(p + e.yz*px ).x + tx(p + e.zy*px ).x)*2.;
    res += tx(p + e.yy*px ).x*4.;
    return res/16.;     
}

void mainImage( out vec4 fragColor, in vec2 fragCoord ){
    vec2 uv = fragCoord/iResolution.xy;
    vec2 pw = 1./iResolution.xy;
    float avgReactDiff = blur(uv);
    vec3 noise = hash33(uv + vec2(53, 43)*iTime)*.6 + .2;
    vec3 e = vec3(1, 0, -1);
    vec2 pwr = pw*1.5; 
    vec2 lap = vec2(tx(uv + e.xy*pwr).y - tx(uv - e.xy*pwr).y, tx(uv + e.yx*pwr).y - tx(uv - e.yx*pwr).y);
    uv = uv + lap*pw*3.0; 
    float newReactDiff = tx(uv).x + (noise.z - 0.5)*0.0025 - 0.002; 
    newReactDiff += dot(tx(uv + (noise.xy-0.5)*pw).xy, vec2(1, -1))*0.145; 
    if(iFrame>9) fragColor.xy = clamp(vec2(newReactDiff, avgReactDiff/.98), 0., 1.);
    else fragColor = vec4(noise, 1.);
}
`;

// Image pass: Raymarched sphere with RD texture - MODIFIED FOR TRANSPARENCY
const IMAGE = `
float sdSphere( vec3 p, float s ) { return length(p)-s; }

mat4 rotationMatrix(vec3 axis, float angle) {
    axis = normalize(axis);
    float s = sin(angle);
    float c = cos(angle);
    float oc = 1.0 - c;
    return mat4(oc * axis.x * axis.x + c,           oc * axis.x * axis.y - axis.z * s,  oc * axis.z * axis.x + axis.y * s,  0.0,
                oc * axis.x * axis.y + axis.z * s,  oc * axis.y * axis.y + c,           oc * axis.y * axis.z - axis.x * s,  0.0,
                oc * axis.z * axis.x - axis.y * s,  oc * axis.y * axis.z + axis.x * s,  oc * axis.z * axis.z + c,           0.0,
                0.0,                                0.0,                                0.0,                                1.0);
}

vec3 hsv2rgb(vec3 c) {
    vec4 K = vec4(1.0, 2.0 / 3.0, 1.0 / 3.0, 3.0);
    vec3 p = abs(fract(c.xxx + K.xyz) * 6.0 - K.www);
    return c.z * mix(K.xxx, clamp(p - K.xxx, 0.0, 1.0), c.y);
}

float hash( float n ) { return fract(sin(n)*753.5453123); }

float noise( in vec3 x ) {
    vec3 p = floor(x);
    vec3 f = fract(x);
    f = f*f*(3.0-2.0*f);
    float n = p.x + p.y*157.0 + 113.0*p.z;
    return mix(mix(mix( hash(n+0.0), hash(n+1.0),f.x),
                   mix( hash(n+157.0), hash(n+158.0),f.x),f.y),
               mix(mix( hash(n+113.0), hash(n+114.0),f.x),
                   mix( hash(n+270.0), hash(n+271.0),f.x),f.y),f.z);
}

vec2 map(in vec3 pos) {    
    pos = pos - vec3(0.0, -1.5, 0.0);
    mat4 mrt = rotationMatrix(vec3(0.0, 1.0, 0.0), sin(iTime / 10.0));
    pos = (vec4(pos, 1.0) * mrt).xyz;
    vec3 p = normalize(pos);
    vec2 uv = vec2(0.0);
    uv.x = 0.5 + atan(p.z, p.x) / (2.*3.14159);
    uv.y = 0.5 - asin(p.y) / 3.14159;
    float y = texture(iChannel0, uv).y;
    float y2 = 0.1 * y;
    float ss = 5.0;
    float sd = sdSphere(pos / ss, 0.4 + y2) * ss;
    return vec2(sd, iTime / 10.0 + y);
}

vec2 castRay( in vec3 ro, in vec3 rd ) {
    float tmin = 1.0;
    float tmax = 20.0;
    float precis = 0.002;
    float t = tmin;
    float m = -1.0;
    for( int i=0; i<50; i++ ) {
        vec2 res = map( ro+rd*t );
        if( res.x<precis || t>tmax ) break;
        t += res.x;
        m = res.y;
    }
    if( t>tmax ) m=-1.0;
    return vec2( t, m );
}

float softshadow( in vec3 ro, in vec3 rd, in float mint, in float tmax ) {
    float res = 1.0;
    float t = mint;
    for( int i=0; i<16; i++ ) {
        float h = map( ro + rd*t ).x;
        res = min( res, 8.0*h/t );
        t += clamp( h, 0.02, 0.10 );
        if( h<0.001 || t>tmax ) break;
    }
    return clamp( res, 0.0, 1.0 );
}

vec3 calcNormal( in vec3 pos ) {
    vec3 eps = vec3( 0.001, 0.0, 0.0 );
    vec3 nor = vec3(
        map(pos+eps.xyy).x - map(pos-eps.xyy).x,
        map(pos+eps.yxy).x - map(pos-eps.yxy).x,
        map(pos+eps.yyx).x - map(pos-eps.yyx).x );
    return normalize(nor);
}

float calcAO( in vec3 pos, in vec3 nor ) {
    float occ = 0.0;
    float sca = 1.0;
    for( int i=0; i<5; i++ ) {
        float hr = 0.01 + 0.12*float(i)/4.0;
        vec3 aopos =  nor * hr + pos;
        float dd = map( aopos ).x;
        occ += -(dd-hr)*sca;
        sca *= 0.95;
    }
    return clamp( 1.0 - 3.0*occ, 0.0, 1.0 );    
}

// MODIFIED: Returns vec4 for transparency
vec4 render( in vec3 ro, in vec3 rd ) { 
    vec2 res = castRay(ro,rd);
    float t = res.x;
    float m = res.y;
    
    if( m < -0.5 ) {
        return vec4(0.0); // Transparent background
    }
    
    vec3 pos = ro + t*rd;
    vec3 nor = calcNormal( pos );
    vec3 ref = reflect( rd, nor );
    vec3 col = hsv2rgb(vec3(m, 1.0, 1.0));
    
    float occ = calcAO( pos, nor );
    vec3 lig = normalize( vec3(-0.6, 0.7, -0.5) );
    float amb = clamp( 0.5+0.5*nor.y, 0.0, 1.0 );
    float dif = clamp( dot( nor, lig ), 0.0, 1.0 );
    float bac = clamp( dot( nor, normalize(vec3(-lig.x,0.0,-lig.z))), 0.0, 1.0 )*clamp( 1.0-pos.y,0.0,1.0);
    float dom = smoothstep( -0.1, 0.1, ref.y );
    float fre = pow( clamp(1.0+dot(nor,rd),0.0,1.0), 2.0 );
    float spe = pow(clamp( dot( ref, lig ), 0.0, 1.0 ),16.0);
    
    dif *= softshadow( pos, lig, 0.02, 2.5 );
    dom *= softshadow( pos, ref, 0.02, 2.5 );

    vec3 lin = vec3(0.0);
    lin += 1.20*dif*vec3(1.00,0.85,0.55);
    lin += 1.20*spe*vec3(1.00,0.85,0.55)*dif;
    lin += 0.20*amb*vec3(0.50,0.70,1.00)*occ;
    lin += 0.30*dom*vec3(0.50,0.70,1.00)*occ;
    lin += 0.30*bac*vec3(0.25,0.25,0.25)*occ;
    lin += 0.40*fre*vec3(1.00,1.00,1.00)*occ;
    col = col*lin;
    col = mix( col, vec3(0.8,0.9,1.0), 1.0-exp( -0.002*t*t ) );
    
    return vec4(clamp(col, 0.0, 1.0), 1.0);
}

mat3 setCamera( in vec3 ro, in vec3 ta, float cr ) {
    vec3 cw = normalize(ta-ro);
    vec3 cp = vec3(sin(cr), cos(cr),0.0);
    vec3 cu = normalize( cross(cw,cp) );
    vec3 cv = normalize( cross(cu,cw) );
    return mat3( cu, cv, cw );
}

void mainImage( out vec4 fragColor, in vec2 fragCoord ) {
    vec2 q = fragCoord.xy/iResolution.xy;
    vec2 p = -1.0+2.0*q;
    p.x *= iResolution.x/iResolution.y;
    vec3 ro = vec3( 3.5, 1.0, 3.5 );
    vec3 ta = vec3( 0.0, -1.5, 0.0 ); // Centered on sphere
    mat3 ca = setCamera( ro, ta, 0.0 );
    vec3 rd = ca * normalize( vec3(p.xy,2.0) );
    vec4 col = render( ro, rd );
    col.rgb = pow( col.rgb, vec3(0.4545) );
    fragColor = col;
}
`;

export const LoadingOrbShader = component$(() => {
    const canvasRef = useSignal<HTMLCanvasElement>();
    const ready = useSignal(false);

    useVisibleTask$(({ cleanup }) => {
        const canvas = canvasRef.value;
        if (!canvas) return;

        // WebGL2 with fallback
        let gl: WebGL2RenderingContext | WebGLRenderingContext | null = canvas.getContext('webgl2', { alpha: true });
        let isWebGL2 = !!gl;
        if (!gl) {
            gl = canvas.getContext('webgl', { alpha: true });
            isWebGL2 = false;
        }
        if (!gl) return;

        // Extensions for float textures
        if (isWebGL2) {
            gl.getExtension('EXT_color_buffer_float');
        } else {
            gl.getExtension('OES_texture_float');
        }
        gl.getExtension('OES_texture_float_linear');

        // Vertex shader
        const vsSrc = isWebGL2 
            ? '#version 300 es\nin vec2 position; void main(){gl_Position=vec4(position,0.,1.);}'
            : 'attribute vec2 position; void main(){gl_Position=vec4(position,0.,1.);}';

        const vs = gl.createShader(gl.VERTEX_SHADER)!;
        gl.shaderSource(vs, vsSrc);
        gl.compileShader(vs);

        // Fragment shader compiler
        function createFragShader(src: string) {
            const fullSrc = isWebGL2 ? `#version 300 es
precision highp float;
precision highp int;
uniform vec3 iResolution;
uniform float iTime;
uniform int iFrame;
uniform vec4 iMouse;
uniform sampler2D iChannel0;
out vec4 fragColor;
#define gl_FragColor fragColor
#define texture2D texture
${src}
void main() { 
    vec4 col = vec4(0.0);
    mainImage(col, gl_FragCoord.xy);
    fragColor = col;
}` : `precision highp float;
precision highp int;
uniform vec3 iResolution;
uniform float iTime;
uniform int iFrame;
uniform vec4 iMouse;
uniform sampler2D iChannel0;
${src}
void main() { mainImage(gl_FragColor, gl_FragCoord.xy); }`;
            
            const fs = gl!.createShader(gl!.FRAGMENT_SHADER)!;
            gl!.shaderSource(fs, fullSrc);
            gl!.compileShader(fs);
            if (!gl!.getShaderParameter(fs, gl!.COMPILE_STATUS)) {
                console.error('Shader compile error:', gl!.getShaderInfoLog(fs));
                return null;
            }
            return fs;
        }

        function createProgram(fragSrc: string) {
            const fs = createFragShader(fragSrc);
            if (!fs) return null;
            const prog = gl!.createProgram()!;
            gl!.attachShader(prog, vs);
            gl!.attachShader(prog, fs);
            gl!.linkProgram(prog);
            return prog;
        }

        const bufferAProg = createProgram(BUFFER_A);
        const imageProg = createProgram(IMAGE);
        if (!bufferAProg || !imageProg) return;

        // Fullscreen quad
        const buf = gl.createBuffer();
        gl.bindBuffer(gl.ARRAY_BUFFER, buf);
        gl.bufferData(gl.ARRAY_BUFFER, new Float32Array([-1,-1, 1,-1, -1,1, 1,1]), gl.STATIC_DRAW);

        // Framebuffers for ping-pong
        function createFBO(w: number, h: number) {
            const tex = gl!.createTexture()!;
            gl!.bindTexture(gl!.TEXTURE_2D, tex);
            const type = gl!.FLOAT;
            const format = isWebGL2 ? (gl as WebGL2RenderingContext).RGBA32F : gl!.RGBA;
            gl!.texImage2D(gl!.TEXTURE_2D, 0, format, w, h, 0, gl!.RGBA, type, null);
            gl!.texParameteri(gl!.TEXTURE_2D, gl!.TEXTURE_MIN_FILTER, gl!.LINEAR);
            gl!.texParameteri(gl!.TEXTURE_2D, gl!.TEXTURE_MAG_FILTER, gl!.LINEAR);
            const fb = gl!.createFramebuffer()!;
            gl!.bindFramebuffer(gl!.FRAMEBUFFER, fb);
            gl!.framebufferTexture2D(gl!.FRAMEBUFFER, gl!.COLOR_ATTACHMENT0, gl!.TEXTURE_2D, tex, 0);
            return { fb, tex };
        }

        let fbo0: { fb: WebGLFramebuffer; tex: WebGLTexture } | null = null;
        let fbo1: { fb: WebGLFramebuffer; tex: WebGLTexture } | null = null;
        let width = 0, height = 0;
        let frame = 0;
        let animId: number;

        function render(time: number) {
            time *= 0.001;
            
            if (canvas.width !== canvas.clientWidth || canvas.height !== canvas.clientHeight) {
                width = canvas.width = canvas.clientWidth;
                height = canvas.height = canvas.clientHeight;
                fbo0 = createFBO(width, height);
                fbo1 = createFBO(width, height);
                frame = 0;
            }

            if (!fbo0 || !fbo1) {
                animId = requestAnimationFrame(render);
                return;
            }

            const read = frame % 2 === 0 ? fbo0 : fbo1;
            const write = frame % 2 === 0 ? fbo1 : fbo0;

            // Buffer A pass (RD simulation)
            gl!.bindFramebuffer(gl!.FRAMEBUFFER, write.fb);
            gl!.viewport(0, 0, width, height);
            gl!.useProgram(bufferAProg);
            gl!.activeTexture(gl!.TEXTURE0);
            gl!.bindTexture(gl!.TEXTURE_2D, read.tex);
            gl!.uniform1i(gl!.getUniformLocation(bufferAProg!, 'iChannel0'), 0);
            gl!.uniform3f(gl!.getUniformLocation(bufferAProg!, 'iResolution'), width, height, 1);
            gl!.uniform1f(gl!.getUniformLocation(bufferAProg!, 'iTime'), time);
            gl!.uniform1i(gl!.getUniformLocation(bufferAProg!, 'iFrame'), frame);
            
            gl!.bindBuffer(gl!.ARRAY_BUFFER, buf);
            const locA = gl!.getAttribLocation(bufferAProg!, 'position');
            gl!.enableVertexAttribArray(locA);
            gl!.vertexAttribPointer(locA, 2, gl!.FLOAT, false, 0, 0);
            gl!.drawArrays(gl!.TRIANGLE_STRIP, 0, 4);

            // Image pass (raymarching)
            gl!.bindFramebuffer(gl!.FRAMEBUFFER, null);
            gl!.viewport(0, 0, width, height);
            gl!.clearColor(0, 0, 0, 0);
            gl!.clear(gl!.COLOR_BUFFER_BIT);
            gl!.useProgram(imageProg);
            gl!.activeTexture(gl!.TEXTURE0);
            gl!.bindTexture(gl!.TEXTURE_2D, write.tex);
            gl!.uniform1i(gl!.getUniformLocation(imageProg!, 'iChannel0'), 0);
            gl!.uniform3f(gl!.getUniformLocation(imageProg!, 'iResolution'), width, height, 1);
            gl!.uniform1f(gl!.getUniformLocation(imageProg!, 'iTime'), time);
            
            const locB = gl!.getAttribLocation(imageProg!, 'position');
            gl!.enableVertexAttribArray(locB);
            gl!.vertexAttribPointer(locB, 2, gl!.FLOAT, false, 0, 0);
            gl!.drawArrays(gl!.TRIANGLE_STRIP, 0, 4);

            frame++;
            if (frame === 10) ready.value = true;
            animId = requestAnimationFrame(render);
        }

        animId = requestAnimationFrame(render);
        cleanup(() => cancelAnimationFrame(animId));
    });

    return (
        <canvas
            ref={canvasRef}
            style={{
                width: '100%',
                height: '100%',
                display: 'block',
                borderRadius: '50%',
            }}
        />
    );
});
