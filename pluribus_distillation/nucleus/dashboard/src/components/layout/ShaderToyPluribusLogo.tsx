/**
 * ShaderToyPluribusLogo.tsx
 *
 * RESTORED: Reaction-Diffusion Shader (Flexi/Eivind Magnus Hvidevold)
 * The "Proper" Pluribus Logo as of ~Jan 1, 2026.
 */

import { component$, useSignal, useVisibleTask$ } from "@builder.io/qwik";

const HEADER = `
precision highp float;
uniform vec2 iResolution;
uniform float iTime;
uniform int iFrame;
uniform vec4 iMouse;
uniform sampler2D iChannel0;

#define texture texture2D
#define outFrag gl_FragColor
`;

const VERT_SRC = `
attribute vec2 position;
void main() {
  gl_Position = vec4(position, 0.0, 1.0);
}
`;

const BUFFER_A = `
// Reaction-diffusion pass.
// Reaction-Diffusion by the Gray-Scott Model

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
void main() {
  mainImage(gl_FragColor, gl_FragCoord.xy);
}`;

const IMAGE = `
// Created by Eivind Magnus Hvidevold emnh/2016.
// Reaction-diffusion by Flexi. Raymarching by inigo quilez.

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
    return mix(mix(mix( hash(n+  0.0), hash(n+  1.0),f.x),
                   mix( hash(n+157.0), hash(n+158.0),f.x),f.y),
               mix(mix( hash(n+113.0), hash(n+114.0),f.x),
                   mix( hash(n+270.0), hash(n+271.0),f.x),f.y),f.z);
}

vec2 map(in vec3 pos) {    
    pos = pos - vec3(0.0, -1.5, 0.0);
    vec2 mo = iMouse.xy/iResolution.xy;
    float ms = 3.14 * 2.0;
    mat4 mrx = rotationMatrix(vec3(1.0, 0.0, 0.0), mo.y * ms);
    mat4 mry = rotationMatrix(vec3(0.0, 1.0, 0.0), mo.x * ms);
    mat4 mrt = rotationMatrix(vec3(0.0, 1.0, 0.0), sin(iTime / 10.0));
    pos = (vec4(pos, 1.0) * mrx * mry * mrt).xyz;
    vec3 p = normalize(pos);
    vec2 uv = vec2(0.0);
    uv.x = 0.5 + atan(p.z, p.x) / (2.*3.14159);
    uv.y = 0.5 - asin(p.y) / 3.14159;
    float y = texture(iChannel0, uv).y;
    float y2 = 0.1 * y;
    float ss = 5.0;
    vec3 sphereO = pos;
    float sd = sdSphere(sphereO / ss, 0.4 + y2) * ss;
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

vec3 render( in vec3 ro, in vec3 rd ) { 
    vec3 col = vec3(0.7, 0.9, 1.0) +rd.y*0.8;
    vec2 res = castRay(ro,rd);
    float t = res.x;
    float m = res.y;
    if( m>-0.5 ) {
        vec3 pos = ro + t*rd;
        vec3 nor = calcNormal( pos );
        vec3 ref = reflect( rd, nor );
        col = hsv2rgb(vec3(m, 1.0, 1.0));
        float occ = calcAO( pos, nor );
        vec3  lig = normalize( vec3(-0.6, 0.7, -0.5) );
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
    }
    return vec3( clamp(col,0.0,1.0) );
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
    vec2 mo = iMouse.xy/iResolution.xy;
    float time = 15.0 + iTime;
    vec3 ro = vec3( 3.5, 1.0, 3.5 );
    vec3 ta = vec3( -0.5, -2.0, -1.0 );
    mat3 ca = setCamera( ro, ta, 0.0 );
    vec3 rd = ca * normalize( vec3(p.xy,2.0) );
    vec3 col = render( ro, rd );
    col = pow( col, vec3(0.4545) );
    fragColor=vec4( col, 1.0 );
} 
void main() {
  mainImage(gl_FragColor, gl_FragCoord.xy);
}`;

export const ShaderToyPluribusLogo = component$(() => {
  const canvasRef = useSignal<HTMLCanvasElement>();

  useVisibleTask$(({ cleanup }) => {
    const canvas = canvasRef.value;
    if (!canvas) return;
    
    const gl = canvas.getContext("webgl", { alpha: true });
    if (!gl) return;
    
    const extFloat = gl.getExtension('OES_texture_float');
    const extLinear = gl.getExtension('OES_texture_float_linear');
    const extHalf = gl.getExtension('OES_texture_half_float');
    const extHalfLinear = gl.getExtension('OES_texture_half_float_linear');

    let texType = gl.UNSIGNED_BYTE;
    if (extFloat && extLinear) {
        texType = gl.FLOAT;
    } else if (extHalf && extHalfLinear) {
        texType = extHalf.HALF_FLOAT_OES;
    } else if (extFloat) {
         texType = gl.FLOAT;
    }

    const compile = (src: string, type: number) => {
        const s = gl.createShader(type)!;
        gl.shaderSource(s, src);
        gl.compileShader(s);
        if (!gl.getShaderParameter(s, gl.COMPILE_STATUS)) {
            console.error("Shader Err:", gl.getShaderInfoLog(s));
            return null;
        }
        return s;
    };
    
    const program = (frag: string) => {
        const p = gl.createProgram()!;
        const vs = compile(VERT_SRC, gl.VERTEX_SHADER);
        const fs = compile(HEADER + frag, gl.FRAGMENT_SHADER);
        if(!vs || !fs) return null;
        gl.attachShader(p, vs);
        gl.attachShader(p, fs);
        gl.linkProgram(p);
        if (!gl.getProgramParameter(p, gl.LINK_STATUS)) {
            console.error("Link Err:", gl.getProgramInfoLog(p));
            return null;
        }
        return p;
    };

    const progBufA = program(BUFFER_A);
    const progImage = program(IMAGE);
    if (!progBufA || !progImage) return;

    const buf = gl.createBuffer();
    gl.bindBuffer(gl.ARRAY_BUFFER, buf);
    gl.bufferData(gl.ARRAY_BUFFER, new Float32Array([-1,-1, 1,-1, -1,1, 1,1]), gl.STATIC_DRAW);

    const getU = (p: WebGLProgram) => ({
        pos: gl.getAttribLocation(p, "position"),
        res: gl.getUniformLocation(p, "iResolution"),
        time: gl.getUniformLocation(p, "iTime"),
        frame: gl.getUniformLocation(p, "iFrame"),
        ch0: gl.getUniformLocation(p, "iChannel0"),
        mouse: gl.getUniformLocation(p, "iMouse")
    });
    
    const uBufA = getU(progBufA);
    const uImage = getU(progImage);

    const fbos = { ping: null as any, pong: null as any };
    let frame = 0;
    let dw = 0, dh = 0;

    const createFBO = (w:number, h:number) => {
        const tex = gl.createTexture();
        gl.bindTexture(gl.TEXTURE_2D, tex);
        gl.texImage2D(gl.TEXTURE_2D, 0, gl.RGBA, w, h, 0, gl.RGBA, texType, null);
        
        const filter = (texType === gl.FLOAT && !extLinear) ? gl.NEAREST : gl.LINEAR;
        gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, filter);
        gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, filter);
        gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_S, gl.REPEAT);
        gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_T, gl.REPEAT);
        
        const fb = gl.createFramebuffer();
        gl.bindFramebuffer(gl.FRAMEBUFFER, fb);
        gl.framebufferTexture2D(gl.FRAMEBUFFER, gl.COLOR_ATTACHMENT0, gl.TEXTURE_2D, tex, 0);
        return { fb, tex };
    };

    let raf = 0;
    const render = (t: number) => {
        const pixelRatio = window.devicePixelRatio || 1;
        const rect = canvas.getBoundingClientRect();
        const w = Math.floor(rect.width * pixelRatio);
        const h = Math.floor(rect.height * pixelRatio);
        
        if (w !== dw || h !== dh) {
            dw = w; dh = h;
            canvas.width = w; canvas.height = h;
            fbos.ping = createFBO(w, h);
            fbos.pong = createFBO(w, h);
            frame = 0;
        }
        if (!fbos.ping) return;

        const time = t * 0.001;
        const read = frame % 2 === 0 ? fbos.ping : fbos.pong;
        const write = frame % 2 === 0 ? fbos.pong : fbos.ping;
        
        // Pass 1
        gl.bindFramebuffer(gl.FRAMEBUFFER, write.fb);
        gl.viewport(0, 0, dw, dh);
        gl.useProgram(progBufA);
        gl.bindBuffer(gl.ARRAY_BUFFER, buf);
        gl.enableVertexAttribArray(uBufA.pos);
        gl.vertexAttribPointer(uBufA.pos, 2, gl.FLOAT, false, 0, 0);
        
        gl.activeTexture(gl.TEXTURE0);
        gl.bindTexture(gl.TEXTURE_2D, read.tex);
        gl.uniform1i(uBufA.ch0, 0);
        gl.uniform2f(uBufA.res, dw, dh);
        gl.uniform1f(uBufA.time, time);
        gl.uniform1i(uBufA.frame, frame);
        gl.uniform4f(uBufA.mouse, 0, 0, 0, 0);
        gl.drawArrays(gl.TRIANGLE_STRIP, 0, 4);
        
        // Pass 2
        gl.bindFramebuffer(gl.FRAMEBUFFER, null);
        gl.viewport(0, 0, dw, dh);
        gl.useProgram(progImage);
        gl.bindBuffer(gl.ARRAY_BUFFER, buf);
        gl.enableVertexAttribArray(uImage.pos);
        gl.vertexAttribPointer(uImage.pos, 2, gl.FLOAT, false, 0, 0);
        
        gl.activeTexture(gl.TEXTURE0);
        gl.bindTexture(gl.TEXTURE_2D, write.tex);
        gl.uniform1i(uImage.ch0, 0);
        gl.uniform2f(uImage.res, dw, dh);
        gl.uniform1f(uImage.time, time);
        gl.uniform4f(uImage.mouse, 0, 0, 0, 0);
        gl.drawArrays(gl.TRIANGLE_STRIP, 0, 4);
        
        frame++;
        raf = requestAnimationFrame(render);
    };
    raf = requestAnimationFrame(render);
    cleanup(() => cancelAnimationFrame(raf));
  });

  return (
    <div class="header-logo-orb" aria-hidden="true" style={{ width: "64px", height: "64px", flexShrink: 0, borderRadius: "50%", overflow: "hidden" }}>
       <canvas ref={canvasRef} style={{width:"100%", height:"100%"}} />
    </div>
  );
});