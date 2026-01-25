import { component$, useSignal, useVisibleTask$, type QRL } from "@builder.io/qwik";

// ============================================================================
// SHADERTOY REACTION-DIFFUSION SPHERE
// EXACT VERBATIM SOURCE - NO MODIFICATIONS
// ============================================================================

const HEADER = `
precision highp float;
uniform vec3 iResolution;
uniform float iTime;
uniform int iFrame;
uniform vec4 iMouse;
uniform sampler2D iChannel0;
#define texture texture2D
`;

const VERT = `
attribute vec2 position;
void main() {
  gl_Position = vec4(position, 0.0, 1.0);
}
`;

// BUFFER A - EXACT VERBATIM FROM USER
const BUFFER_A = `
// Reaction-diffusion pass.
//
// Here's a really short, non technical explanation:
//
// To begin, sprinkle the buffer with some initial noise on the first few frames (Sometimes, the 
// first frame gets skipped, so you do a few more).
//
// During the buffer loop pass, determine the reaction diffusion value using a combination of the 
// value stored in the buffer's "X" channel, and a the blurred value - stored in the "Y" channel 
// (You can see how that's done in the code below). Blur the value from the "X" channel (the old 
// reaction diffusion value) and store it in "Y", then store the new (reaction diffusion) value 
// in "X." Display either the "X" value  or "Y" buffer value in the "Image" tab, add some window 
// dressing, then repeat the process. Simple... Slightly confusing when I try to explain it, but 
// trust me, it's simple. :)
//
// Anyway, for a more sophisticated explanation, here are a couple of references below:
//
// Reaction-Diffusion by the Gray-Scott Model - http://www.karlsims.com/rd.html
// Reaction-Diffusion Tutorial - http://www.karlsims.com/rd.html

// Cheap vec3 to vec3 hash. Works well enough, but there are other ways.
vec3 hash33(in vec2 p){ 
    float n = sin(dot(p, vec2(41, 289)));    
    return fract(vec3(2097152, 262144, 32768)*n); 
}

// Serves no other purpose than to save having to write this out all the time. I could write a 
// "define," but I'm pretty sure this'll be inlined.
vec4 tx(in vec2 p){ return texture(iChannel0, p); }

// Weighted blur function. Pretty standard.
float blur(in vec2 p){
    
    // Used to move to adjoining pixels. - uv + vec2(-1, 1)*px, uv + vec2(1, 0)*px, etc.
    vec3 e = vec3(1, 0, -1);
    vec2 px = 1./iResolution.xy;
    
    // Weighted 3x3 blur, or a cheap and nasty Gaussian blur approximation.
	float res = 0.0;
    // Four corners. Those receive the least weight.
	res += tx(p + e.xx*px ).x + tx(p + e.xz*px ).x + tx(p + e.zx*px ).x + tx(p + e.zz*px ).x;
    // Four sides, which are given a little more weight.
    res += (tx(p + e.xy*px ).x + tx(p + e.yx*px ).x + tx(p + e.yz*px ).x + tx(p + e.zy*px ).x)*2.;
	// The center pixel, which we're giving the most weight to, as you'd expect.
	res += tx(p + e.yy*px ).x*4.;
    // Normalizing.
    return res/16.;     
    
}

// The reaction diffusion loop.
// 
void mainImage( out vec4 fragColor, in vec2 fragCoord ){

    
	vec2 uv = fragCoord/iResolution.xy; // Screen coordinates. Range: [0, 1]
    vec2 pw = 1./iResolution.xy; // Relative pixel width. Used for neighboring pixels, etc.
    
    
    // The blurred pixel. This is the result that's used in the "Image" tab. It's also reused
    // in the next frame in the reaction diffusion process (see below).
	float avgReactDiff = blur(uv);

    
	// The noise value. Because the result is blurred, we can get away with plain old static noise.
    // However, smooth noise, and various kinds of noise textures will work, too.
    vec3 noise = hash33(uv + vec2(53, 43)*iTime)*.6 + .2;

    // Used to move to adjoining pixels. - uv + vec2(-1, 1)*px, uv + vec2(1, 0)*px, etc.
    vec3 e = vec3(1, 0, -1);
    
    // Gradient epsilon value. The "1.5" figure was trial and error, but was based on the 3x3 blur radius.
    vec2 pwr = pw*1.5; 
    
    // Use the blurred pixels (stored in the Y-Channel) to obtain the gradient. I haven't put too much 
    // thought into this, but the gradient of a pixel on a blurred pixel grid (average neighbors), would 
    // be analogous to a Laplacian operator on a 2D discreet grid. Laplacians tend to be used to describe 
    // chemical flow, so... Sounds good, anyway. :)
    //
    // Seriously, though, take a look at the formula for the reacion-diffusion process, and you'll see
    // that the following few lines are simply putting it into effect.
    
    // Gradient of the blurred pixels from the previous frame.
	vec2 lap = vec2(tx(uv + e.xy*pwr).y - tx(uv - e.xy*pwr).y, tx(uv + e.yx*pwr).y - tx(uv - e.yx*pwr).y);//
    
    // Add some diffusive expansion, scaled down to the order of a pixel width.
    uv = uv + lap*pw*3.0; 
    
    // Stochastic decay. Ie: A differention equation, influenced by noise.
    // You need the decay, otherwise things would keep increasing, which in this case means a white screen.
    float newReactDiff = tx(uv).x + (noise.z - 0.5)*0.0025 - 0.002; 
    
    // Reaction-diffusion.
	newReactDiff += dot(tx(uv + (noise.xy-0.5)*pw).xy, vec2(1, -1))*0.145; 

    
    // Storing the reaction diffusion value in the X channel, and avgReactDiff (the blurred pixel value) 
    // in the Y channel. However, for the first few frames, we add some noise. Normally, one frame would 
    // be enough, but for some weird reason, it doesn't always get stored on the very first frame.
    if(iFrame>9) fragColor.xy = clamp(vec2(newReactDiff, avgReactDiff/.98), 0., 1.);
    else fragColor = vec4(noise, 1.);
    
}
void main() { mainImage(gl_FragColor, gl_FragCoord.xy); }
`;

// IMAGE - EXACT VERBATIM FROM USER
const IMAGE = `
// Created by Eivind Magnus Hvidevold emnh/2016.
// Reaction-diffusion by Flexi.
// Raymarching by inigo quilez - iq/2013.
// License Creative Commons Attribution-NonCommercial-ShareAlike 3.0 Unported License.

float sdSphere( vec3 p, float s )
{
    return length(p)-s;
}

//----------------------------------------------------------------------

mat4 rotationMatrix(vec3 axis, float angle)
{
    axis = normalize(axis);
    float s = sin(angle);
    float c = cos(angle);
    float oc = 1.0 - c;
    
    return mat4(oc * axis.x * axis.x + c,           oc * axis.x * axis.y - axis.z * s,  oc * axis.z * axis.x + axis.y * s,  0.0,
                oc * axis.x * axis.y + axis.z * s,  oc * axis.y * axis.y + c,           oc * axis.y * axis.z - axis.x * s,  0.0,
                oc * axis.z * axis.x - axis.y * s,  oc * axis.y * axis.z + axis.x * s,  oc * axis.z * axis.z + c,           0.0,
                0.0,                                0.0,                                0.0,                                1.0);
}

vec3 hsv2rgb(vec3 c)
{
    vec4 K = vec4(1.0, 2.0 / 3.0, 1.0 / 3.0, 3.0);
    vec3 p = abs(fract(c.xxx + K.xyz) * 6.0 - K.www);
    return c.z * mix(K.xxx, clamp(p - K.xxx, 0.0, 1.0), c.y);
}

float hash( float n ) { return fract(sin(n)*753.5453123); }

float noise( in vec3 x )
{
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
    vec3 o = pos;
    
    pos = pos - vec3(0.0, -1.5, 0.0);
    vec2 mo = iMouse.xy/iResolution.xy;
    float ms = 3.14 * 2.0;
    mat4 mrx = rotationMatrix(vec3(1.0, 0.0, 0.0), mo.y * ms);
    mat4 mry = rotationMatrix(vec3(0.0, 1.0, 0.0), mo.x * ms);
    mat4 mrt = rotationMatrix(vec3(0.0, 1.0, 0.0), sin(iTime / 10.0));
    
    pos = (vec4(pos, 1.0) * mrx * mry * mrt).xyz;
    
    // uv mapping
    vec3 p = normalize(pos);
    vec2 uv = vec2(0.0);
    uv.x = 0.5 + atan(p.z, p.x) / (2.*3.14159);
    uv.y = 0.5 - asin(p.y) / 3.14159;
    
    float y = texture(iChannel0, uv).y;
    float y2 = 0.1 * y;
    
    float ss = 5.0;
    vec3 sphereO = pos; // - vec3(0.0, 0.25, 1.0);
    
    float sd = 0.0;
	sd = sdSphere(sphereO / ss, 0.4 + y2) * ss;
    
    return vec2(sd, iTime / 10.0 + y); //sd + iTime / 10.0);
}

vec2 castRay( in vec3 ro, in vec3 rd )
{
    float tmin = 1.0;
    float tmax = 20.0;
    
#if 0
    float tp1 = (0.0-ro.y)/rd.y; if( tp1>0.0 ) tmax = min( tmax, tp1 );
    float tp2 = (1.6-ro.y)/rd.y; if( tp2>0.0 ) { if( ro.y>1.6 ) tmin = max( tmin, tp2 );
                                                 else           tmax = min( tmax, tp2 ); }
#endif
    
	float precis = 0.002;
    float t = tmin;
    float m = -1.0;
    for( int i=0; i<50; i++ )
    {
	    vec2 res = map( ro+rd*t );
        if( res.x<precis || t>tmax ) break;
        t += res.x;
	    m = res.y;
    }

    if( t>tmax ) m=-1.0;
    return vec2( t, m );
}


float softshadow( in vec3 ro, in vec3 rd, in float mint, in float tmax )
{
	float res = 1.0;
    float t = mint;
    for( int i=0; i<16; i++ )
    {
		float h = map( ro + rd*t ).x;
        res = min( res, 8.0*h/t );
        t += clamp( h, 0.02, 0.10 );
        if( h<0.001 || t>tmax ) break;
    }
    return clamp( res, 0.0, 1.0 );

}

vec3 calcNormal( in vec3 pos )
{
	vec3 eps = vec3( 0.001, 0.0, 0.0 );
	vec3 nor = vec3(
	    map(pos+eps.xyy).x - map(pos-eps.xyy).x,
	    map(pos+eps.yxy).x - map(pos-eps.yxy).x,
	    map(pos+eps.yyx).x - map(pos-eps.yyx).x );
	return normalize(nor);
}

float calcAO( in vec3 pos, in vec3 nor )
{
	float occ = 0.0;
    float sca = 1.0;
    for( int i=0; i<5; i++ )
    {
        float hr = 0.01 + 0.12*float(i)/4.0;
        vec3 aopos =  nor * hr + pos;
        float dd = map( aopos ).x;
        occ += -(dd-hr)*sca;
        sca *= 0.95;
    }
    return clamp( 1.0 - 3.0*occ, 0.0, 1.0 );    
}

vec3 render( in vec3 ro, in vec3 rd )
{ 
    vec3 col = vec3(0.7, 0.9, 1.0) +rd.y*0.8;
    vec2 res = castRay(ro,rd);
    float t = res.x;
	float m = res.y;
    if( m>-0.5 )
    {
        vec3 pos = ro + t*rd;
        vec3 nor = calcNormal( pos );
        vec3 ref = reflect( rd, nor );
        
        // material        
		/* col = 0.45 + 0.3*sin( vec3(0.05,0.08,0.10)*(m-1.0) );
        
		
        if( m<1.5 )
        {
            
            float f = mod( floor(5.0*pos.z) + floor(5.0*pos.x), 2.0);
            col = 0.4 + 0.1*f*vec3(1.0);
        }*/
		col = hsv2rgb(vec3(m, 1.0, 1.0));

        // lighting        
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

mat3 setCamera( in vec3 ro, in vec3 ta, float cr )
{
	vec3 cw = normalize(ta-ro);
	vec3 cp = vec3(sin(cr), cos(cr),0.0);
	vec3 cu = normalize( cross(cw,cp) );
	vec3 cv = normalize( cross(cu,cw) );
    return mat3( cu, cv, cw );
}

void mainImage( out vec4 fragColor, in vec2 fragCoord )
{
	vec2 q = fragCoord.xy/iResolution.xy;
    vec2 p = -1.0+2.0*q;
	p.x *= iResolution.x/iResolution.y;
    vec2 mo = iMouse.xy/iResolution.xy;
		 
	float time = 15.0 + iTime;

	// camera	
	// vec3 ro = vec3( -0.5+3.5*cos(0.1*time + 6.0*mo.x), 1.0 + 2.0*mo.y, 0.5 + 3.5*sin(0.1*time + 6.0*mo.x) );
    //vec3 ro = vec3( -0.5+3.5*cos(6.0*mo.x), 1.0 + 2.0*mo.y, 0.5 + 3.5*sin(6.0*mo.x) );
    vec3 ro = vec3( 3.5, 1.0, 3.5 );
	vec3 ta = vec3( -0.5, -2.0, -1.0 );
	
	// camera-to-world transformation
    mat3 ca = setCamera( ro, ta, 0.0 );
    
    // ray direction
	vec3 rd = ca * normalize( vec3(p.xy,2.0) );

    // render	
    vec3 col = render( ro, rd );

	col = pow( col, vec3(0.4545) );

    fragColor=vec4( col, 1.0 );
}
void main() { mainImage(gl_FragColor, gl_FragCoord.xy); }
`;

interface Props {
    onReady$?: QRL<() => void>;
}

export const LoadingOrbShader = component$<Props>(({ onReady$ }) => {
    const canvasRef = useSignal<HTMLCanvasElement>();

    useVisibleTask$(({ cleanup }) => {
        const canvas = canvasRef.value;
        if (!canvas) return;

        // Try WebGL2 first for better float texture support
        let gl: WebGLRenderingContext | WebGL2RenderingContext | null = canvas.getContext("webgl2");
        let isWebGL2 = true;

        if (!gl) {
            gl = canvas.getContext("webgl");
            isWebGL2 = false;
        }

        if (!gl) {
            console.error("WebGL not supported");
            return;
        }

        console.log("Using WebGL" + (isWebGL2 ? "2" : "1"));

        // Extensions for float textures (WebGL1 only needs these)
        let extFloat: OES_texture_float | null = null;
        let extFloatLinear: OES_texture_float_linear | null = null;

        if (!isWebGL2) {
            extFloat = gl.getExtension('OES_texture_float');
            extFloatLinear = gl.getExtension('OES_texture_float_linear');
            console.log("OES_texture_float:", !!extFloat, "OES_texture_float_linear:", !!extFloatLinear);
        }

        // Determine texture type - prefer FLOAT for precision
        let texType: number;
        let texInternalFormat: number;

        if (isWebGL2) {
            texType = (gl as WebGL2RenderingContext).FLOAT;
            texInternalFormat = (gl as WebGL2RenderingContext).RGBA32F;
            // Enable float texture filtering in WebGL2
            gl.getExtension('EXT_color_buffer_float');
        } else if (extFloat) {
            texType = gl.FLOAT;
            texInternalFormat = gl.RGBA;
        } else {
            texType = gl.UNSIGNED_BYTE;
            texInternalFormat = gl.RGBA;
            console.warn("Float textures not supported - R-D may not work correctly");
        }

        // Compile shader
        const compileShader = (src: string, type: number): WebGLShader | null => {
            const shader = gl!.createShader(type);
            if (!shader) return null;
            gl!.shaderSource(shader, src);
            gl!.compileShader(shader);
            if (!gl!.getShaderParameter(shader, gl!.COMPILE_STATUS)) {
                console.error("Shader compile error:", gl!.getShaderInfoLog(shader));
                return null;
            }
            return shader;
        };

        // Adjust header for WebGL2
        const getHeader = () => {
            if (isWebGL2) {
                return `#version 300 es
precision highp float;
uniform vec3 iResolution;
uniform float iTime;
uniform int iFrame;
uniform vec4 iMouse;
uniform sampler2D iChannel0;
out vec4 fragColor;
#define gl_FragColor fragColor
`;
            }
            return HEADER;
        };

        const getVert = () => {
            if (isWebGL2) {
                return `#version 300 es
in vec2 position;
void main() {
  gl_Position = vec4(position, 0.0, 1.0);
}
`;
            }
            return VERT;
        };

        // Create program
        const createProgram = (fragSrc: string): WebGLProgram | null => {
            const vs = compileShader(getVert(), gl!.VERTEX_SHADER);
            const fs = compileShader(getHeader() + fragSrc, gl!.FRAGMENT_SHADER);
            if (!vs || !fs) return null;

            const prog = gl!.createProgram();
            if (!prog) return null;
            gl!.attachShader(prog, vs);
            gl!.attachShader(prog, fs);
            gl!.linkProgram(prog);
            if (!gl!.getProgramParameter(prog, gl!.LINK_STATUS)) {
                console.error("Program link error:", gl!.getProgramInfoLog(prog));
                return null;
            }
            return prog;
        };

        const progA = createProgram(BUFFER_A);
        const progImg = createProgram(IMAGE);
        if (!progA || !progImg) {
            console.error("Failed to create shader programs");
            return;
        }

        // Fullscreen quad
        const quadBuf = gl.createBuffer();
        gl.bindBuffer(gl.ARRAY_BUFFER, quadBuf);
        gl.bufferData(gl.ARRAY_BUFFER, new Float32Array([-1, -1, 1, -1, -1, 1, 1, 1]), gl.STATIC_DRAW);

        // Uniform locations
        const getUniforms = (prog: WebGLProgram) => ({
            position: gl!.getAttribLocation(prog, "position"),
            iResolution: gl!.getUniformLocation(prog, "iResolution"),
            iTime: gl!.getUniformLocation(prog, "iTime"),
            iFrame: gl!.getUniformLocation(prog, "iFrame"),
            iMouse: gl!.getUniformLocation(prog, "iMouse"),
            iChannel0: gl!.getUniformLocation(prog, "iChannel0"),
        });

        const uA = getUniforms(progA);
        const uImg = getUniforms(progImg);

        // Create framebuffer with texture
        const createFBO = (w: number, h: number) => {
            const tex = gl!.createTexture();
            gl!.bindTexture(gl!.TEXTURE_2D, tex);

            if (isWebGL2) {
                (gl as WebGL2RenderingContext).texImage2D(gl!.TEXTURE_2D, 0, texInternalFormat, w, h, 0, gl!.RGBA, texType, null);
            } else {
                gl!.texImage2D(gl!.TEXTURE_2D, 0, texInternalFormat, w, h, 0, gl!.RGBA, texType, null);
            }

            // Use LINEAR filtering if supported, otherwise NEAREST
            const canFilter = isWebGL2 || extFloatLinear || texType === gl!.UNSIGNED_BYTE;
            const filter = canFilter ? gl!.LINEAR : gl!.NEAREST;

            gl!.texParameteri(gl!.TEXTURE_2D, gl!.TEXTURE_MIN_FILTER, filter);
            gl!.texParameteri(gl!.TEXTURE_2D, gl!.TEXTURE_MAG_FILTER, filter);
            gl!.texParameteri(gl!.TEXTURE_2D, gl!.TEXTURE_WRAP_S, gl!.REPEAT);
            gl!.texParameteri(gl!.TEXTURE_2D, gl!.TEXTURE_WRAP_T, gl!.REPEAT);

            const fb = gl!.createFramebuffer();
            gl!.bindFramebuffer(gl!.FRAMEBUFFER, fb);
            gl!.framebufferTexture2D(gl!.FRAMEBUFFER, gl!.COLOR_ATTACHMENT0, gl!.TEXTURE_2D, tex, 0);

            // Check framebuffer completeness
            const status = gl!.checkFramebufferStatus(gl!.FRAMEBUFFER);
            if (status !== gl!.FRAMEBUFFER_COMPLETE) {
                console.error("FBO not complete:", status);
            }

            gl!.bindFramebuffer(gl!.FRAMEBUFFER, null);
            return { fb, tex };
        };

        let fbo0: ReturnType<typeof createFBO> | null = null;
        let fbo1: ReturnType<typeof createFBO> | null = null;
        let frame = 0;
        let width = 0, height = 0;

        let animId = 0;
        const render = (time: number) => {
            const dpr = window.devicePixelRatio || 1;
            const rect = canvas.getBoundingClientRect();
            const w = Math.floor(rect.width * dpr);
            const h = Math.floor(rect.height * dpr);

            if (w !== width || h !== height || !fbo0 || !fbo1) {
                width = w; height = h;
                canvas.width = w; canvas.height = h;
                fbo0 = createFBO(w, h);
                fbo1 = createFBO(w, h);
                frame = 0;
                console.log("FBOs created:", w, "x", h);
            }

            const t = time * 0.001;
            const readFBO = frame % 2 === 0 ? fbo0 : fbo1;
            const writeFBO = frame % 2 === 0 ? fbo1 : fbo0;

            // PASS 1: Buffer A (reaction-diffusion)
            gl!.bindFramebuffer(gl!.FRAMEBUFFER, writeFBO!.fb);
            gl!.viewport(0, 0, width, height);
            gl!.useProgram(progA);

            gl!.bindBuffer(gl!.ARRAY_BUFFER, quadBuf);
            gl!.enableVertexAttribArray(uA.position);
            gl!.vertexAttribPointer(uA.position, 2, gl!.FLOAT, false, 0, 0);

            gl!.activeTexture(gl!.TEXTURE0);
            gl!.bindTexture(gl!.TEXTURE_2D, readFBO!.tex);
            gl!.uniform1i(uA.iChannel0, 0);
            gl!.uniform3f(uA.iResolution, width, height, 1.0);
            gl!.uniform1f(uA.iTime, t);
            gl!.uniform1i(uA.iFrame, frame);
            gl!.uniform4f(uA.iMouse, 0, 0, 0, 0);
            gl!.drawArrays(gl!.TRIANGLE_STRIP, 0, 4);

            // PASS 2: Image (raymarching) - reads from the buffer we just wrote
            gl!.bindFramebuffer(gl!.FRAMEBUFFER, null);
            gl!.viewport(0, 0, width, height);
            gl!.useProgram(progImg);

            gl!.bindBuffer(gl!.ARRAY_BUFFER, quadBuf);
            gl!.enableVertexAttribArray(uImg.position);
            gl!.vertexAttribPointer(uImg.position, 2, gl!.FLOAT, false, 0, 0);

            gl!.activeTexture(gl!.TEXTURE0);
            gl!.bindTexture(gl!.TEXTURE_2D, writeFBO!.tex);
            gl!.uniform1i(uImg.iChannel0, 0);
            gl!.uniform3f(uImg.iResolution, width, height, 1.0);
            gl!.uniform1f(uImg.iTime, t);
            gl!.uniform4f(uImg.iMouse, 0, 0, 0, 0);
            gl!.drawArrays(gl!.TRIANGLE_STRIP, 0, 4);

            frame++;
            if (frame === 5 && onReady$) {
                onReady$();
            }
            animId = requestAnimationFrame(render);
        };

        animId = requestAnimationFrame(render);
        cleanup(() => cancelAnimationFrame(animId));
    });

    // CSS handles circular masking - shader is UNMODIFIED
    return (
        <div class="loading-orb-container" style={{
            width: "100%",
            height: "100%",
            borderRadius: "50%",
            overflow: "hidden",
            background: "transparent"
        }}>
            <canvas ref={canvasRef} style={{ width: "100%", height: "100%", display: "block" }} />
        </div>
    );
});
