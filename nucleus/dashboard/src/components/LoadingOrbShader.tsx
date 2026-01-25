import { component$, useVisibleTask$, useSignal, noSerialize } from '@builder.io/qwik';

export const LoadingOrbShader = component$(() => {
    const canvasRef = useSignal<HTMLCanvasElement>();

    useVisibleTask$(({ cleanup }) => {
        const canvas = canvasRef.value;
        if (!canvas) return;

        // SHADER SOURCES (EXACT COPY-PASTE FROM USER)
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
`;

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
`;

        // WEBGL INITIALIZATION
        let gl = canvas.getContext('webgl2', { alpha: true }) as WebGL2RenderingContext | null;
        let isWebGL2 = !!gl;
        if (!gl) {
            gl = canvas.getContext('webgl', { alpha: true }) as WebGLRenderingContext | null;
            isWebGL2 = false;
        }
        if (!gl) return;

        // init extensions
        isWebGL2 ? gl.getExtension('EXT_color_buffer_float') : gl.getExtension('OES_texture_float');
        gl.getExtension('OES_texture_float_linear');

        function createShader(src: string, type: number) {
            if (!gl) return null;
            const s = gl.createShader(type);
            if (!s) return null;

            const commonHeader = isWebGL2 ?
                `#version 300 es
            precision highp float;
            precision highp int;
            uniform vec3 iResolution;
            uniform float iTime;
            uniform float iTimeDelta;
            uniform float iFrameRate;
            uniform int iFrame;
            uniform vec4 iMouse;
            uniform vec4 iDate;
            uniform vec3 iChannelResolution[4];
            uniform float iChannelTime[4];
            uniform sampler2D iChannel0;
            
            in vec2 position;
            out vec4 fragColor;
            #define gl_FragColor fragColor
            #define texture2D texture` :
                `precision highp float;
            precision highp int;
            uniform vec3 iResolution;
            uniform float iTime;
            uniform float iTimeDelta;
            uniform float iFrameRate;
            uniform int iFrame;
            uniform vec4 iMouse;
            uniform vec4 iDate;
            uniform vec3 iChannelResolution[4];
            uniform float iChannelTime[4];
            uniform sampler2D iChannel0;
            
            attribute vec2 position;`;

            const wrapper = isWebGL2 ?
                `${commonHeader}
            ${src}
            void main() { 
                vec4 col = vec4(0.0);
                mainImage(col, gl_FragCoord.xy);
                fragColor = col;
            }` :
                `${commonHeader}
            ${src}
            void main() { 
                mainImage(gl_FragColor, gl_FragCoord.xy);
            }`;

            gl.shaderSource(s, wrapper);
            gl.compileShader(s);
            if (!gl.getShaderParameter(s, gl.COMPILE_STATUS)) {
                console.error('Shader Compile Error:', gl.getShaderInfoLog(s));
                return null;
            }
            return s;
        }

        const vsSrc = isWebGL2 ?
            `#version 300 es
        in vec2 position; void main(){gl_Position=vec4(position,0.,1.);}` :
            `attribute vec2 position; void main(){gl_Position=vec4(position,0.,1.);}`;

        const vs = gl.createShader(gl.VERTEX_SHADER);
        if (!vs) return;
        gl.shaderSource(vs, vsSrc);
        gl.compileShader(vs);

        function createProg(fragSrc: string) {
            if (!gl || !vs) return null;
            const fs = createShader(fragSrc, gl.FRAGMENT_SHADER);
            if (!fs) return null;
            const p = gl.createProgram();
            if (!p) return null;
            gl.attachShader(p, vs);
            gl.attachShader(p, fs);
            gl.linkProgram(p);
            return p;
        }

        const bufferAProg = createProg(BUFFER_A);
        if (!bufferAProg) return;

        // TRANSPARENCY INJECTION
        let MOD_IMAGE = IMAGE;
        MOD_IMAGE = MOD_IMAGE.replace('vec3 render( in vec3 ro, in vec3 rd )', 'vec4 render( in vec3 ro, in vec3 rd )');
        MOD_IMAGE = MOD_IMAGE.replace('return vec3( clamp(col,0.0,1.0) );', 'return vec4(0.0); // Transparent background');
        MOD_IMAGE = MOD_IMAGE.replace('col = mix( col, vec3(0.8,0.9,1.0), 1.0-exp( -0.002*t*t ) );', 'col = mix( col, vec3(0.8,0.9,1.0), 1.0-exp( -0.002*t*t ) ); return vec4(clamp(col, 0.0, 1.0), 1.0);');
        MOD_IMAGE = MOD_IMAGE.replace('vec3 col = render( ro, rd );', 'vec4 col = render( ro, rd );');
        MOD_IMAGE = MOD_IMAGE.replace('col = pow( col, vec3(0.4545) );', 'col.rgb = pow( col.rgb, vec3(0.4545) );');
        MOD_IMAGE = MOD_IMAGE.replace('fragColor=vec4( col, 1.0 );', 'fragColor = col;');

        const imageProg = createProg(MOD_IMAGE);
        if (!imageProg) return;

        const buf = gl.createBuffer();
        gl.bindBuffer(gl.ARRAY_BUFFER, buf);
        gl.bufferData(gl.ARRAY_BUFFER, new Float32Array([-1, -1, 1, -1, -1, 1, 1, 1]), gl.STATIC_DRAW);

        // FBO Setup
        function createFBO(w: number, h: number) {
            if (!gl) return null;
            const t = gl.createTexture();
            gl.bindTexture(gl.TEXTURE_2D, t);

            let type = gl.UNSIGNED_BYTE;
            let format = gl.RGBA;

            if (isWebGL2) {
                type = gl.FLOAT;
                format = (gl as WebGL2RenderingContext).RGBA32F;
            } else {
                // WebGL1 float texture check? simpler to assume float ext exists if we got this far
                // but strictly:
                const ext = gl.getExtension('OES_texture_float');
                if (ext) type = gl.FLOAT;
            }

            gl.texImage2D(gl.TEXTURE_2D, 0, format, w, h, 0, gl.RGBA, type, null);
            gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.LINEAR);
            gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, gl.LINEAR);

            const f = gl.createFramebuffer();
            gl.bindFramebuffer(gl.FRAMEBUFFER, f);
            gl.framebufferTexture2D(gl.FRAMEBUFFER, gl.COLOR_ATTACHMENT0, gl.TEXTURE_2D, t, 0);
            return { f, t };
        }

        let fbo0: any, fbo1: any;
        let width = 0, height = 0;
        let frame = 0;
        let animationFrameId: number;
        let startTime = Date.now();

        function render() {
            if (!gl || !bufferAProg || !imageProg) return;

            const time = (Date.now() - startTime) * 0.001;

            if (canvas!.width !== canvas!.clientWidth || !fbo0) {
                width = canvas!.width = canvas!.clientWidth;
                height = canvas!.height = canvas!.clientHeight;
                fbo0 = createFBO(width, height);
                fbo1 = createFBO(width, height);
                frame = 0;
            }

            if (!fbo0 || !fbo1) return;

            const read = frame % 2 === 0 ? fbo0 : fbo1;
            const write = frame % 2 === 0 ? fbo1 : fbo0;

            // Pass 1: Buffer A
            gl.bindFramebuffer(gl.FRAMEBUFFER, write.f);
            gl.viewport(0, 0, width, height);
            gl.useProgram(bufferAProg);

            gl.activeTexture(gl.TEXTURE0);
            gl.bindTexture(gl.TEXTURE_2D, read.t);
            gl.uniform1i(gl.getUniformLocation(bufferAProg, "iChannel0"), 0);
            gl.uniform3f(gl.getUniformLocation(bufferAProg, "iResolution"), width, height, 1);
            gl.uniform1f(gl.getUniformLocation(bufferAProg, "iTime"), time);
            gl.uniform1i(gl.getUniformLocation(bufferAProg, "iFrame"), frame);

            gl.bindBuffer(gl.ARRAY_BUFFER, buf);
            const locA = gl.getAttribLocation(bufferAProg, "position");
            gl.enableVertexAttribArray(locA);
            gl.vertexAttribPointer(locA, 2, gl.FLOAT, false, 0, 0);
            gl.drawArrays(gl.TRIANGLE_STRIP, 0, 4);

            // Pass 2: Image
            gl.bindFramebuffer(gl.FRAMEBUFFER, null);
            gl.viewport(0, 0, width, height);
            gl.clearColor(0, 0, 0, 0);
            gl.clear(gl.COLOR_BUFFER_BIT); // Clear transparent

            gl.useProgram(imageProg);
            gl.activeTexture(gl.TEXTURE0);
            gl.bindTexture(gl.TEXTURE_2D, write.t);
            gl.uniform1i(gl.getUniformLocation(imageProg, "iChannel0"), 0);
            gl.uniform3f(gl.getUniformLocation(imageProg, "iResolution"), width, height, 1);
            gl.uniform1f(gl.getUniformLocation(imageProg, "iTime"), time);

            const locB = gl.getAttribLocation(imageProg, "position");
            gl.enableVertexAttribArray(locB);
            gl.vertexAttribPointer(locB, 2, gl.FLOAT, false, 0, 0);
            gl.drawArrays(gl.TRIANGLE_STRIP, 0, 4);

            frame++;
            animationFrameId = requestAnimationFrame(render);
        }

        render();

        cleanup(() => cancelAnimationFrame(animationFrameId));
    });

    return <canvas ref={canvasRef} style={{ width: '100%', height: '100%', display: 'block' }} />;
});
