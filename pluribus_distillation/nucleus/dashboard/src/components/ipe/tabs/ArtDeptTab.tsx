/**
 * ArtDeptTab.tsx
 *
 * Art Department shader editor tab for IPE.
 * Allows shader preview, editing, and swapping.
 */

import {
  component$,
  useSignal,
  useStore,
  useVisibleTask$,
  $,
  type QRL,
} from '@builder.io/qwik';
import type { IPEContext, UniformValue } from '../../../lib/ipe';

interface ArtDeptTabProps {
  context: IPEContext;
  onShaderChange$?: QRL<(shader: string) => void>;
  onUniformChange$?: QRL<(name: string, value: number | number[]) => void>;
}

type SubTab = 'preview' | 'code' | 'uniforms';

export const ArtDeptTab = component$<ArtDeptTabProps>(({
  context,
  onShaderChange$,
  onUniformChange$,
}) => {
  const activeSubTab = useSignal<SubTab>('uniforms');
  const shaderCode = useSignal(context.shaderContext?.fragmentSource || '');
  const compileError = useSignal<string | null>(null);

  const hasShader = context.elementType === 'shader' || context.elementType === 'canvas';
  const uniforms = context.shaderContext?.uniforms || {};

  if (!hasShader) {
    return (
      <div class="text-center py-8 text-gray-500">
        <div class="text-4xl mb-4">🎨</div>
        <div>This element is not a shader/canvas.</div>
        <div class="text-xs mt-2">
          Select a WebGL canvas element to edit its shader.
        </div>
      </div>
    );
  }

  return (
    <div class="space-y-4">
      {/* Shader info header */}
      <div class="flex items-center justify-between p-3 rounded-lg bg-white/5">
        <div>
          <div class="text-sm font-medium text-gray-300">WebGL Shader</div>
          <div class="text-xs text-gray-500">
            {context.shaderContext?.canvasSize?.[0]} × {context.shaderContext?.canvasSize?.[1]}
            {' • '}
            WebGL {context.shaderContext?.webglVersion}
          </div>
        </div>
        <div class="flex items-center gap-2">
          <span class="px-2 py-1 rounded bg-orange-500/20 text-orange-300 text-xs">
            {Object.keys(uniforms).length} uniforms
          </span>
        </div>
      </div>

      {/* Sub-tabs */}
      <div class="flex gap-1 p-1 rounded-lg bg-white/5">
        {(['preview', 'uniforms', 'code'] as SubTab[]).map(tab => (
          <button
            key={tab}
            type="button"
            class={[
              'flex-1 px-3 py-1.5 rounded text-sm font-medium transition-colors',
              activeSubTab.value === tab
                ? 'bg-white/10 text-white'
                : 'text-gray-400 hover:text-white',
            ]}
            onClick$={() => { activeSubTab.value = tab; }}
          >
            {tab === 'preview' && '👁️ Preview'}
            {tab === 'uniforms' && '🎛️ Uniforms'}
            {tab === 'code' && '📝 GLSL'}
          </button>
        ))}
      </div>

      {/* Sub-tab content */}
      <div class="min-h-[200px]">
        {activeSubTab.value === 'preview' && (
          <ShaderPreview context={context} />
        )}

        {activeSubTab.value === 'uniforms' && (
          <UniformsEditor
            uniforms={uniforms}
            onUniformChange$={onUniformChange$}
          />
        )}

        {activeSubTab.value === 'code' && (
          <CodeEditor
            code={shaderCode.value}
            error={compileError.value}
            onCodeChange$={(code) => {
              shaderCode.value = code;
              onShaderChange$?.(code);
            }}
          />
        )}
      </div>

      {/* Compile error */}
      {compileError.value && (
        <div class="p-3 rounded-lg bg-red-500/10 border border-red-500/30 text-red-300 text-xs font-mono">
          {compileError.value}
        </div>
      )}
    </div>
  );
});

// ============================================================================
// Sub-components
// ============================================================================

interface ShaderPreviewProps {
  context: IPEContext;
}

const ShaderPreview = component$<ShaderPreviewProps>(({ context }) => {
  return (
    <div class="space-y-4">
      <div class="text-sm text-gray-400">
        Live shader preview coming in Phase 3
      </div>

      {/* Shader info */}
      <div class="p-3 rounded-lg bg-black/30 space-y-2 text-xs">
        <div class="flex justify-between">
          <span class="text-gray-500">Canvas size:</span>
          <span class="text-gray-300 font-mono">
            {context.shaderContext?.canvasSize?.[0]} × {context.shaderContext?.canvasSize?.[1]}
          </span>
        </div>
        <div class="flex justify-between">
          <span class="text-gray-500">WebGL version:</span>
          <span class="text-gray-300 font-mono">
            {context.shaderContext?.webglVersion}
          </span>
        </div>
        <div class="flex justify-between">
          <span class="text-gray-500">Active uniforms:</span>
          <span class="text-gray-300 font-mono">
            {context.shaderContext?.programInfo?.activeUniforms}
          </span>
        </div>
        <div class="flex justify-between">
          <span class="text-gray-500">Active attributes:</span>
          <span class="text-gray-300 font-mono">
            {context.shaderContext?.programInfo?.activeAttributes}
          </span>
        </div>
      </div>

      {/* Placeholder preview */}
      <div
        class={[
          'aspect-video rounded-lg overflow-hidden',
          'bg-gradient-to-br from-purple-900/50 to-blue-900/50',
          'flex items-center justify-center',
          'border border-[var(--glass-border)]',
        ]}
      >
        <div class="text-center text-gray-500">
          <div class="text-2xl mb-2">🎭</div>
          <div class="text-xs">Shader Preview</div>
        </div>
      </div>
    </div>
  );
});

interface UniformsEditorProps {
  uniforms: Record<string, UniformValue>;
  onUniformChange$?: QRL<(name: string, value: number | number[]) => void>;
}

const UniformsEditor = component$<UniformsEditorProps>(({
  uniforms,
  onUniformChange$,
}) => {
  const uniformEntries = Object.entries(uniforms);

  if (uniformEntries.length === 0) {
    return (
      <div class="text-center py-8 text-gray-500">
        <div>No editable uniforms detected.</div>
        <div class="text-xs mt-2">
          Shader introspection may be limited.
        </div>
      </div>
    );
  }

  return (
    <div class="space-y-3">
      {uniformEntries.map(([name, uniform]) => (
        <UniformControl
          key={name}
          name={name}
          uniform={uniform}
          onChange$={(value) => onUniformChange$?.(name, value)}
        />
      ))}
    </div>
  );
});

interface UniformControlProps {
  name: string;
  uniform: UniformValue;
  onChange$: QRL<(value: number | number[]) => void>;
}

const UniformControl = component$<UniformControlProps>(({
  name,
  uniform,
  onChange$,
}) => {
  const value = useSignal(uniform.value);

  // Determine control type based on uniform type
  const isVector = ['vec2', 'vec3', 'vec4'].includes(uniform.type);
  const isColor = isVector && (
    name.toLowerCase().includes('color') ||
    name.toLowerCase().includes('colour')
  );

  if (uniform.type === 'float' || uniform.type === 'int') {
    const numValue = typeof value.value === 'number' ? value.value : 0;
    const min = uniform.min ?? 0;
    const max = uniform.max ?? (uniform.type === 'int' ? 100 : 10);
    const step = uniform.step ?? (uniform.type === 'int' ? 1 : 0.01);

    return (
      <div class="p-2 rounded-lg bg-white/5">
        <div class="flex items-center justify-between mb-2">
          <span class="text-xs font-mono text-orange-400">{name}</span>
          <span class="text-xs text-gray-500">{uniform.type}</span>
        </div>
        <div class="flex items-center gap-2">
          <input
            type="range"
            min={min}
            max={max}
            step={step}
            value={numValue}
            class="flex-1 h-1.5 rounded appearance-none bg-white/20"
            onInput$={(e) => {
              const newVal = parseFloat((e.target as HTMLInputElement).value);
              value.value = newVal;
              onChange$(newVal);
            }}
          />
          <input
            type="number"
            value={numValue}
            step={step}
            class={[
              'w-16 px-2 py-1 rounded text-xs font-mono text-right',
              'bg-black/30 border border-[var(--glass-border)] text-gray-300',
            ]}
            onInput$={(e) => {
              const newVal = parseFloat((e.target as HTMLInputElement).value);
              value.value = newVal;
              onChange$(newVal);
            }}
          />
        </div>
      </div>
    );
  }

  if (isVector) {
    const vecValue = Array.isArray(value.value) ? value.value : [0, 0, 0, 0];
    const components = uniform.type === 'vec2' ? 2 : uniform.type === 'vec3' ? 3 : 4;
    const labels = ['X', 'Y', 'Z', 'W'].slice(0, components);

    if (isColor && components >= 3) {
      // Color picker for vec3/vec4 that represent colors
      const r = Math.round(vecValue[0] * 255);
      const g = Math.round(vecValue[1] * 255);
      const b = Math.round(vecValue[2] * 255);
      const hexColor = `#${r.toString(16).padStart(2, '0')}${g.toString(16).padStart(2, '0')}${b.toString(16).padStart(2, '0')}`;

      return (
        <div class="p-2 rounded-lg bg-white/5">
          <div class="flex items-center justify-between mb-2">
            <span class="text-xs font-mono text-orange-400">{name}</span>
            <span class="text-xs text-gray-500">{uniform.type}</span>
          </div>
          <div class="flex items-center gap-2">
            <input
              type="color"
              value={hexColor}
              class="w-8 h-8 rounded border border-[var(--glass-border-active)] cursor-pointer"
              onInput$={(e) => {
                const hex = (e.target as HTMLInputElement).value;
                const newR = parseInt(hex.slice(1, 3), 16) / 255;
                const newG = parseInt(hex.slice(3, 5), 16) / 255;
                const newB = parseInt(hex.slice(5, 7), 16) / 255;
                const newVal = [newR, newG, newB, vecValue[3] ?? 1];
                value.value = newVal;
                onChange$(newVal);
              }}
            />
            <span class="text-xs font-mono text-gray-300">{hexColor}</span>
          </div>
        </div>
      );
    }

    // Generic vector sliders
    return (
      <div class="p-2 rounded-lg bg-white/5">
        <div class="flex items-center justify-between mb-2">
          <span class="text-xs font-mono text-orange-400">{name}</span>
          <span class="text-xs text-gray-500">{uniform.type}</span>
        </div>
        <div class="space-y-1">
          {labels.map((label, i) => (
            <div key={label} class="flex items-center gap-2">
              <span class="w-4 text-xs text-gray-500">{label}</span>
              <input
                type="range"
                min={-10}
                max={10}
                step={0.01}
                value={vecValue[i]}
                class="flex-1 h-1 rounded appearance-none bg-white/20"
                onInput$={(e) => {
                  const newVec = [...vecValue];
                  newVec[i] = parseFloat((e.target as HTMLInputElement).value);
                  value.value = newVec;
                  onChange$(newVec);
                }}
              />
              <span class="w-12 text-xs text-gray-400 text-right font-mono">
                {vecValue[i].toFixed(2)}
              </span>
            </div>
          ))}
        </div>
      </div>
    );
  }

  // Fallback for unsupported types
  return (
    <div class="p-2 rounded-lg bg-white/5">
      <div class="flex items-center justify-between">
        <span class="text-xs font-mono text-orange-400">{name}</span>
        <span class="text-xs text-gray-500">{uniform.type}</span>
      </div>
      <div class="text-xs text-gray-500 mt-1">
        {JSON.stringify(uniform.value)}
      </div>
    </div>
  );
});

interface CodeEditorProps {
  code: string;
  error: string | null;
  onCodeChange$: QRL<(code: string) => void>;
}

const CodeEditor = component$<CodeEditorProps>(({
  code,
  error,
  onCodeChange$,
}) => {
  return (
    <div class="space-y-2">
      <div class="flex items-center justify-between">
        <span class="text-xs text-gray-400">Fragment Shader (GLSL)</span>
        <button
          type="button"
          class="px-2 py-1 rounded text-xs bg-white/10 hover:bg-white/20 text-gray-300 transition-colors"
          onClick$={() => navigator.clipboard.writeText(code)}
        >
          Copy
        </button>
      </div>

      <textarea
        value={code}
        class={[
          'w-full h-64 p-3 rounded-lg font-mono text-xs',
          'bg-black/50 text-green-400',
          'border border-[var(--glass-border)] focus:border-blue-500/50 focus:outline-none',
          'resize-none',
          error ? 'border-red-500/50' : '',
        ]}
        spellcheck={false}
        onInput$={(e) => onCodeChange$((e.target as HTMLTextAreaElement).value)}
      />

      <div class="p-2 rounded-lg bg-yellow-500/10 border border-yellow-500/30">
        <div class="text-yellow-300 text-xs font-medium">
          CodeMirror integration coming in Phase 3
        </div>
        <div class="text-yellow-200/70 text-xs mt-1">
          Full syntax highlighting and autocomplete for GLSL
        </div>
      </div>
    </div>
  );
});

export default ArtDeptTab;
