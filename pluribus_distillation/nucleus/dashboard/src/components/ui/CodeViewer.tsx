import { component$, useComputed$ } from '@builder.io/qwik';

interface Props {
  code: string;
  lang?: string;
}

// Simple regex-based syntaxizer for "Neon" effect
// Maps tokens to our global CSS vars (primary=cyan, secondary=purple, accent=pink)
function tokenize(code: string): string {
  if (!code) return '';
  
  // Escape HTML first
  let html = code
    .replace(/&/g, '&amp;')
    .replace(/</g, '&lt;')
    .replace(/>/g, '&gt;');

  // 1. Keywords (Purple/Secondary)
  html = html.replace(/\b(import|export|const|let|var|function|class|interface|return|if|else|for|while|await|async|from)\b/g, 
    '<span class="text-secondary neon-token">$&</span>');

  // 2. Types/Classes (Pink/Accent)
  html = html.replace(/\b([A-Z][a-zA-Z0-9_]*)\b/g, 
    '<span class="text-accent neon-token">$&</span>');

  // 3. Strings (Green)
  html = html.replace(/(['`"])(\.*?)\1/g, 
    '<span class="text-green-400">$1$2$1</span>');

  // 4. Comments (Muted)
  html = html.replace(/(\/\/.*$)/gm, 
    '<span class="text-muted-foreground italic">$1</span>');

  // 5. Special Chars (Cyan/Primary)
  html = html.replace(/([{}[\].(),;])/g, 
    '<span class="text-primary/70">$1</span>');

  return html;
}

export const CodeViewer = component$<Props>(({ code }) => {
  const lines = useComputed$(() => {
    const rawHtml = tokenize(code);
    return rawHtml.split('\n');
  });

  return (
    <div class="font-mono text-xs overflow-auto h-full bg-[#08080a] relative group">
      {/* Line Numbers + Code */}
      <div class="flex min-h-full">
        {/* Gutter */}
        <div class="flex-shrink-0 flex flex-col text-right select-none bg-[#0c0c0e] border-r border-[var(--glass-border-subtle)] py-4 px-2 text-muted-foreground/40 w-10">
          {lines.value.map((_, i) => (
            <div key={i} class="h-5 leading-5">{i + 1}</div>
          ))}
        </div>

        {/* Content */}
        <div class="flex-1 py-4">
          {lines.value.map((lineHtml, i) => (
            <div 
              key={i} 
              class="h-5 leading-5 px-4 hover:bg-white/5 transition-colors duration-75 flex items-center relative group/line"
            >
              {/* Active Line Glow (Left Marker) */}
              <div class="absolute left-0 top-0 bottom-0 w-0.5 bg-primary opacity-0 group-hover/line:opacity-100 transition-opacity shadow-[0_0_8px_#06b6d4]" />
              
              {/* The Code Line */}
              <span 
                class="whitespace-pre" 
                dangerouslySetInnerHTML={lineHtml || ' '}
              />
            </div>
          ))}
        </div>
      </div>
    </div>
  );
});
