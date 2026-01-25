import { component$, type Signal, type QRL, $ } from '@builder.io/qwik';
import { CodeViewer } from '../ui/CodeViewer';

interface RhizomeViewProps {
  currentPath: Signal<string>;
  fileTree: Signal<any[]>;
  selectedFile: Signal<string | null>;
  fileContent: Signal<string>;
  navigateUp: QRL<() => void>;
  enterDir: QRL<(name: string) => void>;
  loadFile: QRL<(path: string) => void>;
}

export const RhizomeView = component$<RhizomeViewProps>((props) => {
  const { currentPath, fileTree, selectedFile, fileContent, navigateUp, enterDir, loadFile } = props;

  return (
    <div class="grid grid-cols-12 gap-6 h-full">
      {/* File Tree */}
      <div class="col-span-3 glass-surface-elevated flex flex-col h-[calc(100vh-200px)]">
        <div class="p-3 border-b border-[var(--glass-border)] flex items-center justify-between">
          <h3 class="font-semibold text-sm text-[var(--glass-text-primary)]">Explorer</h3>
          <span class="glass-chip text-[10px] mono truncate max-w-[150px]">{currentPath.value}</span>
        </div>
        <div class="p-2 border-b border-[var(--glass-border)]">
          <button
            onClick$={navigateUp}
            disabled={currentPath.value === '/'}
            class="glass-interactive w-full text-left px-2 py-1 text-xs disabled:opacity-50"
          >
            .. (Up)
          </button>
        </div>
        <div class="flex-1 overflow-auto p-2 space-y-1">
          {fileTree.value.map((entry, idx) => (
            <button
              key={entry.name}
              onClick$={() => entry.type === 'dir' ? enterDir(entry.name) : loadFile(entry.name)}
              class={`glass-animate-enter-fade w-full text-left px-2 py-1.5 rounded text-xs font-mono flex items-center gap-2 glass-hover-lift ${
                selectedFile.value === entry.name ? 'glass-gradient-border text-[var(--glass-accent-cyan)]' : 'text-[var(--glass-text-secondary)]'
              }`}
              style={{ '--stagger': idx } as any}
            >
              <span>{entry.type === 'dir' ? '📁' : '📄'}</span>
              <span class="truncate">{entry.name}</span>
            </button>
          ))}
        </div>
      </div>

      {/* Code Viewer */}
      <div class="col-span-9 glass-surface-elevated flex flex-col h-[calc(100vh-200px)]">
        <div class="p-3 border-b border-[var(--glass-border)] flex items-center gap-3">
          <h3 class="font-semibold text-sm text-[var(--glass-text-primary)]">
            {selectedFile.value || 'No file selected'}
          </h3>
          {selectedFile.value && (
            <span class="glass-chip glass-chip-accent-emerald text-[10px]">Active</span>
          )}
        </div>
        <div class="flex-1 overflow-hidden bg-[#08080a]">
          {selectedFile.value ? (
            <div class="glass-animate-enter h-full">
              <CodeViewer code={fileContent.value} />
            </div>
          ) : (
            <div class="flex items-center justify-center h-full text-[var(--glass-text-tertiary)]">
              Select a file to view content
            </div>
          )}
        </div>
      </div>
    </div>
  );
});
