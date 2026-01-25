/**
 * MountpointsPanel - Floating Cloud Storage Browser with Deep Pipeline Integration
 *
 * VMware Clarity / Lit-style hierarchical tree browser for Google Drive mounts
 * with magical colorful glowing animations and deep integration with:
 * - STRp pipeline for categorization/distillation
 * - SemOps semiotic analysis/destructuring
 * - Knowledge Graph (KG) node creation
 * - Vector encoding via rag_vector
 * - Browser localStorage IR cache
 *
 * Architecture:
 * - FileTree: Recursive expand/collapse with lazy loading
 * - FileViewer: Content display with syntax highlighting
 * - ActionBar: Pipeline integration controls
 * - LocalIR: IndexedDB-backed local search cache
 */

import { component$, useSignal, useStore, useVisibleTask$, $, type QRL } from '@builder.io/qwik';
import { Button } from './ui/Button';
import { Input } from './ui/Input';

// ═══════════════════════════════════════════════════════════════════════════════
// TYPES
// ═══════════════════════════════════════════════════════════════════════════════

export interface MountedDrive {
  name: string;
  email: string;
  mountpoint: string;
  status: 'mounted' | 'unmounted' | 'connecting';
}

export interface FileNode {
  name: string;
  path: string;
  isDir: boolean;
  size: number;
  modTime?: string;
  mimeType?: string;
  children?: FileNode[];
  loaded?: boolean;
  expanded?: boolean;
}

export interface FileContent {
  path: string;
  content: string;
  mimeType: string;
  size: number;
  encoding?: string;
  preview?: string; // Base64 for images
}

export interface PipelineAction {
  id: string;
  type: 'strp' | 'semops' | 'kg' | 'vector' | 'cache';
  status: 'pending' | 'running' | 'complete' | 'error';
  label: string;
  result?: unknown;
}

export interface LocalIREntry {
  path: string;
  content: string;
  embedding?: number[];
  keywords: string[];
  timestamp: number;
  source: string;
}

// ═══════════════════════════════════════════════════════════════════════════════
// LOCAL IR CACHE (IndexedDB wrapper)
// ═══════════════════════════════════════════════════════════════════════════════

const LOCAL_IR_DB = 'pluribus_local_ir';
const LOCAL_IR_STORE = 'documents';

const openLocalIR = async (): Promise<IDBDatabase> => {
  return new Promise((resolve, reject) => {
    const request = indexedDB.open(LOCAL_IR_DB, 1);
    request.onerror = () => reject(request.error);
    request.onsuccess = () => resolve(request.result);
    request.onupgradeneeded = () => {
      const db = request.result;
      if (!db.objectStoreNames.contains(LOCAL_IR_STORE)) {
        const store = db.createObjectStore(LOCAL_IR_STORE, { keyPath: 'path' });
        store.createIndex('keywords', 'keywords', { multiEntry: true });
        store.createIndex('timestamp', 'timestamp');
        store.createIndex('source', 'source');
      }
    };
  });
};

const cacheToLocalIR = async (entry: LocalIREntry): Promise<void> => {
  const db = await openLocalIR();
  return new Promise((resolve, reject) => {
    const tx = db.transaction(LOCAL_IR_STORE, 'readwrite');
    tx.objectStore(LOCAL_IR_STORE).put(entry);
    tx.oncomplete = () => resolve();
    tx.onerror = () => reject(tx.error);
  });
};

const searchLocalIR = async (query: string): Promise<LocalIREntry[]> => {
  const db = await openLocalIR();
  return new Promise((resolve, reject) => {
    const tx = db.transaction(LOCAL_IR_STORE, 'readonly');
    const store = tx.objectStore(LOCAL_IR_STORE);
    const results: LocalIREntry[] = [];
    const request = store.openCursor();
    request.onsuccess = () => {
      const cursor = request.result;
      if (cursor) {
        const entry = cursor.value as LocalIREntry;
        if (entry.content.toLowerCase().includes(query.toLowerCase()) ||
            entry.keywords.some(k => k.toLowerCase().includes(query.toLowerCase()))) {
          results.push(entry);
        }
        cursor.continue();
      } else {
        resolve(results);
      }
    };
    request.onerror = () => reject(request.error);
  });
};

// ═══════════════════════════════════════════════════════════════════════════════
// UTILITY FUNCTIONS
// ═══════════════════════════════════════════════════════════════════════════════

const formatSize = (bytes: number): string => {
  if (bytes === 0) return '—';
  const k = 1024;
  const sizes = ['B', 'KB', 'MB', 'GB', 'TB'];
  const i = Math.floor(Math.log(bytes) / Math.log(k));
  return `${(bytes / Math.pow(k, i)).toFixed(1)} ${sizes[i]}`;
};

const getFileIcon = (name: string, isDir: boolean): string => {
  if (isDir) return '📁';
  const ext = name.split('.').pop()?.toLowerCase() || '';
  const icons: Record<string, string> = {
    pdf: '📕', md: '📝', txt: '📄', json: '📋', yaml: '⚙️', yml: '⚙️',
    py: '🐍', ts: '💠', tsx: '⚛️', js: '📜', jsx: '⚛️',
    png: '🖼️', jpg: '🖼️', jpeg: '🖼️', gif: '🎞️', svg: '🎨',
    mp3: '🎵', mp4: '🎬', wav: '🔊', webm: '🎬',
    zip: '📦', tar: '📦', gz: '📦', rar: '📦',
    doc: '📘', docx: '📘', xls: '📊', xlsx: '📊', ppt: '📽️',
  };
  return icons[ext] || '📄';
};

const extractKeywords = (content: string): string[] => {
  // Simple keyword extraction - find unique words > 3 chars
  const words = content.toLowerCase().match(/\b[a-z]{4,}\b/g) || [];
  const counts = new Map<string, number>();
  words.forEach(w => counts.set(w, (counts.get(w) || 0) + 1));
  return [...counts.entries()]
    .sort((a, b) => b[1] - a[1])
    .slice(0, 20)
    .map(([word]) => word);
};

// ═══════════════════════════════════════════════════════════════════════════════
// FILE TREE NODE COMPONENT
// ═══════════════════════════════════════════════════════════════════════════════

interface FileTreeNodeProps {
  node: FileNode;
  depth: number;
  onSelect$: QRL<(node: FileNode) => void>;
  onExpand$: QRL<(node: FileNode) => void>;
  selectedPath: string;
}

export const FileTreeNode = component$<FileTreeNodeProps>(({
  node,
  depth,
  onSelect$,
  onExpand$,
  selectedPath,
}) => {
  const isSelected = selectedPath === node.path;
  const indent = depth * 16;

  return (
    <div class="select-none">
      <div
        class={`
          flex items-center gap-2 px-2 py-1.5 cursor-pointer rounded-md
          transition-all duration-200 group
          ${isSelected
            ? 'bg-gradient-to-r from-cyan-500/20 to-purple-500/20 border border-cyan-500/30 shadow-[0_0_15px_rgba(6,182,212,0.3)]'
            : 'hover:bg-muted/40 border border-transparent'
          }
        `}
        style={{ paddingLeft: `${indent + 8}px` }}
        onClick$={async () => {
          if (node.isDir) {
            await onExpand$(node);
          } else {
            await onSelect$(node);
          }
        }}
      >
        {/* Expand/Collapse Arrow */}
        {node.isDir && (
          <span class={`
            text-xs transition-transform duration-200
            ${node.expanded ? 'rotate-90' : ''}
          `}>
            ▶
          </span>
        )}

        {/* Icon */}
        <span class={`
          text-base transition-all duration-300
          ${isSelected ? 'scale-110 drop-shadow-[0_0_8px_rgba(6,182,212,0.8)]' : ''}
        `}>
          {getFileIcon(node.name, node.isDir)}
        </span>

        {/* Name */}
        <span class={`
          flex-1 text-sm truncate
          ${isSelected ? 'text-cyan-300 font-medium' : 'text-foreground'}
        `}>
          {node.name}
        </span>

        {/* Size badge */}
        {!node.isDir && (
          <span class="text-[10px] text-muted-foreground opacity-0 group-hover:opacity-100 transition-opacity">
            {formatSize(node.size)}
          </span>
        )}
      </div>

      {/* Children */}
      {node.isDir && node.expanded && node.children && (
        <div class="animate-in slide-in-from-top-2 duration-200">
          {node.children.map((child) => (
            <FileTreeNode
              key={child.path}
              node={child}
              depth={depth + 1}
              onSelect$={onSelect$}
              onExpand$={onExpand$}
              selectedPath={selectedPath}
            />
          ))}
        </div>
      )}
    </div>
  );
});

// ═══════════════════════════════════════════════════════════════════════════════
// PIPELINE ACTION BUTTON
// ═══════════════════════════════════════════════════════════════════════════════

interface ActionButtonProps {
  label: string;
  icon: string;
  color: string;
  glowColor: string;
  active: boolean;
  loading: boolean;
  onClick$: QRL<() => void>;
  tooltip?: string;
}

export const ActionButton = component$<ActionButtonProps>(({
  label,
  icon,
  color,
  glowColor,
  active,
  loading,
  onClick$,
  tooltip,
}) => {
  return (
    <Button
      variant="tonal"
      onClick$={onClick$}
      disabled={loading}
      title={tooltip}
      class={`
        text-xs font-medium h-auto py-2
        ${active
                            ? `${color} shadow-[0_0_20px_var(--glow-color)] scale-105`
                            : `bg-muted/30 text-muted-foreground hover:${color} hover:shadow-[0_0_10px_var(--glow-color)]`
                          }
                          style={{ '--glow-color': glowColor } as any}        }
        ${loading ? 'animate-pulse' : ''}
      `}
    >
      <span class="flex items-center gap-1.5">
        <span class={loading ? 'animate-spin' : ''}>{icon}</span>
        <span>{label}</span>
      </span>

      {/* Glow ring animation when active */}
      {active && (
        <span class={`
          absolute inset-0 rounded-lg
          animate-ping opacity-30
          ${color}
        `} />
      )}
    </Button>
  );
});

// ═══════════════════════════════════════════════════════════════════════════════
// CONTENT VIEWER WITH MAGICAL ANIMATIONS
// ═══════════════════════════════════════════════════════════════════════════════

interface ContentViewerProps {
  content: FileContent | null;
  loading: boolean;
  onClose$: QRL<() => void>;
  actions: PipelineAction[];
  onAction$: QRL<(actionType: string) => void>;
}

export const ContentViewer = component$<ContentViewerProps>(({
  content,
  loading,
  onClose$,
  actions,
  onAction$,
}) => {
  if (!content && !loading) return null;

  const isImage = content?.mimeType?.startsWith('image/');
  const isCode = ['application/json', 'text/x-python', 'application/javascript', 'text/typescript']
    .some(m => content?.mimeType?.includes(m)) ||
    content?.path?.match(/\.(py|ts|tsx|js|jsx|json|yaml|yml|md)$/);

  return (
    <div class={`
      fixed inset-4 z-50 flex flex-col
      bg-background/95 backdrop-blur-xl
      rounded-2xl border border-cyan-500/30
      shadow-[0_0_60px_rgba(6,182,212,0.2),0_0_120px_rgba(168,85,247,0.1)]
      animate-in zoom-in-95 fade-in duration-300
      overflow-hidden
    `}>
      {/* Animated gradient border */}
      <div class="absolute inset-0 rounded-2xl pointer-events-none overflow-hidden">
        <div class="absolute inset-0 bg-gradient-to-r from-cyan-500/20 via-purple-500/20 to-pink-500/20 animate-gradient-x" />
      </div>

      {/* Header */}
      <div class="relative flex items-center justify-between p-4 border-b border-border/50">
        <div class="flex items-center gap-3">
          <span class="text-2xl drop-shadow-[0_0_10px_rgba(6,182,212,0.5)]">
            {content ? getFileIcon(content.path, false) : '📄'}
          </span>
          <div>
            <h2 class="font-semibold text-foreground">
              {content?.path.split('/').pop() || 'Loading...'}
            </h2>
            <p class="text-xs text-muted-foreground">
              {content?.mimeType} • {content ? formatSize(content.size) : ''}
            </p>
          </div>
        </div>

        <Button
          variant="icon"
          icon="close"
          onClick$={onClose$}
          class="text-muted-foreground hover:text-foreground"
        />
      </div>

      {/* Pipeline Actions Bar */}
      <div class="relative flex items-center gap-2 p-3 border-b border-border/30 bg-muted/20 overflow-x-auto">
        <span class="text-xs text-muted-foreground mr-2">Pipeline:</span>

        <ActionButton
          label="STRp Distill"
          icon="🔮"
          color="bg-purple-500/20 text-purple-400"
          glowColor="rgba(168,85,247,0.5)"
          active={actions.some(a => a.type === 'strp' && a.status === 'running')}
          loading={actions.some(a => a.type === 'strp' && a.status === 'running')}
          onClick$={() => onAction$('strp')}
          tooltip="Create STRp distillation request"
        />

        <ActionButton
          label="SemOps"
          icon="🧠"
          color="bg-cyan-500/20 text-cyan-400"
          glowColor="rgba(6,182,212,0.5)"
          active={actions.some(a => a.type === 'semops' && a.status === 'running')}
          loading={actions.some(a => a.type === 'semops' && a.status === 'running')}
          onClick$={() => onAction$('semops')}
          tooltip="Semiotic analysis & destructuring"
        />

        <ActionButton
          label="KG Node"
          icon="🕸️"
          color="bg-green-500/20 text-green-400"
          glowColor="rgba(34,197,94,0.5)"
          active={actions.some(a => a.type === 'kg' && a.status === 'running')}
          loading={actions.some(a => a.type === 'kg' && a.status === 'running')}
          onClick$={() => onAction$('kg')}
          tooltip="Create Knowledge Graph node"
        />

        <ActionButton
          label="Vector"
          icon="📐"
          color="bg-orange-500/20 text-orange-400"
          glowColor="rgba(249,115,22,0.5)"
          active={actions.some(a => a.type === 'vector' && a.status === 'running')}
          loading={actions.some(a => a.type === 'vector' && a.status === 'running')}
          onClick$={() => onAction$('vector')}
          tooltip="Generate vector embedding"
        />

        <ActionButton
          label="Cache IR"
          icon="💾"
          color="bg-blue-500/20 text-blue-400"
          glowColor="rgba(59,130,246,0.5)"
          active={actions.some(a => a.type === 'cache' && a.status === 'running')}
          loading={actions.some(a => a.type === 'cache' && a.status === 'running')}
          onClick$={() => onAction$('cache')}
          tooltip="Cache to local IR store"
        />

        {/* Action status badges */}
        <div class="flex-1" />
        {actions.filter(a => a.status === 'complete').map(a => (
          <span
            key={a.id}
            class="text-[10px] px-2 py-0.5 rounded-full bg-green-500/20 text-green-400 border border-green-500/30"
          >
            ✓ {a.label}
          </span>
        ))}
      </div>

      {/* Content Area */}
      <div class="relative flex-1 overflow-auto p-4">
        {loading ? (
          <div class="flex items-center justify-center h-full">
            <div class="flex flex-col items-center gap-4">
              <div class="w-16 h-16 rounded-full border-4 border-cyan-500/30 border-t-cyan-500 animate-spin" />
              <p class="text-muted-foreground animate-pulse">Loading content...</p>
            </div>
          </div>
        ) : isImage && content?.preview ? (
          <div class="flex items-center justify-center h-full">
            <img
              src={`data:${content.mimeType};base64,${content.preview}`}
              alt={content.path}
              class="max-w-full max-h-full object-contain rounded-lg shadow-[0_0_30px_rgba(6,182,212,0.2)]"
            />
          </div>
        ) : (
          <pre class={`
            text-sm font-mono whitespace-pre-wrap break-words
            ${isCode ? 'bg-black/30 p-4 rounded-lg border border-border/30' : ''}
          `}>
            {content?.content || 'No content'}
          </pre>
        )}
      </div>

      {/* Footer with file path */}
      <div class="relative p-3 border-t border-border/30 bg-muted/10">
        <code class="text-xs text-muted-foreground">
          {content?.path}
        </code>
      </div>
    </div>
  );
});

// ═══════════════════════════════════════════════════════════════════════════════
// MAIN MOUNTPOINTS PANEL
// ═══════════════════════════════════════════════════════════════════════════════

export const MountpointsPanel = component$(() => {
  // State
  const drives = useStore<MountedDrive[]>([]);
  const fileTree = useStore<Record<string, FileNode[]>>({});
  const selectedFile = useSignal<FileNode | null>(null);
  const fileContent = useSignal<FileContent | null>(null);
  const contentLoading = useSignal(false);
  const actions = useStore<PipelineAction[]>([]);
  const isMinimized = useSignal(false);
  const searchQuery = useSignal('');
  const searchResults = useStore<LocalIREntry[]>([]);
  const activeTab = useSignal<'tree' | 'search'>('tree');

  // Fetch mounted drives
  const fetchDrives = $(async () => {
    try {
      const res = await fetch('/api/cloud/status');
      if (res.ok) {
        const data = await res.json();
        drives.length = 0;
        for (const [name, info] of Object.entries(data.remotes || {})) {
          const remote = info as { configured: boolean; mounted: boolean; mountpoint?: string; account_email?: string };
          if (remote.configured) {
            drives.push({
              name,
              email: remote.account_email || name,
              mountpoint: remote.mountpoint || `/mnt/gdrive/${name}`,
              status: remote.mounted ? 'mounted' : 'unmounted',
            });
          }
        }
      }
    } catch (e) {
      console.error('Failed to fetch drives:', e);
    }
  });

  // Fetch directory contents
  const fetchDirectory = $(async (drive: string, path: string = ''): Promise<FileNode[]> => {
    try {
      const res = await fetch(`/api/cloud/browse/${drive}?path=${encodeURIComponent(path)}`);
      if (res.ok) {
        const data = await res.json();
        return (data.files || []).map((f: { Name?: string; name?: string; Size?: number; size?: number; IsDir?: boolean; isDir?: boolean; ModTime?: string }) => ({
          name: f.Name || f.name || '',
          path: path ? `${path}/${f.Name || f.name}` : (f.Name || f.name || ''),
          isDir: f.IsDir || f.isDir || false,
          size: f.Size || f.size || 0,
          modTime: f.ModTime,
          children: undefined,
          loaded: false,
          expanded: false,
        }));
      }
    } catch (e) {
      console.error('Failed to fetch directory:', e);
    }
    return [];
  });

  // Expand directory node
  const expandNode = $(async (node: FileNode, driveName: string) => {
    if (!node.isDir) return;

    node.expanded = !node.expanded;

    if (!node.loaded && node.expanded) {
      const children = await fetchDirectory(driveName, node.path);
      node.children = children;
      node.loaded = true;
    }
  });

  // Select file and load content
  const selectFile = $(async (node: FileNode, driveName: string) => {
    selectedFile.value = node;
    contentLoading.value = true;
    fileContent.value = null;
    actions.length = 0;

    try {
      const res = await fetch(`/api/cloud/file/${driveName}?path=${encodeURIComponent(node.path)}`);
      if (res.ok) {
        const data = await res.json();
        fileContent.value = {
          path: node.path,
          content: data.content || '',
          mimeType: data.mimeType || 'text/plain',
          size: data.size || 0,
          preview: data.preview,
        };
      }
    } catch (e) {
      console.error('Failed to load file:', e);
    }

    contentLoading.value = false;
  });

  // Pipeline action handlers
  const handlePipelineAction = $(async (actionType: string) => {
    if (!fileContent.value) return;

    const actionId = `${actionType}-${Date.now()}`;
    const action: PipelineAction = {
      id: actionId,
      type: actionType as PipelineAction['type'],
      status: 'running',
      label: actionType.toUpperCase(),
    };
    actions.push(action);

    try {
      switch (actionType) {
        case 'strp': {
          // Create STRp distillation request
          const res = await fetch('/api/emit', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
              topic: 'strp.request.distill',
              payload: {
                source: 'mountpoints',
                path: fileContent.value.path,
                content_preview: fileContent.value.content.slice(0, 500),
                request_type: 'categorize',
              },
            }),
          });
          action.status = res.ok ? 'complete' : 'error';
          break;
        }

        case 'semops': {
          // Semiotic analysis request
          const res = await fetch('/api/emit', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
              topic: 'semops.analyze',
              payload: {
                path: fileContent.value.path,
                content: fileContent.value.content.slice(0, 2000),
                operations: ['destructure', 'label', 'learn'],
              },
            }),
          });
          action.status = res.ok ? 'complete' : 'error';
          break;
        }

        case 'kg': {
          // Knowledge Graph node creation
          const res = await fetch('/api/emit', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
              topic: 'kg.node.create',
              payload: {
                type: 'artifact',
                label: fileContent.value.path.split('/').pop(),
                path: fileContent.value.path,
                source: 'cloud_drive',
                tags: ['imported', 'gdrive'],
              },
            }),
          });
          action.status = res.ok ? 'complete' : 'error';
          break;
        }

        case 'vector': {
          // Vector encoding request
          const res = await fetch('/api/emit', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
              topic: 'rag.index.add',
              payload: {
                path: fileContent.value.path,
                content: fileContent.value.content,
                source: 'cloud_drive',
                metadata: {
                  mimeType: fileContent.value.mimeType,
                  size: fileContent.value.size,
                },
              },
            }),
          });
          action.status = res.ok ? 'complete' : 'error';
          break;
        }

        case 'cache': {
          // Cache to local IR (IndexedDB)
          const keywords = extractKeywords(fileContent.value.content);
          await cacheToLocalIR({
            path: fileContent.value.path,
            content: fileContent.value.content,
            keywords,
            timestamp: Date.now(),
            source: 'cloud_drive',
          });
          action.status = 'complete';
          break;
        }
      }
    } catch (e) {
      console.error(`Pipeline action ${actionType} failed:`, e);
      action.status = 'error';
    }
  });

  // Search local IR
  const handleSearch = $(async () => {
    if (!searchQuery.value.trim()) {
      searchResults.length = 0;
      return;
    }
    const results = await searchLocalIR(searchQuery.value);
    searchResults.length = 0;
    results.forEach(r => searchResults.push(r));
  });

  // Initialize
  useVisibleTask$(({ cleanup }) => {
    fetchDrives();
    const interval = setInterval(fetchDrives, 30000);
    cleanup(() => clearInterval(interval));
  });

  // Load root directories for mounted drives
  useVisibleTask$(({ track }) => {
    track(() => drives.length);
    drives.forEach(async (drive) => {
      if (drive.status === 'mounted' && !fileTree[drive.name]) {
        const root = await fetchDirectory(drive.name, '');
        fileTree[drive.name] = root;
      }
    });
  });

  const mountedDrives = drives.filter(d => d.status === 'mounted');

  return (
    <>
      {/* Floating Panel */}
      <div class={`
        fixed right-4 top-20 z-40
        transition-all duration-500 ease-out
        ${isMinimized.value ? 'w-12' : 'w-80'}
      `}>
        {/* Header */}
        <div class={`
          flex items-center gap-2 p-3 rounded-t-xl
          bg-gradient-to-r from-cyan-500/10 via-purple-500/10 to-pink-500/10
          border border-cyan-500/30 border-b-0
          backdrop-blur-lg
          cursor-pointer
          shadow-[0_-10px_40px_rgba(6,182,212,0.1)]
        `}
        onClick$={() => isMinimized.value = !isMinimized.value}
        >
          <span class="text-xl animate-pulse drop-shadow-[0_0_10px_rgba(6,182,212,0.5)]">
            ☁️
          </span>
          {!isMinimized.value && (
            <>
              <span class="flex-1 font-semibold text-sm">Mountpoints</span>
              <span class="text-xs text-muted-foreground">
                {mountedDrives.length} mounted
              </span>
            </>
          )}
        </div>

        {/* Body */}
        {!isMinimized.value && (
          <div class={`
            bg-background/95 backdrop-blur-xl
            border border-cyan-500/20 border-t-0 rounded-b-xl
            shadow-[0_20px_60px_rgba(6,182,212,0.1),0_10px_30px_rgba(168,85,247,0.1)]
            max-h-[60vh] overflow-hidden flex flex-col
          `}>
            {/* Tabs */}
            <div class="flex border-b border-border/30">
              <Button
                variant={activeTab.value === 'tree' ? 'tonal' : 'text'}
                class={`flex-1 rounded-none text-xs font-medium border-b-2 transition-colors ${
                  activeTab.value === 'tree'
                    ? 'text-cyan-400 border-cyan-400'
                    : 'text-muted-foreground border-transparent'
                }`}
                onClick$={() => activeTab.value = 'tree'}
              >
                📂 Files
              </Button>
              <Button
                variant={activeTab.value === 'search' ? 'tonal' : 'text'}
                class={`flex-1 rounded-none text-xs font-medium border-b-2 transition-colors ${
                  activeTab.value === 'search'
                    ? 'text-cyan-400 border-cyan-400'
                    : 'text-muted-foreground border-transparent'
                }`}
                onClick$={() => activeTab.value = 'search'}
              >
                🔍 Local IR
              </Button>
            </div>

            {/* Tree View */}
            {activeTab.value === 'tree' && (
              <div class="flex-1 overflow-auto p-2">
                {mountedDrives.length === 0 ? (
                  <div class="text-center py-8 text-muted-foreground text-sm">
                    <p>No drives mounted</p>
                    <p class="text-xs mt-2">Go to ☁️ Cloud tab to connect</p>
                  </div>
                ) : (
                  mountedDrives.map((drive) => (
                    <div key={drive.name} class="mb-4">
                      <div class="flex items-center gap-2 px-2 py-1.5 bg-muted/30 rounded-lg mb-1">
                        <span class="text-base">💾</span>
                        <span class="flex-1 text-xs font-medium truncate">{drive.email}</span>
                        <span class="w-2 h-2 rounded-full bg-green-400 animate-pulse" />
                      </div>
                      <div class="pl-2">
                        {fileTree[drive.name]?.map((node) => (
                          <FileTreeNode
                            key={node.path}
                            node={node}
                            depth={0}
                            selectedPath={selectedFile.value?.path || ''}
                            onSelect$={(n) => selectFile(n, drive.name)}
                            onExpand$={(n) => expandNode(n, drive.name)}
                          />
                        ))}
                      </div>
                    </div>
                  ))
                )}
              </div>
            )}

            {/* Search View */}
            {activeTab.value === 'search' && (
              <div class="flex-1 overflow-auto p-2">
                <div class="flex gap-2 mb-3">
                  <Input
                    type="search"
                    label=""
                    placeholder="Search local cache..."
                    class="flex-1"
                    value={searchQuery.value}
                    onInput$={(_, el) => searchQuery.value = el.value}
                    onKeyDown$={(e) => e.key === 'Enter' && handleSearch()}
                  />
                  <Button
                    variant="tonal"
                    class="bg-cyan-500/20 text-cyan-400 hover:bg-cyan-500/30"
                    onClick$={handleSearch}
                  >
                    🔍
                  </Button>
                </div>

                {searchResults.length > 0 ? (
                  <div class="space-y-2">
                    {searchResults.map((result, i) => (
                      <div
                        key={i}
                        class="p-2 rounded-lg bg-muted/20 border border-border/30 hover:border-cyan-500/30 cursor-pointer"
                      >
                        <div class="text-xs font-medium truncate">{result.path}</div>
                        <div class="text-[10px] text-muted-foreground line-clamp-2 mt-1">
                          {result.content.slice(0, 100)}...
                        </div>
                        <div class="flex gap-1 mt-1 flex-wrap">
                          {result.keywords.slice(0, 5).map(k => (
                            <span key={k} class="text-[9px] px-1 py-0.5 bg-cyan-500/10 text-cyan-400 rounded">
                              {k}
                            </span>
                          ))}
                        </div>
                      </div>
                    ))}
                  </div>
                ) : (
                  <div class="text-center py-8 text-muted-foreground text-sm">
                    <p>Search your local document cache</p>
                    <p class="text-xs mt-2">Documents cached via "Cache IR" action</p>
                  </div>
                )}
              </div>
            )}
          </div>
        )}
      </div>

      {/* Content Viewer Overlay */}
      <ContentViewer
        content={fileContent.value}
        loading={contentLoading.value}
        actions={actions}
        onClose$={() => {
          selectedFile.value = null;
          fileContent.value = null;
        }}
        onAction$={handlePipelineAction}
      />

      {/* CSS for gradient animation */}
      <style>
        {`
          @keyframes gradient-x {
            0%, 100% { background-position: 0% 50%; }
            50% { background-position: 100% 50%; }
          }
          .animate-gradient-x {
            background-size: 200% 200%;
            animation: gradient-x 3s ease infinite;
          }
        `}
      </style>
    </>
  );
});

export default MountpointsPanel;