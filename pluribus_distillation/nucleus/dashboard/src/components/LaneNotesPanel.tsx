/**
 * LaneNotesPanel - Notes and comments for lanes
 *
 * Phase 3, Iteration 21 of OITERATE lanes-widget-enhancement
 *
 * Features:
 * - Add notes with markdown support
 * - Threaded replies
 * - @mentions for agents
 * - Timestamp display
 * - Pin important notes
 * - Emit bus events for note changes
 */

import {
  component$,
  useSignal,
  useComputed$,
  $,
  type QRL,
} from '@builder.io/qwik';

// ============================================================================
// Types
// ============================================================================

export interface Note {
  id: string;
  content: string;
  author: string;
  createdAt: string;
  updatedAt?: string;
  pinned?: boolean;
  parentId?: string;
  mentions?: string[];
}

export interface NoteEvent {
  type: 'add' | 'edit' | 'delete' | 'pin' | 'unpin';
  laneId: string;
  note: Note;
  actor: string;
  timestamp: string;
}

export interface LaneNotesPanelProps {
  /** Lane ID */
  laneId: string;
  /** Lane name for display */
  laneName: string;
  /** Current notes */
  notes: Note[];
  /** Available agents for @mentions */
  availableAgents?: string[];
  /** Callback when notes change */
  onNoteChange$?: QRL<(event: NoteEvent) => void>;
  /** Current actor/author */
  actor?: string;
  /** Compact mode */
  compact?: boolean;
}

// ============================================================================
// Helpers
// ============================================================================

function formatTimestamp(ts: string): string {
  try {
    const date = new Date(ts);
    const now = new Date();
    const diffMs = now.getTime() - date.getTime();
    const diffMins = Math.floor(diffMs / (1000 * 60));
    const diffHours = Math.floor(diffMins / 60);
    const diffDays = Math.floor(diffHours / 24);

    if (diffMins < 1) return 'just now';
    if (diffMins < 60) return `${diffMins}m ago`;
    if (diffHours < 24) return `${diffHours}h ago`;
    if (diffDays < 7) return `${diffDays}d ago`;
    return date.toLocaleDateString('en-US', { month: 'short', day: 'numeric' });
  } catch {
    return ts.slice(0, 10);
  }
}

function generateId(): string {
  return `note-${Date.now().toString(36)}-${Math.random().toString(36).slice(2, 6)}`;
}

function extractMentions(content: string): string[] {
  const mentions = content.match(/@[\w-]+/g) || [];
  return mentions.map(m => m.slice(1));
}

function highlightMentions(content: string): string {
  return content.replace(/@([\w-]+)/g, '<span class="text-primary font-medium">@$1</span>');
}

// ============================================================================
// Component
// ============================================================================

export const LaneNotesPanel = component$<LaneNotesPanelProps>(({
  laneId,
  laneName,
  notes: initialNotes,
  availableAgents = [],
  onNoteChange$,
  actor = 'dashboard',
  compact = false,
}) => {
  // State
  const notes = useSignal<Note[]>(initialNotes);
  const newContent = useSignal('');
  const replyingTo = useSignal<string | null>(null);
  const editingId = useSignal<string | null>(null);
  const editContent = useSignal('');
  const showMentionDropdown = useSignal(false);
  const mentionFilter = useSignal('');

  // Computed
  const rootNotes = useComputed$(() =>
    notes.value.filter(n => !n.parentId).sort((a, b) => {
      // Pinned first, then by date
      if (a.pinned && !b.pinned) return -1;
      if (!a.pinned && b.pinned) return 1;
      return new Date(b.createdAt).getTime() - new Date(a.createdAt).getTime();
    })
  );

  const getReplies = $((parentId: string): Note[] =>
    notes.value.filter(n => n.parentId === parentId)
      .sort((a, b) => new Date(a.createdAt).getTime() - new Date(b.createdAt).getTime())
  );

  const stats = useComputed$(() => ({
    total: notes.value.length,
    pinned: notes.value.filter(n => n.pinned).length,
    threads: rootNotes.value.length,
  }));

  // Emit event helper
  const emitEvent = $(async (type: NoteEvent['type'], note: Note) => {
    const event: NoteEvent = {
      type,
      laneId,
      note,
      actor,
      timestamp: new Date().toISOString(),
    };

    console.log(`[LaneNotesPanel] Emitting: operator.lanes.note.${type}`, event);

    if (onNoteChange$) {
      await onNoteChange$(event);
    }
  });

  // Add new note
  const addNote = $(async (parentId?: string) => {
    const content = newContent.value.trim();
    if (!content) return;

    const newNote: Note = {
      id: generateId(),
      content,
      author: actor,
      createdAt: new Date().toISOString(),
      parentId,
      mentions: extractMentions(content),
    };

    notes.value = [...notes.value, newNote];
    await emitEvent('add', newNote);

    newContent.value = '';
    replyingTo.value = null;
  });

  // Save edit
  const saveEdit = $(async () => {
    if (!editingId.value || !editContent.value.trim()) return;

    const idx = notes.value.findIndex(n => n.id === editingId.value);
    if (idx === -1) return;

    const updated: Note = {
      ...notes.value[idx],
      content: editContent.value.trim(),
      updatedAt: new Date().toISOString(),
      mentions: extractMentions(editContent.value),
    };

    const newNotes = [...notes.value];
    newNotes[idx] = updated;
    notes.value = newNotes;

    await emitEvent('edit', updated);

    editingId.value = null;
    editContent.value = '';
  });

  // Delete note
  const deleteNote = $(async (noteId: string) => {
    const note = notes.value.find(n => n.id === noteId);
    if (!note) return;

    // Delete note and all replies
    notes.value = notes.value.filter(n => n.id !== noteId && n.parentId !== noteId);
    await emitEvent('delete', note);
  });

  // Toggle pin
  const togglePin = $(async (noteId: string) => {
    const idx = notes.value.findIndex(n => n.id === noteId);
    if (idx === -1) return;

    const updated = { ...notes.value[idx], pinned: !notes.value[idx].pinned };
    const newNotes = [...notes.value];
    newNotes[idx] = updated;
    notes.value = newNotes;

    await emitEvent(updated.pinned ? 'pin' : 'unpin', updated);
  });

  // Handle @mention input
  const handleInput = $((e: InputEvent) => {
    const target = e.target as HTMLTextAreaElement;
    const value = target.value;
    newContent.value = value;

    // Check for @ trigger
    const lastAt = value.lastIndexOf('@');
    if (lastAt !== -1 && (lastAt === 0 || value[lastAt - 1] === ' ')) {
      const partial = value.slice(lastAt + 1).split(/\s/)[0];
      mentionFilter.value = partial;
      showMentionDropdown.value = true;
    } else {
      showMentionDropdown.value = false;
    }
  });

  // Insert mention
  const insertMention = $((agentId: string) => {
    const lastAt = newContent.value.lastIndexOf('@');
    if (lastAt !== -1) {
      newContent.value = newContent.value.slice(0, lastAt) + `@${agentId} `;
    }
    showMentionDropdown.value = false;
  });

  // Render note component
  const NoteItem = $((note: Note, isReply: boolean = false) => (
    <div
      class={`${isReply ? 'ml-6 border-l-2 border-border/30 pl-3' : ''} ${
        note.pinned ? 'bg-amber-500/5 border-l-2 border-l-amber-500' : ''
      }`}
    >
      <div class="p-3">
        {/* Header */}
        <div class="flex items-center justify-between mb-1">
          <div class="flex items-center gap-2">
            <span class="text-xs font-medium text-foreground">@{note.author}</span>
            <span class="text-[9px] text-muted-foreground">{formatTimestamp(note.createdAt)}</span>
            {note.updatedAt && (
              <span class="text-[8px] text-muted-foreground/50">(edited)</span>
            )}
            {note.pinned && (
              <span class="text-[9px] text-amber-400">📌 pinned</span>
            )}
          </div>
          <div class="flex items-center gap-1">
            {!isReply && (
              <button
                onClick$={() => togglePin(note.id)}
                class={`w-5 h-5 flex items-center justify-center rounded text-[10px] transition-colors ${
                  note.pinned
                    ? 'bg-amber-500/20 text-amber-400'
                    : 'bg-muted/20 text-muted-foreground hover:bg-muted/40'
                }`}
                title={note.pinned ? 'Unpin' : 'Pin'}
              >
                📌
              </button>
            )}
            <button
              onClick$={() => {
                editingId.value = note.id;
                editContent.value = note.content;
              }}
              class="w-5 h-5 flex items-center justify-center rounded bg-muted/20 text-muted-foreground hover:bg-muted/40 text-[10px]"
              title="Edit"
            >
              ✏️
            </button>
            <button
              onClick$={() => deleteNote(note.id)}
              class="w-5 h-5 flex items-center justify-center rounded bg-red-500/20 text-red-400 hover:bg-red-500/30 text-[10px]"
              title="Delete"
            >
              ×
            </button>
          </div>
        </div>

        {/* Content */}
        {editingId.value === note.id ? (
          <div class="space-y-2">
            <textarea
              value={editContent.value}
              onInput$={(e) => { editContent.value = (e.target as HTMLTextAreaElement).value; }}
              class="w-full px-2 py-1.5 text-xs rounded bg-card border border-border/50 text-foreground focus:outline-none focus:border-primary/50 resize-none"
              rows={3}
            />
            <div class="flex gap-2">
              <button
                onClick$={saveEdit}
                class="px-2 py-1 text-[10px] rounded bg-primary text-primary-foreground"
              >
                Save
              </button>
              <button
                onClick$={() => { editingId.value = null; }}
                class="px-2 py-1 text-[10px] rounded bg-muted/30 text-muted-foreground"
              >
                Cancel
              </button>
            </div>
          </div>
        ) : (
          <div
            class="text-xs text-foreground/90 whitespace-pre-wrap"
            dangerouslySetInnerHTML={highlightMentions(note.content)}
          />
        )}

        {/* Reply button */}
        {!isReply && editingId.value !== note.id && (
          <button
            onClick$={() => { replyingTo.value = note.id; }}
            class="mt-2 text-[9px] text-muted-foreground hover:text-foreground transition-colors"
          >
            ↪ Reply
          </button>
        )}
      </div>
    </div>
  ));

  return (
    <div class="rounded-lg border border-border bg-card">
      {/* Header */}
      <div class="flex items-center justify-between p-3 border-b border-border/50">
        <div class="flex items-center gap-2">
          <span class="text-xs font-semibold text-muted-foreground">NOTES</span>
          <span class="text-[10px] px-2 py-0.5 rounded bg-blue-500/20 text-blue-400 border border-blue-500/30">
            {stats.value.total} notes
          </span>
          {stats.value.pinned > 0 && (
            <span class="text-[10px] px-2 py-0.5 rounded bg-amber-500/20 text-amber-400 border border-amber-500/30">
              {stats.value.pinned} pinned
            </span>
          )}
        </div>
        <span class="text-[9px] text-muted-foreground">{laneName}</span>
      </div>

      {/* Notes list */}
      <div class={`overflow-y-auto ${compact ? 'max-h-[200px]' : 'max-h-[350px]'}`}>
        {rootNotes.value.length === 0 ? (
          <div class="p-6 text-center">
            <div class="text-2xl mb-2">📝</div>
            <div class="text-xs text-muted-foreground">No notes yet</div>
          </div>
        ) : (
          rootNotes.value.map(note => (
            <div key={note.id} class="border-b border-border/30">
              {NoteItem(note, false)}

              {/* Reply form */}
              {replyingTo.value === note.id && (
                <div class="ml-6 p-3 bg-muted/10 border-l-2 border-primary/30">
                  <textarea
                    value={newContent.value}
                    onInput$={handleInput}
                    placeholder="Write a reply..."
                    class="w-full px-2 py-1.5 text-xs rounded bg-card border border-border/50 text-foreground placeholder:text-muted-foreground/50 focus:outline-none focus:border-primary/50 resize-none"
                    rows={2}
                  />
                  <div class="flex gap-2 mt-2">
                    <button
                      onClick$={() => addNote(note.id)}
                      disabled={!newContent.value.trim()}
                      class="px-2 py-1 text-[10px] rounded bg-primary text-primary-foreground disabled:opacity-50"
                    >
                      Reply
                    </button>
                    <button
                      onClick$={() => { replyingTo.value = null; newContent.value = ''; }}
                      class="px-2 py-1 text-[10px] rounded bg-muted/30 text-muted-foreground"
                    >
                      Cancel
                    </button>
                  </div>
                </div>
              )}

              {/* Replies */}
              {notes.value.filter(r => r.parentId === note.id).map(reply => (
                <div key={reply.id}>{NoteItem(reply, true)}</div>
              ))}
            </div>
          ))
        )}
      </div>

      {/* Add note form */}
      <div class="p-3 border-t border-border/50 relative">
        <textarea
          value={newContent.value}
          onInput$={handleInput}
          placeholder="Add a note... (use @agent to mention)"
          class="w-full px-2 py-1.5 text-xs rounded bg-muted/10 border border-border/30 text-foreground placeholder:text-muted-foreground/50 focus:outline-none focus:border-primary/50 resize-none"
          rows={2}
        />

        {/* Mention dropdown */}
        {showMentionDropdown.value && availableAgents.length > 0 && (
          <div class="absolute bottom-full left-3 mb-1 w-48 rounded border border-border bg-card shadow-lg max-h-32 overflow-y-auto z-10">
            {availableAgents
              .filter(a => a.toLowerCase().includes(mentionFilter.value.toLowerCase()))
              .map(agent => (
                <button
                  key={agent}
                  onClick$={() => insertMention(agent)}
                  class="w-full px-2 py-1.5 text-left text-xs text-foreground hover:bg-muted/20 transition-colors"
                >
                  @{agent}
                </button>
              ))
            }
          </div>
        )}

        <div class="flex justify-between items-center mt-2">
          <span class="text-[9px] text-muted-foreground">
            Posting as @{actor}
          </span>
          <button
            onClick$={() => addNote()}
            disabled={!newContent.value.trim()}
            class="px-3 py-1.5 text-[10px] rounded bg-primary text-primary-foreground font-medium disabled:opacity-50 disabled:cursor-not-allowed"
          >
            Add Note
          </button>
        </div>
      </div>
    </div>
  );
});

export default LaneNotesPanel;
