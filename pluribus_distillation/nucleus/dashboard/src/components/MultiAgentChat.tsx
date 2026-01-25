/**
 * MultiAgentChat - Agent-to-agent communication interface
 *
 * Phase 5, Iteration 44 of OITERATE lanes-widget-enhancement
 *
 * Features:
 * - Multi-agent chat interface
 * - Threaded conversations
 * - Broadcast and direct messages
 * - Message status tracking
 * - Channel management
 */

import {
  component$,
  useSignal,
  useStore,
  useComputed$,
  $,
  type QRL,
} from '@builder.io/qwik';

// ============================================================================
// Types
// ============================================================================

export interface ChatMessage {
  id: string;
  channelId: string;
  fromAgent: string;
  fromAgentName: string;
  content: string;
  timestamp: string;
  type: 'message' | 'system' | 'task' | 'alert';
  replyTo?: string;
  reactions?: { emoji: string; agents: string[] }[];
  status: 'sent' | 'delivered' | 'read';
}

export interface ChatChannel {
  id: string;
  name: string;
  type: 'broadcast' | 'direct' | 'group';
  participants: string[];
  unreadCount: number;
  lastMessage?: ChatMessage;
}

export interface ChatAgent {
  id: string;
  name: string;
  status: 'online' | 'busy' | 'offline';
  color: string;
}

export interface MultiAgentChatProps {
  /** Current agent ID (viewer) */
  currentAgentId: string;
  /** Available channels */
  channels: ChatChannel[];
  /** All messages */
  messages: ChatMessage[];
  /** Participating agents */
  agents: ChatAgent[];
  /** Callback when message is sent */
  onSendMessage$?: QRL<(channelId: string, content: string, replyTo?: string) => void>;
  /** Callback when channel is created */
  onCreateChannel$?: QRL<(name: string, type: ChatChannel['type'], participants: string[]) => void>;
}

// ============================================================================
// Helpers
// ============================================================================

function getStatusColor(status: string): string {
  switch (status) {
    case 'online': return 'bg-emerald-400';
    case 'busy': return 'bg-amber-400';
    case 'offline': return 'bg-gray-400';
    default: return 'bg-gray-400';
  }
}

function formatTime(dateStr: string): string {
  try {
    const date = new Date(dateStr);
    const now = new Date();
    const diffMs = now.getTime() - date.getTime();
    const diffMins = Math.floor(diffMs / (1000 * 60));

    if (diffMins < 1) return 'now';
    if (diffMins < 60) return `${diffMins}m`;
    if (diffMins < 1440) return date.toLocaleTimeString('en-US', { hour: '2-digit', minute: '2-digit' });
    return date.toLocaleDateString('en-US', { month: 'short', day: 'numeric' });
  } catch {
    return dateStr;
  }
}

function getTypeIcon(type: string): string {
  switch (type) {
    case 'system': return '⚙';
    case 'task': return '📋';
    case 'alert': return '⚠';
    default: return '';
  }
}

// ============================================================================
// Component
// ============================================================================

export const MultiAgentChat = component$<MultiAgentChatProps>(({
  currentAgentId,
  channels,
  messages,
  agents,
  onSendMessage$,
  onCreateChannel$,
}) => {
  // State
  const selectedChannelId = useSignal<string | null>(channels[0]?.id || null);
  const newMessage = useSignal('');
  const replyingTo = useSignal<string | null>(null);
  const showCreateChannel = useSignal(false);

  const newChannel = useStore({
    name: '',
    type: 'group' as ChatChannel['type'],
    participants: [] as string[],
  });

  // Computed
  const selectedChannel = useComputed$(() =>
    channels.find(c => c.id === selectedChannelId.value)
  );

  const channelMessages = useComputed$(() =>
    messages
      .filter(m => m.channelId === selectedChannelId.value)
      .sort((a, b) => new Date(a.timestamp).getTime() - new Date(b.timestamp).getTime())
  );

  const replyingToMessage = useComputed$(() =>
    replyingTo.value ? messages.find(m => m.id === replyingTo.value) : null
  );

  const totalUnread = useComputed$(() =>
    channels.reduce((sum, c) => sum + c.unreadCount, 0)
  );

  // Actions
  const sendMessage = $(async () => {
    if (!newMessage.value.trim() || !selectedChannelId.value) return;

    if (onSendMessage$) {
      await onSendMessage$(selectedChannelId.value, newMessage.value, replyingTo.value || undefined);
    }

    newMessage.value = '';
    replyingTo.value = null;
  });

  const createChannel = $(async () => {
    if (!newChannel.name || newChannel.participants.length === 0) return;

    if (onCreateChannel$) {
      await onCreateChannel$(newChannel.name, newChannel.type, newChannel.participants);
    }

    showCreateChannel.value = false;
    newChannel.name = '';
    newChannel.type = 'group';
    newChannel.participants = [];
  });

  const toggleParticipant = $((agentId: string) => {
    if (newChannel.participants.includes(agentId)) {
      newChannel.participants = newChannel.participants.filter(id => id !== agentId);
    } else {
      newChannel.participants = [...newChannel.participants, agentId];
    }
  });

  return (
    <div class="rounded-lg border border-border bg-card">
      {/* Header */}
      <div class="flex items-center justify-between p-3 border-b border-border/50">
        <div class="flex items-center gap-2">
          <span class="text-xs font-semibold text-muted-foreground">AGENT CHAT</span>
          {totalUnread.value > 0 && (
            <span class="text-[9px] px-2 py-0.5 rounded-full bg-red-500/20 text-red-400">
              {totalUnread.value}
            </span>
          )}
        </div>
        <button
          onClick$={() => { showCreateChannel.value = true; }}
          class="text-[10px] px-2 py-1 rounded bg-primary/20 text-primary hover:bg-primary/30 transition-colors"
        >
          + Channel
        </button>
      </div>

      {/* Main content */}
      <div class="grid grid-cols-4 gap-0 min-h-[350px]">
        {/* Channel list */}
        <div class="border-r border-border/30 max-h-[400px] overflow-y-auto">
          {/* Online agents */}
          <div class="p-2 border-b border-border/30">
            <div class="text-[8px] font-semibold text-muted-foreground mb-2">AGENTS</div>
            <div class="flex flex-wrap gap-1">
              {agents.filter(a => a.status !== 'offline').map(agent => (
                <div
                  key={agent.id}
                  class="flex items-center gap-1 px-1.5 py-0.5 rounded bg-muted/20"
                  title={agent.status}
                >
                  <div class={`w-1.5 h-1.5 rounded-full ${getStatusColor(agent.status)}`} />
                  <span class="text-[8px]" style={{ color: agent.color }}>
                    {agent.name}
                  </span>
                </div>
              ))}
            </div>
          </div>

          {/* Channels */}
          <div class="p-2">
            <div class="text-[8px] font-semibold text-muted-foreground mb-2">CHANNELS</div>
            <div class="space-y-1">
              {channels.map(channel => (
                <div
                  key={channel.id}
                  onClick$={() => { selectedChannelId.value = channel.id; }}
                  class={`p-2 rounded cursor-pointer transition-colors ${
                    selectedChannelId.value === channel.id
                      ? 'bg-primary/10 border border-primary/30'
                      : 'hover:bg-muted/10'
                  }`}
                >
                  <div class="flex items-center justify-between">
                    <div class="flex items-center gap-1">
                      <span class="text-[9px]">
                        {channel.type === 'broadcast' ? '📢' :
                         channel.type === 'direct' ? '💬' : '#'}
                      </span>
                      <span class="text-[10px] font-medium text-foreground">{channel.name}</span>
                    </div>
                    {channel.unreadCount > 0 && (
                      <span class="text-[8px] px-1.5 py-0.5 rounded-full bg-red-500/20 text-red-400">
                        {channel.unreadCount}
                      </span>
                    )}
                  </div>
                  {channel.lastMessage && (
                    <div class="text-[8px] text-muted-foreground mt-1 truncate">
                      <span style={{ color: agents.find(a => a.id === channel.lastMessage?.fromAgent)?.color }}>
                        {channel.lastMessage.fromAgentName}:
                      </span>{' '}
                      {channel.lastMessage.content.slice(0, 30)}
                    </div>
                  )}
                </div>
              ))}
            </div>
          </div>
        </div>

        {/* Chat area */}
        <div class="col-span-3 flex flex-col">
          {selectedChannel.value ? (
            <>
              {/* Channel header */}
              <div class="p-3 border-b border-border/30 bg-muted/5">
                <div class="flex items-center justify-between">
                  <div class="flex items-center gap-2">
                    <span class="text-[10px]">
                      {selectedChannel.value.type === 'broadcast' ? '📢' :
                       selectedChannel.value.type === 'direct' ? '💬' : '#'}
                    </span>
                    <span class="text-xs font-medium text-foreground">{selectedChannel.value.name}</span>
                  </div>
                  <span class="text-[9px] text-muted-foreground">
                    {selectedChannel.value.participants.length} participants
                  </span>
                </div>
              </div>

              {/* Messages */}
              <div class="flex-grow overflow-y-auto p-3 space-y-3 max-h-[260px]">
                {channelMessages.value.map(message => {
                  const agent = agents.find(a => a.id === message.fromAgent);
                  const isOwn = message.fromAgent === currentAgentId;
                  const replyMsg = message.replyTo
                    ? messages.find(m => m.id === message.replyTo)
                    : null;

                  return (
                    <div
                      key={message.id}
                      class={`flex ${isOwn ? 'justify-end' : 'justify-start'}`}
                    >
                      <div class={`max-w-[80%] ${isOwn ? 'order-2' : ''}`}>
                        {/* Reply reference */}
                        {replyMsg && (
                          <div class="text-[8px] text-muted-foreground mb-1 pl-2 border-l-2 border-muted">
                            ↩ {replyMsg.fromAgentName}: {replyMsg.content.slice(0, 30)}...
                          </div>
                        )}

                        <div
                          class={`rounded-lg p-2 ${
                            message.type === 'system' ? 'bg-blue-500/10 border border-blue-500/30' :
                            message.type === 'alert' ? 'bg-red-500/10 border border-red-500/30' :
                            message.type === 'task' ? 'bg-purple-500/10 border border-purple-500/30' :
                            isOwn ? 'bg-primary/20' : 'bg-muted/20'
                          }`}
                        >
                          {!isOwn && (
                            <div class="flex items-center gap-1 mb-1">
                              <span class="text-[9px] font-medium" style={{ color: agent?.color }}>
                                {message.fromAgentName}
                              </span>
                              {message.type !== 'message' && (
                                <span class="text-[8px]">{getTypeIcon(message.type)}</span>
                              )}
                            </div>
                          )}
                          <div class="text-[10px] text-foreground">{message.content}</div>
                          <div class="flex items-center justify-between mt-1">
                            <div class="flex items-center gap-1">
                              {message.reactions?.map((r, i) => (
                                <span key={i} class="text-[9px] bg-muted/30 px-1 rounded">
                                  {r.emoji} {r.agents.length}
                                </span>
                              ))}
                            </div>
                            <div class="flex items-center gap-1 text-[8px] text-muted-foreground">
                              <span>{formatTime(message.timestamp)}</span>
                              {isOwn && (
                                <span>
                                  {message.status === 'read' ? '✓✓' :
                                   message.status === 'delivered' ? '✓' : '○'}
                                </span>
                              )}
                            </div>
                          </div>
                        </div>

                        {/* Reply button */}
                        {!isOwn && (
                          <button
                            onClick$={() => { replyingTo.value = message.id; }}
                            class="text-[8px] text-muted-foreground hover:text-foreground mt-0.5"
                          >
                            Reply
                          </button>
                        )}
                      </div>
                    </div>
                  );
                })}

                {channelMessages.value.length === 0 && (
                  <div class="flex items-center justify-center h-full text-[10px] text-muted-foreground">
                    No messages yet
                  </div>
                )}
              </div>

              {/* Reply indicator */}
              {replyingToMessage.value && (
                <div class="px-3 py-1 bg-muted/10 border-t border-border/30 flex items-center justify-between">
                  <div class="text-[9px] text-muted-foreground">
                    ↩ Replying to{' '}
                    <span class="text-foreground">{replyingToMessage.value.fromAgentName}</span>
                  </div>
                  <button
                    onClick$={() => { replyingTo.value = null; }}
                    class="text-muted-foreground hover:text-foreground"
                  >
                    ✕
                  </button>
                </div>
              )}

              {/* Input */}
              <div class="p-3 border-t border-border/30">
                <div class="flex items-center gap-2">
                  <input
                    type="text"
                    value={newMessage.value}
                    onInput$={(e) => { newMessage.value = (e.target as HTMLInputElement).value; }}
                    onKeyDown$={(e) => {
                      if (e.key === 'Enter' && !e.shiftKey) sendMessage();
                    }}
                    class="flex-grow px-3 py-2 text-[10px] rounded bg-card border border-border/50"
                    placeholder="Type a message..."
                  />
                  <button
                    onClick$={sendMessage}
                    disabled={!newMessage.value.trim()}
                    class="px-3 py-2 text-[10px] rounded bg-primary text-primary-foreground hover:bg-primary/90 disabled:opacity-50"
                  >
                    Send
                  </button>
                </div>
              </div>
            </>
          ) : (
            <div class="flex items-center justify-center h-full text-[10px] text-muted-foreground">
              Select a channel to start chatting
            </div>
          )}
        </div>
      </div>

      {/* Create Channel Modal */}
      {showCreateChannel.value && (
        <div class="fixed inset-0 bg-black/50 flex items-center justify-center z-50">
          <div class="bg-card rounded-lg border border-border p-4 w-80">
            <div class="text-xs font-semibold text-foreground mb-4">Create Channel</div>

            <div class="space-y-3">
              <div>
                <label class="text-[9px] text-muted-foreground block mb-1">Channel Name</label>
                <input
                  type="text"
                  value={newChannel.name}
                  onInput$={(e) => { newChannel.name = (e.target as HTMLInputElement).value; }}
                  class="w-full px-2 py-1.5 text-xs rounded bg-card border border-border/50"
                  placeholder="channel-name"
                />
              </div>

              <div>
                <label class="text-[9px] text-muted-foreground block mb-1">Type</label>
                <div class="flex gap-2">
                  {(['group', 'direct', 'broadcast'] as const).map(type => (
                    <button
                      key={type}
                      onClick$={() => { newChannel.type = type; }}
                      class={`flex-1 px-2 py-1 text-[9px] rounded transition-colors ${
                        newChannel.type === type
                          ? 'bg-primary/20 text-primary'
                          : 'bg-muted/20 text-muted-foreground'
                      }`}
                    >
                      {type}
                    </button>
                  ))}
                </div>
              </div>

              <div>
                <label class="text-[9px] text-muted-foreground block mb-1">Participants</label>
                <div class="flex flex-wrap gap-1 max-h-24 overflow-y-auto">
                  {agents.filter(a => a.id !== currentAgentId).map(agent => (
                    <button
                      key={agent.id}
                      onClick$={() => toggleParticipant(agent.id)}
                      class={`px-2 py-1 text-[9px] rounded transition-colors ${
                        newChannel.participants.includes(agent.id)
                          ? 'bg-primary/20 text-primary'
                          : 'bg-muted/20 text-muted-foreground'
                      }`}
                    >
                      @{agent.name}
                    </button>
                  ))}
                </div>
              </div>
            </div>

            <div class="flex items-center gap-2 mt-4">
              <button
                onClick$={() => { showCreateChannel.value = false; }}
                class="flex-1 px-3 py-1.5 text-xs rounded bg-muted/30 text-muted-foreground"
              >
                Cancel
              </button>
              <button
                onClick$={createChannel}
                disabled={!newChannel.name || newChannel.participants.length === 0}
                class="flex-1 px-3 py-1.5 text-xs rounded bg-primary text-primary-foreground disabled:opacity-50"
              >
                Create
              </button>
            </div>
          </div>
        </div>
      )}
    </div>
  );
});

export default MultiAgentChat;
