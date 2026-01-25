/**
 * Undo/Redo System for Lanes
 *
 * Phase 7, Iteration 55 of OITERATE lanes-widget-enhancement
 *
 * Features:
 * - Command pattern implementation
 * - Undo stack with configurable depth
 * - Redo support
 * - Command grouping for batch operations
 * - Keyboard shortcuts (Ctrl+Z, Ctrl+Shift+Z)
 * - Persistence support
 */

import type { Lane, LaneAction } from './store';

// ============================================================================
// Types
// ============================================================================

export interface Command {
  id: string;
  type: string;
  description: string;
  timestamp: number;
  execute: () => void;
  undo: () => void;
  data?: unknown;
}

export interface CommandGroup {
  id: string;
  description: string;
  commands: Command[];
  timestamp: number;
}

export interface UndoRedoConfig {
  /** Maximum stack size */
  maxStackSize: number;
  /** Enable keyboard shortcuts */
  enableKeyboardShortcuts: boolean;
  /** Persist to localStorage */
  persistState: boolean;
  /** Storage key */
  storageKey: string;
  /** Group timeout in ms (commands within this window are grouped) */
  groupTimeout: number;
}

export type UndoRedoListener = (canUndo: boolean, canRedo: boolean) => void;

// ============================================================================
// Default Config
// ============================================================================

const DEFAULT_CONFIG: UndoRedoConfig = {
  maxStackSize: 100,
  enableKeyboardShortcuts: true,
  persistState: false,
  storageKey: 'pblanes-undo',
  groupTimeout: 500,
};

// ============================================================================
// Command Factory
// ============================================================================

export function createLaneUpdateCommand(
  id: string,
  previousState: Partial<Lane>,
  newState: Partial<Lane>,
  applyUpdate: (id: string, changes: Partial<Lane>) => void
): Command {
  return {
    id: `lane-update-${id}-${Date.now()}`,
    type: 'lane.update',
    description: `Update lane "${id}"`,
    timestamp: Date.now(),
    execute: () => applyUpdate(id, newState),
    undo: () => applyUpdate(id, previousState),
    data: { laneId: id, previousState, newState },
  };
}

export function createLaneAddCommand(
  lane: Lane,
  addLane: (lane: Lane) => void,
  removeLane: (id: string) => void
): Command {
  return {
    id: `lane-add-${lane.id}-${Date.now()}`,
    type: 'lane.add',
    description: `Add lane "${lane.name}"`,
    timestamp: Date.now(),
    execute: () => addLane(lane),
    undo: () => removeLane(lane.id),
    data: { lane },
  };
}

export function createLaneRemoveCommand(
  lane: Lane,
  addLane: (lane: Lane) => void,
  removeLane: (id: string) => void
): Command {
  return {
    id: `lane-remove-${lane.id}-${Date.now()}`,
    type: 'lane.remove',
    description: `Remove lane "${lane.name}"`,
    timestamp: Date.now(),
    execute: () => removeLane(lane.id),
    undo: () => addLane(lane),
    data: { lane },
  };
}

// ============================================================================
// Undo/Redo Manager
// ============================================================================

export class UndoRedoManager {
  private config: UndoRedoConfig;
  private undoStack: Array<Command | CommandGroup> = [];
  private redoStack: Array<Command | CommandGroup> = [];
  private listeners: Set<UndoRedoListener> = new Set();
  private pendingGroup: Command[] = [];
  private groupTimer: ReturnType<typeof setTimeout> | null = null;
  private keyboardCleanup: (() => void) | null = null;

  constructor(config: Partial<UndoRedoConfig> = {}) {
    this.config = { ...DEFAULT_CONFIG, ...config };

    if (this.config.persistState) {
      this.loadState();
    }

    if (this.config.enableKeyboardShortcuts && typeof window !== 'undefined') {
      this.setupKeyboardShortcuts();
    }
  }

  // ============================================================================
  // Public Methods
  // ============================================================================

  /**
   * Execute a command and add to undo stack
   */
  execute(command: Command): void {
    command.execute();
    this.addToUndoStack(command);
    this.redoStack = []; // Clear redo stack on new action
    this.notifyListeners();
    this.persistIfEnabled();
  }

  /**
   * Execute multiple commands as a group
   */
  executeGroup(commands: Command[], description: string): void {
    const group: CommandGroup = {
      id: `group-${Date.now()}`,
      description,
      commands,
      timestamp: Date.now(),
    };

    for (const cmd of commands) {
      cmd.execute();
    }

    this.undoStack.push(group);
    this.trimStack(this.undoStack);
    this.redoStack = [];
    this.notifyListeners();
    this.persistIfEnabled();
  }

  /**
   * Add a command to pending group (for auto-grouping)
   */
  addToPendingGroup(command: Command): void {
    command.execute();
    this.pendingGroup.push(command);

    // Reset timer
    if (this.groupTimer) {
      clearTimeout(this.groupTimer);
    }

    this.groupTimer = setTimeout(() => {
      this.flushPendingGroup();
    }, this.config.groupTimeout);
  }

  /**
   * Flush pending group immediately
   */
  flushPendingGroup(): void {
    if (this.groupTimer) {
      clearTimeout(this.groupTimer);
      this.groupTimer = null;
    }

    if (this.pendingGroup.length === 0) return;

    if (this.pendingGroup.length === 1) {
      this.addToUndoStack(this.pendingGroup[0]);
    } else {
      const group: CommandGroup = {
        id: `group-${Date.now()}`,
        description: `${this.pendingGroup.length} changes`,
        commands: [...this.pendingGroup],
        timestamp: Date.now(),
      };
      this.undoStack.push(group);
      this.trimStack(this.undoStack);
    }

    this.pendingGroup = [];
    this.redoStack = [];
    this.notifyListeners();
    this.persistIfEnabled();
  }

  /**
   * Undo the last action
   */
  undo(): boolean {
    this.flushPendingGroup();

    if (this.undoStack.length === 0) return false;

    const item = this.undoStack.pop()!;

    if (this.isGroup(item)) {
      // Undo in reverse order
      for (let i = item.commands.length - 1; i >= 0; i--) {
        item.commands[i].undo();
      }
    } else {
      item.undo();
    }

    this.redoStack.push(item);
    this.notifyListeners();
    this.persistIfEnabled();

    return true;
  }

  /**
   * Redo the last undone action
   */
  redo(): boolean {
    this.flushPendingGroup();

    if (this.redoStack.length === 0) return false;

    const item = this.redoStack.pop()!;

    if (this.isGroup(item)) {
      for (const cmd of item.commands) {
        cmd.execute();
      }
    } else {
      item.execute();
    }

    this.undoStack.push(item);
    this.notifyListeners();
    this.persistIfEnabled();

    return true;
  }

  /**
   * Check if undo is available
   */
  canUndo(): boolean {
    return this.undoStack.length > 0 || this.pendingGroup.length > 0;
  }

  /**
   * Check if redo is available
   */
  canRedo(): boolean {
    return this.redoStack.length > 0;
  }

  /**
   * Get undo stack (for display)
   */
  getUndoStack(): Array<{ id: string; description: string; timestamp: number }> {
    return this.undoStack.map(item => ({
      id: item.id,
      description: this.isGroup(item) ? item.description : item.description,
      timestamp: item.timestamp,
    }));
  }

  /**
   * Get redo stack (for display)
   */
  getRedoStack(): Array<{ id: string; description: string; timestamp: number }> {
    return this.redoStack.map(item => ({
      id: item.id,
      description: this.isGroup(item) ? item.description : item.description,
      timestamp: item.timestamp,
    }));
  }

  /**
   * Subscribe to undo/redo state changes
   */
  subscribe(listener: UndoRedoListener): () => void {
    this.listeners.add(listener);
    // Immediate notification
    listener(this.canUndo(), this.canRedo());
    return () => this.listeners.delete(listener);
  }

  /**
   * Clear all history
   */
  clear(): void {
    this.undoStack = [];
    this.redoStack = [];
    this.pendingGroup = [];
    if (this.groupTimer) {
      clearTimeout(this.groupTimer);
      this.groupTimer = null;
    }
    this.notifyListeners();
    this.persistIfEnabled();
  }

  /**
   * Dispose resources
   */
  dispose(): void {
    if (this.keyboardCleanup) {
      this.keyboardCleanup();
      this.keyboardCleanup = null;
    }
    if (this.groupTimer) {
      clearTimeout(this.groupTimer);
      this.groupTimer = null;
    }
    this.listeners.clear();
  }

  // ============================================================================
  // Private Methods
  // ============================================================================

  private isGroup(item: Command | CommandGroup): item is CommandGroup {
    return 'commands' in item;
  }

  private addToUndoStack(command: Command): void {
    this.undoStack.push(command);
    this.trimStack(this.undoStack);
  }

  private trimStack(stack: Array<Command | CommandGroup>): void {
    while (stack.length > this.config.maxStackSize) {
      stack.shift();
    }
  }

  private notifyListeners(): void {
    const canUndo = this.canUndo();
    const canRedo = this.canRedo();
    this.listeners.forEach(listener => listener(canUndo, canRedo));
  }

  private setupKeyboardShortcuts(): void {
    const handler = (e: KeyboardEvent) => {
      // Ignore if in input/textarea
      const target = e.target as HTMLElement;
      if (target.tagName === 'INPUT' || target.tagName === 'TEXTAREA') return;

      // Ctrl+Z or Cmd+Z for undo
      if ((e.ctrlKey || e.metaKey) && e.key === 'z' && !e.shiftKey) {
        e.preventDefault();
        this.undo();
      }

      // Ctrl+Shift+Z or Cmd+Shift+Z for redo
      // Also Ctrl+Y on Windows
      if (
        ((e.ctrlKey || e.metaKey) && e.shiftKey && e.key === 'z') ||
        ((e.ctrlKey || e.metaKey) && e.key === 'y')
      ) {
        e.preventDefault();
        this.redo();
      }
    };

    window.addEventListener('keydown', handler);
    this.keyboardCleanup = () => window.removeEventListener('keydown', handler);
  }

  private persistIfEnabled(): void {
    if (!this.config.persistState) return;
    if (typeof localStorage === 'undefined') return;

    try {
      // Only persist descriptions, not full commands (which may have functions)
      const state = {
        undoDescriptions: this.getUndoStack(),
        redoDescriptions: this.getRedoStack(),
      };
      localStorage.setItem(this.config.storageKey, JSON.stringify(state));
    } catch (e) {
      // Ignore storage errors
    }
  }

  private loadState(): void {
    if (typeof localStorage === 'undefined') return;

    try {
      const stored = localStorage.getItem(this.config.storageKey);
      if (stored) {
        // Note: We can only restore descriptions, not actual undo functionality
        // This is just for UI display of recent actions
        // Actual undo/redo requires fresh commands
      }
    } catch (e) {
      // Ignore storage errors
    }
  }
}

// ============================================================================
// Singleton
// ============================================================================

let globalManager: UndoRedoManager | null = null;

export function getGlobalUndoRedoManager(config?: Partial<UndoRedoConfig>): UndoRedoManager {
  if (!globalManager) {
    globalManager = new UndoRedoManager(config);
  }
  return globalManager;
}

export function resetGlobalUndoRedoManager(): void {
  if (globalManager) {
    globalManager.dispose();
  }
  globalManager = null;
}

export default UndoRedoManager;
