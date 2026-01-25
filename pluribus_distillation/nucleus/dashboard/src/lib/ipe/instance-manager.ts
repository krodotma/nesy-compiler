/**
 * IPE Instance Manager
 *
 * Manages instance-specific style overrides.
 * Supports applying unique IDs to elements and injecting scoped CSS.
 */

import { generateInstanceId, type IPEContext } from './context-capture';
import { generateInstanceCSS, type ThemeStyleProps } from './token-bridge';

// ============================================================================
// Types
// ============================================================================

export interface InstanceOverride {
  instanceId: string;
  selector: string;
  styles: Partial<ThemeStyleProps>;
  shaderOverride?: string;
  purpose?: string;
  createdAt: string;
  updatedAt: string;
}

export interface InstanceManager {
  /** Register an element for instance-specific editing */
  register(element: Element): string;

  /** Apply styles to a registered instance */
  applyStyles(instanceId: string, styles: Partial<ThemeStyleProps>, purpose?: string): void;

  /** Get all registered instances */
  getInstances(): Map<string, InstanceOverride>;

  /** Remove an instance */
  remove(instanceId: string): void;

  /** Clear all instances */
  clear(): void;

  /** Export all instances to JSON */
  export(): string;

  /** Import instances from JSON */
  import(json: string): void;
}

// ============================================================================
// Implementation
// ============================================================================

class InstanceManagerImpl implements InstanceManager {
  private instances = new Map<string, InstanceOverride>();
  private styleElements = new Map<string, HTMLStyleElement>();
  private observers = new Map<string, MutationObserver>();

  /**
   * Register an element for instance-specific editing
   */
  register(element: Element): string {
    const instanceId = generateInstanceId(element);

    // Already registered?
    if (this.instances.has(instanceId)) {
      return instanceId;
    }

    // Add data attribute
    element.setAttribute('data-ipe-id', instanceId);

    // Create instance record
    const now = new Date().toISOString();
    this.instances.set(instanceId, {
      instanceId,
      selector: getSelector(element),
      styles: {},
      createdAt: now,
      updatedAt: now,
    });

    // Set up mutation observer to track element removal
    const observer = new MutationObserver((mutations) => {
      for (const mutation of mutations) {
        for (const removed of mutation.removedNodes) {
          if (removed === element || (removed instanceof Element && removed.contains(element))) {
            this.cleanup(instanceId);
          }
        }
      }
    });

    if (element.parentElement) {
      observer.observe(element.parentElement, { childList: true, subtree: true });
      this.observers.set(instanceId, observer);
    }

    return instanceId;
  }

  /**
   * Apply styles to a registered instance
   */
  applyStyles(instanceId: string, styles: Partial<ThemeStyleProps>, purpose?: string): void {
    const instance = this.instances.get(instanceId);
    if (!instance) {
      console.warn(`[IPE] Instance not found: ${instanceId}`);
      return;
    }

    // Merge styles
    instance.styles = { ...instance.styles, ...styles };
    instance.updatedAt = new Date().toISOString();
    if (purpose) {
      instance.purpose = purpose;
    }

    // Generate CSS
    const css = generateInstanceCSS(instanceId, instance.styles);

    // Inject or update style element
    let styleEl = this.styleElements.get(instanceId);
    if (!styleEl) {
      styleEl = document.createElement('style');
      styleEl.setAttribute('data-ipe-instance', instanceId);
      document.head.appendChild(styleEl);
      this.styleElements.set(instanceId, styleEl);
    }

    styleEl.textContent = css;

    // Dispatch event
    window.dispatchEvent(new CustomEvent('ipe:instance:update', {
      detail: { instanceId, styles: instance.styles }
    }));
  }

  /**
   * Get all registered instances
   */
  getInstances(): Map<string, InstanceOverride> {
    return new Map(this.instances);
  }

  /**
   * Get a specific instance
   */
  getInstance(instanceId: string): InstanceOverride | undefined {
    return this.instances.get(instanceId);
  }

  /**
   * Remove an instance
   */
  remove(instanceId: string): void {
    this.cleanup(instanceId);
    this.instances.delete(instanceId);

    // Remove data attribute from element
    const element = document.querySelector(`[data-ipe-id="${instanceId}"]`);
    if (element) {
      element.removeAttribute('data-ipe-id');
    }

    // Dispatch event
    window.dispatchEvent(new CustomEvent('ipe:instance:remove', {
      detail: { instanceId }
    }));
  }

  /**
   * Clear all instances
   */
  clear(): void {
    for (const instanceId of this.instances.keys()) {
      this.remove(instanceId);
    }
  }

  /**
   * Export all instances to JSON
   */
  export(): string {
    const data = Array.from(this.instances.values());
    return JSON.stringify(data, null, 2);
  }

  /**
   * Import instances from JSON
   */
  import(json: string): void {
    try {
      const data = JSON.parse(json) as InstanceOverride[];
      for (const instance of data) {
        this.instances.set(instance.instanceId, instance);

        // Try to find and register the element
        const element = document.querySelector(instance.selector);
        if (element) {
          element.setAttribute('data-ipe-id', instance.instanceId);
          this.applyStyles(instance.instanceId, instance.styles);
        }
      }
    } catch (error) {
      console.error('[IPE] Failed to import instances:', error);
    }
  }

  /**
   * Cleanup resources for an instance
   */
  private cleanup(instanceId: string): void {
    // Remove style element
    const styleEl = this.styleElements.get(instanceId);
    if (styleEl) {
      styleEl.remove();
      this.styleElements.delete(instanceId);
    }

    // Disconnect observer
    const observer = this.observers.get(instanceId);
    if (observer) {
      observer.disconnect();
      this.observers.delete(instanceId);
    }
  }
}

// ============================================================================
// Singleton
// ============================================================================

let manager: InstanceManagerImpl | null = null;

/**
 * Get the singleton instance manager
 */
export function getInstanceManager(): InstanceManager {
  if (!manager) {
    manager = new InstanceManagerImpl();
  }
  return manager;
}

// ============================================================================
// Utility Functions
// ============================================================================

/**
 * Get a unique CSS selector for an element
 */
function getSelector(element: Element): string {
  const parts: string[] = [];
  let current: Element | null = element;

  while (current && current !== document.body) {
    let selector = current.tagName.toLowerCase();

    if (current.id) {
      selector = `#${CSS.escape(current.id)}`;
      parts.unshift(selector);
      break; // ID is unique, no need to go further
    }

    const parent = current.parentElement;
    if (parent) {
      const siblings = Array.from(parent.children);
      const index = siblings.indexOf(current);
      selector += `:nth-child(${index + 1})`;
    }

    parts.unshift(selector);
    current = parent;
  }

  return parts.join(' > ');
}

/**
 * Find element by instance ID
 */
export function findElementByInstanceId(instanceId: string): Element | null {
  return document.querySelector(`[data-ipe-id="${instanceId}"]`);
}

/**
 * Check if an element has an instance ID
 */
export function hasInstanceId(element: Element): boolean {
  return element.hasAttribute('data-ipe-id');
}

/**
 * Get instance ID from element
 */
export function getElementInstanceId(element: Element): string | null {
  return element.getAttribute('data-ipe-id');
}

// ============================================================================
// Default Export
// ============================================================================

export default {
  getInstanceManager,
  findElementByInstanceId,
  hasInstanceId,
  getElementInstanceId,
};
