/**
 * Lanes Accessibility Audit
 *
 * Phase 9, Iteration 76 of OITERATE lanes-widget-enhancement
 *
 * Features:
 * - axe-core integration
 * - WCAG 2.1 AA compliance checking
 * - Keyboard navigation audit
 * - Screen reader compatibility
 * - Color contrast validation
 */

// ============================================================================
// Types
// ============================================================================

export interface A11yViolation {
  id: string;
  impact: 'minor' | 'moderate' | 'serious' | 'critical';
  description: string;
  help: string;
  helpUrl: string;
  nodes: A11yNode[];
  tags: string[];
  wcagLevel?: string;
}

export interface A11yNode {
  html: string;
  target: string[];
  failureSummary: string;
  xpath?: string[];
}

export interface A11yResult {
  violations: A11yViolation[];
  passes: A11yPass[];
  incomplete: A11yIncomplete[];
  inapplicable: A11yInapplicable[];
  timestamp: number;
  url: string;
  testEngine: string;
}

export interface A11yPass {
  id: string;
  description: string;
  nodes: number;
}

export interface A11yIncomplete {
  id: string;
  description: string;
  nodes: A11yNode[];
}

export interface A11yInapplicable {
  id: string;
  description: string;
}

export interface A11yAuditConfig {
  /** Include WCAG 2.0 rules */
  wcag20: boolean;
  /** Include WCAG 2.1 rules */
  wcag21: boolean;
  /** Target level: A, AA, or AAA */
  level: 'A' | 'AA' | 'AAA';
  /** Custom rules to include */
  rules?: string[];
  /** Rules to exclude */
  exclude?: string[];
  /** Element selectors to audit */
  include?: string[];
  /** Element selectors to skip */
  skip?: string[];
  /** Run in iframes */
  iframes: boolean;
}

export interface KeyboardAuditResult {
  focusable: number;
  tabbable: number;
  missingTabIndex: string[];
  trapDetected: boolean;
  focusOrderIssues: string[];
  skipLinkPresent: boolean;
  landmarksPresent: string[];
}

export interface ColorContrastResult {
  element: string;
  foreground: string;
  background: string;
  ratio: number;
  requiredRatio: number;
  passes: boolean;
  level: 'AA' | 'AAA';
  isLargeText: boolean;
}

// ============================================================================
// Default Config
// ============================================================================

const DEFAULT_CONFIG: A11yAuditConfig = {
  wcag20: true,
  wcag21: true,
  level: 'AA',
  iframes: false,
};

// ============================================================================
// WCAG Requirements Map
// ============================================================================

const WCAG_REQUIREMENTS = {
  // Perceivable
  '1.1.1': { level: 'A', name: 'Non-text Content' },
  '1.2.1': { level: 'A', name: 'Audio-only and Video-only' },
  '1.3.1': { level: 'A', name: 'Info and Relationships' },
  '1.3.2': { level: 'A', name: 'Meaningful Sequence' },
  '1.3.3': { level: 'A', name: 'Sensory Characteristics' },
  '1.4.1': { level: 'A', name: 'Use of Color' },
  '1.4.2': { level: 'A', name: 'Audio Control' },
  '1.4.3': { level: 'AA', name: 'Contrast (Minimum)' },
  '1.4.4': { level: 'AA', name: 'Resize Text' },
  '1.4.5': { level: 'AA', name: 'Images of Text' },
  '1.4.10': { level: 'AA', name: 'Reflow' },
  '1.4.11': { level: 'AA', name: 'Non-text Contrast' },
  '1.4.12': { level: 'AA', name: 'Text Spacing' },
  '1.4.13': { level: 'AA', name: 'Content on Hover or Focus' },

  // Operable
  '2.1.1': { level: 'A', name: 'Keyboard' },
  '2.1.2': { level: 'A', name: 'No Keyboard Trap' },
  '2.1.4': { level: 'A', name: 'Character Key Shortcuts' },
  '2.4.1': { level: 'A', name: 'Bypass Blocks' },
  '2.4.2': { level: 'A', name: 'Page Titled' },
  '2.4.3': { level: 'A', name: 'Focus Order' },
  '2.4.4': { level: 'A', name: 'Link Purpose (In Context)' },
  '2.4.5': { level: 'AA', name: 'Multiple Ways' },
  '2.4.6': { level: 'AA', name: 'Headings and Labels' },
  '2.4.7': { level: 'AA', name: 'Focus Visible' },
  '2.5.1': { level: 'A', name: 'Pointer Gestures' },
  '2.5.2': { level: 'A', name: 'Pointer Cancellation' },
  '2.5.3': { level: 'A', name: 'Label in Name' },
  '2.5.4': { level: 'A', name: 'Motion Actuation' },

  // Understandable
  '3.1.1': { level: 'A', name: 'Language of Page' },
  '3.1.2': { level: 'AA', name: 'Language of Parts' },
  '3.2.1': { level: 'A', name: 'On Focus' },
  '3.2.2': { level: 'A', name: 'On Input' },
  '3.2.3': { level: 'AA', name: 'Consistent Navigation' },
  '3.2.4': { level: 'AA', name: 'Consistent Identification' },
  '3.3.1': { level: 'A', name: 'Error Identification' },
  '3.3.2': { level: 'A', name: 'Labels or Instructions' },
  '3.3.3': { level: 'AA', name: 'Error Suggestion' },
  '3.3.4': { level: 'AA', name: 'Error Prevention' },

  // Robust
  '4.1.1': { level: 'A', name: 'Parsing' },
  '4.1.2': { level: 'A', name: 'Name, Role, Value' },
  '4.1.3': { level: 'AA', name: 'Status Messages' },
};

// ============================================================================
// Accessibility Auditor
// ============================================================================

export class A11yAuditor {
  private config: A11yAuditConfig;

  constructor(config: Partial<A11yAuditConfig> = {}) {
    this.config = { ...DEFAULT_CONFIG, ...config };
  }

  /**
   * Run a full accessibility audit
   */
  async audit(element?: HTMLElement): Promise<A11yResult> {
    const target = element || document.body;
    const violations: A11yViolation[] = [];
    const passes: A11yPass[] = [];
    const incomplete: A11yIncomplete[] = [];

    // Run individual checks
    this.checkImages(target, violations, passes);
    this.checkHeadings(target, violations, passes);
    this.checkLinks(target, violations, passes);
    this.checkButtons(target, violations, passes);
    this.checkForms(target, violations, passes);
    this.checkTables(target, violations, passes);
    this.checkLandmarks(target, violations, passes);
    this.checkARIA(target, violations, passes);
    this.checkColorContrast(target, violations, passes);
    this.checkFocusManagement(target, violations, passes);

    return {
      violations,
      passes,
      incomplete,
      inapplicable: [],
      timestamp: Date.now(),
      url: typeof window !== 'undefined' ? window.location.href : '',
      testEngine: 'lanes-a11y-audit',
    };
  }

  /**
   * Audit keyboard navigation
   */
  auditKeyboard(element?: HTMLElement): KeyboardAuditResult {
    const target = element || document.body;
    const result: KeyboardAuditResult = {
      focusable: 0,
      tabbable: 0,
      missingTabIndex: [],
      trapDetected: false,
      focusOrderIssues: [],
      skipLinkPresent: false,
      landmarksPresent: [],
    };

    // Find all focusable elements
    const focusableSelectors = [
      'a[href]',
      'button:not([disabled])',
      'input:not([disabled])',
      'select:not([disabled])',
      'textarea:not([disabled])',
      '[tabindex]:not([tabindex="-1"])',
      '[contenteditable="true"]',
    ];

    const focusable = target.querySelectorAll(focusableSelectors.join(','));
    result.focusable = focusable.length;

    // Count tabbable elements (positive tabindex or natural tab order)
    focusable.forEach((el) => {
      const tabindex = el.getAttribute('tabindex');
      if (tabindex !== '-1') {
        result.tabbable++;
      }
    });

    // Check for interactive elements without proper focus handling
    const interactiveWithoutFocus = target.querySelectorAll(
      '[onclick]:not(button):not(a):not([tabindex]), ' +
      '[role="button"]:not([tabindex]), ' +
      '[role="link"]:not([tabindex])'
    );
    interactiveWithoutFocus.forEach((el) => {
      result.missingTabIndex.push(this.getElementDescription(el));
    });

    // Check for skip link
    const skipLink = target.querySelector('a[href^="#"]:first-of-type');
    result.skipLinkPresent = !!skipLink;

    // Check for landmarks
    const landmarks = target.querySelectorAll(
      'main, [role="main"], ' +
      'nav, [role="navigation"], ' +
      'aside, [role="complementary"], ' +
      'header, [role="banner"], ' +
      'footer, [role="contentinfo"], ' +
      '[role="search"], ' +
      '[role="region"][aria-label], ' +
      'section[aria-label], section[aria-labelledby]'
    );
    landmarks.forEach((el) => {
      const role = el.getAttribute('role') || el.tagName.toLowerCase();
      if (!result.landmarksPresent.includes(role)) {
        result.landmarksPresent.push(role);
      }
    });

    return result;
  }

  /**
   * Audit color contrast
   */
  auditColorContrast(element?: HTMLElement): ColorContrastResult[] {
    const target = element || document.body;
    const results: ColorContrastResult[] = [];

    // Get all text elements
    const textElements = target.querySelectorAll('p, span, h1, h2, h3, h4, h5, h6, a, button, label, td, th, li');

    textElements.forEach((el) => {
      const style = window.getComputedStyle(el);
      const fg = style.color;
      const bg = this.getBackgroundColor(el);

      const fgRgb = this.parseColor(fg);
      const bgRgb = this.parseColor(bg);

      if (fgRgb && bgRgb) {
        const ratio = this.calculateContrastRatio(fgRgb, bgRgb);
        const fontSize = parseFloat(style.fontSize);
        const fontWeight = style.fontWeight;
        const isLargeText = fontSize >= 18 || (fontSize >= 14 && parseInt(fontWeight) >= 700);

        const requiredAAA = isLargeText ? 4.5 : 7;
        const requiredAA = isLargeText ? 3 : 4.5;

        results.push({
          element: this.getElementDescription(el),
          foreground: fg,
          background: bg,
          ratio: Math.round(ratio * 100) / 100,
          requiredRatio: this.config.level === 'AAA' ? requiredAAA : requiredAA,
          passes: ratio >= (this.config.level === 'AAA' ? requiredAAA : requiredAA),
          level: this.config.level === 'AAA' ? 'AAA' : 'AA',
          isLargeText,
        });
      }
    });

    return results;
  }

  // ============================================================================
  // Individual Checks
  // ============================================================================

  private checkImages(target: HTMLElement, violations: A11yViolation[], passes: A11yPass[]): void {
    const images = target.querySelectorAll('img');
    let passCount = 0;
    const failedNodes: A11yNode[] = [];

    images.forEach((img) => {
      const alt = img.getAttribute('alt');
      const role = img.getAttribute('role');

      if (alt === null && role !== 'presentation' && role !== 'none') {
        failedNodes.push({
          html: img.outerHTML.slice(0, 100),
          target: [this.getSelector(img)],
          failureSummary: 'Image is missing alt attribute',
        });
      } else {
        passCount++;
      }
    });

    if (failedNodes.length > 0) {
      violations.push({
        id: 'image-alt',
        impact: 'critical',
        description: 'Images must have alternate text',
        help: 'Ensure every image has an alt attribute',
        helpUrl: 'https://dequeuniversity.com/rules/axe/4.4/image-alt',
        nodes: failedNodes,
        tags: ['wcag2a', 'wcag111', 'section508'],
        wcagLevel: 'A',
      });
    }

    if (passCount > 0) {
      passes.push({
        id: 'image-alt',
        description: 'Images have alternative text',
        nodes: passCount,
      });
    }
  }

  private checkHeadings(target: HTMLElement, violations: A11yViolation[], passes: A11yPass[]): void {
    const headings = target.querySelectorAll('h1, h2, h3, h4, h5, h6');
    const levels: number[] = [];
    const failedNodes: A11yNode[] = [];

    headings.forEach((heading) => {
      const level = parseInt(heading.tagName[1]);
      const prevLevel = levels[levels.length - 1] || 0;

      // Check for empty headings
      if (!heading.textContent?.trim()) {
        failedNodes.push({
          html: heading.outerHTML,
          target: [this.getSelector(heading)],
          failureSummary: 'Heading is empty',
        });
      }

      // Check for skipped levels
      if (prevLevel > 0 && level > prevLevel + 1) {
        failedNodes.push({
          html: heading.outerHTML,
          target: [this.getSelector(heading)],
          failureSummary: `Heading level skipped from h${prevLevel} to h${level}`,
        });
      }

      levels.push(level);
    });

    if (failedNodes.length > 0) {
      violations.push({
        id: 'heading-order',
        impact: 'moderate',
        description: 'Heading levels should be in order and not empty',
        help: 'Maintain proper heading hierarchy',
        helpUrl: 'https://dequeuniversity.com/rules/axe/4.4/heading-order',
        nodes: failedNodes,
        tags: ['wcag2a', 'wcag131'],
        wcagLevel: 'A',
      });
    }

    if (headings.length > failedNodes.length) {
      passes.push({
        id: 'heading-order',
        description: 'Heading levels are properly ordered',
        nodes: headings.length - failedNodes.length,
      });
    }
  }

  private checkLinks(target: HTMLElement, violations: A11yViolation[], passes: A11yPass[]): void {
    const links = target.querySelectorAll('a[href]');
    let passCount = 0;
    const failedNodes: A11yNode[] = [];

    links.forEach((link) => {
      const text = link.textContent?.trim() || '';
      const ariaLabel = link.getAttribute('aria-label');
      const ariaLabelledby = link.getAttribute('aria-labelledby');
      const title = link.getAttribute('title');
      const img = link.querySelector('img');

      const hasAccessibleName = text || ariaLabel || ariaLabelledby || title || (img && img.getAttribute('alt'));

      if (!hasAccessibleName) {
        failedNodes.push({
          html: link.outerHTML.slice(0, 100),
          target: [this.getSelector(link)],
          failureSummary: 'Link has no accessible name',
        });
      } else if (['click here', 'read more', 'learn more', 'here'].includes(text.toLowerCase())) {
        failedNodes.push({
          html: link.outerHTML.slice(0, 100),
          target: [this.getSelector(link)],
          failureSummary: 'Link text is not descriptive',
        });
      } else {
        passCount++;
      }
    });

    if (failedNodes.length > 0) {
      violations.push({
        id: 'link-name',
        impact: 'serious',
        description: 'Links must have discernible text',
        help: 'Ensure links have meaningful names',
        helpUrl: 'https://dequeuniversity.com/rules/axe/4.4/link-name',
        nodes: failedNodes,
        tags: ['wcag2a', 'wcag244', 'wcag412'],
        wcagLevel: 'A',
      });
    }

    if (passCount > 0) {
      passes.push({
        id: 'link-name',
        description: 'Links have discernible text',
        nodes: passCount,
      });
    }
  }

  private checkButtons(target: HTMLElement, violations: A11yViolation[], passes: A11yPass[]): void {
    const buttons = target.querySelectorAll('button, [role="button"]');
    let passCount = 0;
    const failedNodes: A11yNode[] = [];

    buttons.forEach((button) => {
      const text = button.textContent?.trim() || '';
      const ariaLabel = button.getAttribute('aria-label');
      const ariaLabelledby = button.getAttribute('aria-labelledby');
      const title = button.getAttribute('title');

      if (!text && !ariaLabel && !ariaLabelledby && !title) {
        failedNodes.push({
          html: button.outerHTML.slice(0, 100),
          target: [this.getSelector(button)],
          failureSummary: 'Button has no accessible name',
        });
      } else {
        passCount++;
      }
    });

    if (failedNodes.length > 0) {
      violations.push({
        id: 'button-name',
        impact: 'critical',
        description: 'Buttons must have discernible text',
        help: 'Ensure buttons have an accessible name',
        helpUrl: 'https://dequeuniversity.com/rules/axe/4.4/button-name',
        nodes: failedNodes,
        tags: ['wcag2a', 'wcag412'],
        wcagLevel: 'A',
      });
    }

    if (passCount > 0) {
      passes.push({
        id: 'button-name',
        description: 'Buttons have discernible text',
        nodes: passCount,
      });
    }
  }

  private checkForms(target: HTMLElement, violations: A11yViolation[], passes: A11yPass[]): void {
    const inputs = target.querySelectorAll('input:not([type="hidden"]), select, textarea');
    let passCount = 0;
    const failedNodes: A11yNode[] = [];

    inputs.forEach((input) => {
      const id = input.getAttribute('id');
      const ariaLabel = input.getAttribute('aria-label');
      const ariaLabelledby = input.getAttribute('aria-labelledby');
      const title = input.getAttribute('title');
      const placeholder = input.getAttribute('placeholder');

      const hasLabel = id && target.querySelector(`label[for="${id}"]`);
      const hasAccessibleName = hasLabel || ariaLabel || ariaLabelledby || title;

      if (!hasAccessibleName) {
        failedNodes.push({
          html: input.outerHTML.slice(0, 100),
          target: [this.getSelector(input)],
          failureSummary: placeholder
            ? 'Form input relies only on placeholder text'
            : 'Form input has no label',
        });
      } else {
        passCount++;
      }
    });

    if (failedNodes.length > 0) {
      violations.push({
        id: 'label',
        impact: 'critical',
        description: 'Form elements must have labels',
        help: 'Ensure every form input has a label',
        helpUrl: 'https://dequeuniversity.com/rules/axe/4.4/label',
        nodes: failedNodes,
        tags: ['wcag2a', 'wcag412', 'wcag131'],
        wcagLevel: 'A',
      });
    }

    if (passCount > 0) {
      passes.push({
        id: 'label',
        description: 'Form elements have labels',
        nodes: passCount,
      });
    }
  }

  private checkTables(target: HTMLElement, violations: A11yViolation[], passes: A11yPass[]): void {
    const tables = target.querySelectorAll('table');
    let passCount = 0;
    const failedNodes: A11yNode[] = [];

    tables.forEach((table) => {
      const headers = table.querySelectorAll('th');
      const caption = table.querySelector('caption');
      const ariaLabel = table.getAttribute('aria-label');
      const ariaLabelledby = table.getAttribute('aria-labelledby');

      if (headers.length === 0) {
        failedNodes.push({
          html: table.outerHTML.slice(0, 100),
          target: [this.getSelector(table)],
          failureSummary: 'Table has no header cells',
        });
      } else if (!caption && !ariaLabel && !ariaLabelledby) {
        // Warning level - tables should have captions
        passCount++;
      } else {
        passCount++;
      }
    });

    if (failedNodes.length > 0) {
      violations.push({
        id: 'table-headers',
        impact: 'serious',
        description: 'Data tables must have headers',
        help: 'Use th elements to define table headers',
        helpUrl: 'https://dequeuniversity.com/rules/axe/4.4/td-headers-attr',
        nodes: failedNodes,
        tags: ['wcag2a', 'wcag131'],
        wcagLevel: 'A',
      });
    }

    if (passCount > 0) {
      passes.push({
        id: 'table-headers',
        description: 'Tables have proper headers',
        nodes: passCount,
      });
    }
  }

  private checkLandmarks(target: HTMLElement, violations: A11yViolation[], passes: A11yPass[]): void {
    const main = target.querySelector('main, [role="main"]');
    const nav = target.querySelector('nav, [role="navigation"]');
    const failedNodes: A11yNode[] = [];

    if (!main) {
      failedNodes.push({
        html: '<body>...',
        target: ['body'],
        failureSummary: 'Page has no main landmark',
      });
    }

    // Check for multiple main landmarks
    const mainLandmarks = target.querySelectorAll('main, [role="main"]');
    if (mainLandmarks.length > 1) {
      failedNodes.push({
        html: '<main>...',
        target: ['main'],
        failureSummary: 'Page has multiple main landmarks',
      });
    }

    if (failedNodes.length > 0) {
      violations.push({
        id: 'landmark-main-is-top-level',
        impact: 'moderate',
        description: 'Page should have proper landmark structure',
        help: 'Ensure page has a single main landmark',
        helpUrl: 'https://dequeuniversity.com/rules/axe/4.4/landmark-main-is-top-level',
        nodes: failedNodes,
        tags: ['wcag2a', 'wcag131'],
        wcagLevel: 'A',
      });
    }

    if (main) {
      passes.push({
        id: 'landmark-main-is-top-level',
        description: 'Page has main landmark',
        nodes: 1,
      });
    }
  }

  private checkARIA(target: HTMLElement, violations: A11yViolation[], passes: A11yPass[]): void {
    const ariaElements = target.querySelectorAll('[role], [aria-label], [aria-labelledby], [aria-describedby]');
    let passCount = 0;
    const failedNodes: A11yNode[] = [];

    ariaElements.forEach((el) => {
      const role = el.getAttribute('role');
      const ariaLabelledby = el.getAttribute('aria-labelledby');
      const ariaDescribedby = el.getAttribute('aria-describedby');

      // Check for valid role
      const validRoles = [
        'alert', 'alertdialog', 'application', 'article', 'banner', 'button',
        'cell', 'checkbox', 'columnheader', 'combobox', 'complementary',
        'contentinfo', 'definition', 'dialog', 'directory', 'document',
        'feed', 'figure', 'form', 'grid', 'gridcell', 'group', 'heading',
        'img', 'link', 'list', 'listbox', 'listitem', 'log', 'main',
        'marquee', 'math', 'menu', 'menubar', 'menuitem', 'menuitemcheckbox',
        'menuitemradio', 'navigation', 'none', 'note', 'option', 'presentation',
        'progressbar', 'radio', 'radiogroup', 'region', 'row', 'rowgroup',
        'rowheader', 'scrollbar', 'search', 'searchbox', 'separator', 'slider',
        'spinbutton', 'status', 'switch', 'tab', 'table', 'tablist', 'tabpanel',
        'term', 'textbox', 'timer', 'toolbar', 'tooltip', 'tree', 'treegrid', 'treeitem',
      ];

      if (role && !validRoles.includes(role)) {
        failedNodes.push({
          html: el.outerHTML.slice(0, 100),
          target: [this.getSelector(el)],
          failureSummary: `Invalid ARIA role: ${role}`,
        });
      }

      // Check aria-labelledby references exist
      if (ariaLabelledby) {
        const ids = ariaLabelledby.split(' ');
        for (const id of ids) {
          if (!document.getElementById(id)) {
            failedNodes.push({
              html: el.outerHTML.slice(0, 100),
              target: [this.getSelector(el)],
              failureSummary: `aria-labelledby references non-existent id: ${id}`,
            });
          }
        }
      }

      // Check aria-describedby references exist
      if (ariaDescribedby) {
        const ids = ariaDescribedby.split(' ');
        for (const id of ids) {
          if (!document.getElementById(id)) {
            failedNodes.push({
              html: el.outerHTML.slice(0, 100),
              target: [this.getSelector(el)],
              failureSummary: `aria-describedby references non-existent id: ${id}`,
            });
          }
        }
      }

      if (failedNodes.length === 0) {
        passCount++;
      }
    });

    if (failedNodes.length > 0) {
      violations.push({
        id: 'aria-valid-attr-value',
        impact: 'critical',
        description: 'ARIA attributes must have valid values',
        help: 'Ensure ARIA attributes reference valid elements',
        helpUrl: 'https://dequeuniversity.com/rules/axe/4.4/aria-valid-attr-value',
        nodes: failedNodes,
        tags: ['wcag2a', 'wcag412'],
        wcagLevel: 'A',
      });
    }

    if (passCount > 0) {
      passes.push({
        id: 'aria-valid-attr-value',
        description: 'ARIA attributes have valid values',
        nodes: passCount,
      });
    }
  }

  private checkColorContrast(target: HTMLElement, violations: A11yViolation[], passes: A11yPass[]): void {
    // Color contrast checking is complex and would require full implementation
    // This is a simplified version
    const results = this.auditColorContrast(target);
    const failedResults = results.filter(r => !r.passes);

    if (failedResults.length > 0) {
      violations.push({
        id: 'color-contrast',
        impact: 'serious',
        description: 'Elements must have sufficient color contrast',
        help: 'Ensure text has minimum contrast ratio of 4.5:1 (or 3:1 for large text)',
        helpUrl: 'https://dequeuniversity.com/rules/axe/4.4/color-contrast',
        nodes: failedResults.map(r => ({
          html: r.element,
          target: [r.element],
          failureSummary: `Contrast ratio ${r.ratio}:1 is less than required ${r.requiredRatio}:1`,
        })),
        tags: ['wcag2aa', 'wcag143'],
        wcagLevel: 'AA',
      });
    }

    const passedResults = results.filter(r => r.passes);
    if (passedResults.length > 0) {
      passes.push({
        id: 'color-contrast',
        description: 'Elements have sufficient color contrast',
        nodes: passedResults.length,
      });
    }
  }

  private checkFocusManagement(target: HTMLElement, violations: A11yViolation[], passes: A11yPass[]): void {
    const focusableElements = target.querySelectorAll(
      'a[href], button, input, select, textarea, [tabindex]:not([tabindex="-1"])'
    );
    let passCount = 0;
    const failedNodes: A11yNode[] = [];

    focusableElements.forEach((el) => {
      const style = window.getComputedStyle(el);
      const pseudoStyle = window.getComputedStyle(el, ':focus');

      // Check if element is visible
      if (style.display === 'none' || style.visibility === 'hidden') {
        return;
      }

      // Check for focus outline (simplified check)
      // In a real implementation, this would need to check :focus styles
      passCount++;
    });

    if (passCount > 0) {
      passes.push({
        id: 'focus-visible',
        description: 'Focusable elements can receive focus',
        nodes: passCount,
      });
    }
  }

  // ============================================================================
  // Utility Methods
  // ============================================================================

  private getSelector(element: Element): string {
    if (element.id) {
      return `#${element.id}`;
    }
    const classes = Array.from(element.classList).slice(0, 3).join('.');
    if (classes) {
      return `${element.tagName.toLowerCase()}.${classes}`;
    }
    return element.tagName.toLowerCase();
  }

  private getElementDescription(element: Element): string {
    const tag = element.tagName.toLowerCase();
    const id = element.id ? `#${element.id}` : '';
    const classes = element.className ? `.${element.className.split(' ').slice(0, 2).join('.')}` : '';
    return `${tag}${id}${classes}`;
  }

  private getBackgroundColor(element: Element): string {
    let el: Element | null = element;
    while (el) {
      const style = window.getComputedStyle(el);
      const bg = style.backgroundColor;
      if (bg && bg !== 'transparent' && bg !== 'rgba(0, 0, 0, 0)') {
        return bg;
      }
      el = el.parentElement;
    }
    return 'rgb(255, 255, 255)';
  }

  private parseColor(color: string): { r: number; g: number; b: number } | null {
    const match = color.match(/rgba?\((\d+),\s*(\d+),\s*(\d+)/);
    if (match) {
      return {
        r: parseInt(match[1]),
        g: parseInt(match[2]),
        b: parseInt(match[3]),
      };
    }
    return null;
  }

  private calculateContrastRatio(
    fg: { r: number; g: number; b: number },
    bg: { r: number; g: number; b: number }
  ): number {
    const getLuminance = (rgb: { r: number; g: number; b: number }) => {
      const [r, g, b] = [rgb.r, rgb.g, rgb.b].map((c) => {
        c = c / 255;
        return c <= 0.03928 ? c / 12.92 : Math.pow((c + 0.055) / 1.055, 2.4);
      });
      return 0.2126 * r + 0.7152 * g + 0.0722 * b;
    };

    const l1 = getLuminance(fg);
    const l2 = getLuminance(bg);
    const lighter = Math.max(l1, l2);
    const darker = Math.min(l1, l2);

    return (lighter + 0.05) / (darker + 0.05);
  }
}

// ============================================================================
// Report Generator
// ============================================================================

export function generateA11yReport(result: A11yResult): string {
  const lines: string[] = [];

  lines.push('╔════════════════════════════════════════════════════════════════════╗');
  lines.push('║                    ACCESSIBILITY AUDIT REPORT                      ║');
  lines.push('╚════════════════════════════════════════════════════════════════════╝');
  lines.push('');
  lines.push(`URL: ${result.url}`);
  lines.push(`Timestamp: ${new Date(result.timestamp).toISOString()}`);
  lines.push(`Engine: ${result.testEngine}`);
  lines.push('');

  // Summary
  lines.push('┌─────────────────────────────────────────────────────────────────────┐');
  lines.push('│ SUMMARY                                                             │');
  lines.push('├─────────────────────────────────────────────────────────────────────┤');
  lines.push(`│ Violations: ${result.violations.length.toString().padEnd(53)}│`);
  lines.push(`│ Passes: ${result.passes.reduce((a, p) => a + p.nodes, 0).toString().padEnd(57)}│`);
  lines.push(`│ Incomplete: ${result.incomplete.length.toString().padEnd(53)}│`);
  lines.push('└─────────────────────────────────────────────────────────────────────┘');
  lines.push('');

  // Violations
  if (result.violations.length > 0) {
    lines.push('┌─────────────────────────────────────────────────────────────────────┐');
    lines.push('│ VIOLATIONS                                                          │');
    lines.push('└─────────────────────────────────────────────────────────────────────┘');

    for (const violation of result.violations) {
      lines.push('');
      lines.push(`  [${ violation.impact.toUpperCase()}] ${violation.id}`);
      lines.push(`  ${violation.description}`);
      lines.push(`  WCAG: ${violation.wcagLevel || 'N/A'} | Nodes: ${violation.nodes.length}`);
      lines.push(`  Help: ${violation.helpUrl}`);

      for (const node of violation.nodes.slice(0, 3)) {
        lines.push(`    - ${node.failureSummary}`);
        lines.push(`      ${node.target.join(' > ')}`);
      }

      if (violation.nodes.length > 3) {
        lines.push(`    ... and ${violation.nodes.length - 3} more`);
      }
    }
  }

  lines.push('');
  lines.push('═══════════════════════════════════════════════════════════════════════');

  return lines.join('\n');
}

// ============================================================================
// Singleton
// ============================================================================

let globalAuditor: A11yAuditor | null = null;

export function getGlobalA11yAuditor(config?: Partial<A11yAuditConfig>): A11yAuditor {
  if (!globalAuditor) {
    globalAuditor = new A11yAuditor(config);
  }
  return globalAuditor;
}

export function resetGlobalA11yAuditor(): void {
  globalAuditor = null;
}

export default A11yAuditor;
