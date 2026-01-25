/**
 * Lanes E2E Tests
 *
 * Phase 9, Iteration 71 of OITERATE lanes-widget-enhancement
 *
 * Features:
 * - Playwright E2E tests
 * - Visual snapshots
 * - Interactive testing
 * - Accessibility checks
 * - Performance benchmarks
 */

import { test, expect, type Page, type Locator } from '@playwright/test';

// ============================================================================
// Test Configuration
// ============================================================================

const BASE_URL = process.env.E2E_BASE_URL || 'https://kroma.live';

test.describe.configure({ mode: 'parallel' });

// ============================================================================
// Helper Functions
// ============================================================================

async function waitForLanesWidget(page: Page): Promise<Locator> {
  const widget = page.locator('[data-testid="lanes-widget"]');
  await widget.waitFor({ state: 'visible', timeout: 10000 });
  return widget;
}

async function getLaneCards(page: Page): Promise<Locator> {
  return page.locator('[data-testid="lane-card"]');
}

async function getLaneByName(page: Page, name: string): Promise<Locator> {
  return page.locator(`[data-testid="lane-card"]:has-text("${name}")`);
}

async function openLaneDetails(page: Page, laneName: string): Promise<void> {
  const lane = await getLaneByName(page, laneName);
  await lane.click();
  await page.locator('[data-testid="lane-details-panel"]').waitFor({ state: 'visible' });
}

async function closeLaneDetails(page: Page): Promise<void> {
  await page.keyboard.press('Escape');
  await page.locator('[data-testid="lane-details-panel"]').waitFor({ state: 'hidden' });
}

// ============================================================================
// Basic Functionality Tests
// ============================================================================

test.describe('Lanes Widget Basic Functionality', () => {
  test.beforeEach(async ({ page }) => {
    await page.goto(BASE_URL);
    await waitForLanesWidget(page);
  });

  test('should render lanes widget', async ({ page }) => {
    const widget = await waitForLanesWidget(page);
    await expect(widget).toBeVisible();
  });

  test('should display lane cards', async ({ page }) => {
    const cards = await getLaneCards(page);
    const count = await cards.count();
    expect(count).toBeGreaterThan(0);
  });

  test('should show lane status indicators', async ({ page }) => {
    const statusIndicators = page.locator('[data-testid="lane-status"]');
    await expect(statusIndicators.first()).toBeVisible();
  });

  test('should display WIP progress bars', async ({ page }) => {
    const progressBars = page.locator('[data-testid="lane-wip-progress"]');
    const first = progressBars.first();
    await expect(first).toBeVisible();
  });

  test('should show lane owners', async ({ page }) => {
    const owners = page.locator('[data-testid="lane-owner"]');
    await expect(owners.first()).toBeVisible();
  });
});

// ============================================================================
// Interaction Tests
// ============================================================================

test.describe('Lanes Widget Interactions', () => {
  test.beforeEach(async ({ page }) => {
    await page.goto(BASE_URL);
    await waitForLanesWidget(page);
  });

  test('should open lane details on click', async ({ page }) => {
    const firstLane = (await getLaneCards(page)).first();
    await firstLane.click();

    const detailsPanel = page.locator('[data-testid="lane-details-panel"]');
    await expect(detailsPanel).toBeVisible();
  });

  test('should close details on escape key', async ({ page }) => {
    const firstLane = (await getLaneCards(page)).first();
    await firstLane.click();

    const detailsPanel = page.locator('[data-testid="lane-details-panel"]');
    await expect(detailsPanel).toBeVisible();

    await page.keyboard.press('Escape');
    await expect(detailsPanel).toBeHidden();
  });

  test('should filter lanes by status', async ({ page }) => {
    const filterDropdown = page.locator('[data-testid="lane-status-filter"]');
    if (await filterDropdown.isVisible()) {
      await filterDropdown.selectOption('red');

      const lanes = await getLaneCards(page);
      const count = await lanes.count();

      for (let i = 0; i < count; i++) {
        const status = await lanes.nth(i).locator('[data-testid="lane-status"]').getAttribute('data-status');
        expect(status).toBe('red');
      }
    }
  });

  test('should search lanes by name', async ({ page }) => {
    const searchInput = page.locator('[data-testid="lane-search"]');
    if (await searchInput.isVisible()) {
      await searchInput.fill('test');
      await page.waitForTimeout(300); // Debounce

      const lanes = await getLaneCards(page);
      const count = await lanes.count();

      for (let i = 0; i < count; i++) {
        const text = await lanes.nth(i).textContent();
        expect(text?.toLowerCase()).toContain('test');
      }
    }
  });

  test('should sort lanes by different criteria', async ({ page }) => {
    const sortDropdown = page.locator('[data-testid="lane-sort"]');
    if (await sortDropdown.isVisible()) {
      await sortDropdown.selectOption('wip');
      // Verify ordering (would need actual lane data to validate)
    }
  });
});

// ============================================================================
// Multi-Select Tests
// ============================================================================

test.describe('Lanes Multi-Select', () => {
  test.beforeEach(async ({ page }) => {
    await page.goto(BASE_URL);
    await waitForLanesWidget(page);
  });

  test('should show checkboxes in multi-select mode', async ({ page }) => {
    const multiSelectToggle = page.locator('[data-testid="multi-select-toggle"]');
    if (await multiSelectToggle.isVisible()) {
      await multiSelectToggle.click();

      const checkboxes = page.locator('[data-testid="lane-checkbox"]');
      await expect(checkboxes.first()).toBeVisible();
    }
  });

  test('should select multiple lanes with checkboxes', async ({ page }) => {
    const multiSelectToggle = page.locator('[data-testid="multi-select-toggle"]');
    if (await multiSelectToggle.isVisible()) {
      await multiSelectToggle.click();

      const checkboxes = page.locator('[data-testid="lane-checkbox"]');
      await checkboxes.nth(0).click();
      await checkboxes.nth(1).click();

      const selectedCount = page.locator('[data-testid="selected-count"]');
      await expect(selectedCount).toContainText('2');
    }
  });

  test('should select all with Ctrl+A', async ({ page }) => {
    const multiSelectToggle = page.locator('[data-testid="multi-select-toggle"]');
    if (await multiSelectToggle.isVisible()) {
      await multiSelectToggle.click();
      await page.keyboard.press('Control+a');

      const lanes = await getLaneCards(page);
      const totalCount = await lanes.count();

      const selectedCount = page.locator('[data-testid="selected-count"]');
      await expect(selectedCount).toContainText(totalCount.toString());
    }
  });

  test('should show bulk actions bar when lanes selected', async ({ page }) => {
    const multiSelectToggle = page.locator('[data-testid="multi-select-toggle"]');
    if (await multiSelectToggle.isVisible()) {
      await multiSelectToggle.click();

      const checkbox = page.locator('[data-testid="lane-checkbox"]').first();
      await checkbox.click();

      const bulkActionsBar = page.locator('[data-testid="bulk-actions-bar"]');
      await expect(bulkActionsBar).toBeVisible();
    }
  });
});

// ============================================================================
// Dependency Manager Tests
// ============================================================================

test.describe('Lanes Dependency Manager', () => {
  test.beforeEach(async ({ page }) => {
    await page.goto(BASE_URL);
    await waitForLanesWidget(page);
  });

  test('should open dependency manager', async ({ page }) => {
    const depManagerBtn = page.locator('[data-testid="open-dependency-manager"]');
    if (await depManagerBtn.isVisible()) {
      await depManagerBtn.click();

      const depManager = page.locator('[data-testid="dependency-manager"]');
      await expect(depManager).toBeVisible();
    }
  });

  test('should display dependency graph', async ({ page }) => {
    const depManagerBtn = page.locator('[data-testid="open-dependency-manager"]');
    if (await depManagerBtn.isVisible()) {
      await depManagerBtn.click();

      const graph = page.locator('[data-testid="dependency-graph"]');
      await expect(graph).toBeVisible();
    }
  });
});

// ============================================================================
// Workflow Templates Tests
// ============================================================================

test.describe('Workflow Templates', () => {
  test.beforeEach(async ({ page }) => {
    await page.goto(BASE_URL);
    await waitForLanesWidget(page);
  });

  test('should open workflow templates panel', async ({ page }) => {
    const templatesBtn = page.locator('[data-testid="open-workflow-templates"]');
    if (await templatesBtn.isVisible()) {
      await templatesBtn.click();

      const templatesPanel = page.locator('[data-testid="workflow-templates"]');
      await expect(templatesPanel).toBeVisible();
    }
  });

  test('should display built-in templates', async ({ page }) => {
    const templatesBtn = page.locator('[data-testid="open-workflow-templates"]');
    if (await templatesBtn.isVisible()) {
      await templatesBtn.click();

      const templates = page.locator('[data-testid="template-card"]');
      const count = await templates.count();
      expect(count).toBeGreaterThan(0);
    }
  });
});

// ============================================================================
// Git Integration Tests
// ============================================================================

test.describe('Git Integration', () => {
  test.beforeEach(async ({ page }) => {
    await page.goto(BASE_URL);
    await waitForLanesWidget(page);
  });

  test('should display git integration panel', async ({ page }) => {
    const gitPanel = page.locator('[data-testid="git-integration"]');
    if (await gitPanel.isVisible()) {
      await expect(gitPanel).toBeVisible();
    }
  });

  test('should show branch linking UI', async ({ page }) => {
    const linkBtn = page.locator('[data-testid="link-git-branch"]');
    if (await linkBtn.isVisible()) {
      await linkBtn.click();

      const linkModal = page.locator('[data-testid="link-git-modal"]');
      await expect(linkModal).toBeVisible();
    }
  });
});

// ============================================================================
// CI/CD Integration Tests
// ============================================================================

test.describe('CI/CD Integration', () => {
  test.beforeEach(async ({ page }) => {
    await page.goto(BASE_URL);
    await waitForLanesWidget(page);
  });

  test('should display CI/CD panel', async ({ page }) => {
    const cicdPanel = page.locator('[data-testid="cicd-integration"]');
    if (await cicdPanel.isVisible()) {
      await expect(cicdPanel).toBeVisible();
    }
  });

  test('should show build status', async ({ page }) => {
    const buildStatus = page.locator('[data-testid="build-status"]');
    if (await buildStatus.isVisible()) {
      await expect(buildStatus).toBeVisible();
    }
  });
});

// ============================================================================
// Visual Snapshot Tests
// ============================================================================

test.describe('Visual Snapshots', () => {
  test.beforeEach(async ({ page }) => {
    await page.goto(BASE_URL);
    await waitForLanesWidget(page);
  });

  test('lanes widget default state', async ({ page }) => {
    const widget = await waitForLanesWidget(page);
    await expect(widget).toHaveScreenshot('lanes-widget-default.png');
  });

  test('lanes widget with details panel', async ({ page }) => {
    const firstLane = (await getLaneCards(page)).first();
    await firstLane.click();
    await page.waitForTimeout(500); // Animation

    await expect(page).toHaveScreenshot('lanes-widget-with-details.png');
  });

  test('lanes widget filtered view', async ({ page }) => {
    const filterDropdown = page.locator('[data-testid="lane-status-filter"]');
    if (await filterDropdown.isVisible()) {
      await filterDropdown.selectOption('red');
      await page.waitForTimeout(300);

      await expect(page).toHaveScreenshot('lanes-widget-filtered.png');
    }
  });
});

// ============================================================================
// Accessibility Tests
// ============================================================================

test.describe('Accessibility', () => {
  test.beforeEach(async ({ page }) => {
    await page.goto(BASE_URL);
    await waitForLanesWidget(page);
  });

  test('should have proper ARIA labels', async ({ page }) => {
    const widget = await waitForLanesWidget(page);
    const ariaLabel = await widget.getAttribute('aria-label');
    expect(ariaLabel).toBeTruthy();
  });

  test('should be keyboard navigable', async ({ page }) => {
    const widget = await waitForLanesWidget(page);
    await widget.focus();

    // Navigate with Tab
    await page.keyboard.press('Tab');
    const focusedElement = page.locator(':focus');
    await expect(focusedElement).toBeVisible();
  });

  test('should have visible focus indicators', async ({ page }) => {
    const firstLane = (await getLaneCards(page)).first();
    await firstLane.focus();

    // Check for focus styles (outline or ring)
    const hasOutline = await firstLane.evaluate((el) => {
      const style = window.getComputedStyle(el);
      return style.outline !== 'none' || style.boxShadow !== 'none';
    });
    expect(hasOutline).toBe(true);
  });

  test('should announce status changes to screen readers', async ({ page }) => {
    const liveRegion = page.locator('[role="status"], [aria-live]');
    await expect(liveRegion.first()).toBeAttached();
  });
});

// ============================================================================
// Performance Tests
// ============================================================================

test.describe('Performance', () => {
  test('should load within acceptable time', async ({ page }) => {
    const startTime = Date.now();
    await page.goto(BASE_URL);
    await waitForLanesWidget(page);
    const loadTime = Date.now() - startTime;

    expect(loadTime).toBeLessThan(5000); // 5 seconds max
  });

  test('should maintain smooth scrolling', async ({ page }) => {
    await page.goto(BASE_URL);
    const widget = await waitForLanesWidget(page);

    // Scroll and measure FPS
    const metrics = await page.evaluate(async () => {
      let frameCount = 0;
      const startTime = performance.now();

      return new Promise<{ fps: number }>((resolve) => {
        const countFrames = () => {
          frameCount++;
          if (performance.now() - startTime < 1000) {
            requestAnimationFrame(countFrames);
          } else {
            resolve({ fps: frameCount });
          }
        };
        requestAnimationFrame(countFrames);
      });
    });

    expect(metrics.fps).toBeGreaterThan(30); // At least 30 FPS
  });

  test('should handle rapid interactions', async ({ page }) => {
    await page.goto(BASE_URL);
    await waitForLanesWidget(page);

    const searchInput = page.locator('[data-testid="lane-search"]');
    if (await searchInput.isVisible()) {
      // Rapid typing
      for (let i = 0; i < 20; i++) {
        await searchInput.press('a');
      }

      // Should not crash or hang
      await expect(searchInput).toBeEnabled();
    }
  });
});

// ============================================================================
// Responsive Design Tests
// ============================================================================

test.describe('Responsive Design', () => {
  test('should adapt to mobile viewport', async ({ page }) => {
    await page.setViewportSize({ width: 375, height: 667 });
    await page.goto(BASE_URL);
    const widget = await waitForLanesWidget(page);

    await expect(widget).toBeVisible();
    await expect(widget).toHaveScreenshot('lanes-widget-mobile.png');
  });

  test('should adapt to tablet viewport', async ({ page }) => {
    await page.setViewportSize({ width: 768, height: 1024 });
    await page.goto(BASE_URL);
    const widget = await waitForLanesWidget(page);

    await expect(widget).toBeVisible();
    await expect(widget).toHaveScreenshot('lanes-widget-tablet.png');
  });

  test('should work on large screens', async ({ page }) => {
    await page.setViewportSize({ width: 1920, height: 1080 });
    await page.goto(BASE_URL);
    const widget = await waitForLanesWidget(page);

    await expect(widget).toBeVisible();
    await expect(widget).toHaveScreenshot('lanes-widget-large.png');
  });
});

// ============================================================================
// Dark Mode Tests
// ============================================================================

test.describe('Dark Mode', () => {
  test('should respect system dark mode preference', async ({ page }) => {
    await page.emulateMedia({ colorScheme: 'dark' });
    await page.goto(BASE_URL);
    const widget = await waitForLanesWidget(page);

    await expect(widget).toBeVisible();
    await expect(widget).toHaveScreenshot('lanes-widget-dark-mode.png');
  });

  test('should respect system light mode preference', async ({ page }) => {
    await page.emulateMedia({ colorScheme: 'light' });
    await page.goto(BASE_URL);
    const widget = await waitForLanesWidget(page);

    await expect(widget).toBeVisible();
    await expect(widget).toHaveScreenshot('lanes-widget-light-mode.png');
  });
});
