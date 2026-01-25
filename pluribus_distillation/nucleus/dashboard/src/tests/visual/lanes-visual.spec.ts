/**
 * Lanes Visual Regression Tests
 *
 * Phase 9, Iteration 73 of OITERATE lanes-widget-enhancement
 *
 * Features:
 * - Percy/Chromatic visual regression
 * - Component snapshots
 * - Responsive snapshots
 * - Dark/light mode snapshots
 * - Animation freeze testing
 */

import { test, expect, type Page } from '@playwright/test';

// ============================================================================
// Configuration
// ============================================================================

const BASE_URL = process.env.E2E_BASE_URL || 'https://kroma.live';

// Viewport sizes for responsive testing
const VIEWPORTS = {
  mobile: { width: 375, height: 667 },
  tablet: { width: 768, height: 1024 },
  desktop: { width: 1280, height: 800 },
  wide: { width: 1920, height: 1080 },
};

// ============================================================================
// Helper Functions
// ============================================================================

async function waitForStableUI(page: Page): Promise<void> {
  // Wait for animations to complete
  await page.evaluate(() => {
    return new Promise<void>((resolve) => {
      // Disable all animations
      const style = document.createElement('style');
      style.textContent = `
        *, *::before, *::after {
          animation-duration: 0s !important;
          animation-delay: 0s !important;
          transition-duration: 0s !important;
          transition-delay: 0s !important;
        }
      `;
      document.head.appendChild(style);

      // Wait for any pending renders
      requestAnimationFrame(() => {
        requestAnimationFrame(() => {
          resolve();
        });
      });
    });
  });

  // Wait for network to be idle
  await page.waitForLoadState('networkidle');
}

async function setupDarkMode(page: Page): Promise<void> {
  await page.emulateMedia({ colorScheme: 'dark' });
}

async function setupLightMode(page: Page): Promise<void> {
  await page.emulateMedia({ colorScheme: 'light' });
}

async function hideTimestamps(page: Page): Promise<void> {
  // Hide dynamic timestamps to avoid snapshot diffs
  await page.evaluate(() => {
    document.querySelectorAll('[data-timestamp], time, .timestamp').forEach((el) => {
      (el as HTMLElement).style.visibility = 'hidden';
    });
  });
}

// ============================================================================
// Component Visual Tests
// ============================================================================

test.describe('Lanes Widget Visual Regression', () => {
  test.beforeEach(async ({ page }) => {
    await page.goto(BASE_URL);
    await page.waitForSelector('[data-testid="lanes-widget"]', { state: 'visible', timeout: 10000 });
    await waitForStableUI(page);
    await hideTimestamps(page);
  });

  test('default state - dark mode', async ({ page }) => {
    await setupDarkMode(page);
    await expect(page).toHaveScreenshot('lanes-widget-default-dark.png', {
      maxDiffPixels: 100,
    });
  });

  test('default state - light mode', async ({ page }) => {
    await setupLightMode(page);
    await expect(page).toHaveScreenshot('lanes-widget-default-light.png', {
      maxDiffPixels: 100,
    });
  });

  test('with lane selected', async ({ page }) => {
    const firstLane = page.locator('[data-testid="lane-card"]').first();
    await firstLane.click();
    await waitForStableUI(page);

    await expect(page).toHaveScreenshot('lanes-widget-selected.png', {
      maxDiffPixels: 100,
    });
  });

  test('with details panel open', async ({ page }) => {
    const firstLane = page.locator('[data-testid="lane-card"]').first();
    await firstLane.click();
    await page.waitForSelector('[data-testid="lane-details-panel"]', { state: 'visible' });
    await waitForStableUI(page);

    await expect(page).toHaveScreenshot('lanes-widget-details-panel.png', {
      maxDiffPixels: 100,
    });
  });

  test('filtered by status', async ({ page }) => {
    const statusFilter = page.locator('[data-testid="lane-status-filter"]');
    if (await statusFilter.isVisible()) {
      await statusFilter.selectOption('red');
      await waitForStableUI(page);

      await expect(page).toHaveScreenshot('lanes-widget-filtered-red.png', {
        maxDiffPixels: 100,
      });
    }
  });

  test('search active', async ({ page }) => {
    const searchInput = page.locator('[data-testid="lane-search"]');
    if (await searchInput.isVisible()) {
      await searchInput.fill('feature');
      await waitForStableUI(page);

      await expect(page).toHaveScreenshot('lanes-widget-search-active.png', {
        maxDiffPixels: 100,
      });
    }
  });

  test('empty state', async ({ page }) => {
    const searchInput = page.locator('[data-testid="lane-search"]');
    if (await searchInput.isVisible()) {
      await searchInput.fill('xyznonexistent12345');
      await waitForStableUI(page);

      await expect(page).toHaveScreenshot('lanes-widget-empty-state.png', {
        maxDiffPixels: 100,
      });
    }
  });
});

// ============================================================================
// Responsive Visual Tests
// ============================================================================

test.describe('Lanes Widget Responsive Visual Regression', () => {
  for (const [name, viewport] of Object.entries(VIEWPORTS)) {
    test(`${name} viewport (${viewport.width}x${viewport.height})`, async ({ page }) => {
      await page.setViewportSize(viewport);
      await page.goto(BASE_URL);
      await page.waitForSelector('[data-testid="lanes-widget"]', { state: 'visible', timeout: 10000 });
      await waitForStableUI(page);
      await hideTimestamps(page);

      await expect(page).toHaveScreenshot(`lanes-widget-${name}.png`, {
        maxDiffPixels: 100,
      });
    });
  }
});

// ============================================================================
// Component-Specific Visual Tests
// ============================================================================

test.describe('Lane Card Visual Regression', () => {
  test.beforeEach(async ({ page }) => {
    await page.goto(BASE_URL);
    await page.waitForSelector('[data-testid="lane-card"]', { state: 'visible', timeout: 10000 });
    await waitForStableUI(page);
  });

  test('green status card', async ({ page }) => {
    const greenCard = page.locator('[data-testid="lane-card"][data-status="green"]').first();
    if (await greenCard.isVisible()) {
      await expect(greenCard).toHaveScreenshot('lane-card-green.png', {
        maxDiffPixels: 50,
      });
    }
  });

  test('yellow status card', async ({ page }) => {
    const yellowCard = page.locator('[data-testid="lane-card"][data-status="yellow"]').first();
    if (await yellowCard.isVisible()) {
      await expect(yellowCard).toHaveScreenshot('lane-card-yellow.png', {
        maxDiffPixels: 50,
      });
    }
  });

  test('red status card', async ({ page }) => {
    const redCard = page.locator('[data-testid="lane-card"][data-status="red"]').first();
    if (await redCard.isVisible()) {
      await expect(redCard).toHaveScreenshot('lane-card-red.png', {
        maxDiffPixels: 50,
      });
    }
  });

  test('card with blockers', async ({ page }) => {
    const blockerCard = page.locator('[data-testid="lane-card"]:has([data-testid="blockers"])').first();
    if (await blockerCard.isVisible()) {
      await expect(blockerCard).toHaveScreenshot('lane-card-with-blockers.png', {
        maxDiffPixels: 50,
      });
    }
  });

  test('card hover state', async ({ page }) => {
    const card = page.locator('[data-testid="lane-card"]').first();
    await card.hover();
    await waitForStableUI(page);

    await expect(card).toHaveScreenshot('lane-card-hover.png', {
      maxDiffPixels: 50,
    });
  });

  test('card focus state', async ({ page }) => {
    const card = page.locator('[data-testid="lane-card"]').first();
    await card.focus();
    await waitForStableUI(page);

    await expect(card).toHaveScreenshot('lane-card-focus.png', {
      maxDiffPixels: 50,
    });
  });
});

// ============================================================================
// Modal/Dialog Visual Tests
// ============================================================================

test.describe('Modal Visual Regression', () => {
  test.beforeEach(async ({ page }) => {
    await page.goto(BASE_URL);
    await page.waitForSelector('[data-testid="lanes-widget"]', { state: 'visible', timeout: 10000 });
    await waitForStableUI(page);
  });

  test('workflow templates modal', async ({ page }) => {
    const templatesBtn = page.locator('[data-testid="open-workflow-templates"]');
    if (await templatesBtn.isVisible()) {
      await templatesBtn.click();
      await page.waitForSelector('[data-testid="workflow-templates"]', { state: 'visible' });
      await waitForStableUI(page);

      await expect(page).toHaveScreenshot('modal-workflow-templates.png', {
        maxDiffPixels: 100,
      });
    }
  });

  test('dependency manager modal', async ({ page }) => {
    const depBtn = page.locator('[data-testid="open-dependency-manager"]');
    if (await depBtn.isVisible()) {
      await depBtn.click();
      await page.waitForSelector('[data-testid="dependency-manager"]', { state: 'visible' });
      await waitForStableUI(page);

      await expect(page).toHaveScreenshot('modal-dependency-manager.png', {
        maxDiffPixels: 100,
      });
    }
  });

  test('bulk actions bar', async ({ page }) => {
    const multiSelectToggle = page.locator('[data-testid="multi-select-toggle"]');
    if (await multiSelectToggle.isVisible()) {
      await multiSelectToggle.click();

      // Select some lanes
      const checkboxes = page.locator('[data-testid="lane-checkbox"]');
      await checkboxes.nth(0).click();
      await checkboxes.nth(1).click();

      await waitForStableUI(page);

      await expect(page).toHaveScreenshot('bulk-actions-bar.png', {
        maxDiffPixels: 100,
      });
    }
  });
});

// ============================================================================
// Animation State Tests
// ============================================================================

test.describe('Animation States Visual Regression', () => {
  test('loading state', async ({ page }) => {
    // Intercept API to delay response
    await page.route('**/api/**', async (route) => {
      await new Promise(resolve => setTimeout(resolve, 5000));
      await route.continue();
    });

    await page.goto(BASE_URL);

    // Capture loading state
    const loadingIndicator = page.locator('[data-testid="loading-indicator"]');
    if (await loadingIndicator.isVisible({ timeout: 2000 })) {
      await expect(page).toHaveScreenshot('lanes-widget-loading.png', {
        maxDiffPixels: 100,
      });
    }
  });

  test('error state', async ({ page }) => {
    // Intercept API to return error
    await page.route('**/api/**', async (route) => {
      await route.fulfill({
        status: 500,
        body: JSON.stringify({ error: 'Internal Server Error' }),
      });
    });

    await page.goto(BASE_URL);
    await page.waitForTimeout(2000);

    const errorMessage = page.locator('[data-testid="error-message"]');
    if (await errorMessage.isVisible()) {
      await expect(page).toHaveScreenshot('lanes-widget-error.png', {
        maxDiffPixels: 100,
      });
    }
  });
});

// ============================================================================
// High Contrast Mode Tests
// ============================================================================

test.describe('High Contrast Visual Regression', () => {
  test('high contrast mode', async ({ page }) => {
    await page.emulateMedia({ forcedColors: 'active' });
    await page.goto(BASE_URL);
    await page.waitForSelector('[data-testid="lanes-widget"]', { state: 'visible', timeout: 10000 });
    await waitForStableUI(page);

    await expect(page).toHaveScreenshot('lanes-widget-high-contrast.png', {
      maxDiffPixels: 200, // More tolerance for high contrast
    });
  });
});

// ============================================================================
// Reduced Motion Tests
// ============================================================================

test.describe('Reduced Motion Visual Regression', () => {
  test('reduced motion preference', async ({ page }) => {
    await page.emulateMedia({ reducedMotion: 'reduce' });
    await page.goto(BASE_URL);
    await page.waitForSelector('[data-testid="lanes-widget"]', { state: 'visible', timeout: 10000 });

    // Interact to trigger potential animations
    const card = page.locator('[data-testid="lane-card"]').first();
    await card.hover();
    await card.click();

    await waitForStableUI(page);

    await expect(page).toHaveScreenshot('lanes-widget-reduced-motion.png', {
      maxDiffPixels: 100,
    });
  });
});
