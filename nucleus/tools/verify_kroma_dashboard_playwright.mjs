import fs from "node:fs";
import path from "node:path";
import process from "node:process";
import crypto from "node:crypto";
import { spawnSync } from "node:child_process";
import { createRequire } from "node:module";
import { nowIsoCompact, classifyFailure } from "./kroma_live_audit_utils.mjs";

const require = createRequire(import.meta.url);
let chromium;
try {
  ({ chromium } = require("playwright"));
} catch {
  const dashRequire = createRequire(new URL("../dashboard/package.json", import.meta.url));
  ({ chromium } = dashRequire("playwright"));
}

function emitBusEvent(topic, data) {
  if (!process.env.PLURIBUS_BUS_DIR) return;
  const busDir = process.env.PLURIBUS_BUS_DIR;
  const busPy = path.join(path.dirname(new URL(import.meta.url).pathname), "agent_bus.py");
  const actor = process.env.PLURIBUS_ACTOR || "kroma_live_audit";
  const payload = JSON.stringify(data);
  const cmd = [
    process.execPath,
    busPy,
    "--bus-dir",
    busDir,
    "pub",
    "--topic",
    topic,
    "--kind",
    "metric",
    "--level",
    "info",
    "--actor",
    actor,
    "--data",
    payload,
  ];
  const result = spawnSync("python3", cmd.slice(1), { stdio: "ignore" });
  if (result.status !== 0) {
    const event = {
      id: crypto.randomUUID(),
      ts: Date.now() / 1000,
      iso: new Date().toISOString(),
      topic,
      kind: "metric",
      level: "info",
      actor,
      data,
    };
    try {
      fs.appendFileSync(path.join(busDir, "events.ndjson"), JSON.stringify(event) + "\n");
    } catch {
      // ignore
    }
  }
}

async function main() {
  const baseURL = process.env.KROMA_BASE_URL || process.env.E2E_BASE_URL || "https://kroma.live";
  const outDir =
    process.env.PLURIBUS_ARTIFACT_DIR ||
    "/pluribus/.pluribus/bus/artifacts/kroma_live";
  fs.mkdirSync(outDir, { recursive: true, mode: 0o700 });

  const ts = nowIsoCompact();
  const screenshotPath = path.join(outDir, `dashboard_${ts}.png`);
  const tracePath = path.join(outDir, `dashboard_${ts}.zip`);
  const reportPath = path.join(outDir, `dashboard_${ts}.json`);

  const report = {
    ts_iso: ts,
    base_url: baseURL,
    ok: false,
    title: null,
    status: null,
    console_errors: [],
    page_errors: [],
    request_failures: [],
    response_samples: [],
    proxy_checks: [],
    module_probes: [],
    notifications: {
      trigger_visible: false,
      panel_visible: false,
      panel_opened: false,
      card_count: 0,
    },
    notes: [],
    screenshot: screenshotPath,
    trace: tracePath,
  };

  const browser = await chromium.launch({ headless: true });
  const context = await browser.newContext({ viewport: { width: 1440, height: 900 } });
  await context.tracing.start({ screenshots: true, snapshots: true, sources: true });
  const page = await context.newPage();

  page.on("console", (msg) => {
    if (msg.type() === "error") {
      report.console_errors.push({ type: msg.type(), text: msg.text(), location: msg.location() });
    }
  });
  page.on("pageerror", (err) => {
    report.page_errors.push({ message: err.message, stack: err.stack });
  });
  page.on("requestfailed", (req) => {
    report.request_failures.push({
      url: req.url(),
      method: req.method(),
      failure: req.failure(),
      resource_type: req.resourceType(),
      category: classifyFailure(req.url()),
    });
  });
  page.on("response", (resp) => {
    const url = resp.url();
    if (url.includes("/api/") || url.includes("/src/") || url.includes("/@")) {
      report.response_samples.push({ url, status: resp.status() });
    }
  });

  try {
    const response = await page.goto(baseURL, { waitUntil: "networkidle", timeout: 60_000 });
    report.status = response?.status?.() ?? null;
    report.title = await page.title();

    await page.waitForTimeout(3_000);

    const notificationTrigger = page.locator(
      '[data-testid="notification-trigger"], .notification-bell, button:has-text("Notifications"), [aria-label="Notifications"]'
    ).first();
    report.notifications.trigger_visible = await notificationTrigger.isVisible().catch(() => false);

    const notificationPanel = page.locator(
      '[data-testid="notification-sidepanel"], .sidepanel, .notification-panel, .edge-panel'
    ).first();
    report.notifications.panel_visible = await notificationPanel.isVisible().catch(() => false);

    if (report.notifications.trigger_visible) {
      await notificationTrigger.click().catch(() => {});
      await page.waitForTimeout(750);
      report.notifications.panel_opened = await notificationPanel.isVisible().catch(() => false);
    }

    report.notifications.card_count = await page
      .locator('.event-item, .notification-card, [data-event-id], .bus-event')
      .count()
      .catch(() => 0);

    await page.screenshot({ path: screenshotPath, fullPage: true });

    const proxyChecks = [
      { name: "session", method: "GET", path: "/api/session" },
      { name: "agents", method: "GET", path: "/api/agents" },
      { name: "browser-status", method: "GET", path: "/api/browser/status" },
      { name: "emit", method: "OPTIONS", path: "/api/emit" },
    ];

    for (const check of proxyChecks) {
      const url = `${baseURL.replace(/\/$/, "")}${check.path}`;
      try {
        const resp = await context.request.fetch(url, { method: check.method, timeout: 15_000 });
        const status = resp.status();
        report.proxy_checks.push({
          name: check.name,
          url,
          method: check.method,
          status,
          ok: status < 500 && status !== 404,
        });
      } catch (err) {
        report.proxy_checks.push({
          name: check.name,
          url,
          method: check.method,
          status: null,
          ok: false,
          error: String(err),
        });
      }
    }

    const moduleProbe = `${baseURL.replace(/\/$/, "")}/src/lib/telemetry/error-collector.ts`;
    try {
      const resp = await context.request.fetch(moduleProbe, { method: "GET", timeout: 15_000 });
      report.module_probes.push({ url: moduleProbe, status: resp.status() });
    } catch (err) {
      report.module_probes.push({ url: moduleProbe, status: null, error: String(err) });
    }

    const criticalFailures = report.request_failures.filter((f) => f.category === "api" || f.category === "module");
    const failedProxy = report.proxy_checks.some((p) => !p.ok);
    const requireNotifications = process.env.KROMA_REQUIRE_NOTIFICATIONS !== "0";
    const notificationsOk =
      report.notifications.trigger_visible ||
      report.notifications.panel_visible ||
      report.notifications.panel_opened;
    if (requireNotifications && !notificationsOk) {
      report.notes.push("notification_panel_missing");
    }
    report.ok =
      report.page_errors.length === 0 &&
      report.console_errors.length === 0 &&
      criticalFailures.length === 0 &&
      !failedProxy &&
      (!requireNotifications || notificationsOk);
  } catch (err) {
    report.notes.push(String(err));
    try {
      await page.screenshot({ path: screenshotPath, fullPage: true });
    } catch {
      // ignore
    }
    report.ok = false;
  } finally {
    await context.tracing.stop({ path: tracePath }).catch(() => {});
    await context.close();
    await browser.close();
    fs.writeFileSync(reportPath, JSON.stringify(report, null, 2) + "\n", { mode: 0o600 });
  }

  emitBusEvent("qa.live.check.summary", {
    base_url: baseURL,
    ok: report.ok,
    console_errors: report.console_errors.length,
    page_errors: report.page_errors.length,
    request_failures: report.request_failures.length,
    proxy_failed: report.proxy_checks.some((p) => !p.ok),
    notifications_ok:
      report.notifications.trigger_visible ||
      report.notifications.panel_visible ||
      report.notifications.panel_opened,
    notifications_cards: report.notifications.card_count,
    report: reportPath,
    screenshot: screenshotPath,
    trace: tracePath,
  });

  process.stdout.write(JSON.stringify({ ok: report.ok, report: reportPath }) + "\n");
  process.exit(report.ok ? 0 : 2);
}

await main();
