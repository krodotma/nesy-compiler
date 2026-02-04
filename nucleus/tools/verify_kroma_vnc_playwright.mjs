import fs from "node:fs";
import path from "node:path";
import process from "node:process";

import { createRequire } from "node:module";

const require = createRequire(import.meta.url);
let chromium;
try {
  ({ chromium } = require("playwright"));
} catch {
  const dashRequire = createRequire(new URL("../dashboard/package.json", import.meta.url));
  ({ chromium } = dashRequire("playwright"));
}

function nowIsoCompact() {
  const d = new Date();
  const pad = (n) => String(n).padStart(2, "0");
  return [
    d.getUTCFullYear(),
    pad(d.getUTCMonth() + 1),
    pad(d.getUTCDate()),
    "T",
    pad(d.getUTCHours()),
    pad(d.getUTCMinutes()),
    pad(d.getUTCSeconds()),
    "Z",
  ].join("");
}

async function main() {
  const baseURL = process.env.KROMA_BASE_URL || "https://kroma.live";
  const outDir =
    process.env.PLURIBUS_ARTIFACT_DIR ||
    "/pluribus/.pluribus/bus/artifacts/kroma_vnc";
  fs.mkdirSync(outDir, { recursive: true, mode: 0o700 });

  const ts = nowIsoCompact();
  const screenshotPath = path.join(outDir, `vnc_auth_${ts}.png`);
  const reportPath = path.join(outDir, `vnc_auth_${ts}.json`);

  const report = {
    ts_iso: ts,
    base_url: baseURL,
    ok: false,
    steps: [],
    console_errors: [],
    page_responses: [],
    page_failures: [],
    iframe: {
      src: null,
      loaded: false,
      url: null,
      title: null,
    },
    notes: [],
  };

  const browser = await chromium.launch({ headless: true });
  const context = await browser.newContext({
    viewport: { width: 1440, height: 900 },
  });
  const page = await context.newPage();

  page.on("console", (msg) => {
    if (msg.type() === "error") {
      report.console_errors.push({ type: msg.type(), text: msg.text() });
    }
  });
  page.on("requestfailed", (req) => {
    report.page_failures.push({
      url: req.url(),
      method: req.method(),
      failure: req.failure(),
    });
  });
  page.on("response", (resp) => {
    const url = resp.url();
    if (url.includes("/vnc/") || url.includes("/api/browser/") || url.includes("/api/")) {
      report.page_responses.push({
        url,
        status: resp.status(),
      });
    }
  });

  try {
    const url = new URL(baseURL);
    url.searchParams.set("view", "browser-auth");
    report.steps.push({ step: "goto", url: url.toString() });
    await page.goto(url.toString(), { waitUntil: "domcontentloaded", timeout: 60_000 });

    const iframe = page.locator('iframe[title="noVNC Browser Access"]').first();
    await iframe.waitFor({ timeout: 30_000 });

    report.iframe.src = await iframe.getAttribute("src");
    report.steps.push({ step: "iframe_visible", src: report.iframe.src });

    // Best-effort: if the iframe is blocked/refused, contentFrame() can be null.
    const frame = await iframe.contentFrame();
    if (frame) {
      report.iframe.loaded = true;
      // Playwright has shipped both `frame.url()` and `frame.url` across versions.
      report.iframe.url = typeof frame.url === "function" ? frame.url() : frame.url;
      report.iframe.title =
        typeof frame.title === "function" ? await frame.title().catch(() => null) : null;
      report.steps.push({ step: "iframe_loaded", url: report.iframe.url, title: report.iframe.title });
    } else {
      report.notes.push("iframe.contentFrame() returned null (often indicates refused-to-connect or blocked frame).");
    }

    await page.waitForTimeout(1_000);
    await page.screenshot({ path: screenshotPath, fullPage: true });
    report.ok = Boolean(frame);
  } catch (err) {
    report.notes.push(String(err));
    await page.screenshot({ path: screenshotPath, fullPage: true }).catch(() => {});
    report.ok = false;
  } finally {
    await context.close();
    await browser.close();
    fs.writeFileSync(reportPath, JSON.stringify(report, null, 2) + "\n", { mode: 0o600 });
  }

  process.stdout.write(JSON.stringify({ ok: report.ok, screenshot: screenshotPath, report: reportPath }) + "\n");
  process.exit(report.ok ? 0 : 2);
}

await main();
