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

function readVncPassword() {
  const envPassword = (process.env.VNC_PASSWORD || "").trim();
  if (envPassword) return { password: envPassword, source: "env" };

  const passwordFile = process.env.VNC_PASSWORD_FILE || "/pluribus/.pluribus/vnc_password.txt";
  try {
    const pw = fs.readFileSync(passwordFile, "utf8").replace(/\r?\n/g, "").slice(0, 64);
    if (pw) return { password: pw, source: passwordFile };
  } catch {
    // Ignore.
  }
  return { password: "", source: "missing" };
}

async function waitForConnected(frame, report) {
  try {
    await frame.locator("html.noVNC_connected").waitFor({ timeout: 30_000 });
    report.vnc.connected = true;
    return true;
  } catch {
    report.vnc.connected = false;
    return false;
  }
}

async function main() {
  const baseURL = process.env.KROMA_BASE_URL || "https://kroma.live";
  const outDir =
    process.env.PLURIBUS_ARTIFACT_DIR || "/pluribus/.pluribus/bus/artifacts/kroma_vnc";
  fs.mkdirSync(outDir, { recursive: true, mode: 0o700 });

  const ts = nowIsoCompact();
  const screenshotPath = path.join(outDir, `vnc_connect_${ts}.png`);
  const reportPath = path.join(outDir, `vnc_connect_${ts}.json`);

  const { password, source } = readVncPassword();
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
    vnc: {
      password_source: source,
      password_len: password.length,
      connected: false,
      novnc_status_text: null,
    },
    notes: [],
  };

  if (!password) {
    report.notes.push("Missing VNC password (set VNC_PASSWORD or ensure VNC_PASSWORD_FILE exists).");
  }

  const browser = await chromium.launch({ headless: true });
  const context = await browser.newContext({ viewport: { width: 1440, height: 900 } });
  const page = await context.newPage();

  page.on("console", (msg) => {
    if (msg.type() === "error") {
      report.console_errors.push({ type: msg.type(), text: msg.text() });
    }
  });
  page.on("requestfailed", (req) => {
    report.page_failures.push({ url: req.url(), method: req.method(), failure: req.failure() });
  });
  page.on("response", (resp) => {
    const url = resp.url();
    if (url.includes("/vnc/") || url.includes("/api/browser/") || url.includes("/api/")) {
      report.page_responses.push({ url, status: resp.status() });
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

    const frame = await iframe.contentFrame();
    if (!frame) {
      report.notes.push(
        "iframe.contentFrame() returned null (often indicates refused-to-connect or blocked frame).",
      );
      throw new Error("noVNC iframe not accessible");
    }

    report.iframe.loaded = true;
    report.iframe.url = typeof frame.url === "function" ? frame.url() : frame.url;
    report.iframe.title =
      typeof frame.title === "function" ? await frame.title().catch(() => null) : null;
    report.steps.push({ step: "iframe_loaded", url: report.iframe.url, title: report.iframe.title });

    // If we are already connected, short-circuit.
    report.steps.push({ step: "check_connected" });
    const alreadyConnected = await frame
      .locator("html.noVNC_connected")
      .isVisible()
      .catch(() => false);
    if (alreadyConnected) {
      report.vnc.connected = true;
      report.steps.push({ step: "already_connected" });
    } else if (password) {
      report.steps.push({ step: "click_connect" });
      const connectButton = frame.locator("#noVNC_connect_button");
      await connectButton.waitFor({ state: "visible", timeout: 30_000 });
      // noVNC animates/relayouts; Playwright can refuse "unstable" clicks. Use DOM click instead.
      await connectButton.evaluate((el) => el.click());

      report.steps.push({ step: "fill_password" });
      const pwInput = frame.locator("#noVNC_password_input");
      await pwInput.waitFor({ state: "visible", timeout: 30_000 });
      await pwInput.fill(password);

      report.steps.push({ step: "send_credentials" });
      const credButton = frame.locator("#noVNC_credentials_button");
      await credButton.waitFor({ state: "visible", timeout: 30_000 });
      await credButton.evaluate((el) => el.click());

      report.steps.push({ step: "wait_connected" });
      await waitForConnected(frame, report);
    }

    report.vnc.novnc_status_text = await frame
      .locator("#noVNC_status")
      .innerText()
      .catch(() => null);

    await page.waitForTimeout(1_000);
    await page.screenshot({ path: screenshotPath, fullPage: true });

    report.ok = Boolean(report.vnc.connected);
  } catch (err) {
    report.notes.push(String(err));
    await page.screenshot({ path: screenshotPath, fullPage: true }).catch(() => {});
    report.ok = false;
  } finally {
    await context.close();
    await browser.close();
    fs.writeFileSync(reportPath, JSON.stringify(report, null, 2) + "\n", { mode: 0o600 });
  }

  process.stdout.write(
    JSON.stringify({ ok: report.ok, connected: report.vnc.connected, screenshot: screenshotPath, report: reportPath }) +
      "\n",
  );
  process.exit(report.ok ? 0 : 2);
}

await main();
