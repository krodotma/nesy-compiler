#!/usr/bin/env node
import assert from "node:assert/strict";
import { classifyFailure, nowIsoCompact } from "../kroma_live_audit_utils.mjs";

let passed = 0;
let failed = 0;

function test(name, fn) {
  try {
    fn();
    console.log(`[PASS] ${name}`);
    passed += 1;
  } catch (err) {
    console.log(`[FAIL] ${name}: ${err.message}`);
    failed += 1;
  }
}

test("classifyFailure detects api", () => {
  assert.equal(classifyFailure("https://kroma.live/api/session"), "api");
});

test("classifyFailure detects module", () => {
  assert.equal(classifyFailure("https://kroma.live/src/app.tsx"), "module");
  assert.equal(classifyFailure("https://kroma.live/@vite/client"), "module");
});

test("classifyFailure detects asset", () => {
  assert.equal(classifyFailure("https://kroma.live/assets/app.css"), "asset");
});

test("classifyFailure detects ws", () => {
  assert.equal(classifyFailure("wss://kroma.live/ws/bus"), "ws");
});

test("classifyFailure falls back to other", () => {
  assert.equal(classifyFailure("https://kroma.live/favicon.ico"), "other");
});

test("nowIsoCompact formats UTC timestamp", () => {
  const stamp = nowIsoCompact(new Date("2025-12-23T04:00:01Z"));
  assert.equal(stamp, "20251223T040001Z");
});

if (failed > 0) {
  console.error(`Tests failed: ${failed}`);
  process.exit(1);
}

console.log(`Tests passed: ${passed}`);
