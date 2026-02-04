export function nowIsoCompact(date = new Date()) {
  const pad = (n) => String(n).padStart(2, "0");
  return [
    date.getUTCFullYear(),
    pad(date.getUTCMonth() + 1),
    pad(date.getUTCDate()),
    "T",
    pad(date.getUTCHours()),
    pad(date.getUTCMinutes()),
    pad(date.getUTCSeconds()),
    "Z",
  ].join("");
}

export function classifyFailure(url) {
  if (url.includes("/api/")) return "api";
  if (url.includes("/src/") || url.includes("/@")) return "module";
  if (url.includes("/assets/")) return "asset";
  if (url.includes("/ws/")) return "ws";
  return "other";
}
