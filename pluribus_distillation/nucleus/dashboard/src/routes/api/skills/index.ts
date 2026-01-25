/**
 * Skills API - Leverages existing registry infrastructure
 * =========================================================
 *
 * Aggregates skills from:
 * - pbskills_operator.py discovery (SKILL.md files)
 * - registry_emitter.py registries (semops, cagent, effects)
 *
 * Protocol: DKIN v30 | registry_protocol_v1
 * Specification: nucleus/specs/pbskills_operator_v2.md
 */

import type { RequestHandler } from "@builder.io/qwik-city";
import { spawn } from "child_process";

interface SkillItem {
  id: string;
  name: string;
  description: string;
  source: "pbskills" | "semops" | "cagent" | "effects" | "oss";
  tower_level?: number;
  mastery?: number;
  tags?: string[];
  paradigms?: string[];
  status: "ok" | "warn" | "wait";
}

interface SkillsResponse {
  timestamp: string;
  total: number;
  sources: {
    pbskills: number;
    registries: number;
    oss: number;
  };
  skills: SkillItem[];
}

/**
 * Run pbskills_operator.py list command to get discovered skills
 */
async function discoverSkills(): Promise<SkillItem[]> {
  return new Promise((resolve) => {
    const skills: SkillItem[] = [];
    const timeout = setTimeout(() => resolve(skills), 1500);

    try {
      const proc = spawn("python3", [
        "/pluribus/nucleus/tools/pbskills_operator.py",
        "list",
        "--json",
      ], {
        cwd: "/pluribus",
        env: { ...process.env, PYTHONUNBUFFERED: "1" },
      });

      let stdout = "";
      proc.stdout?.on("data", (data) => { stdout += data.toString(); });
      proc.on("close", () => {
        clearTimeout(timeout);
        try {
          const parsed = JSON.parse(stdout);
          if (Array.isArray(parsed)) {
            for (const s of parsed) {
              skills.push({
                id: s.id || "unknown",
                name: s.name || s.id || "Skill",
                description: s.description || "",
                source: "pbskills",
                tower_level: s.tower_level || 1,
                mastery: s.mastery?.score || 0,
                tags: s.tags || [],
                paradigms: s.paradigms || [],
                status: (s.mastery?.score || 0) > 0.6 ? "ok" : "warn",
              });
            }
          }
        } catch {
          // JSON parse failed, use fallback
        }
        resolve(skills);
      });
      proc.on("error", () => {
        clearTimeout(timeout);
        resolve(skills);
      });
    } catch {
      clearTimeout(timeout);
      resolve(skills);
    }
  });
}

/**
 * Run registry_emitter.py to get registry entries
 */
async function getRegistrySkills(): Promise<SkillItem[]> {
  return new Promise((resolve) => {
    const skills: SkillItem[] = [];
    const timeout = setTimeout(() => resolve(skills), 1500);

    try {
      const proc = spawn("python3", [
        "/pluribus/nucleus/tools/registry_emitter.py",
        "emit",
        "--registry", "semops",
        "--registry", "cagent_registry",
        "--registry", "effects_registry",
        "--dry-run",
      ], {
        cwd: "/pluribus",
        env: { ...process.env, PYTHONUNBUFFERED: "1" },
      });

      let stdout = "";
      proc.stdout?.on("data", (data) => { stdout += data.toString(); });
      proc.on("close", () => {
        clearTimeout(timeout);
        try {
          const parsed = JSON.parse(stdout);
          // Map registry entries to skill format
          for (const [regId, info] of Object.entries(parsed) as [string, { entry_count?: number }][]) {
            const count = info?.entry_count || 0;
            skills.push({
              id: `registry:${regId}`,
              name: regId.replace(/_/g, " ").replace(/\b\w/g, c => c.toUpperCase()),
              description: `${count} entries from ${regId} registry`,
              source: regId === "semops" ? "semops" : regId === "cagent_registry" ? "cagent" : "effects",
              tags: ["registry", regId],
              status: count > 0 ? "ok" : "wait",
            });
          }
        } catch {
          // JSON parse failed
        }
        resolve(skills);
      });
      proc.on("error", () => {
        clearTimeout(timeout);
        resolve(skills);
      });
    } catch {
      clearTimeout(timeout);
      resolve(skills);
    }
  });
}

/**
 * Scan OSS skills directories for skill.md files
 */
async function getOssSkills(): Promise<SkillItem[]> {
  const skills: SkillItem[] = [];
  const ossDirs = [
    ".agent/skills",
    ".codex/skills",
    ".gemini/skills",
    ".github/skills",
    "membrane/oss-skills",
  ];

  for (const dir of ossDirs) {
    try {
      const { readdirSync, existsSync, statSync } = await import("fs");
      const fullPath = `/pluribus/${dir}`;
      if (!existsSync(fullPath)) continue;

      const entries = readdirSync(fullPath);
      for (const entry of entries) {
        const entryPath = `${fullPath}/${entry}`;
        if (statSync(entryPath).isDirectory()) {
          skills.push({
            id: `oss:${dir}:${entry}`,
            name: entry.replace(/-/g, " ").replace(/\b\w/g, c => c.toUpperCase()),
            description: `OSS skill from ${dir}`,
            source: "oss",
            tags: ["oss", dir.split("/")[0]],
            status: "ok",
          });
        }
      }
    } catch {
      // Directory doesn't exist or can't be read
    }
  }

  return skills;
}

export const onGet: RequestHandler = async ({ json }) => {
  // Fetch skills from all sources in parallel
  const [pbskills, registrySkills, ossSkills] = await Promise.all([
    discoverSkills(),
    getRegistrySkills(),
    getOssSkills(),
  ]);

  const allSkills = [...pbskills, ...registrySkills, ...ossSkills];

  const response: SkillsResponse = {
    timestamp: new Date().toISOString(),
    total: allSkills.length,
    sources: {
      pbskills: pbskills.length,
      registries: registrySkills.length,
      oss: ossSkills.length,
    },
    skills: allSkills.slice(0, 50), // Limit for LoadingOverlay performance
  };

  json(200, response);
};
