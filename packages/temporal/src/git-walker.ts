/**
 * GitWalker: Fast Git History Traversal
 *
 * Step 17 of NeSy Evolution Phase 2 (Temporal Archeology)
 *
 * Traverses git log with patches to extract:
 * - Commit metadata (author, date, message)
 * - File changes (additions, deletions, modifications)
 * - Diff hunks for semantic analysis
 *
 * Uses shell git commands for performance (vs isomorphic-git).
 */

import { spawn } from 'child_process';
import * as path from 'path';

export interface GitCommit {
  hash: string;
  shortHash: string;
  author: string;
  authorEmail: string;
  date: Date;
  message: string;
  files: GitFileChange[];
}

export interface GitFileChange {
  path: string;
  oldPath?: string;  // For renames
  status: 'A' | 'M' | 'D' | 'R' | 'C';  // Added, Modified, Deleted, Renamed, Copied
  additions: number;
  deletions: number;
  hunks?: DiffHunk[];
}

export interface DiffHunk {
  oldStart: number;
  oldCount: number;
  newStart: number;
  newCount: number;
  content: string;
}

export interface GitWalkerConfig {
  repoPath: string;
  branch?: string;
  since?: Date;
  until?: Date;
  maxCommits?: number;
  includeDiffs?: boolean;
  fileFilter?: RegExp;
}

const DEFAULT_CONFIG: Partial<GitWalkerConfig> = {
  branch: 'HEAD',
  maxCommits: 1000,
  includeDiffs: true,
};

/**
 * GitWalker: Iterate through git history efficiently.
 */
export class GitWalker {
  private config: GitWalkerConfig;

  constructor(config: GitWalkerConfig) {
    this.config = { ...DEFAULT_CONFIG, ...config };
  }

  /**
   * Walk through commits, yielding each one.
   */
  async *walk(): AsyncGenerator<GitCommit> {
    const args = this.buildLogArgs();
    const output = await this.runGit(['log', ...args]);

    const commits = this.parseLogOutput(output);
    for (const commit of commits) {
      if (this.config.fileFilter) {
        commit.files = commit.files.filter(f =>
          this.config.fileFilter!.test(f.path)
        );
      }
      yield commit;
    }
  }

  /**
   * Get all commits as an array.
   */
  async getCommits(): Promise<GitCommit[]> {
    const commits: GitCommit[] = [];
    for await (const commit of this.walk()) {
      commits.push(commit);
    }
    return commits;
  }

  /**
   * Get file history for a specific path.
   */
  async getFileHistory(filePath: string): Promise<GitCommit[]> {
    const args = [
      'log',
      '--format=%H|%h|%an|%ae|%aI|%s',
      '--numstat',
      '--follow',
      '-M',  // Detect renames
      `--max-count=${this.config.maxCommits}`,
      '--',
      filePath,
    ];

    const output = await this.runGit(args);
    return this.parseLogOutput(output);
  }

  /**
   * Get blame information for a file.
   */
  async getBlame(filePath: string): Promise<BlameLine[]> {
    const args = [
      'blame',
      '--porcelain',
      filePath,
    ];

    try {
      const output = await this.runGit(args);
      return this.parseBlameOutput(output);
    } catch {
      return [];
    }
  }

  /**
   * Get file content at a specific commit.
   */
  async getFileAtCommit(filePath: string, commitHash: string): Promise<string | null> {
    try {
      return await this.runGit(['show', `${commitHash}:${filePath}`]);
    } catch {
      return null;
    }
  }

  /**
   * Get diff between two commits for a file.
   */
  async getDiff(
    filePath: string,
    fromCommit: string,
    toCommit: string
  ): Promise<DiffHunk[]> {
    const args = [
      'diff',
      fromCommit,
      toCommit,
      '--',
      filePath,
    ];

    try {
      const output = await this.runGit(args);
      return this.parseDiffHunks(output);
    } catch {
      return [];
    }
  }

  /**
   * Get list of all authors with commit counts.
   */
  async getAuthors(): Promise<Map<string, number>> {
    const output = await this.runGit([
      'shortlog',
      '-sne',
      this.config.branch || 'HEAD',
    ]);

    const authors = new Map<string, number>();
    const lines = output.trim().split('\n');

    for (const line of lines) {
      const match = line.match(/^\s*(\d+)\s+(.+?)\s+<(.+)>$/);
      if (match) {
        const [, count, name] = match;
        authors.set(name, parseInt(count, 10));
      }
    }

    return authors;
  }

  /**
   * Build git log arguments.
   */
  private buildLogArgs(): string[] {
    const args = [
      '--format=%H|%h|%an|%ae|%aI|%s',
      '--numstat',
    ];

    if (this.config.maxCommits) {
      args.push(`--max-count=${this.config.maxCommits}`);
    }

    if (this.config.since) {
      args.push(`--since=${this.config.since.toISOString()}`);
    }

    if (this.config.until) {
      args.push(`--until=${this.config.until.toISOString()}`);
    }

    args.push(this.config.branch || 'HEAD');

    return args;
  }

  /**
   * Parse git log output.
   */
  private parseLogOutput(output: string): GitCommit[] {
    const commits: GitCommit[] = [];
    const lines = output.split('\n');
    let currentCommit: GitCommit | null = null;

    for (const line of lines) {
      if (line.includes('|')) {
        // Commit header line
        const parts = line.split('|');
        if (parts.length >= 6) {
          if (currentCommit) {
            commits.push(currentCommit);
          }
          currentCommit = {
            hash: parts[0],
            shortHash: parts[1],
            author: parts[2],
            authorEmail: parts[3],
            date: new Date(parts[4]),
            message: parts.slice(5).join('|'),
            files: [],
          };
        }
      } else if (currentCommit && line.trim()) {
        // Numstat line: additions deletions path
        const match = line.match(/^(\d+|-)\t(\d+|-)\t(.+)$/);
        if (match) {
          const [, add, del, filePath] = match;
          const additions = add === '-' ? 0 : parseInt(add, 10);
          const deletions = del === '-' ? 0 : parseInt(del, 10);

          // Check for rename: oldPath => newPath
          const renameMatch = filePath.match(/^(.+?)\s*=>\s*(.+)$/);
          if (renameMatch) {
            currentCommit.files.push({
              path: renameMatch[2].trim(),
              oldPath: renameMatch[1].trim(),
              status: 'R',
              additions,
              deletions,
            });
          } else {
            currentCommit.files.push({
              path: filePath,
              status: additions > 0 && deletions > 0 ? 'M' :
                      additions > 0 ? 'A' : 'D',
              additions,
              deletions,
            });
          }
        }
      }
    }

    if (currentCommit) {
      commits.push(currentCommit);
    }

    return commits;
  }

  /**
   * Parse git blame output.
   */
  private parseBlameOutput(output: string): BlameLine[] {
    const lines: BlameLine[] = [];
    const chunks = output.split(/^([a-f0-9]{40})/gm).filter(Boolean);

    for (let i = 0; i < chunks.length; i += 2) {
      const hash = chunks[i];
      const content = chunks[i + 1] || '';

      const authorMatch = content.match(/^author (.+)$/m);
      const timeMatch = content.match(/^author-time (\d+)$/m);
      const lineMatch = content.match(/^\t(.*)$/m);

      if (authorMatch && lineMatch) {
        lines.push({
          hash,
          author: authorMatch[1],
          timestamp: timeMatch ? new Date(parseInt(timeMatch[1], 10) * 1000) : new Date(),
          content: lineMatch[1],
        });
      }
    }

    return lines;
  }

  /**
   * Parse diff hunks from diff output.
   */
  private parseDiffHunks(output: string): DiffHunk[] {
    const hunks: DiffHunk[] = [];
    const hunkRegex = /^@@\s+-(\d+)(?:,(\d+))?\s+\+(\d+)(?:,(\d+))?\s+@@(.*)$/gm;
    let match;

    while ((match = hunkRegex.exec(output)) !== null) {
      const [, oldStart, oldCount, newStart, newCount] = match;

      // Find hunk content (until next @@ or end)
      const startIdx = match.index + match[0].length;
      const nextMatch = output.indexOf('\n@@', startIdx);
      const endIdx = nextMatch === -1 ? output.length : nextMatch;
      const content = output.slice(startIdx, endIdx).trim();

      hunks.push({
        oldStart: parseInt(oldStart, 10),
        oldCount: parseInt(oldCount || '1', 10),
        newStart: parseInt(newStart, 10),
        newCount: parseInt(newCount || '1', 10),
        content,
      });
    }

    return hunks;
  }

  /**
   * Run a git command and return output.
   */
  private runGit(args: string[]): Promise<string> {
    return new Promise((resolve, reject) => {
      const proc = spawn('git', args, {
        cwd: this.config.repoPath,
        stdio: ['ignore', 'pipe', 'pipe'],
      });

      let stdout = '';
      let stderr = '';

      proc.stdout.on('data', (data) => { stdout += data; });
      proc.stderr.on('data', (data) => { stderr += data; });

      proc.on('close', (code) => {
        if (code === 0) {
          resolve(stdout);
        } else {
          reject(new Error(`git ${args[0]} failed: ${stderr}`));
        }
      });

      proc.on('error', reject);
    });
  }
}

export interface BlameLine {
  hash: string;
  author: string;
  timestamp: Date;
  content: string;
}

/**
 * Quick function to get commit history for a repo.
 */
export async function getCommitHistory(
  repoPath: string,
  options?: Partial<Omit<GitWalkerConfig, 'repoPath'>>
): Promise<GitCommit[]> {
  const walker = new GitWalker({ repoPath, ...options });
  return walker.getCommits();
}

/**
 * Get file history with full details.
 */
export async function getFileHistory(
  repoPath: string,
  filePath: string
): Promise<GitCommit[]> {
  const walker = new GitWalker({ repoPath });
  return walker.getFileHistory(filePath);
}
