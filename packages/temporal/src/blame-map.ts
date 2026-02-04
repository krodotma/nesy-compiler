/**
 * BlameMap: Author Archetype Analysis
 *
 * Step 20 of NeSy Evolution Phase 2 (Temporal Archeology)
 *
 * Maps code blocks to Author Archetypes:
 * - The Refactorer: Improves structure without adding features
 * - The Patcher: Quick fixes, often introduces tech debt
 * - The Builder: Adds new features and capabilities
 * - The Optimizer: Performance improvements
 * - The Documenter: Adds comments and documentation
 * - The Cleaner: Removes dead code, simplifies
 *
 * Uses git blame + commit message analysis.
 */

import type { GitCommit, BlameLine } from './git-walker';

export type AuthorArchetype =
  | 'refactorer'
  | 'patcher'
  | 'builder'
  | 'optimizer'
  | 'documenter'
  | 'cleaner'
  | 'unknown';

export interface AuthorProfile {
  name: string;
  email?: string;
  primaryArchetype: AuthorArchetype;
  archetypeDistribution: Map<AuthorArchetype, number>;
  totalCommits: number;
  totalAdditions: number;
  totalDeletions: number;
  filesModified: Set<string>;
  activePeriod: { first: Date; last: Date };
  commitPatterns: CommitPattern;
}

export interface CommitPattern {
  avgCommitSize: number;
  avgMessageLength: number;
  prefersSmallCommits: boolean;
  usesConventionalCommits: boolean;
  peakHour: number;
}

export interface BlameSegment {
  startLine: number;
  endLine: number;
  author: string;
  commitHash: string;
  date: Date;
  archetype: AuthorArchetype;
  age: number;  // Days since this segment was written
}

export interface FileBlameMap {
  path: string;
  segments: BlameSegment[];
  authorDistribution: Map<string, number>;  // Author -> line count
  archetypeDistribution: Map<AuthorArchetype, number>;  // Archetype -> line count
  oldestSegment: BlameSegment | null;
  newestSegment: BlameSegment | null;
  dominantAuthor: string;
  dominantArchetype: AuthorArchetype;
}

/**
 * Classify a commit into an archetype based on message and changes.
 */
export function classifyCommit(commit: GitCommit): AuthorArchetype {
  const msg = commit.message.toLowerCase();

  // Check for conventional commit prefixes
  if (msg.startsWith('refactor') || msg.startsWith('refact')) {
    return 'refactorer';
  }
  if (msg.startsWith('fix') || msg.startsWith('bugfix') || msg.startsWith('hotfix')) {
    return 'patcher';
  }
  if (msg.startsWith('feat') || msg.startsWith('feature') || msg.startsWith('add')) {
    return 'builder';
  }
  if (msg.startsWith('perf') || msg.startsWith('optim')) {
    return 'optimizer';
  }
  if (msg.startsWith('docs') || msg.startsWith('comment')) {
    return 'documenter';
  }
  if (msg.startsWith('chore') || msg.startsWith('clean') || msg.startsWith('remove')) {
    return 'cleaner';
  }

  // Keyword analysis
  const keywords: Record<AuthorArchetype, string[]> = {
    refactorer: ['refactor', 'restructure', 'reorganize', 'simplify', 'extract', 'rename'],
    patcher: ['fix', 'patch', 'bug', 'issue', 'error', 'crash', 'broken'],
    builder: ['add', 'implement', 'create', 'new', 'feature', 'support'],
    optimizer: ['optimize', 'performance', 'speed', 'cache', 'reduce', 'improve'],
    documenter: ['document', 'comment', 'readme', 'jsdoc', 'describe'],
    cleaner: ['remove', 'delete', 'clean', 'unused', 'deprecated', 'dead code'],
    unknown: [],
  };

  for (const [archetype, words] of Object.entries(keywords)) {
    if (words.some(word => msg.includes(word))) {
      return archetype as AuthorArchetype;
    }
  }

  // Heuristic based on change pattern
  const totalChanges = commit.files.reduce(
    (sum, f) => sum + f.additions + f.deletions,
    0
  );
  const totalAdditions = commit.files.reduce((sum, f) => sum + f.additions, 0);
  const totalDeletions = commit.files.reduce((sum, f) => sum + f.deletions, 0);

  // More deletions than additions = cleaner
  if (totalDeletions > totalAdditions * 1.5) {
    return 'cleaner';
  }

  // Many additions, few deletions = builder
  if (totalAdditions > totalDeletions * 2) {
    return 'builder';
  }

  // Balanced changes = refactorer
  if (totalAdditions > 0 && totalDeletions > 0) {
    const ratio = Math.min(totalAdditions, totalDeletions) / Math.max(totalAdditions, totalDeletions);
    if (ratio > 0.7) {
      return 'refactorer';
    }
  }

  return 'unknown';
}

/**
 * Build an author profile from their commits.
 */
export function buildAuthorProfile(
  author: string,
  commits: GitCommit[]
): AuthorProfile {
  const archetypeCounts = new Map<AuthorArchetype, number>();
  const filesModified = new Set<string>();
  let totalAdditions = 0;
  let totalDeletions = 0;
  let totalMessageLength = 0;
  let totalCommitSize = 0;
  let conventionalCount = 0;

  const dates: Date[] = [];

  for (const commit of commits) {
    dates.push(commit.date);

    // Classify commit
    const archetype = classifyCommit(commit);
    archetypeCounts.set(archetype, (archetypeCounts.get(archetype) || 0) + 1);

    // Track files
    for (const file of commit.files) {
      filesModified.add(file.path);
      totalAdditions += file.additions;
      totalDeletions += file.deletions;
      totalCommitSize += file.additions + file.deletions;
    }

    // Message analysis
    totalMessageLength += commit.message.length;
    if (/^(feat|fix|docs|style|refactor|perf|test|chore|ci|build)(\(.+\))?:/.test(commit.message)) {
      conventionalCount++;
    }
  }

  // Find primary archetype
  let primaryArchetype: AuthorArchetype = 'unknown';
  let maxCount = 0;
  for (const [arch, count] of archetypeCounts) {
    if (count > maxCount) {
      maxCount = count;
      primaryArchetype = arch;
    }
  }

  // Calculate activity period
  const sortedDates = dates.sort((a, b) => a.getTime() - b.getTime());
  const first = sortedDates[0] || new Date();
  const last = sortedDates[sortedDates.length - 1] || new Date();

  // Calculate peak hour
  const hourCounts = new Array(24).fill(0);
  for (const date of dates) {
    hourCounts[date.getHours()]++;
  }
  const peakHour = hourCounts.indexOf(Math.max(...hourCounts));

  return {
    name: author,
    primaryArchetype,
    archetypeDistribution: archetypeCounts,
    totalCommits: commits.length,
    totalAdditions,
    totalDeletions,
    filesModified,
    activePeriod: { first, last },
    commitPatterns: {
      avgCommitSize: commits.length > 0 ? totalCommitSize / commits.length : 0,
      avgMessageLength: commits.length > 0 ? totalMessageLength / commits.length : 0,
      prefersSmallCommits: totalCommitSize / commits.length < 50,
      usesConventionalCommits: conventionalCount / commits.length > 0.5,
      peakHour,
    },
  };
}

/**
 * Build a blame map for a file.
 */
export function buildFileBlameMap(
  path: string,
  blameLines: BlameLine[],
  authorProfiles: Map<string, AuthorProfile>
): FileBlameMap {
  const segments: BlameSegment[] = [];
  const authorCounts = new Map<string, number>();
  const archetypeCounts = new Map<AuthorArchetype, number>();
  const now = new Date();

  let currentSegment: BlameSegment | null = null;

  for (let i = 0; i < blameLines.length; i++) {
    const line = blameLines[i];
    const lineNum = i + 1;

    // Get author's archetype
    const profile = authorProfiles.get(line.author);
    const archetype = profile?.primaryArchetype || 'unknown';

    // Count lines
    authorCounts.set(line.author, (authorCounts.get(line.author) || 0) + 1);
    archetypeCounts.set(archetype, (archetypeCounts.get(archetype) || 0) + 1);

    // Build segments
    if (currentSegment &&
        currentSegment.author === line.author &&
        currentSegment.commitHash === line.hash) {
      // Extend current segment
      currentSegment.endLine = lineNum;
    } else {
      // Start new segment
      if (currentSegment) {
        segments.push(currentSegment);
      }
      currentSegment = {
        startLine: lineNum,
        endLine: lineNum,
        author: line.author,
        commitHash: line.hash,
        date: line.timestamp,
        archetype,
        age: (now.getTime() - line.timestamp.getTime()) / (1000 * 60 * 60 * 24),
      };
    }
  }

  if (currentSegment) {
    segments.push(currentSegment);
  }

  // Find dominant author and archetype
  let dominantAuthor = 'unknown';
  let maxAuthorLines = 0;
  for (const [author, count] of authorCounts) {
    if (count > maxAuthorLines) {
      maxAuthorLines = count;
      dominantAuthor = author;
    }
  }

  let dominantArchetype: AuthorArchetype = 'unknown';
  let maxArchetypeLines = 0;
  for (const [arch, count] of archetypeCounts) {
    if (count > maxArchetypeLines) {
      maxArchetypeLines = count;
      dominantArchetype = arch;
    }
  }

  // Find oldest/newest segments
  const sortedSegments = [...segments].sort((a, b) => a.date.getTime() - b.date.getTime());
  const oldestSegment = sortedSegments[0] || null;
  const newestSegment = sortedSegments[sortedSegments.length - 1] || null;

  return {
    path,
    segments,
    authorDistribution: authorCounts,
    archetypeDistribution: archetypeCounts,
    oldestSegment,
    newestSegment,
    dominantAuthor,
    dominantArchetype,
  };
}

/**
 * Get archetype description for documentation.
 */
export function getArchetypeDescription(archetype: AuthorArchetype): string {
  const descriptions: Record<AuthorArchetype, string> = {
    refactorer: 'Improves code structure without adding features. Values clean architecture.',
    patcher: 'Quick fixes to address immediate issues. May introduce tech debt.',
    builder: 'Adds new features and capabilities. Focuses on functionality.',
    optimizer: 'Improves performance and efficiency. Data-driven approach.',
    documenter: 'Adds comments, documentation, and explanations. Values clarity.',
    cleaner: 'Removes dead code and simplifies. Reduces complexity.',
    unknown: 'Commit pattern does not match known archetypes.',
  };
  return descriptions[archetype];
}
