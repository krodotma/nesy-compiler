/**
 * TemporalSignal: Schema for Git-Derived Metrics
 *
 * Step 18 of NeSy Evolution Phase 2 (Temporal Archeology)
 *
 * Defines temporal signals extracted from git history:
 * - commit_freq: How often the file is touched
 * - author_entropy: How many different authors
 * - churn_rate: Ratio of changes to file size
 * - refactor_velocity: Speed of structural changes
 *
 * These signals help identify:
 * - Stable, trusted code (low churn, many refactors survived)
 * - Thrash (high churn, low semantic change)
 * - Bus factor risks (single author)
 */

import type { GitCommit, GitFileChange } from './git-walker';

/**
 * Temporal signal for a single file.
 */
export interface TemporalSignal {
  /** File path */
  path: string;

  /** Commit frequency (commits per month) */
  commitFreq: number;

  /** Author entropy (0 = single author, 1 = evenly distributed) */
  authorEntropy: number;

  /** Unique author count */
  authorCount: number;

  /** Churn rate (total changes / current size) */
  churnRate: number;

  /** Total lines added historically */
  totalAdditions: number;

  /** Total lines deleted historically */
  totalDeletions: number;

  /** Net change (additions - deletions) */
  netChange: number;

  /** Refactor velocity (structural changes per month) */
  refactorVelocity: number;

  /** Stability score (0-1, higher = more stable) */
  stabilityScore: number;

  /** Age in days since first commit */
  ageInDays: number;

  /** Days since last modification */
  daysSinceLastChange: number;

  /** First commit date */
  createdAt: Date;

  /** Last commit date */
  lastModifiedAt: Date;

  /** Primary author (most commits) */
  primaryAuthor: string;

  /** Recent activity (commits in last 30 days) */
  recentActivity: number;

  /** Bus factor (1 = single point of failure) */
  busFactor: number;

  /** Rename count (times file was renamed) */
  renameCount: number;
}

/**
 * Temporal signal aggregated at directory level.
 */
export interface DirectoryTemporalSignal {
  path: string;
  fileCount: number;
  avgCommitFreq: number;
  avgAuthorEntropy: number;
  avgChurnRate: number;
  avgStabilityScore: number;
  hotspots: string[];  // Files with highest churn
  coldspots: string[]; // Files with no recent changes
}

/**
 * Time-based activity pattern.
 */
export interface ActivityPattern {
  /** Hour of day distribution (0-23) */
  hourlyDistribution: number[];
  /** Day of week distribution (0-6, Sunday=0) */
  dailyDistribution: number[];
  /** Monthly commit counts */
  monthlyTrend: Map<string, number>;
  /** Peak activity hours */
  peakHours: number[];
  /** Most active day */
  mostActiveDay: number;
}

/**
 * Calculate temporal signal for a file from its commit history.
 */
export function calculateTemporalSignal(
  path: string,
  commits: GitCommit[],
  currentSize: number = 0
): TemporalSignal {
  if (commits.length === 0) {
    return createEmptySignal(path);
  }

  // Sort by date (oldest first)
  const sorted = [...commits].sort((a, b) =>
    a.date.getTime() - b.date.getTime()
  );

  const firstCommit = sorted[0];
  const lastCommit = sorted[sorted.length - 1];
  const now = new Date();

  // Calculate age
  const ageInDays = (now.getTime() - firstCommit.date.getTime()) / (1000 * 60 * 60 * 24);
  const daysSinceLastChange = (now.getTime() - lastCommit.date.getTime()) / (1000 * 60 * 60 * 24);
  const monthsActive = Math.max(1, ageInDays / 30);

  // Aggregate file changes
  let totalAdditions = 0;
  let totalDeletions = 0;
  let renameCount = 0;
  const authorCommits = new Map<string, number>();

  for (const commit of commits) {
    // Count author commits
    const count = authorCommits.get(commit.author) || 0;
    authorCommits.set(commit.author, count + 1);

    // Sum changes for this file
    for (const file of commit.files) {
      if (file.path === path || file.oldPath === path) {
        totalAdditions += file.additions;
        totalDeletions += file.deletions;
        if (file.status === 'R') {
          renameCount++;
        }
      }
    }
  }

  // Calculate metrics
  const commitFreq = commits.length / monthsActive;
  const authorEntropy = calculateEntropy(authorCommits);
  const authorCount = authorCommits.size;
  const churnRate = currentSize > 0
    ? (totalAdditions + totalDeletions) / currentSize
    : totalAdditions + totalDeletions;
  const netChange = totalAdditions - totalDeletions;

  // Estimate refactor velocity (significant changes)
  const significantChanges = commits.filter(c => {
    const fileChange = c.files.find(f => f.path === path);
    return fileChange && (fileChange.additions + fileChange.deletions) > 10;
  });
  const refactorVelocity = significantChanges.length / monthsActive;

  // Calculate stability score
  const stabilityScore = calculateStabilityScore(
    commitFreq,
    churnRate,
    daysSinceLastChange,
    authorCount
  );

  // Find primary author
  let primaryAuthor = 'unknown';
  let maxCommits = 0;
  for (const [author, count] of authorCommits) {
    if (count > maxCommits) {
      maxCommits = count;
      primaryAuthor = author;
    }
  }

  // Recent activity (last 30 days)
  const thirtyDaysAgo = new Date(now.getTime() - 30 * 24 * 60 * 60 * 1000);
  const recentActivity = commits.filter(c => c.date >= thirtyDaysAgo).length;

  // Bus factor (simplified: count of authors with >10% of commits)
  const threshold = commits.length * 0.1;
  const significantAuthors = Array.from(authorCommits.values())
    .filter(count => count >= threshold).length;
  const busFactor = Math.max(1, significantAuthors);

  return {
    path,
    commitFreq,
    authorEntropy,
    authorCount,
    churnRate,
    totalAdditions,
    totalDeletions,
    netChange,
    refactorVelocity,
    stabilityScore,
    ageInDays,
    daysSinceLastChange,
    createdAt: firstCommit.date,
    lastModifiedAt: lastCommit.date,
    primaryAuthor,
    recentActivity,
    busFactor,
    renameCount,
  };
}

/**
 * Calculate Shannon entropy for author distribution.
 */
function calculateEntropy(authorCommits: Map<string, number>): number {
  const total = Array.from(authorCommits.values()).reduce((a, b) => a + b, 0);
  if (total === 0) return 0;

  let entropy = 0;
  for (const count of authorCommits.values()) {
    if (count > 0) {
      const p = count / total;
      entropy -= p * Math.log2(p);
    }
  }

  // Normalize by max possible entropy (log2 of author count)
  const maxEntropy = Math.log2(authorCommits.size);
  return maxEntropy > 0 ? entropy / maxEntropy : 0;
}

/**
 * Calculate stability score (0-1, higher = more stable).
 */
function calculateStabilityScore(
  commitFreq: number,
  churnRate: number,
  daysSinceLastChange: number,
  authorCount: number
): number {
  // Factors that increase stability:
  // - Lower commit frequency (not constantly changing)
  // - Lower churn rate (changes are meaningful)
  // - More days since last change (settled)
  // - Multiple authors (reviewed code)

  // Normalize each factor to 0-1
  const freqScore = 1 / (1 + commitFreq / 5);  // Penalize > 5 commits/month
  const churnScore = 1 / (1 + churnRate / 10); // Penalize > 10x size in changes
  const ageScore = Math.min(1, daysSinceLastChange / 90); // Max at 90 days stable
  const authorScore = Math.min(1, authorCount / 3);  // Max at 3+ authors

  // Weighted average
  return (
    freqScore * 0.3 +
    churnScore * 0.3 +
    ageScore * 0.2 +
    authorScore * 0.2
  );
}

/**
 * Create empty signal for files with no history.
 */
function createEmptySignal(path: string): TemporalSignal {
  const now = new Date();
  return {
    path,
    commitFreq: 0,
    authorEntropy: 0,
    authorCount: 0,
    churnRate: 0,
    totalAdditions: 0,
    totalDeletions: 0,
    netChange: 0,
    refactorVelocity: 0,
    stabilityScore: 0,
    ageInDays: 0,
    daysSinceLastChange: 0,
    createdAt: now,
    lastModifiedAt: now,
    primaryAuthor: 'unknown',
    recentActivity: 0,
    busFactor: 1,
    renameCount: 0,
  };
}

/**
 * Calculate activity pattern from commits.
 */
export function calculateActivityPattern(
  commits: GitCommit[]
): ActivityPattern {
  const hourly = new Array(24).fill(0);
  const daily = new Array(7).fill(0);
  const monthly = new Map<string, number>();

  for (const commit of commits) {
    const date = commit.date;
    hourly[date.getHours()]++;
    daily[date.getDay()]++;

    const monthKey = `${date.getFullYear()}-${String(date.getMonth() + 1).padStart(2, '0')}`;
    monthly.set(monthKey, (monthly.get(monthKey) || 0) + 1);
  }

  // Find peak hours (top 3)
  const sortedHours = hourly
    .map((count, hour) => ({ hour, count }))
    .sort((a, b) => b.count - a.count);
  const peakHours = sortedHours.slice(0, 3).map(h => h.hour);

  // Find most active day
  const mostActiveDay = daily.indexOf(Math.max(...daily));

  return {
    hourlyDistribution: hourly,
    dailyDistribution: daily,
    monthlyTrend: monthly,
    peakHours,
    mostActiveDay,
  };
}

/**
 * Aggregate temporal signals for a directory.
 */
export function aggregateDirectorySignals(
  dirPath: string,
  signals: TemporalSignal[]
): DirectoryTemporalSignal {
  if (signals.length === 0) {
    return {
      path: dirPath,
      fileCount: 0,
      avgCommitFreq: 0,
      avgAuthorEntropy: 0,
      avgChurnRate: 0,
      avgStabilityScore: 0,
      hotspots: [],
      coldspots: [],
    };
  }

  const avgCommitFreq = signals.reduce((s, sig) => s + sig.commitFreq, 0) / signals.length;
  const avgAuthorEntropy = signals.reduce((s, sig) => s + sig.authorEntropy, 0) / signals.length;
  const avgChurnRate = signals.reduce((s, sig) => s + sig.churnRate, 0) / signals.length;
  const avgStabilityScore = signals.reduce((s, sig) => s + sig.stabilityScore, 0) / signals.length;

  // Find hotspots (highest churn)
  const sortedByChurn = [...signals].sort((a, b) => b.churnRate - a.churnRate);
  const hotspots = sortedByChurn.slice(0, 5).map(s => s.path);

  // Find coldspots (no activity in 90+ days)
  const coldspots = signals
    .filter(s => s.daysSinceLastChange > 90)
    .map(s => s.path);

  return {
    path: dirPath,
    fileCount: signals.length,
    avgCommitFreq,
    avgAuthorEntropy,
    avgChurnRate,
    avgStabilityScore,
    hotspots,
    coldspots,
  };
}

/**
 * Classify a file based on its temporal signal.
 */
export function classifyByTemporal(
  signal: TemporalSignal
): 'stable' | 'active' | 'thrash' | 'abandoned' | 'new' {
  // New file (< 7 days old)
  if (signal.ageInDays < 7) {
    return 'new';
  }

  // Abandoned (no activity in 180+ days)
  if (signal.daysSinceLastChange > 180) {
    return 'abandoned';
  }

  // Thrash (high churn, low stability)
  if (signal.churnRate > 5 && signal.stabilityScore < 0.3) {
    return 'thrash';
  }

  // Active (recent changes, moderate churn)
  if (signal.recentActivity > 0 && signal.commitFreq > 1) {
    return 'active';
  }

  // Stable (everything else)
  return 'stable';
}
