/**
 * EditTripletExtractor: Mine Git History for Refactoring Patterns
 *
 * Step 53 of NeSy Evolution Plan
 *
 * Extracts (PreState, Diff, PostState) tuples from git history
 * to teach refactoring patterns to the learning system.
 */

import { GitWalker, GitCommit, GitFileChange } from './git-walker';

export interface EditTriplet {
  preState: string;
  diff: string;
  postState: string;
  isGood: boolean;
  source: string;
}

export interface ExtractionConfig {
  repoPath: string;
  filePatterns: string[];
  maxCommits: number;
  minDiffLines: number;
}

const DEFAULT_CONFIG: Partial<ExtractionConfig> = {
  maxCommits: 500,
  minDiffLines: 3,
  filePatterns: ['*.ts', '*.tsx', '*.js', '*.jsx'],
};

export class EditTripletExtractor {
  private config: ExtractionConfig;
  private gitWalker: GitWalker;

  constructor(config: ExtractionConfig) {
    this.config = { ...DEFAULT_CONFIG, ...config } as ExtractionConfig;
    this.gitWalker = new GitWalker({
      repoPath: config.repoPath,
      maxCommits: this.config.maxCommits,
      includeDiffs: true,
    });
  }

  async extractFromGitLog(): Promise<EditTriplet[]> {
    const triplets: EditTriplet[] = [];
    const commits = await this.gitWalker.getCommits();

    for (const commit of commits) {
      const fileTriplets = await this.extractFromCommit(commit);
      triplets.push(...fileTriplets);
    }

    return this.filterValidTriplets(triplets);
  }

  async getCommitDiff(commitHash: string): Promise<{ preState: string; diff: string; postState: string }> {
    const parentHash = `${commitHash}^`;

    const commits = await this.runGitShow(commitHash);
    if (commits.files.length === 0) {
      return { preState: '', diff: '', postState: '' };
    }

    const file = commits.files[0];
    const preState = await this.gitWalker.getFileAtCommit(file.path, parentHash) || '';
    const postState = await this.gitWalker.getFileAtCommit(file.path, commitHash) || '';
    const diffHunks = await this.gitWalker.getDiff(file.path, parentHash, commitHash);
    const diff = diffHunks.map(h => h.content).join('\n');

    return { preState, diff, postState };
  }

  filterValidTriplets(triplets: EditTriplet[]): EditTriplet[] {
    return triplets.filter(t => {
      if (!t.preState && !t.postState) return false;
      if (!t.diff || t.diff.trim().length === 0) return false;
      
      const diffLines = t.diff.split('\n').filter(l => l.startsWith('+') || l.startsWith('-')).length;
      if (diffLines < this.config.minDiffLines) return false;

      return true;
    });
  }

  private async extractFromCommit(commit: GitCommit): Promise<EditTriplet[]> {
    const triplets: EditTriplet[] = [];
    const parentHash = `${commit.hash}^`;

    for (const file of commit.files) {
      if (!this.matchesFilePatterns(file.path)) continue;
      if (file.status === 'A' || file.status === 'D') continue;

      try {
        const preState = await this.gitWalker.getFileAtCommit(file.path, parentHash);
        const postState = await this.gitWalker.getFileAtCommit(file.path, commit.hash);
        const diffHunks = await this.gitWalker.getDiff(file.path, parentHash, commit.hash);

        if (!preState || !postState) continue;

        const diff = diffHunks.map(h => h.content).join('\n');

        triplets.push({
          preState,
          diff,
          postState,
          isGood: this.inferQuality(commit, file),
          source: `${commit.shortHash}:${file.path}`,
        });
      } catch {
        // Skip files that can't be retrieved
      }
    }

    return triplets;
  }

  private matchesFilePatterns(filePath: string): boolean {
    return this.config.filePatterns.some(pattern => {
      const regex = new RegExp(pattern.replace(/\*/g, '.*').replace(/\?/g, '.'));
      return regex.test(filePath);
    });
  }

  private inferQuality(commit: GitCommit, file: GitFileChange): boolean {
    const positivePatterns = [
      /refactor/i,
      /clean/i,
      /improve/i,
      /simplif/i,
      /fix/i,
      /optimize/i,
    ];

    const negativePatterns = [
      /wip/i,
      /tmp/i,
      /hack/i,
      /revert/i,
    ];

    const msg = commit.message;
    
    if (negativePatterns.some(p => p.test(msg))) return false;
    if (positivePatterns.some(p => p.test(msg))) return true;
    
    const ratio = file.additions / (file.deletions || 1);
    return ratio >= 0.5 && ratio <= 2.0;
  }

  private async runGitShow(commitHash: string): Promise<GitCommit> {
    const walker = new GitWalker({
      repoPath: this.config.repoPath,
      maxCommits: 1,
    });
    const commits = await walker.getCommits();
    return commits.find(c => c.hash === commitHash || c.shortHash === commitHash) || {
      hash: commitHash,
      shortHash: commitHash.slice(0, 7),
      author: '',
      authorEmail: '',
      date: new Date(),
      message: '',
      files: [],
    };
  }
}

export async function extractEditTriplets(
  repoPath: string,
  options?: Partial<Omit<ExtractionConfig, 'repoPath'>>
): Promise<EditTriplet[]> {
  const extractor = new EditTripletExtractor({
    repoPath,
    filePatterns: options?.filePatterns || ['*.ts', '*.tsx', '*.js', '*.jsx'],
    maxCommits: options?.maxCommits || 500,
    minDiffLines: options?.minDiffLines || 3,
  });
  return extractor.extractFromGitLog();
}
