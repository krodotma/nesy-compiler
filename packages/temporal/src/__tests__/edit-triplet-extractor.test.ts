import { describe, it, expect, vi, beforeEach } from 'vitest';
import { EditTripletExtractor, ExtractionConfig, EditTriplet } from '../edit-triplet-extractor';
import { GitWalker } from '../git-walker';

vi.mock('../git-walker', () => ({
  GitWalker: vi.fn(),
}));

describe('EditTripletExtractor', () => {
  const mockConfig: ExtractionConfig = {
    repoPath: '/mock/repo',
    filePatterns: ['*.ts', '*.tsx'],
    maxCommits: 100,
    minDiffLines: 3,
  };

  const mockCommits = [
    {
      hash: 'abc123def456',
      shortHash: 'abc123d',
      author: 'Test Author',
      authorEmail: 'test@example.com',
      date: new Date('2024-01-15'),
      message: 'refactor: improve function clarity',
      files: [
        { path: 'src/utils.ts', status: 'M' as const, additions: 10, deletions: 5 },
      ],
    },
    {
      hash: 'def456abc789',
      shortHash: 'def456a',
      author: 'Test Author',
      authorEmail: 'test@example.com',
      date: new Date('2024-01-14'),
      message: 'wip: temporary hack',
      files: [
        { path: 'src/temp.ts', status: 'M' as const, additions: 20, deletions: 2 },
      ],
    },
  ];

  let mockGetCommits: ReturnType<typeof vi.fn>;
  let mockGetFileAtCommit: ReturnType<typeof vi.fn>;
  let mockGetDiff: ReturnType<typeof vi.fn>;

  beforeEach(() => {
    vi.clearAllMocks();

    mockGetCommits = vi.fn().mockResolvedValue(mockCommits);
    mockGetFileAtCommit = vi.fn().mockImplementation((filePath: string, commitHash: string) => {
      if (commitHash.endsWith('^')) {
        return Promise.resolve('const x = 1;\nconst y = 2;\n');
      }
      return Promise.resolve('const x = 1;\nconst y = 2;\nconst z = 3;\n');
    });
    mockGetDiff = vi.fn().mockResolvedValue([
      {
        oldStart: 1,
        oldCount: 2,
        newStart: 1,
        newCount: 3,
        content: '+const z = 3;\n-// old comment\n+// new comment\n+// another line',
      },
    ]);

    (GitWalker as unknown as ReturnType<typeof vi.fn>).mockImplementation(() => ({
      getCommits: mockGetCommits,
      getFileAtCommit: mockGetFileAtCommit,
      getDiff: mockGetDiff,
    }));
  });

  describe('constructor', () => {
    it('creates extractor with config', () => {
      const extractor = new EditTripletExtractor(mockConfig);
      expect(extractor).toBeInstanceOf(EditTripletExtractor);
    });
  });

  describe('extractFromGitLog', () => {
    it('extracts triplets from git commits', async () => {
      const extractor = new EditTripletExtractor(mockConfig);
      const triplets = await extractor.extractFromGitLog();

      expect(mockGetCommits).toHaveBeenCalled();
      expect(triplets.length).toBeGreaterThan(0);
    });

    it('includes preState, diff, and postState', async () => {
      const extractor = new EditTripletExtractor(mockConfig);
      const triplets = await extractor.extractFromGitLog();

      for (const triplet of triplets) {
        expect(triplet).toHaveProperty('preState');
        expect(triplet).toHaveProperty('diff');
        expect(triplet).toHaveProperty('postState');
        expect(triplet).toHaveProperty('isGood');
        expect(triplet).toHaveProperty('source');
      }
    });
  });

  describe('getCommitDiff', () => {
    it('returns diff data for a commit', async () => {
      const extractor = new EditTripletExtractor(mockConfig);
      const result = await extractor.getCommitDiff('abc123def456');

      expect(result).toHaveProperty('preState');
      expect(result).toHaveProperty('diff');
      expect(result).toHaveProperty('postState');
    });
  });

  describe('filterValidTriplets', () => {
    it('filters out triplets with empty states', () => {
      const extractor = new EditTripletExtractor(mockConfig);

      const triplets: EditTriplet[] = [
        { preState: '', diff: '', postState: '', isGood: true, source: 'test1' },
        { preState: 'code', diff: '+line\n-line\n+line', postState: 'modified', isGood: true, source: 'test2' },
      ];

      const filtered = extractor.filterValidTriplets(triplets);
      expect(filtered).toHaveLength(1);
      expect(filtered[0].source).toBe('test2');
    });

    it('filters out triplets with insufficient diff lines', () => {
      const extractor = new EditTripletExtractor(mockConfig);

      const triplets: EditTriplet[] = [
        { preState: 'code', diff: '+one', postState: 'modified', isGood: true, source: 'small' },
        { preState: 'code', diff: '+line1\n-line2\n+line3\n-line4', postState: 'modified', isGood: true, source: 'big' },
      ];

      const filtered = extractor.filterValidTriplets(triplets);
      expect(filtered).toHaveLength(1);
      expect(filtered[0].source).toBe('big');
    });

    it('filters out triplets with empty diff', () => {
      const extractor = new EditTripletExtractor(mockConfig);

      const triplets: EditTriplet[] = [
        { preState: 'code', diff: '', postState: 'modified', isGood: true, source: 'empty' },
        { preState: 'code', diff: '   ', postState: 'modified', isGood: true, source: 'whitespace' },
      ];

      const filtered = extractor.filterValidTriplets(triplets);
      expect(filtered).toHaveLength(0);
    });
  });

  describe('quality inference', () => {
    it('marks refactor commits as good', async () => {
      const extractor = new EditTripletExtractor(mockConfig);
      const triplets = await extractor.extractFromGitLog();

      const refactorTriplet = triplets.find(t => t.source.includes('abc123'));
      expect(refactorTriplet?.isGood).toBe(true);
    });

    it('marks wip/hack commits as not good', async () => {
      const extractor = new EditTripletExtractor(mockConfig);
      const triplets = await extractor.extractFromGitLog();

      const wipTriplet = triplets.find(t => t.source.includes('def456'));
      expect(wipTriplet?.isGood).toBe(false);
    });
  });
});
