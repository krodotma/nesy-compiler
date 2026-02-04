import { z } from 'zod';

export const TemporalSignalSchema = z.object({
  file_path: z.string(),
  commit_freq: z.number().min(0),      // Commits per week
  author_entropy: z.number().min(0).max(1), // Diversity of authors (0=single, 1=chaos)
  churn_rate: z.number().min(0),       // Lines changed per commit
  refactor_velocity: z.number().min(0), // Frequency of large diffs
  last_modified: z.number(),           // Unix timestamp
  age_days: z.number().min(0),
});

export type TemporalSignal = z.infer<typeof TemporalSignalSchema>;

export const ThrashDetectorSchema = z.object({
  is_thrashing: z.boolean(),
  severity: z.number().min(0).max(100), // 0-100 score
  reason: z.string().optional(),
});

export type ThrashDetector = z.infer<typeof ThrashDetectorSchema>;
