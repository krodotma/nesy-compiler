/**
 * topicFilter.ts - Topic glob matching utilities
 *
 * Topic filters are dot-delimited with `*` wildcards:
 * - `strp.*` matches `strp.request` but not `strp.request.extra`
 * - `a2a.*.*` matches exactly 3 segments
 * - `*.request` matches any topic ending in `.request`
 *
 * `*` does not cross segment boundaries (it never matches `.`).
 */

function escapeRegexLiteral(text: string): string {
  return text.replace(/[\\^$.*+?()[\]{}|]/g, '\\$&');
}

export function compileTopicFilter(filter: string): RegExp | null {
  const raw = (filter || '').trim();
  if (!raw) return null;

  const parts = raw.split('.').map((part) => {
    if (part === '*') return '[^.]+';

    let segment = '';
    for (const ch of part) {
      if (ch === '*') segment += '[^.]*';
      else segment += escapeRegexLiteral(ch);
    }
    return segment;
  });

  return new RegExp(`^${parts.join('\\.')}$`);
}

export function matchesTopicFilter(topic: string, filter: string): boolean {
  const re = compileTopicFilter(filter);
  if (!re) return true;
  return re.test(topic);
}

