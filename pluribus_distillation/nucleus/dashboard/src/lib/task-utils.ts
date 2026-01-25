export function countCheckboxes(text: string): { pending: number; completed: number } {
  let pending = 0;
  let completed = 0;
  for (const line of text.split('\n')) {
    const s = line.trim();
    if (!s.startsWith('- [')) continue;
    if (s.startsWith('- [x]') || s.startsWith('- [X]')) completed += 1;
    else if (s.startsWith('- [ ]')) pending += 1;
  }
  return { pending, completed };
}

export function parseTaskStatus(text: string): string {
  // Common patterns in this repo: "**Status:** Active" or "Status: Active"
  const lines = text.split('\n').slice(0, 60);
  for (const raw of lines) {
    const line = raw.trim();
    const m1 = line.match(/^\*\*Status:\*\*\s*(.+)\s*$/i);
    if (m1?.[1]) return m1[1].trim();
    const m2 = line.match(/^Status:\s*(.+)\s*$/i);
    if (m2?.[1]) return m2[1].trim();
  }
  return 'unknown';
}
