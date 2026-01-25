export interface ViewManifestSummary {
  id: string;
  name: string;
  version: string;
  description?: string;
  path: string;
}

const DEFAULT_MANIFESTS: ViewManifestSummary[] = [
  {
    id: 'lanes',
    name: 'Lanes Status',
    version: '1.0.0',
    description: 'Multi-agent work lanes with WIP progress tracking',
    path: '/api/fs/nucleus/views/manifests/lanes.view.json',
  },
];

const fetchManifest = async (entry: ViewManifestSummary): Promise<ViewManifestSummary | null> => {
  try {
    const res = await fetch(entry.path, { cache: 'no-store' });
    if (!res.ok) return null;
    const raw = await res.text();
    const payload = JSON.parse(raw) as Partial<ViewManifestSummary> & { id?: string };
    return {
      id: String(payload.id || entry.id),
      name: String(payload.name || entry.name),
      version: String(payload.version || entry.version),
      description: payload.description || entry.description,
      path: entry.path,
    };
  } catch {
    return null;
  }
};

export const loadViewManifests = async (): Promise<ViewManifestSummary[]> => {
  const results = await Promise.all(DEFAULT_MANIFESTS.map(fetchManifest));
  const manifests = results.filter((item): item is ViewManifestSummary => !!item);
  return manifests.length ? manifests : DEFAULT_MANIFESTS.slice();
};
