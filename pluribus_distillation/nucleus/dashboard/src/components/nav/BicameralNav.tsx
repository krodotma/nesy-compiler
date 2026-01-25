import { component$, useVisibleTask$, type QRL } from '@builder.io/qwik';
import { useTracking } from '../../lib/telemetry/use-tracking';

// M3 Components (Step 65 - Filter Chip Navigation)
import '@material/web/chips/filter-chip.js';
import '@material/web/elevation/elevation.js';

type ViewId =
  | 'home'
  | 'studio'
  | 'bus'
  | 'events'
  | 'agents'
  | 'requests'
  | 'leads'
  | 'sota'
  | 'semops'
  | 'dkin'
  | 'services'
  | 'rhizome'
  | 'git'
  | 'types'
  | 'terminal'
  | 'plurichat'
  | 'webllm'
  | 'voice'
  | 'distill'
  | 'diagnostics'
  | 'generative'
  | 'browser-auth'
  | 'metatest'
  | 'registry-atlas';

interface NavItem {
  id: ViewId;
  label: string;
  hint?: string;
}

interface BicameralNavProps {
  activeView: ViewId;
  onSelect$: QRL<(view: ViewId) => void>;
}

const CORE_VIEWS: NavItem[] = [{ id: 'home', label: '🏠 Home' }];

const DEVOPS_VIEWS: NavItem[] = [
  { id: 'bus', label: '🧭 Bus', hint: 'Nerve center' },
  { id: 'services', label: '⚙️ Services' },
  { id: 'registry-atlas', label: '🗺️ Registry', hint: 'Registry topology atlas' },
  { id: 'diagnostics', label: '🔬 Diagnostics' },
  { id: 'metatest', label: '🧪 MetaTest', hint: 'Test coverage' },
  { id: 'terminal', label: '💻 Terminal' },
];

const MLOPS_VIEWS: NavItem[] = [
  { id: 'webllm', label: '🧩 WebLLM' },
  { id: 'voice', label: '🎙️ Voice' },
  { id: 'sota', label: '🔬 SOTA' },
  { id: 'semops', label: '🧠 SemOps' },
  { id: 'types', label: '🗂️ Types', hint: 'Sextet + AuOM schema tree' },
  { id: 'dkin', label: '🧬 DKIN' },
];

const EVOCODE_VIEWS: NavItem[] = [
  { id: 'studio', label: '🧪 Studio' },
  { id: 'rhizome', label: '🌳 Rhizome' },
  { id: 'git', label: '📦 Git' },
  { id: 'distill', label: '🧪 Distill' },
  { id: 'generative', label: '🎨 Generative' },
  { id: 'plurichat', label: '🗣️ PluriChat' },
];

const BUS_SUBVIEWS: NavItem[] = [
  { id: 'events', label: '📜 Events' },
  { id: 'agents', label: '🤖 Agents' },
  { id: 'requests', label: '📋 Requests' },
  { id: 'leads', label: '📋 Leads', hint: 'Content curation queue' },
];

const BUS_FAMILY = new Set<ViewId>(['bus', 'events', 'agents', 'requests', 'leads']);

export const BicameralNav = component$<BicameralNavProps>(({ activeView, onSelect$ }) => {
  // Track component mount for LoadingRegistry
  useTracking("comp:bicameral-nav");
  useVisibleTask$(() => {
    (window as any).__loadingRegistry?.entry("comp:bicameral-nav");
    (window as any).__loadingRegistry?.exit("comp:bicameral-nav");
  });

  const isBusFamilyActive = BUS_FAMILY.has(activeView);
  const buttonClass = (active: boolean) =>
    `nav-3d-inline-btn px-3 py-2 text-xs font-medium rounded-md transition-all btn-laser ${
      active
        ? 'is-active bg-primary/15 text-primary border border-primary/40'
        : 'bg-muted/20 text-muted-foreground border border-border/50 hover:text-foreground hover:bg-muted/40'
    }`;

  return (
    <nav class="bicameral-nav border-b border-border px-6 py-4 flex-shrink-0 overflow-x-auto">
      <md-elevation class="bicameral-nav-elevation"></md-elevation>
      <div class="flex flex-col gap-3">
        <div class="flex flex-wrap items-center gap-2">
          <span class="text-[10px] uppercase tracking-[0.3em] text-muted-foreground">Core</span>
          {CORE_VIEWS.map((view) => (
            <md-filter-chip
              key={view.id}
              class="bicameral-chip bicameral-chip-core"
              label={view.label}
              selected={activeView === view.id}
              onClick$={() => onSelect$(view.id)}
              title={view.hint}
            ></md-filter-chip>
          ))}
        </div>

        <div class="grid grid-cols-1 gap-3 lg:grid-cols-3">
          <section class="bicameral-section bicameral-section-devops space-y-2">
            <div class="flex items-center gap-2 text-[11px] font-semibold uppercase tracking-[0.2em] text-muted-foreground">
              <span class="bicameral-indicator h-2 w-2 rounded-full bg-primary/70" />
              DevOps
            </div>
            <div class="flex flex-wrap gap-2">
              {DEVOPS_VIEWS.map((view) => {
                const isActive = view.id === 'bus' ? isBusFamilyActive : activeView === view.id;
                return (
                  <md-filter-chip
                    key={view.id}
                    class="bicameral-chip bicameral-chip-devops"
                    label={view.label}
                    selected={isActive}
                    onClick$={() => onSelect$(view.id)}
                    title={view.hint}
                  ></md-filter-chip>
                );
              })}
            </div>
          </section>

          <section class="bicameral-section bicameral-section-mlops space-y-2">
            <div class="flex items-center gap-2 text-[11px] font-semibold uppercase tracking-[0.2em] text-muted-foreground">
              <span class="bicameral-indicator h-2 w-2 rounded-full bg-cyan-400/70" />
              MLOps
            </div>
            <div class="flex flex-wrap gap-2">
              {MLOPS_VIEWS.map((view) => (
                <md-filter-chip
                  key={view.id}
                  class="bicameral-chip bicameral-chip-mlops"
                  label={view.label}
                  selected={activeView === view.id}
                  onClick$={() => onSelect$(view.id)}
                  title={view.hint}
                ></md-filter-chip>
              ))}
            </div>
          </section>

          <section class="bicameral-section bicameral-section-evocode space-y-2">
            <div class="flex items-center gap-2 text-[11px] font-semibold uppercase tracking-[0.2em] text-muted-foreground">
              <span class="bicameral-indicator h-2 w-2 rounded-full bg-emerald-400/70" />
              EvoCode
            </div>
            <div class="flex flex-wrap gap-2">
              {EVOCODE_VIEWS.map((view) => (
                <md-filter-chip
                  key={view.id}
                  class="bicameral-chip bicameral-chip-evocode"
                  label={view.label}
                  selected={activeView === view.id}
                  onClick$={() => onSelect$(view.id)}
                  title={view.hint}
                ></md-filter-chip>
              ))}
            </div>
          </section>
        </div>

        {isBusFamilyActive && (
          <div class="bicameral-subviews flex flex-wrap items-center gap-2">
            <span class="text-[10px] uppercase tracking-[0.25em] text-muted-foreground">Bus Views</span>
            {BUS_SUBVIEWS.map((view) => (
              <md-filter-chip
                key={view.id}
                class="bicameral-chip bicameral-chip-bus"
                label={view.label}
                selected={activeView === view.id}
                onClick$={() => onSelect$(view.id)}
              ></md-filter-chip>
            ))}
          </div>
        )}
      </div>
    </nav>
  );
});
