import { component$, type QRL } from '@builder.io/qwik';
import { TYPES_SCHEMA, TYPES_LAYOUT, resolveTypeAxes } from '../../lib/types-schema';

// M3 Components - MotifTypesPanel
import '@material/web/elevation/elevation.js';
import '@material/web/ripple/ripple.js';
import '@material/web/button/text-button.js';

interface MotifTypesPanelProps {
  onOpenTypes$: QRL<() => void>;
}

export const MotifTypesPanel = component$<MotifTypesPanelProps>(({ onOpenTypes$ }) => {
  const topNodes = TYPES_SCHEMA.children ?? [];
  const motifNode = TYPES_LAYOUT.indexById['shape-motif-superpattern'];
  const motifChildren = motifNode?.children ?? [];
  const motifAxes = motifNode ? resolveTypeAxes(motifNode) : null;

  return (
    <section class="motif-panel">
      <div class="motif-panel-header">
        <div>
          <div class="motif-panel-overline">Motif + Types</div>
          <h3 class="motif-panel-title">Motif Superpattern and Public-Domain Types</h3>
          <p class="motif-panel-subtitle">
            The motif superpattern binds rhizomes, offspring, and guarded transfer while the Types Atlas
            formalizes common knowledge, events, planes, and things.
          </p>
        </div>
        <button class="motif-panel-cta" onClick$={onOpenTypes$}>
          Open Types Tree
        </button>
      </div>

      <div class="motif-panel-grid">
        <div class="motif-orbit">
          <div class="motif-core">
            <div class="motif-core-title">Motif Superpattern</div>
            <div class="motif-core-subtitle">Rhizome lineage kernel</div>
            {motifAxes && (
              <div class="motif-core-meta">
                <span class="types-chip axis">{motifAxes.polymorphism}</span>
                <span class="types-chip axis">{motifAxes.mutability}</span>
                <span class="types-chip axis">{motifAxes.agency}</span>
              </div>
            )}
          </div>
          {motifChildren.map((child, index) => (
            <div
              key={child.id}
              class="motif-satellite"
              style={{ '--angle': `${(index / Math.max(motifChildren.length, 1)) * 360}deg` } as Record<string, string>}
            >
              <span>{child.label}</span>
            </div>
          ))}
        </div>

        <div class="types-preview">
          <div class="types-preview-grid">
            {topNodes.map((node) => {
              const nodeAxes = resolveTypeAxes(node);
              return (
                <div key={node.id} class="types-preview-card">
                  <div class="types-preview-title">{node.label}</div>
                  <div class="types-preview-summary">{node.summary}</div>
                  <div class="types-preview-purpose">{nodeAxes.teleology.purpose}</div>
                  <div class="types-preview-tags">
                    {node.semantics.map((item) => (
                      <span key={item} class="types-chip semantics">{item}</span>
                    ))}
                  </div>
                  <div class="types-preview-axes">
                    <span class="types-chip axis">{nodeAxes.mutability}</span>
                    <span class="types-chip axis">{nodeAxes.polymorphism}</span>
                    <span class="types-chip axis">{nodeAxes.scope}</span>
                  </div>
                </div>
              );
            })}
          </div>
        </div>
      </div>

      <div class="motif-panel-footer">
        <span class="motif-footer-label">Sextet + AuOM</span>
        <span class="motif-footer-text">
          Objects, processes, types, shapes, symbols, and observers are tagged with mechanism flows and AuOM boundaries.
        </span>
      </div>
    </section>
  );
});

export default MotifTypesPanel;
