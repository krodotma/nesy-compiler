/**
 * BicameralNav3D - WHIPPERSNAPPER Enhanced 3D Navigation
 * ======================================================
 *
 * Features:
 * - 3D perspective transforms for DevOps ↔ EvoCode domain switching
 * - Fullscreen overlay modal with backdrop blur (Hakim.se style)
 * - Spring physics animations for smooth transitions
 * - Expandable/collapsible with 3D card flip effect
 * - Particle burst on domain switch
 */

import {
  component$,
  useSignal,
  useStore,
  useVisibleTask$,
  $,
  type QRL,
} from '@builder.io/qwik';

// M3 Components - BicameralNav3D
import '@material/web/elevation/elevation.js';
import '@material/web/ripple/ripple.js';
import '@material/web/button/filled-button.js';
import '@material/web/chips/filter-chip.js';

type ViewId =
  | 'home'
  | 'human'
  | 'studio'
  | 'bus'
  | 'events'
  | 'agents'
  | 'requests'
  | 'sota'
  | 'semops'
  | 'dkin'
  | 'services'
  | 'rhizome'
  | 'git'
  | 'terminal'
  | 'plurichat'
  | 'webllm'
  | 'voice'
  | 'distill'
  | 'diagnostics'
  | 'generative'
  | 'types'
  | 'browser-auth'
  | 'metatest'
  | 'skills'
  | 'personas'
  | 'patterns'
  | 'library'
  | 'memory'
  | 'dojo'
  | 'economy';

type Domain = 'devops' | 'evocode';

interface NavItem {
  id: ViewId;
  label: string;
  icon: string;
  hint?: string;
}

interface BicameralNav3DProps {
  activeView: ViewId;
  onSelect$: QRL<(view: ViewId) => void>;
}

const DEVOPS_VIEWS: NavItem[] = [
  { id: 'bus', label: 'Bus', icon: '🧭', hint: 'Nerve center' },
  { id: 'services', label: 'Services', icon: '⚙️' },
  { id: 'diagnostics', label: 'Diagnostics', icon: '🔬' },
  { id: 'metatest', label: 'MetaTest', icon: '🧪', hint: 'Test coverage' },
  { id: 'terminal', label: 'Terminal', icon: '💻' },
];

const MLOPS_VIEWS: NavItem[] = [
  { id: 'webllm', label: 'WebLLM', icon: '🧩' },
  { id: 'voice', label: 'Voice', icon: '🎙️' },
  { id: 'sota', label: 'SOTA', icon: '🔬' },
  { id: 'semops', label: 'SemOps', icon: '🧠' },
  { id: 'types', label: 'Types', icon: '🗂️' },
  { id: 'dkin', label: 'DKIN', icon: '🧬' },
];

const EVOCODE_VIEWS: NavItem[] = [
  { id: 'studio', label: 'Studio', icon: '🧪' },
  { id: 'rhizome', label: 'Rhizome', icon: '🌳' },
  { id: 'git', label: 'Git', icon: '📦' },
  { id: 'distill', label: 'Distill', icon: '🧪' },
  { id: 'generative', label: 'Generative', icon: '🎨' },
  { id: 'plurichat', label: 'PluriChat', icon: '🗣️' },
];

const LEARNING_VIEWS: NavItem[] = [
  { id: 'skills', label: 'Skills', icon: '⚡', hint: 'Elite capabilities' },
  { id: 'personas', label: 'Personas', icon: '🎭', hint: 'Agent identities' },
  { id: 'patterns', label: 'Patterns', icon: '🧠', hint: 'Cognitive design' },
  { id: 'library', label: 'Library', icon: '📚', hint: 'Knowledge base' },
  { id: 'memory', label: 'Memory', icon: '🧠', hint: 'Hippocampus' },
  { id: 'dojo', label: 'Dojo', icon: '🥋', hint: 'Training' },
  { id: 'economy', label: 'Economy', icon: '⚖️', hint: 'Resources' },
];

const BUS_SUBVIEWS: NavItem[] = [
  { id: 'events', label: 'Events', icon: '📜' },
  { id: 'agents', label: 'Agents', icon: '🤖' },
  { id: 'requests', label: 'Requests', icon: '📋' },
];

const BUS_FAMILY: ViewId[] = ['bus', 'events', 'agents', 'requests'];
const DEVOPS_IDS: ViewId[] = [...DEVOPS_VIEWS, ...MLOPS_VIEWS].map(v => v.id);

export const BicameralNav3D = component$<BicameralNav3DProps>(({ activeView, onSelect$ }) => {
  const isOpen = useSignal(false);
  const activeDomain = useSignal<Domain>(DEVOPS_IDS.includes(activeView) ? 'devops' : 'evocode');
  const isFlipping = useSignal(false);
  const springState = useStore({
    rotateY: 0,
    scale: 1,
    blur: 0,
  });

  const isBusFamilyActive = BUS_FAMILY.includes(activeView);

  // Spring physics for smooth transitions
  useVisibleTask$(({ track, cleanup }) => {
    track(() => activeDomain.value);
    track(() => isOpen.value);

    let animationId: number;
    let velocity = { rotateY: 0, scale: 0, blur: 0 };
    const smoothTime = 0.15;

    const animate = () => {
      const targetRotateY = activeDomain.value === 'devops' ? 0 : 180;
      const targetScale = isOpen.value ? 1 : 0.95;
      const targetBlur = isOpen.value ? 12 : 0;

      // Smooth damp interpolation
      const dampFactor = 1 - Math.exp(-10 * (1/60));

      springState.rotateY += (targetRotateY - springState.rotateY) * dampFactor;
      springState.scale += (targetScale - springState.scale) * dampFactor;
      springState.blur += (targetBlur - springState.blur) * dampFactor;

      const settled =
        Math.abs(springState.rotateY - targetRotateY) < 0.1 &&
        Math.abs(springState.scale - targetScale) < 0.001 &&
        Math.abs(springState.blur - targetBlur) < 0.1;

      if (!settled) {
        animationId = requestAnimationFrame(animate);
      }
    };

    animationId = requestAnimationFrame(animate);
    cleanup(() => cancelAnimationFrame(animationId));
  });

  const toggleMenu = $(() => {
    isOpen.value = !isOpen.value;
  });

  const emitParticleBurst = $(() => {
    const container = document.getElementById('nav-particles');
    if (!container) return;

    // Create 12 particles in a burst pattern
    for (let i = 0; i < 12; i++) {
      const particle = document.createElement('div');
      particle.className = 'nav-3d-particle';
      const angle = (i / 12) * Math.PI * 2;
      const distance = 60 + Math.random() * 40;
      const tx = Math.cos(angle) * distance;
      const ty = Math.sin(angle) * distance;
      particle.style.setProperty('--tx', `${tx}px`);
      particle.style.setProperty('--ty', `${ty}px`);
      particle.style.left = '50%';
      particle.style.top = '50%';
      container.appendChild(particle);

      // Remove particle after animation
      setTimeout(() => particle.remove(), 800);
    }
  });

  const switchDomain = $((domain: Domain) => {
    if (domain !== activeDomain.value) {
      isFlipping.value = true;
      activeDomain.value = domain;
      emitParticleBurst();
      setTimeout(() => {
        isFlipping.value = false;
      }, 600);
    }
  });

  const selectView = $((viewId: ViewId) => {
    onSelect$(viewId);
    // Auto-close on mobile
    if (window.innerWidth < 1024) {
      isOpen.value = false;
    }
  });

  const NavButton = component$<{ item: NavItem; isActive: boolean }>(
    ({ item, isActive }) => (
      <button
        class={`nav-3d-btn ${isActive ? 'active' : ''}`}
        onClick$={() => selectView(item.id)}
        title={item.hint}
      >
        <span class="nav-3d-icon">{item.icon}</span>
        <span class="nav-3d-label">{item.label}</span>
        {isActive && <span class="nav-3d-indicator" />}
      </button>
    )
  );

  return (
    <>
      {/* Backdrop blur overlay */}
      <div
        class={`nav-3d-backdrop ${isOpen.value ? 'active' : ''}`}
        style={{
          backdropFilter: `blur(${springState.blur}px)`,
          WebkitBackdropFilter: `blur(${springState.blur}px)`,
        }}
        onClick$={toggleMenu}
      />

      {/* Floating trigger button */}
      <button
        class={`nav-3d-trigger ${isOpen.value ? 'active' : ''}`}
        onClick$={toggleMenu}
        aria-label="Toggle navigation"
        aria-expanded={isOpen.value}
      >
        <div class="nav-3d-trigger-icon">
          <span />
          <span />
          <span />
        </div>
      </button>

      {/* Main navigation overlay */}
      <nav
        class={`nav-3d-overlay ${isOpen.value ? 'open' : ''}`}
        style={{
          transform: `scale(${springState.scale})`,
        }}
      >
        {/* Core buttons - always visible */}
        <div class="nav-3d-home">
          <button
            class={`nav-3d-home-btn ${activeView === 'home' ? 'active' : ''}`}
            onClick$={() => selectView('home')}
          >
            <span class="nav-3d-icon">🏠</span>
            <span>Home</span>
          </button>
          <button
            class={`nav-3d-home-btn ${activeView === 'human' ? 'active' : ''}`}
            onClick$={() => selectView('human')}
            title="Temporal orchestration lens"
          >
            <span class="nav-3d-icon">🧭</span>
            <span>Human</span>
          </button>
        </div>

        {/* Domain switcher - 3D flip cards */}
        <div class="nav-3d-domain-switcher">
          <button
            class={`nav-3d-domain-tab ${activeDomain.value === 'devops' ? 'active' : ''}`}
            onClick$={() => switchDomain('devops')}
          >
            <span class="domain-indicator devops" />
            DevOps
          </button>
          <button
            class={`nav-3d-domain-tab ${activeDomain.value === 'evocode' ? 'active' : ''}`}
            onClick$={() => switchDomain('evocode')}
          >
            <span class="domain-indicator evocode" />
            EvoCode
          </button>
        </div>

        {/* 3D Flip Container */}
        <div class="nav-3d-flip-container">
          <div
            class={`nav-3d-flipper ${isFlipping.value ? 'flipping' : ''}`}
            style={{
              transform: `rotateY(${springState.rotateY}deg)`,
            }}
          >
            {/* Front face - DevOps */}
            <div class="nav-3d-face nav-3d-front">
              <div class="nav-3d-section">
                <div class="nav-3d-section-title">DevOps</div>
                <div class="nav-3d-grid">
                  {DEVOPS_VIEWS.map((view) => {
                    const isActive = view.id === 'bus' ? isBusFamilyActive : activeView === view.id;
                    return <NavButton key={view.id} item={view} isActive={isActive} />;
                  })}
                </div>
              </div>
              <div class="nav-3d-section">
                <div class="nav-3d-section-title mlops">MLOps</div>
                <div class="nav-3d-grid">
                  {MLOPS_VIEWS.map((view) => (
                    <NavButton key={view.id} item={view} isActive={activeView === view.id} />
                  ))}
                </div>
              </div>
            </div>

            {/* Back face - EvoCode */}
            <div class="nav-3d-face nav-3d-back">
              <div class="nav-3d-section">
                <div class="nav-3d-section-title">EvoCode</div>
                <div class="nav-3d-grid">
                  {EVOCODE_VIEWS.map((view) => (
                    <NavButton key={view.id} item={view} isActive={activeView === view.id} />
                  ))}
                </div>
              </div>
              <div class="nav-3d-section">
                <div class="nav-3d-section-title learning" style={{ color: '#FCD34D' }}>Learning</div>
                <div class="nav-3d-grid">
                  {LEARNING_VIEWS.map((view) => (
                    <NavButton key={view.id} item={view} isActive={activeView === view.id} />
                  ))}
                </div>
              </div>
            </div>
          </div>
        </div>

        {/* Bus subviews - slide in when bus family active */}
        {isBusFamilyActive && (
          <div class="nav-3d-subviews">
            <span class="nav-3d-subviews-label">Bus Views</span>
            <div class="nav-3d-subviews-list">
              {BUS_SUBVIEWS.map((view) => (
                <NavButton key={view.id} item={view} isActive={activeView === view.id} />
              ))}
            </div>
          </div>
        )}

        {/* Particle burst effect container */}
        <div class="nav-3d-particles" id="nav-particles" />
      </nav>
    </>
  );
});

export default BicameralNav3D;
