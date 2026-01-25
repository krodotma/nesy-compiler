/**
 * RingGuardPanel - Security Ring Visualization (Glassmorphism Edition)
 *
 * Displays the SCI/SAP-inspired compartmentalization status:
 * - Ring distribution (Kernel, Operator, Application, Ephemeral)
 * - Active compartments (PQC, OMEGA, EVOLUTION, etc.)
 * - Enforcement mode (observe vs enforce)
 * - Violation counts
 *
 * Historical inspiration: Manhattan Project badge colors
 * - WHITE (Ring 0) - Senior scientists, full access
 * - GOLD (Ring 1) - Operators
 * - BLUE (Ring 2) - Application workers
 * - RED (Ring 3) - Ephemeral/escorted
 *
 * Design: Glass prism + deep neon + M3 motion tokens
 * @version 2.0.0 - Glassmorphism + Accessibility update
 */

import React, { useState, useEffect, useCallback } from "react";

// Manhattan Project badge colors with neon glow enhancements
const RING_CONFIG = {
  0: {
    name: "KERNEL",
    color: "#FFFFFF",
    bgColor: "rgba(255, 255, 255, 0.08)",
    textColor: "#000000",
    description: "Constitutional, PQC",
    glow: "0 0 12px rgba(255, 255, 255, 0.5)",
    ariaLabel: "Ring 0 - Kernel level access for constitutional and post-quantum cryptography systems"
  },
  1: {
    name: "OPERATOR",
    color: "#FFD700",
    bgColor: "rgba(255, 215, 0, 0.08)",
    textColor: "#000000",
    description: "Infrastructure, Bus",
    glow: "0 0 12px rgba(255, 215, 0, 0.5)",
    ariaLabel: "Ring 1 - Operator level access for infrastructure and message bus"
  },
  2: {
    name: "APPLICATION",
    color: "#00BFFF",
    bgColor: "rgba(0, 191, 255, 0.08)",
    textColor: "#FFFFFF",
    description: "Dashboard, Reports",
    glow: "0 0 12px rgba(0, 191, 255, 0.5)",
    ariaLabel: "Ring 2 - Application level access for dashboard and reports"
  },
  3: {
    name: "EPHEMERAL",
    color: "#FF4757",
    bgColor: "rgba(255, 71, 87, 0.08)",
    textColor: "#FFFFFF",
    description: "PAIP, Sandbox",
    glow: "0 0 12px rgba(255, 71, 87, 0.5)",
    ariaLabel: "Ring 3 - Ephemeral access for PAIP isolation and sandbox environments"
  },
};

const COMPARTMENTS = [
  { id: "PQC", label: "Post-Quantum Crypto", color: "#00F3FF" },
  { id: "OMEGA", label: "Omega Protocol", color: "#BC13FE" },
  { id: "EVOLUTION", label: "Evolution Engine", color: "#10B981" },
  { id: "GENESIS", label: "Genesis Module", color: "#F59E0B" },
  { id: "METATOOL", label: "Meta-Tool System", color: "#FF6B6B" },
];

interface RingStatus {
  enforcement_mode: string;
  total_agents: number;
  by_ring: {
    kernel: number;
    operator: number;
    application: number;
    ephemeral: number;
  };
  compartments_active: string[];
  violations_total: number;
}

interface RingGuardPanelProps {
  className?: string;
  compact?: boolean;
  /** Accessible label for screen readers */
  ariaLabel?: string;
}

export const RingGuardPanel: React.FC<RingGuardPanelProps> = ({
  className = "",
  compact = false,
  ariaLabel = "Ring Guard Security Panel",
}) => {
  const [status, setStatus] = useState<RingStatus | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [lastUpdate, setLastUpdate] = useState<Date>(new Date());
  const [isVisible, setIsVisible] = useState(false);

  // Trigger fade-in animation after mount
  useEffect(() => {
    const timer = setTimeout(() => setIsVisible(true), 50);
    return () => clearTimeout(timer);
  }, []);

  const fetchStatus = useCallback(async () => {
    try {
      // In production, this would call a backend endpoint
      // For now, simulate with reasonable defaults
      const mockStatus: RingStatus = {
        enforcement_mode: "observe",
        total_agents: 5,
        by_ring: {
          kernel: 1,
          operator: 2,
          application: 2,
          ephemeral: 0,
        },
        compartments_active: ["OMEGA", "METATOOL"],
        violations_total: 0,
      };

      // Try to fetch from backend if available
      try {
        const resp = await fetch("/api/ring-guard/status", { method: "GET" });
        if (resp.ok) {
          const data = await resp.json();
          setStatus(data);
        } else {
          setStatus(mockStatus);
        }
      } catch {
        // Backend not available, use mock
        setStatus(mockStatus);
      }

      setLastUpdate(new Date());
      setError(null);
    } catch (err) {
      setError(err instanceof Error ? err.message : "Unknown error");
    } finally {
      setLoading(false);
    }
  }, []);

  useEffect(() => {
    fetchStatus();
    const interval = setInterval(fetchStatus, 30000); // Refresh every 30s
    return () => clearInterval(interval);
  }, [fetchStatus]);

  // Loading state with skeleton
  if (loading) {
    return (
      <div
        className={`ring-guard-panel ${className}`}
        style={styles.container}
        role="status"
        aria-label="Loading Ring Guard status"
        aria-busy="true"
      >
        <div style={styles.skeletonContainer}>
          <div style={{ ...styles.skeleton, width: '60%', height: '20px' }} />
          <div style={{ ...styles.skeleton, width: '100%', height: '16px', marginTop: '12px' }} />
          <div style={{ ...styles.skeleton, width: '100%', height: '16px', marginTop: '8px' }} />
          <div style={{ ...styles.skeleton, width: '80%', height: '16px', marginTop: '8px' }} />
        </div>
        <span className="sr-only">Loading security ring status...</span>
      </div>
    );
  }

  // Error state with retry option
  if (error) {
    return (
      <div
        className={`ring-guard-panel ${className}`}
        style={styles.container}
        role="alert"
        aria-live="polite"
      >
        <div style={styles.errorContainer}>
          <div style={styles.errorIcon} aria-hidden="true">!</div>
          <div style={styles.errorText}>
            <span style={styles.errorTitle}>Connection Error</span>
            <span style={styles.errorMessage}>{error}</span>
          </div>
          <button
            style={styles.retryButton}
            onClick={fetchStatus}
            aria-label="Retry loading Ring Guard status"
          >
            Retry
          </button>
        </div>
      </div>
    );
  }

  if (!status) {
    return null;
  }

  const ringCounts = [
    { ring: 0, count: status.by_ring.kernel },
    { ring: 1, count: status.by_ring.operator },
    { ring: 2, count: status.by_ring.application },
    { ring: 3, count: status.by_ring.ephemeral },
  ];

  const maxCount = Math.max(...ringCounts.map((r) => r.count), 1);
  const isEnforcing = status.enforcement_mode === "enforce";

  return (
    <section
      className={`ring-guard-panel ${className}`}
      style={{
        ...styles.container,
        opacity: isVisible ? 1 : 0,
        transform: isVisible ? 'translateY(0)' : 'translateY(8px)',
      }}
      role="region"
      aria-label={ariaLabel}
      aria-live="polite"
    >
      {/* Header with glass effect */}
      <header style={styles.header}>
        <div style={styles.headerLeft}>
          <h2 style={styles.title}>Ring Guard</h2>
          <span
            style={{
              ...styles.badge,
              backgroundColor: isEnforcing
                ? "rgba(255, 71, 87, 0.2)"
                : "rgba(255, 191, 0, 0.2)",
              color: isEnforcing ? "#FF4757" : "#FFBF00",
              borderColor: isEnforcing
                ? "rgba(255, 71, 87, 0.4)"
                : "rgba(255, 191, 0, 0.4)",
              boxShadow: isEnforcing
                ? "0 0 8px rgba(255, 71, 87, 0.3)"
                : "0 0 8px rgba(255, 191, 0, 0.3)",
            }}
            role="status"
            aria-label={`Enforcement mode: ${status.enforcement_mode}`}
          >
            {status.enforcement_mode.toUpperCase()}
          </span>
        </div>
        <div style={styles.headerRight}>
          <span style={styles.agentCount} aria-label={`${status.total_agents} active agents`}>
            <span style={styles.agentDot} aria-hidden="true" />
            {status.total_agents} agents
          </span>
        </div>
      </header>

      {/* Ring Distribution with animated bars */}
      {!compact && (
        <div style={styles.section} role="group" aria-labelledby="ring-dist-title">
          <h3 id="ring-dist-title" style={styles.sectionTitle}>Ring Distribution</h3>
          <div style={styles.ringBars} role="list">
            {ringCounts.map(({ ring, count }, index) => {
              const config = RING_CONFIG[ring as keyof typeof RING_CONFIG];
              const width = (count / maxCount) * 100;
              return (
                <div
                  key={ring}
                  style={{
                    ...styles.ringRow,
                    animationDelay: `${index * 100}ms`,
                  }}
                  role="listitem"
                  aria-label={config.ariaLabel}
                  tabIndex={0}
                >
                  <div style={styles.ringLabel}>
                    <span
                      style={{
                        ...styles.ringBadge,
                        backgroundColor: config.bgColor,
                        color: config.color,
                        boxShadow: config.glow,
                        border: `1px solid ${config.color}`,
                      }}
                      aria-hidden="true"
                    >
                      {ring}
                    </span>
                    <span style={styles.ringName}>{config.name}</span>
                  </div>
                  <div
                    style={styles.barContainer}
                    role="progressbar"
                    aria-valuenow={count}
                    aria-valuemin={0}
                    aria-valuemax={maxCount}
                    aria-label={`${config.name}: ${count} agents`}
                  >
                    <div
                      style={{
                        ...styles.bar,
                        width: `${width}%`,
                        background: `linear-gradient(90deg, ${config.color}dd, ${config.color}88)`,
                        boxShadow: `0 0 8px ${config.color}66`,
                      }}
                    />
                    <span style={styles.barCount} aria-hidden="true">{count}</span>
                  </div>
                </div>
              );
            })}
          </div>
        </div>
      )}

      {/* Compartments with neon chips */}
      <div style={styles.section} role="group" aria-labelledby="compartments-title">
        <h3 id="compartments-title" style={styles.sectionTitle}>Compartments</h3>
        <div style={styles.compartments} role="list">
          {COMPARTMENTS.map((comp) => {
            const isActive = status.compartments_active.includes(comp.id);
            return (
              <span
                key={comp.id}
                style={{
                  ...styles.compartmentTag,
                  backgroundColor: isActive
                    ? `${comp.color}22`
                    : "rgba(255, 255, 255, 0.03)",
                  color: isActive ? comp.color : "rgba(255, 255, 255, 0.4)",
                  borderColor: isActive
                    ? `${comp.color}66`
                    : "rgba(255, 255, 255, 0.1)",
                  boxShadow: isActive ? `0 0 8px ${comp.color}44` : "none",
                }}
                role="listitem"
                aria-label={`${comp.label}: ${isActive ? "Active" : "Inactive"}`}
                title={comp.label}
                tabIndex={0}
              >
                <span
                  style={{
                    ...styles.compartmentDot,
                    backgroundColor: isActive ? comp.color : "rgba(255, 255, 255, 0.2)",
                    boxShadow: isActive ? `0 0 6px ${comp.color}` : "none",
                  }}
                  aria-hidden="true"
                />
                {comp.id}
              </span>
            );
          })}
        </div>
      </div>

      {/* Violations alert with animation */}
      {status.violations_total > 0 && (
        <div style={styles.section} role="alert" aria-live="assertive">
          <div style={styles.violationAlert}>
            <span style={styles.violationIcon} aria-hidden="true">!</span>
            <span style={styles.violationText}>
              <strong>{status.violations_total}</strong> violation{status.violations_total !== 1 ? 's' : ''} recorded
            </span>
            <button
              style={styles.violationDismiss}
              aria-label="View violation details"
            >
              Details
            </button>
          </div>
        </div>
      )}

      {/* Footer with subtle gradient */}
      <footer style={styles.footer}>
        <span style={styles.lastUpdate}>
          <span style={styles.updateIcon} aria-hidden="true" />
          Updated: {lastUpdate.toLocaleTimeString()}
        </span>
        <button
          style={styles.refreshButton}
          onClick={fetchStatus}
          aria-label="Refresh Ring Guard status"
          title="Refresh"
        >
          <span style={styles.refreshIcon}>&#x21bb;</span>
        </button>
      </footer>
    </section>
  );
};

const styles: Record<string, React.CSSProperties> = {
  // Glass container with backdrop blur
  container: {
    background: "rgba(17, 17, 27, 0.8)",
    backdropFilter: "blur(16px)",
    WebkitBackdropFilter: "blur(16px)",
    border: "1px solid rgba(0, 255, 255, 0.08)",
    borderRadius: "16px",
    padding: "20px",
    fontFamily: "'JetBrains Mono', monospace",
    color: "rgba(255, 255, 255, 0.9)",
    fontSize: "13px",
    boxShadow: "0 8px 32px rgba(0, 0, 0, 0.3), inset 0 1px 0 rgba(255, 255, 255, 0.05)",
    transition: "opacity 300ms cubic-bezier(0.05, 0.7, 0.1, 1), transform 300ms cubic-bezier(0.05, 0.7, 0.1, 1)",
    position: "relative",
    overflow: "hidden",
  },
  // Skeleton loading states
  skeletonContainer: {
    padding: "8px 0",
  },
  skeleton: {
    background: "linear-gradient(90deg, rgba(255, 255, 255, 0.03) 25%, rgba(255, 255, 255, 0.08) 50%, rgba(255, 255, 255, 0.03) 75%)",
    backgroundSize: "200% 100%",
    animation: "shimmer 1.5s infinite",
    borderRadius: "4px",
  },
  // Error state
  errorContainer: {
    display: "flex",
    alignItems: "center",
    gap: "12px",
    padding: "12px",
    background: "rgba(255, 71, 87, 0.1)",
    border: "1px solid rgba(255, 71, 87, 0.3)",
    borderRadius: "8px",
  },
  errorIcon: {
    width: "32px",
    height: "32px",
    borderRadius: "50%",
    background: "rgba(255, 71, 87, 0.2)",
    color: "#FF4757",
    display: "flex",
    alignItems: "center",
    justifyContent: "center",
    fontSize: "16px",
    fontWeight: 700,
    flexShrink: 0,
  },
  errorText: {
    display: "flex",
    flexDirection: "column",
    gap: "2px",
    flex: 1,
  },
  errorTitle: {
    fontSize: "13px",
    fontWeight: 600,
    color: "#FF4757",
  },
  errorMessage: {
    fontSize: "11px",
    color: "rgba(255, 71, 87, 0.7)",
  },
  retryButton: {
    background: "rgba(255, 71, 87, 0.2)",
    border: "1px solid rgba(255, 71, 87, 0.4)",
    borderRadius: "6px",
    padding: "6px 12px",
    color: "#FF4757",
    fontSize: "11px",
    fontWeight: 600,
    cursor: "pointer",
    transition: "all 150ms cubic-bezier(0.2, 0, 0, 1)",
  },
  // Header
  header: {
    display: "flex",
    justifyContent: "space-between",
    alignItems: "center",
    marginBottom: "20px",
    paddingBottom: "16px",
    borderBottom: "1px solid rgba(0, 255, 255, 0.08)",
  },
  headerLeft: {
    display: "flex",
    alignItems: "center",
    gap: "12px",
  },
  headerRight: {
    display: "flex",
    alignItems: "center",
  },
  title: {
    fontSize: "16px",
    fontWeight: 600,
    color: "rgba(255, 255, 255, 0.95)",
    margin: 0,
    letterSpacing: "-0.02em",
  },
  badge: {
    fontSize: "10px",
    fontWeight: 600,
    padding: "4px 10px",
    borderRadius: "6px",
    textTransform: "uppercase" as const,
    letterSpacing: "0.05em",
    border: "1px solid",
    transition: "all 200ms cubic-bezier(0.2, 0, 0, 1)",
  },
  agentCount: {
    fontSize: "12px",
    color: "rgba(255, 255, 255, 0.6)",
    display: "flex",
    alignItems: "center",
    gap: "6px",
  },
  agentDot: {
    width: "6px",
    height: "6px",
    borderRadius: "50%",
    background: "#10B981",
    boxShadow: "0 0 8px rgba(16, 185, 129, 0.6)",
  },
  // Sections
  section: {
    marginBottom: "20px",
  },
  sectionTitle: {
    fontSize: "10px",
    fontWeight: 600,
    color: "rgba(255, 255, 255, 0.4)",
    textTransform: "uppercase" as const,
    letterSpacing: "0.1em",
    marginBottom: "12px",
    margin: 0,
  },
  // Ring bars
  ringBars: {
    display: "flex",
    flexDirection: "column" as const,
    gap: "8px",
  },
  ringRow: {
    display: "flex",
    alignItems: "center",
    gap: "12px",
    padding: "4px 0",
    borderRadius: "8px",
    transition: "background 150ms cubic-bezier(0.2, 0, 0, 1)",
    cursor: "default",
  },
  ringLabel: {
    display: "flex",
    alignItems: "center",
    gap: "10px",
    width: "130px",
    flexShrink: 0,
  },
  ringBadge: {
    width: "24px",
    height: "24px",
    borderRadius: "50%",
    display: "flex",
    alignItems: "center",
    justifyContent: "center",
    fontSize: "11px",
    fontWeight: 700,
    transition: "all 200ms cubic-bezier(0.2, 0, 0, 1)",
  },
  ringName: {
    fontSize: "11px",
    color: "rgba(255, 255, 255, 0.6)",
    fontWeight: 500,
  },
  barContainer: {
    flex: 1,
    display: "flex",
    alignItems: "center",
    height: "20px",
    background: "rgba(255, 255, 255, 0.03)",
    borderRadius: "10px",
    overflow: "hidden",
    position: "relative" as const,
    border: "1px solid rgba(255, 255, 255, 0.05)",
  },
  bar: {
    height: "100%",
    borderRadius: "10px",
    transition: "width 500ms cubic-bezier(0.05, 0.7, 0.1, 1)",
    minWidth: "8px",
  },
  barCount: {
    position: "absolute" as const,
    right: "10px",
    fontSize: "10px",
    color: "rgba(255, 255, 255, 0.8)",
    fontWeight: 600,
    textShadow: "0 1px 2px rgba(0, 0, 0, 0.5)",
  },
  // Compartments
  compartments: {
    display: "flex",
    flexWrap: "wrap" as const,
    gap: "8px",
    marginTop: "8px",
  },
  compartmentTag: {
    fontSize: "10px",
    fontWeight: 600,
    padding: "6px 12px",
    borderRadius: "8px",
    letterSpacing: "0.05em",
    display: "flex",
    alignItems: "center",
    gap: "6px",
    border: "1px solid",
    transition: "all 200ms cubic-bezier(0.2, 0, 0, 1)",
    cursor: "default",
  },
  compartmentDot: {
    width: "6px",
    height: "6px",
    borderRadius: "50%",
    transition: "all 200ms cubic-bezier(0.2, 0, 0, 1)",
  },
  // Violation alert
  violationAlert: {
    display: "flex",
    alignItems: "center",
    gap: "12px",
    padding: "12px 16px",
    background: "rgba(255, 71, 87, 0.1)",
    border: "1px solid rgba(255, 71, 87, 0.3)",
    borderRadius: "10px",
    color: "#FF4757",
    fontSize: "12px",
    boxShadow: "0 0 16px rgba(255, 71, 87, 0.15)",
    animation: "pulse-glow 2s ease-in-out infinite",
  },
  violationIcon: {
    width: "24px",
    height: "24px",
    borderRadius: "50%",
    background: "rgba(255, 71, 87, 0.2)",
    color: "#FF4757",
    display: "flex",
    alignItems: "center",
    justifyContent: "center",
    fontSize: "12px",
    fontWeight: 700,
    border: "1px solid rgba(255, 71, 87, 0.4)",
    boxShadow: "0 0 8px rgba(255, 71, 87, 0.3)",
  },
  violationText: {
    flex: 1,
    fontSize: "12px",
  },
  violationDismiss: {
    background: "transparent",
    border: "1px solid rgba(255, 71, 87, 0.4)",
    borderRadius: "6px",
    padding: "4px 10px",
    color: "#FF4757",
    fontSize: "10px",
    fontWeight: 600,
    cursor: "pointer",
    transition: "all 150ms cubic-bezier(0.2, 0, 0, 1)",
  },
  // Footer
  footer: {
    marginTop: "16px",
    paddingTop: "12px",
    borderTop: "1px solid rgba(255, 255, 255, 0.05)",
    display: "flex",
    justifyContent: "space-between",
    alignItems: "center",
  },
  lastUpdate: {
    fontSize: "10px",
    color: "rgba(255, 255, 255, 0.4)",
    display: "flex",
    alignItems: "center",
    gap: "6px",
  },
  updateIcon: {
    width: "4px",
    height: "4px",
    borderRadius: "50%",
    background: "rgba(0, 255, 255, 0.6)",
    boxShadow: "0 0 4px rgba(0, 255, 255, 0.4)",
  },
  refreshButton: {
    background: "rgba(255, 255, 255, 0.03)",
    border: "1px solid rgba(255, 255, 255, 0.1)",
    borderRadius: "6px",
    padding: "6px 8px",
    color: "rgba(255, 255, 255, 0.5)",
    fontSize: "14px",
    cursor: "pointer",
    transition: "all 150ms cubic-bezier(0.2, 0, 0, 1)",
    display: "flex",
    alignItems: "center",
    justifyContent: "center",
  },
  refreshIcon: {
    display: "inline-block",
    transition: "transform 300ms cubic-bezier(0.2, 0, 0, 1)",
  },
};

// Add keyframe styles via CSS injection
if (typeof document !== 'undefined') {
  const styleSheet = document.createElement('style');
  styleSheet.textContent = `
    @keyframes shimmer {
      0% { background-position: 200% 0; }
      100% { background-position: -200% 0; }
    }
    @keyframes pulse-glow {
      0%, 100% { box-shadow: 0 0 16px rgba(255, 71, 87, 0.15); }
      50% { box-shadow: 0 0 24px rgba(255, 71, 87, 0.25); }
    }
    .ring-guard-panel:hover .refresh-icon {
      transform: rotate(180deg);
    }
    .sr-only {
      position: absolute;
      width: 1px;
      height: 1px;
      padding: 0;
      margin: -1px;
      overflow: hidden;
      clip: rect(0, 0, 0, 0);
      white-space: nowrap;
      border: 0;
    }
  `;
  document.head.appendChild(styleSheet);
}

export default RingGuardPanel;
