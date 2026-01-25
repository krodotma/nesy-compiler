/**
 * SystemHealthMonitor - System-wide health monitoring dashboard
 *
 * Phase 5, Iteration 45 of OITERATE lanes-widget-enhancement
 *
 * Features:
 * - Real-time system health metrics
 * - Service status monitoring
 * - Resource utilization
 * - Alert management
 * - Health history trends
 */

import {
  component$,
  useSignal,
  useComputed$,
  useVisibleTask$,
  $,
  type QRL,
} from '@builder.io/qwik';
import { Button } from './ui/Button';
import { Card } from './ui/Card';
import { NeonTitle, NeonBadge, NeonSectionHeader } from './ui/NeonTitle';

// M3 Components - SystemHealthMonitor
import '@material/web/elevation/elevation.js';
import '@material/web/ripple/ripple.js';
import '@material/web/chips/filter-chip.js';
import '@material/web/progress/circular-progress.js';
import '@material/web/button/filled-tonal-button.js';

// ============================================================================
// Types
// ============================================================================

export type HealthStatus = 'healthy' | 'degraded' | 'critical' | 'unknown';

export interface ServiceHealth {
  id: string;
  name: string;
  status: HealthStatus;
  latency: number; // ms
  uptime: number; // percentage
  lastCheck: string;
  metrics: {
    cpu: number;
    memory: number;
    requests: number;
    errors: number;
  };
  incidents: number;
}

export interface HealthAlert {
  id: string;
  serviceId: string;
  serviceName: string;
  severity: 'info' | 'warning' | 'error' | 'critical';
  message: string;
  timestamp: string;
  acknowledged: boolean;
}

export interface HealthMetric {
  timestamp: string;
  value: number;
}

export interface SystemHealthMonitorProps {
  /** Services to monitor */
  services: ServiceHealth[];
  /** Active alerts */
  alerts: HealthAlert[];
  /** Overall system metrics history */
  metricsHistory?: {
    cpu: HealthMetric[];
    memory: HealthMetric[];
    requests: HealthMetric[];
  };
  /** Callback when alert is acknowledged */
  onAcknowledgeAlert$?: QRL<(alertId: string) => void>;
  /** Callback when service is refreshed */
  onRefreshService$?: QRL<(serviceId: string) => void>;
}

// ============================================================================
// Helpers
// ============================================================================

function getStatusColor(status: HealthStatus): string {
  switch (status) {
    case 'healthy': return 'glass-status-healthy';
    case 'degraded': return 'glass-status-warning';
    case 'critical': return 'glass-status-critical';
    case 'unknown': return 'glass-chip';
    default: return 'glass-chip';
  }
}

function getStatusDot(status: HealthStatus): string {
  switch (status) {
    case 'healthy': return 'bg-md-success';
    case 'degraded': return 'bg-md-warning animate-pulse';
    case 'critical': return 'bg-md-error animate-pulse';
    case 'unknown': return 'bg-md-outline';
    default: return 'bg-md-outline';
  }
}

function getSeverityColor(severity: string): string {
  switch (severity) {
    case 'critical': return 'bg-md-error/10 text-md-error border-md-error/20';
    case 'error': return 'bg-md-error/5 text-md-error/80 border-md-error/10';
    case 'warning': return 'bg-md-warning/10 text-md-warning border-md-warning/20';
    case 'info': return 'bg-md-primary/10 text-md-primary border-md-primary/20';
    default: return 'bg-md-surface-variant text-md-on-surface-variant border-md-outline/20';
  }
}

function formatLatency(ms: number): string {
  if (ms < 1000) return `${ms}ms`;
  return `${(ms / 1000).toFixed(2)}s`;
}

function formatUptime(pct: number): string {
  return `${pct.toFixed(2)}%`;
}

function formatTime(dateStr: string): string {
  try {
    return new Date(dateStr).toLocaleTimeString('en-US', {
      hour: '2-digit',
      minute: '2-digit',
    });
  } catch {
    return dateStr;
  }
}

function formatTimeSince(dateStr: string): string {
  try {
    const date = new Date(dateStr);
    const now = new Date();
    const diffMs = now.getTime() - date.getTime();
    const diffSecs = Math.floor(diffMs / 1000);

    if (diffSecs < 60) return `${diffSecs}s ago`;
    if (diffSecs < 3600) return `${Math.floor(diffSecs / 60)}m ago`;
    return `${Math.floor(diffSecs / 3600)}h ago`;
  } catch {
    return dateStr;
  }
}

function getMetricColor(value: number): string {
  if (value >= 90) return 'var(--md-sys-color-error)';
  if (value >= 70) return 'var(--md-sys-color-warning)';
  if (value >= 50) return 'var(--md-sys-color-tertiary)';
  return 'var(--md-sys-color-primary)';
}

// ============================================================================
// Component
// ============================================================================

export const SystemHealthMonitor = component$<SystemHealthMonitorProps>(({
  services,
  alerts,
  metricsHistory,
  onAcknowledgeAlert$,
  onRefreshService$,
}) => {
  const selectedServiceId = useSignal<string | null>(null);
  const showAllAlerts = useSignal(false);
  const lastRefresh = useSignal(new Date().toISOString());

  useVisibleTask$(() => {
    const interval = setInterval(() => {
      lastRefresh.value = new Date().toISOString();
    }, 30000);
    return () => clearInterval(interval);
  });

  const selectedService = useComputed$(() =>
    services.find(s => s.id === selectedServiceId.value)
  );

  const overallHealth = useComputed$(() => {
    if (services.some(s => s.status === 'critical')) return 'critical';
    if (services.some(s => s.status === 'degraded')) return 'degraded';
    if (services.every(s => s.status === 'healthy')) return 'healthy';
    return 'unknown';
  });

  const stats = useComputed$(() => {
    const avgLatency = services.length > 0
      ? Math.round(services.reduce((sum, s) => sum + s.latency, 0) / services.length)
      : 0;
    const avgUptime = services.length > 0
      ? services.reduce((sum, s) => sum + s.uptime, 0) / services.length
      : 0;
    const totalRequests = services.reduce((sum, s) => sum + s.metrics.requests, 0);
    const totalErrors = services.reduce((sum, s) => sum + s.metrics.errors, 0);
    const activeAlerts = alerts.filter(a => !a.acknowledged).length;

    return { avgLatency, avgUptime, totalRequests, totalErrors, activeAlerts };
  });

  const unacknowledgedAlerts = useComputed$(() =>
    alerts.filter(a => !a.acknowledged).slice(0, showAllAlerts.value ? undefined : 5)
  );

  const acknowledgeAlert = $(async (alertId: string) => {
    if (onAcknowledgeAlert$) await onAcknowledgeAlert$(alertId);
  });

  const refreshService = $(async (serviceId: string) => {
    if (onRefreshService$) await onRefreshService$(serviceId);
  });

  const chartWidth = 120;
  const chartHeight = 32;

  const renderMiniChart = (data: HealthMetric[] | undefined, color: string) => {
    if (!data || data.length < 2) return null;
    const values = data.map(d => d.value);
    const max = Math.max(...values, 100);
    const min = Math.min(...values, 0);
    const range = max - min || 1;
    const path = data.map((d, i) => {
      const x = (i / (data.length - 1)) * chartWidth;
      const y = chartHeight - ((d.value - min) / range) * chartHeight;
      return i === 0 ? `M ${x} ${y}` : `L ${x} ${y}`;
    }).join(' ');
    return (
      <svg width={chartWidth} height={chartHeight} class="overflow-visible">
        <path d={path} fill="none" stroke={color} stroke-width="2" stroke-linecap="round" stroke-linejoin="round" />
      </svg>
    );
  };

  return (
    <Card variant="outlined" padding="p-0" class="overflow-hidden glass-surface-elevated glass-animate-enter">
      {/* Header */}
      <div class="flex items-center justify-between p-3 border-b border-glass-border-subtle bg-glass-bg-card/50">
        <div class="flex items-center gap-2">
          <NeonTitle level="span" color="cyan" size="xs" animation="flicker">System Health</NeonTitle>
          <NeonBadge
            color={overallHealth.value === 'healthy' ? 'emerald' : overallHealth.value === 'degraded' ? 'amber' : 'rose'}
            glow
            pulse={overallHealth.value === 'critical'}
          >
            {overallHealth.value.toUpperCase()}
          </NeonBadge>
        </div>
        <div class="flex items-center gap-3">
          <span class="text-[10px] font-medium text-md-on-surface-variant/60">Updated: {formatTimeSince(lastRefresh.value)}</span>
          {stats.value.activeAlerts > 0 && (
            <span class="px-2 py-0.5 rounded-full bg-md-error/10 text-md-error text-[10px] font-bold animate-pulse border border-md-error/20">
              {stats.value.activeAlerts} alerts
            </span>
          )}
        </div>
      </div>

      {/* Summary metrics */}
      <div class="grid grid-cols-5 gap-0 border-b border-glass-border-subtle glass-surface-subtle divide-x divide-glass-border-subtle/50">
        <div class="py-3 text-center">
          <div class="text-xl font-bold text-md-on-surface">{services.length}</div>
          <div class="text-[10px] font-bold uppercase text-md-on-surface-variant/50">Services</div>
        </div>
        <div class="py-3 text-center">
          <div class={`text-xl font-bold ${stats.value.avgLatency > 1000 ? 'text-md-error' : 'text-md-success'}`}>
            {formatLatency(stats.value.avgLatency)}
          </div>
          <div class="text-[10px] font-bold uppercase text-md-on-surface-variant/50">Latency</div>
        </div>
        <div class="py-3 text-center">
          <div class={`text-xl font-bold ${stats.value.avgUptime < 99 ? 'text-md-warning' : 'text-md-success'}`}>
            {formatUptime(stats.value.avgUptime)}
          </div>
          <div class="text-[10px] font-bold uppercase text-md-on-surface-variant/50">Uptime</div>
        </div>
        <div class="py-3 text-center">
          <div class="text-xl font-bold text-md-primary">{stats.value.totalRequests}</div>
          <div class="text-[10px] font-bold uppercase text-md-on-surface-variant/50">Req/min</div>
        </div>
        <div class="py-3 text-center">
          <div class={`text-xl font-bold ${stats.value.totalErrors > 0 ? 'text-md-error' : 'text-md-success'}`}>
            {stats.value.totalErrors}
          </div>
          <div class="text-[10px] font-bold uppercase text-md-on-surface-variant/50">Errors</div>
        </div>
      </div>

      {/* Main content */}
      <div class="grid grid-cols-3 gap-0 min-h-[350px]">
        {/* Service list */}
        <div class="border-r border-md-outline/10 max-h-[400px] overflow-y-auto no-scrollbar bg-md-surface-container-low/30">
          <div class="p-3 border-b border-md-outline/10">
            <NeonTitle level="div" color="cyan" size="xs">Registry</NeonTitle>
          </div>
          <div class="p-2 space-y-1">
            {services.map((service, idx) => (
              <div
                key={service.id}
                onClick$={() => { selectedServiceId.value = service.id; }}
                class={`p-3 rounded-xl cursor-pointer glass-transition-all relative overflow-hidden glass-surface glass-hover-glow glass-animate-enter ${
                  selectedServiceId.value === service.id
                    ? 'ring-1 ring-glass-accent/40 shadow-glass-glow'
                    : ''
                }`}
                style={{ '--stagger': idx } as any}
              >
                <md-ripple></md-ripple>
                <div class="flex items-center justify-between mb-2">
                  <div class="flex items-center gap-2">
                    <div class={`w-2.5 h-2.5 rounded-full ${getStatusDot(service.status)} shadow-sm`} />
                    <span class="text-xs font-bold text-md-on-surface">{service.name}</span>
                  </div>
                  <span class="text-[10px] font-mono font-medium text-md-on-surface-variant/60">{formatLatency(service.latency)}</span>
                </div>

                <div class="grid grid-cols-4 gap-1 text-[9px] font-bold">
                  <div class="text-center">
                    <div style={{ color: getMetricColor(service.metrics.cpu) }}>{service.metrics.cpu}%</div>
                  </div>
                  <div class="text-center">
                    <div style={{ color: getMetricColor(service.metrics.memory) }}>{service.metrics.memory}%</div>
                  </div>
                  <div class="text-center">
                    <div class="text-md-primary">{service.metrics.requests}</div>
                  </div>
                  <div class="text-center">
                    <div class={service.metrics.errors > 0 ? 'text-md-error' : 'text-md-success'}>
                      {service.metrics.errors}
                    </div>
                  </div>
                </div>
              </div>
            ))}
          </div>
        </div>

        {/* Service detail / Alerts */}
        <div class="col-span-2 max-h-[400px] overflow-y-auto no-scrollbar">
          {selectedService.value ? (
            <div class="p-6 space-y-6">
              <div class="flex items-center justify-between">
                <div class="flex items-center gap-3">
                  <div class={`w-4 h-4 rounded-full ${getStatusDot(selectedService.value.status)} shadow-md`} />
                  <h3 class="text-lg font-bold text-md-on-surface tracking-tight">{selectedService.value.name}</h3>
                  <span class={`text-[10px] font-bold px-3 py-1 rounded-full border ${getStatusColor(selectedService.value.status)}`}>
                    {selectedService.value.status.toUpperCase()}
                  </span>
                </div>
                <Button
                  variant="secondary"
                  class="h-8 text-[11px]"
                  icon="refresh"
                  onClick$={() => refreshService(selectedService.value!.id)}
                >
                  Refresh
                </Button>
              </div>

              <div class="grid grid-cols-2 gap-4">
                <Card variant="filled" padding="p-4" class="bg-md-surface-container">
                  <div class="text-[10px] font-bold uppercase tracking-widest text-md-on-surface-variant/60 mb-1">Latency</div>
                  <div class={`text-2xl font-black ${selectedService.value.latency > 1000 ? 'text-md-error' : 'text-md-success'}`}>
                    {formatLatency(selectedService.value.latency)}
                  </div>
                </Card>
                <Card variant="filled" padding="p-4" class="bg-md-surface-container">
                  <div class="text-[10px] font-bold uppercase tracking-widest text-md-on-surface-variant/60 mb-1">Uptime</div>
                  <div class={`text-2xl font-black ${selectedService.value.uptime < 99 ? 'text-md-warning' : 'text-md-success'}`}>
                    {formatUptime(selectedService.value.uptime)}
                  </div>
                </Card>
              </div>

              <div class="grid grid-cols-2 gap-4">
                {[
                  { label: 'CPU Load', value: selectedService.value.metrics.cpu },
                  { label: 'Memory Usage', value: selectedService.value.metrics.memory },
                ].map(metric => (
                  <div key={metric.label} class="space-y-2">
                    <div class="flex items-center justify-between px-1">
                      <span class="text-[10px] font-bold uppercase tracking-widest text-md-on-surface-variant/60">{metric.label}</span>
                      <span class="text-xs font-black" style={{ color: getMetricColor(metric.value) }}>
                        {metric.value}%
                      </span>
                    </div>
                    <div class="h-2 rounded-full bg-md-surface-container-highest overflow-hidden">
                      <div
                        class="h-full rounded-full transition-all duration-500 shadow-sm"
                        style={{
                          width: `${metric.value}%`,
                          backgroundColor: getMetricColor(metric.value),
                        }}
                      />
                    </div>
                  </div>
                ))}
              </div>

              {metricsHistory && (
                <div class="space-y-3">
                  <div class="text-[10px] font-bold uppercase tracking-widest text-md-on-surface-variant/60 px-1">Performance (24h)</div>
                  <div class="grid grid-cols-3 gap-3">
                    <Card variant="outlined" padding="p-3" class="bg-md-surface/50">
                      <div class="text-[9px] font-bold text-md-on-surface-variant/60 mb-2 uppercase">CPU</div>
                      <div class="flex justify-center">{renderMiniChart(metricsHistory.cpu, 'var(--md-sys-color-primary)')}</div>
                    </Card>
                    <Card variant="outlined" padding="p-3" class="bg-md-surface/50">
                      <div class="text-[9px] font-bold text-md-on-surface-variant/60 mb-2 uppercase">Mem</div>
                      <div class="flex justify-center">{renderMiniChart(metricsHistory.memory, 'var(--md-sys-color-tertiary)')}</div>
                    </Card>
                    <Card variant="outlined" padding="p-3" class="bg-md-surface/50">
                      <div class="text-[9px] font-bold text-md-on-surface-variant/60 mb-2 uppercase">Req</div>
                      <div class="flex justify-center">{renderMiniChart(metricsHistory.requests, 'var(--md-sys-color-success)')}</div>
                    </Card>
                  </div>
                </div>
              )}

              <div class="grid grid-cols-2 gap-2 text-[10px] font-bold uppercase tracking-tighter bg-md-surface-container-low p-3 rounded-lg border border-md-outline/5">
                <div>
                  <span class="text-md-on-surface-variant/40">Last Check:</span>
                  <span class="ml-2 text-md-on-surface">{formatTime(selectedService.value.lastCheck)}</span>
                </div>
                <div>
                  <span class="text-md-on-surface-variant/40">Incidents:</span>
                  <span class={`ml-2 ${selectedService.value.incidents > 0 ? 'text-md-error' : 'text-md-success'}`}>
                    {selectedService.value.incidents}
                  </span>
                </div>
              </div>
            </div>
          ) : (
            <div class="p-6 space-y-6">
              <div class="flex items-center justify-between mb-4">
                <NeonTitle level="div" color="amber" size="sm">Active Alerts</NeonTitle>
                <Button variant="text" class="h-7 text-[10px]" onClick$={() => { showAllAlerts.value = !showAllAlerts.value; }}>
                  {showAllAlerts.value ? 'Show Less' : 'View All'}
                </Button>
              </div>

              {unacknowledgedAlerts.value.length > 0 ? (
                <div class="space-y-3">
                  {unacknowledgedAlerts.value.map(alert => (
                    <Card
                      key={alert.id}
                      variant="outlined"
                      padding="p-4"
                      class={`${getSeverityColor(alert.severity)} transition-all hover:bg-md-surface-container-low`}
                    >
                      <div class="flex items-start justify-between gap-4">
                        <div class="flex-1">
                          <div class="flex items-center gap-3 mb-2">
                            <span class={`text-[9px] font-black px-2 py-0.5 rounded-full border ${getSeverityColor(alert.severity)}`}>
                              {alert.severity.toUpperCase()}
                            </span>
                            <span class="text-xs font-black text-md-on-surface uppercase tracking-tight">{alert.serviceName}</span>
                          </div>
                          <div class="text-sm font-medium text-md-on-surface leading-tight mb-2">{alert.message}</div>
                          <div class="text-[10px] font-bold text-md-on-surface-variant/40 uppercase tracking-tighter">
                            {formatTimeSince(alert.timestamp)}
                          </div>
                        </div>
                        <Button
                          variant="tonal"
                          class="h-8 text-[11px] min-w-[60px]"
                          onClick$={() => acknowledgeAlert(alert.id)}
                        >
                          Ack
                        </Button>
                      </div>
                    </Card>
                  ))}
                </div>
              ) : (
                <div class="text-center py-16">
                  <div class="text-5xl mb-4 opacity-20">🛡️</div>
                  <div class="text-sm font-bold text-md-on-surface-variant/60 uppercase tracking-widest">All systems protected</div>
                  <div class="text-[10px] text-md-on-surface-variant/40 mt-1 uppercase">No active alerts detected</div>
                </div>
              )}
            </div>
          )}
        </div>
      </div>

      {/* Footer */}
      <div class="p-3 border-t border-glass-border-subtle text-[10px] font-bold uppercase tracking-[0.2em] text-glass-text-muted text-center glass-surface-subtle">
        {services.length} active services • {alerts.filter(a => !a.acknowledged).length} unacknowledged alerts
      </div>
    </Card>
  );
});

export default SystemHealthMonitor;