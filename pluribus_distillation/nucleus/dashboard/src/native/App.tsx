/**
 * Pluribus Dashboard - Expo Native App
 *
 * React Native shell for the dashboard, built with Expo.
 * Shares state and bus client with web version.
 */

import React, { useEffect, useState, useCallback } from 'react';
import {
  SafeAreaView,
  ScrollView,
  View,
  Text,
  TouchableOpacity,
  StyleSheet,
  StatusBar,
  RefreshControl,
} from 'react-native';

// Types from shared state module
interface ProviderStatus {
  available: boolean;
  lastCheck: string;
  error?: string;
}

interface VPSSession {
  flowMode: 'm' | 'A';
  providers: Record<string, ProviderStatus>;
  fallbackOrder: string[];
  activeFallback: string | null;
  auth: {
    claudeLoggedIn: boolean;
    geminiCliLoggedIn: boolean;
    gcpProject?: string;
  };
}

interface ServiceDef {
  id: string;
  name: string;
  kind: string;
  port?: number;
  tags: string[];
}

interface DashboardState {
  services: ServiceDef[];
  session: VPSSession;
  connected: boolean;
}

// Default state
const defaultState: DashboardState = {
  services: [],
  session: {
    flowMode: 'm',
    providers: {
      'chatgpt-web': { available: false, lastCheck: '' },
      'claude-web': { available: false, lastCheck: '' },
      'gemini-web': { available: false, lastCheck: '' },
    },
    fallbackOrder: [
      'chatgpt-web',
      'claude-web',
      'gemini-web',
    ],
    activeFallback: null,
    auth: {
      claudeLoggedIn: false,
      geminiCliLoggedIn: false,
    },
  },
  connected: false,
};

// Hook for dashboard state
function useDashboard(wsUrl: string) {
  const [state, setState] = useState<DashboardState>(defaultState);
  const [refreshing, setRefreshing] = useState(false);

  useEffect(() => {
    // Connect to WebSocket bridge
    let ws: WebSocket | null = null;
    let reconnectTimer: NodeJS.Timeout | null = null;

    function connect() {
      ws = new WebSocket(wsUrl);

      ws.onopen = () => {
        setState(s => ({ ...s, connected: true }));
        ws?.send(JSON.stringify({ type: 'sync' }));
      };

      ws.onclose = () => {
        setState(s => ({ ...s, connected: false }));
        reconnectTimer = setTimeout(connect, 5000);
      };

      ws.onmessage = (msg) => {
        try {
          const data = JSON.parse(msg.data);
          if (data.type === 'sync' && data.state) {
            setState(s => ({ ...s, ...data.state }));
          } else if (data.type === 'event') {
            // Handle individual events
          }
        } catch (e) {
          // Ignore parse errors
        }
      };
    }

    connect();

    return () => {
      if (reconnectTimer) clearTimeout(reconnectTimer);
      ws?.close();
    };
  }, [wsUrl]);

  const refresh = useCallback(async () => {
    setRefreshing(true);
    // Trigger refresh via WebSocket
    setTimeout(() => setRefreshing(false), 1000);
  }, []);

  return { state, refreshing, refresh };
}

// Provider status indicator
function ProviderIndicator({ name, status }: { name: string; status: ProviderStatus }) {
  return (
    <View style={styles.providerItem}>
      <View style={[
        styles.statusDot,
        { backgroundColor: status.available ? '#4caf50' : '#f44336' }
      ]} />
      <View style={styles.providerInfo}>
        <Text style={styles.providerName}>{name}</Text>
        <Text style={styles.providerStatus}>
          {status.available ? 'Available' : status.error || 'Unavailable'}
        </Text>
      </View>
    </View>
  );
}

// Service list item
function ServiceItem({ service }: { service: ServiceDef }) {
  return (
    <View style={styles.serviceItem}>
      <View style={styles.serviceHeader}>
        <Text style={styles.serviceName}>{service.name}</Text>
        <View style={[styles.kindBadge, kindStyles[service.kind] || styles.kindDefault]}>
          <Text style={styles.kindText}>{service.kind}</Text>
        </View>
      </View>
      <Text style={styles.serviceId}>{service.id}</Text>
      {service.port && <Text style={styles.servicePort}>Port: {service.port}</Text>}
    </View>
  );
}

// Main App component
export default function App() {
  const wsUrl = 'ws://localhost:9200/ws/bus';  // Configure this
  const { state, refreshing, refresh } = useDashboard(wsUrl);
  const [activeTab, setActiveTab] = useState<'vps' | 'services'>('vps');

  return (
    <SafeAreaView style={styles.container}>
      <StatusBar barStyle="dark-content" />

      {/* Header */}
      <View style={styles.header}>
        <Text style={styles.title}>Pluribus Dashboard</Text>
        <View style={[
          styles.connectionStatus,
          { backgroundColor: state.connected ? '#4caf50' : '#f44336' }
        ]} />
      </View>

      {/* Tab Bar */}
      <View style={styles.tabBar}>
        <TouchableOpacity
          style={[styles.tab, activeTab === 'vps' && styles.activeTab]}
          onPress={() => setActiveTab('vps')}
        >
          <Text style={[styles.tabText, activeTab === 'vps' && styles.activeTabText]}>
            VPS Session
          </Text>
        </TouchableOpacity>
        <TouchableOpacity
          style={[styles.tab, activeTab === 'services' && styles.activeTab]}
          onPress={() => setActiveTab('services')}
        >
          <Text style={[styles.tabText, activeTab === 'services' && styles.activeTabText]}>
            Services
          </Text>
        </TouchableOpacity>
      </View>

      {/* Content */}
      <ScrollView
        style={styles.content}
        refreshControl={
          <RefreshControl refreshing={refreshing} onRefresh={refresh} />
        }
      >
        {activeTab === 'vps' && (
          <View style={styles.section}>
            {/* Flow Mode */}
            <View style={styles.card}>
              <Text style={styles.sectionTitle}>Flow Mode</Text>
              <View style={styles.flowModeButtons}>
                <TouchableOpacity
                  style={[
                    styles.flowModeButton,
                    state.session.flowMode === 'm' && styles.flowModeActive,
                    state.session.flowMode === 'm' && styles.flowModeMonitor,
                  ]}
                >
                  <Text style={styles.flowModeText}>M (Monitor)</Text>
                </TouchableOpacity>
                <TouchableOpacity
                  style={[
                    styles.flowModeButton,
                    state.session.flowMode === 'A' && styles.flowModeActive,
                    state.session.flowMode === 'A' && styles.flowModeAuto,
                  ]}
                >
                  <Text style={styles.flowModeText}>A (Auto)</Text>
                </TouchableOpacity>
              </View>
            </View>

            {/* Providers */}
            <View style={styles.card}>
              <Text style={styles.sectionTitle}>Providers</Text>
              {Object.entries(state.session.providers).map(([name, status]) => (
                <ProviderIndicator key={name} name={name} status={status} />
              ))}
            </View>

            {/* Fallback Chain */}
            <View style={styles.card}>
              <Text style={styles.sectionTitle}>Fallback Chain</Text>
              <View style={styles.fallbackChain}>
                {state.session.fallbackOrder.map((provider, i) => (
                  <React.Fragment key={provider}>
                    {i > 0 && <Text style={styles.arrow}>→</Text>}
                    <View style={[
                      styles.fallbackItem,
                      provider === state.session.activeFallback && styles.fallbackActive,
                    ]}>
                      <Text style={styles.fallbackText}>{provider}</Text>
                    </View>
                  </React.Fragment>
                ))}
              </View>
            </View>

            {/* Auth Status */}
            <View style={styles.card}>
              <Text style={styles.sectionTitle}>Authentication</Text>
              <View style={styles.authItem}>
                <View style={[
                  styles.authDot,
                  { backgroundColor: state.session.auth.geminiCliLoggedIn ? '#4caf50' : '#f44336' }
                ]} />
                <Text style={styles.authText}>Gemini CLI</Text>
              </View>
              <View style={styles.authItem}>
                <View style={[
                  styles.authDot,
                  { backgroundColor: state.session.auth.claudeLoggedIn ? '#4caf50' : '#f44336' }
                ]} />
                <Text style={styles.authText}>Claude Code</Text>
              </View>
            </View>
          </View>
        )}

        {activeTab === 'services' && (
          <View style={styles.section}>
            <View style={styles.card}>
              <Text style={styles.sectionTitle}>Registered Services</Text>
              {state.services.length === 0 ? (
                <Text style={styles.emptyText}>No services registered</Text>
              ) : (
                state.services.map(service => (
                  <ServiceItem key={service.id} service={service} />
                ))
              )}
            </View>
          </View>
        )}
      </ScrollView>
    </SafeAreaView>
  );
}

const kindStyles: Record<string, object> = {
  port: { backgroundColor: '#e3f2fd' },
  process: { backgroundColor: '#f3e5f5' },
  composition: { backgroundColor: '#fff3e0' },
};

const styles = StyleSheet.create({
  container: {
    flex: 1,
    backgroundColor: '#f5f5f5',
  },
  header: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    alignItems: 'center',
    padding: 16,
    backgroundColor: '#fff',
    borderBottomWidth: 1,
    borderBottomColor: '#e0e0e0',
  },
  title: {
    fontSize: 20,
    fontWeight: '600',
    color: '#333',
  },
  connectionStatus: {
    width: 12,
    height: 12,
    borderRadius: 6,
  },
  tabBar: {
    flexDirection: 'row',
    backgroundColor: '#fff',
    borderBottomWidth: 1,
    borderBottomColor: '#e0e0e0',
  },
  tab: {
    flex: 1,
    paddingVertical: 12,
    alignItems: 'center',
  },
  activeTab: {
    borderBottomWidth: 2,
    borderBottomColor: '#1976d2',
  },
  tabText: {
    fontSize: 14,
    color: '#666',
  },
  activeTabText: {
    color: '#1976d2',
    fontWeight: '600',
  },
  content: {
    flex: 1,
  },
  section: {
    padding: 16,
  },
  card: {
    backgroundColor: '#fff',
    borderRadius: 8,
    padding: 16,
    marginBottom: 16,
    shadowColor: '#000',
    shadowOffset: { width: 0, height: 1 },
    shadowOpacity: 0.1,
    shadowRadius: 2,
    elevation: 2,
  },
  sectionTitle: {
    fontSize: 16,
    fontWeight: '600',
    color: '#333',
    marginBottom: 12,
  },
  flowModeButtons: {
    flexDirection: 'row',
    gap: 8,
  },
  flowModeButton: {
    flex: 1,
    padding: 12,
    borderRadius: 8,
    borderWidth: 2,
    borderColor: '#e0e0e0',
    alignItems: 'center',
  },
  flowModeActive: {
    borderColor: 'transparent',
  },
  flowModeMonitor: {
    backgroundColor: '#ff9800',
  },
  flowModeAuto: {
    backgroundColor: '#4caf50',
  },
  flowModeText: {
    fontSize: 14,
    fontWeight: '500',
  },
  providerItem: {
    flexDirection: 'row',
    alignItems: 'center',
    paddingVertical: 8,
  },
  statusDot: {
    width: 10,
    height: 10,
    borderRadius: 5,
    marginRight: 12,
  },
  providerInfo: {
    flex: 1,
  },
  providerName: {
    fontSize: 14,
    fontWeight: '500',
    color: '#333',
  },
  providerStatus: {
    fontSize: 12,
    color: '#666',
  },
  fallbackChain: {
    flexDirection: 'row',
    flexWrap: 'wrap',
    alignItems: 'center',
    gap: 8,
  },
  fallbackItem: {
    paddingHorizontal: 12,
    paddingVertical: 6,
    backgroundColor: '#f0f0f0',
    borderRadius: 16,
  },
  fallbackActive: {
    backgroundColor: '#1976d2',
  },
  fallbackText: {
    fontSize: 12,
    color: '#333',
  },
  arrow: {
    color: '#999',
    fontSize: 12,
  },
  authItem: {
    flexDirection: 'row',
    alignItems: 'center',
    paddingVertical: 6,
  },
  authDot: {
    width: 8,
    height: 8,
    borderRadius: 4,
    marginRight: 10,
  },
  authText: {
    fontSize: 14,
    color: '#333',
  },
  serviceItem: {
    paddingVertical: 12,
    borderBottomWidth: 1,
    borderBottomColor: '#f0f0f0',
  },
  serviceHeader: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    alignItems: 'center',
  },
  serviceName: {
    fontSize: 14,
    fontWeight: '500',
    color: '#333',
  },
  kindBadge: {
    paddingHorizontal: 8,
    paddingVertical: 2,
    borderRadius: 4,
  },
  kindDefault: {
    backgroundColor: '#f0f0f0',
  },
  kindText: {
    fontSize: 11,
    color: '#666',
  },
  serviceId: {
    fontSize: 12,
    color: '#666',
    marginTop: 2,
    fontFamily: 'monospace',
  },
  servicePort: {
    fontSize: 12,
    color: '#999',
    marginTop: 2,
  },
  emptyText: {
    textAlign: 'center',
    color: '#999',
    padding: 20,
  },
});
