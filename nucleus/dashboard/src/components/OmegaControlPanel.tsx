/**
 * OmegaControlPanel.tsx
 * Unified control panel integrating all Pluribus capabilities:
 * - Auralux Voice Pipeline (VoiceHUD)
 * - A2A Task Monitor (Star Topology)
 * - Dialogos Status (Streaming + Priority)
 */
import React, { useState } from 'react';
import { VoiceHUD } from './VoiceHUD';
import { A2ATaskMonitor } from './A2ATaskMonitor';
import { DialogosStatusPanel } from './DialogosStatusPanel';
import { MetaLearnerPanel } from './MetaLearnerPanel';

// ============================================================================
// Types
// ============================================================================

type PanelTab = 'voice' | 'a2a' | 'dialogos';

// ============================================================================
// Styles
// ============================================================================

const styles: Record<string, React.CSSProperties> = {
    container: {
        position: 'fixed',
        bottom: '20px',
        right: '20px',
        display: 'flex',
        flexDirection: 'column',
        gap: '12px',
        zIndex: 1000,
    },
    tabBar: {
        display: 'flex',
        gap: '4px',
        background: 'rgba(0, 0, 0, 0.6)',
        padding: '4px',
        borderRadius: '8px',
        backdropFilter: 'blur(10px)',
    },
    tab: {
        padding: '6px 12px',
        borderRadius: '6px',
        border: 'none',
        background: 'transparent',
        color: 'rgba(255, 255, 255, 0.6)',
        cursor: 'pointer',
        fontSize: '12px',
        transition: 'all 0.2s ease',
    },
    tabActive: {
        background: 'rgba(255, 255, 255, 0.15)',
        color: '#fff',
    },
    panel: {
        maxHeight: '400px',
        overflow: 'auto',
    },
    minimized: {
        position: 'fixed',
        bottom: '20px',
        right: '20px',
        width: '48px',
        height: '48px',
        borderRadius: '50%',
        background: 'linear-gradient(135deg, #8b5cf6 0%, #6366f1 100%)',
        display: 'flex',
        alignItems: 'center',
        justifyContent: 'center',
        cursor: 'pointer',
        boxShadow: '0 4px 20px rgba(139, 92, 246, 0.4)',
        zIndex: 1000,
        fontSize: '20px',
    },
};

// ============================================================================
// Component
// ============================================================================

interface OmegaControlPanelProps {
    defaultTab?: PanelTab;
    startMinimized?: boolean;
}

export function OmegaControlPanel({
    defaultTab = 'voice',
    startMinimized = false,
}: OmegaControlPanelProps) {
    const [activeTab, setActiveTab] = useState<PanelTab>(defaultTab);
    const [isMinimized, setIsMinimized] = useState(startMinimized);

    const tabs: { id: PanelTab; icon: string; label: string }[] = [
        { id: 'voice', icon: '🎤', label: 'Voice' },
        { id: 'a2a', icon: '🌟', label: 'A2A' },
        { id: 'dialogos', icon: '🎯', label: 'Dialogos' },
    ];

    if (isMinimized) {
        return (
            <div
                style={styles.minimized}
                onClick={() => setIsMinimized(false)}
                title="Open Omega Control Panel"
            >
                Ω
            </div>
        );
    }

    return (
        <div style={styles.container}>
            {/* Tab Bar */}
            <div style={styles.tabBar}>
                {tabs.map(tab => (
                    <button
                        key={tab.id}
                        style={{
                            ...styles.tab,
                            ...(activeTab === tab.id ? styles.tabActive : {}),
                        }}
                        onClick={() => setActiveTab(tab.id)}
                    >
                        {tab.icon} {tab.label}
                    </button>
                ))}
                <button
                    style={{ ...styles.tab, marginLeft: 'auto' }}
                    onClick={() => setIsMinimized(true)}
                    title="Minimize"
                >
                    −
                </button>
            </div>

            {/* Panel Content */}
            <div style={styles.panel}>
                {activeTab === 'voice' && <VoiceHUD />}
                {activeTab === 'a2a' && <A2ATaskMonitor />}
                {activeTab === 'dialogos' && <DialogosStatusPanel />}
                <MetaLearnerPanel />
            </div>
        </div>
    );
}

export default OmegaControlPanel;
