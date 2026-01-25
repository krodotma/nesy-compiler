import { component$, useStore, useVisibleTask$ } from '@builder.io/qwik';

export const MetaLearnerPanel = component$(() => {
    const state = useStore({ status: 'loading', model: '' });

    useVisibleTask$(() => {
        const poll = async () => {
            try {
                const r = await fetch('http://localhost:8001/health');
                const j = await r.json();
                state.status = j.status;
                state.model = j.model_path ?? 'none';
            } catch (e) {
                state.status = 'error';
                state.model = '';
            }
        };
        poll();
        const iv = setInterval(poll, 5000);
        return () => clearInterval(iv);
    });

    return (
        <div style={{ padding: '8px', background: 'rgba(0,0,0,0.6)', color: '#fff', borderRadius: '8px' }}>
            <h3>MetaLearner</h3>
            <pre>{JSON.stringify(state, null, 2)}</pre>
        </div>
    );
});
