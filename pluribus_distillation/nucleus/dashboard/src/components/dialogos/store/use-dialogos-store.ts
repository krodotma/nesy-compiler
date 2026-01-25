/**
 * Dialogos Store (Logic Core)
 * Author: opus_backend_1
 * Context: Phase 1 Foundation
 */

import { useStore, $, useVisibleTask$, noSerialize } from '@builder.io/qwik';
import type { DialogosState, DialogosAtom, ContextSnapshot } from '../types/dialogos';
import { createBusClient } from '../../../lib/bus/bus-client';
import { IntentRouter } from '../logic/IntentRouter';
import { CommandParser } from '../logic/CommandParser';
import { PBTSOBridge } from '../logic/PBTSOBridge';
import { IPEBridge } from '../logic/IPEBridge';
import { SystemListener } from '../logic/SystemListener';
import { TokenStreamer } from '../logic/TokenStreamer';
import { SnapshotService } from '../logic/SnapshotService';
import { persistence } from './persistence';

export const useDialogosStore = () => {
  const state = useStore<DialogosState>({
    isOpen: false,
    mode: 'rest',
    activeSessionId: null,
    atoms: {},
    timeline: [],
    inputDraft: '',
    pendingAttachments: [],
    isThinking: false
  });

  const busRef = useStore<{ client: any }>({ client: null });
  const pbtsoRef = useStore<{ instance: any }>({ instance: null });
  const ipeRef = useStore<{ instance: any }>({ instance: null });
  const sysRef = useStore<{ instance: any }>({ instance: null });
  // Map of req_id -> Streamer
  const streamers = new Map<string, TokenStreamer>();

  // Bus Connection & Bridge Init & Persistence Load
  useVisibleTask$(({ cleanup }) => {
    const client = createBusClient({ platform: 'browser' });
    busRef.client = noSerialize(client);
    
    const updateAtomState = (id: string, newState: any) => {
      if (state.atoms[id]) {
        state.atoms[id].state = newState;
        persistence.saveAtom(state.atoms[id]);
      }
    };

    const handleIncomingAtom = (atom: DialogosAtom) => {
      state.atoms[atom.id] = atom;
      if (!state.timeline.includes(atom.id)) {
        state.timeline.push(atom.id);
      }
      persistence.saveAtom(atom);
    };

    const pbtsoBridge = new PBTSOBridge({ updateAtomState });
    pbtsoRef.instance = noSerialize(pbtsoBridge);

    const ipeBridge = new IPEBridge();
    ipeRef.instance = noSerialize(ipeBridge);

    const sysListener = new SystemListener(handleIncomingAtom);
    sysRef.instance = noSerialize(sysListener);

    const init = async () => {
      try {
        await persistence.init();
        const history = await persistence.loadTimeline(50);
        history.forEach(atom => {
          state.atoms[atom.id] = atom;
          if (!state.timeline.includes(atom.id)) {
            state.timeline.push(atom.id);
          }
        });

        await client.connect();
        await pbtsoBridge.init();
        await ipeBridge.init();
        await sysListener.init();
        
        client.subscribe('dialogos.ingress.result', (event: any) => {
          handleIncomingAtom(event.data);
        });

        // Streaming Support
        client.subscribe('webllm.infer.partial', (event: any) => {
           const { req_id, token } = event.data;
           const atomId = `resp-${req_id}`;
           
           if (!state.atoms[atomId]) {
             // Initialize response atom on first token
             const responseAtom: DialogosAtom = {
                 id: atomId,
                 timestamp: Date.now(),
                 author: { id: 'webllm', name: 'Agent', role: 'agent' },
                 intent: 'execution',
                 content: { type: 'text', value: '' },
                 context: { url: 'local' },
                 state: 'actualizing',
                 causes: [req_id],
                 effects: []
             };
             handleIncomingAtom(responseAtom);
             streamers.set(req_id, new TokenStreamer((text) => {
                 state.atoms[atomId].content = { type: 'text', value: text };
             }));
           }
           
           const streamer = streamers.get(req_id);
           if (streamer) streamer.append(token);
        });

        client.subscribe('webllm.infer.response', (event: any) => {
          const { req_id, text, ok, error } = event.data;
          const atomId = `resp-${req_id}`;
          
          if (ok) {
             // Finalize
             if (!state.atoms[atomId]) {
                 // If no streaming happened, create full atom
                 const responseAtom: DialogosAtom = {
                     id: atomId,
                     timestamp: Date.now(),
                     author: { id: 'webllm', name: 'Agent', role: 'agent' },
                     intent: 'execution',
                     content: { type: 'text', value: text },
                     context: { url: 'local' },
                     state: 'actualized',
                     causes: [req_id],
                     effects: []
                 };
                 handleIncomingAtom(responseAtom);
             } else {
                 // Ensure final text matches
                 state.atoms[atomId].content = { type: 'text', value: text };
                 state.atoms[atomId].state = 'actualized';
                 persistence.saveAtom(state.atoms[atomId]);
             }
             updateAtomState(req_id, 'actualized');
          } else {
             updateAtomState(req_id, 'rejected');
             console.error('[Dialogos] Inference Error:', error);
          }
          state.isThinking = false;
          streamers.delete(req_id);
        });

      } catch (err) {
        console.error('[Dialogos] Initialization failed:', err);
      }
    };

    init();
    cleanup(() => {
      client.disconnect();
      pbtsoBridge.disconnect();
      ipeBridge.disconnect();
      sysListener.disconnect();
    });
  });

  // Actions
  const submit$ = $(async (text: string) => {
    if (!text.trim()) return;
    
    state.isThinking = true;
    
    const context = await SnapshotService.capture();
    const intent = IntentRouter.route(text);
    const command = CommandParser.parse(text, intent);

    const tempId = `atom-${Date.now()}`;
    const atom: DialogosAtom = {
      id: tempId,
      timestamp: Date.now(),
      author: { id: 'user', name: 'User', role: 'human' },
      intent: intent,
      content: command.structuredContent,
      context: context, 
      state: 'potential',
      causes: [],
      effects: []
    };

    state.atoms[tempId] = atom;
    state.timeline.push(tempId);
    state.inputDraft = '';
    persistence.saveAtom(atom);

    if (intent === 'task' && pbtsoRef.instance) {
      pbtsoRef.instance.dispatch(atom);
    }
    if (intent === 'mutation' && ipeRef.instance) {
      ipeRef.instance.dispatchMutation(atom);
    }

    if (busRef.client) {
      await busRef.client.publish({
        topic: 'webllm.infer.request',
        kind: 'command',
        actor: 'dialogos-ingress',
        data: {
          req_id: tempId,
          prompt: text,
          temperature: 0.7,
          session_id: state.activeSessionId,
          stream: true // Enable streaming
        }
      });
    }
  });

  const setMode$ = $((mode: 'rest' | 'active' | 'full') => {
    state.mode = mode;
  });

  return { state, submit$, setMode$ };
};
