import { WebSocket } from 'ws';
// @ts-ignore
import wrtc from 'wrtc';
import { encodeSkyEnvelope, decodeSkyEnvelope } from '../sky/codec.node.js';
import { getSkyEnvelopeType } from '../sky/proto.node.js';
import { SKY_MAGIC_V1, SKY_VERSION_V1 } from '../sky/constants.js';

const PEER_ID: string = process.argv[2] || `peer-${Math.random().toString(36).slice(2, 8)}`;
const TARGET_PEER_ID: string | undefined = process.argv[3] || undefined;

enum SignalType {
  OFFER = 0,
  ANSWER = 1,
  ICE_CANDIDATE = 2,
  ICE_RESTART = 3,
}

const ws = new WebSocket(`ws://localhost:9200/sky?peerId=${PEER_ID}`);

const pc = new wrtc.RTCPeerConnection({
  iceServers: [{ urls: 'stun:stun.l.google.com:19302' }],
});

pc.onicecandidate = (event: RTCPeerConnectionIceEvent) => {
  if (event.candidate) {
    const SkyEnvelope = getSkyEnvelopeType();
    const message = SkyEnvelope.create({
      magic: SKY_MAGIC_V1,
      version: SKY_VERSION_V1,
      ts_ms: Date.now(),
      swarm_id: 'default',
      source_peer_id: PEER_ID,
      target_peer_id: TARGET_PEER_ID,
      signal: {
        type: SignalType.ICE_CANDIDATE,
        candidate: event.candidate.candidate,
        sdp_mid: event.candidate.sdpMid || '',
        sdp_mline_index: event.candidate.sdpMLineIndex || 0,
      },
    });
    const encoded = encodeSkyEnvelope(message.toJSON());
    ws.send(encoded);
  }
};

pc.ondatachannel = (event: RTCDataChannelEvent) => {
  const dc = event.channel;
  dc.onmessage = (e: MessageEvent) => {
    if (e.data === 'heartbeat') {
      console.log(`[${PEER_ID}] Received heartbeat from remote peer`);
      dc.send('heartbeat-ack');
    } else if (e.data === 'heartbeat-ack') {
      console.log(`[${PEER_ID}] Received heartbeat-ack from remote peer`);
    } else {
      console.log(`[${PEER_ID}] Received message: ${e.data}`);
    }
  };
  dc.onopen = () => {
    console.log(`[${PEER_ID}] Data channel open`);
    // Start sending heartbeats
    setInterval(() => {
      if (dc.readyState === 'open') {
        dc.send('heartbeat');
      }
    }, 5000);
  };
};

ws.on('open', async () => {
  console.log(`[${PEER_ID}] Connected to signaling server`);

  if (TARGET_PEER_ID) {
    const dc = pc.createDataChannel('chat');
    dc.onmessage = (e: MessageEvent) => {
      if (e.data === 'heartbeat') {
        console.log(`[${PEER_ID}] Received heartbeat from ${TARGET_PEER_ID}`);
        dc.send('heartbeat-ack');
      } else if (e.data === 'heartbeat-ack') {
        console.log(`[${PEER_ID}] Received heartbeat-ack from ${TARGET_PEER_ID}`);
      } else {
        console.log(`[${PEER_ID}] Received message: ${e.data}`);
      }
    };
    dc.onopen = () => {
        console.log(`[${PEER_ID}] Data channel open`);
        dc.send('Hello from the other side!');
        // Start sending heartbeats
        setInterval(() => {
          if (dc.readyState === 'open') {
            dc.send('heartbeat');
          }
        }, 5000);
    }

    const offer = await pc.createOffer();
    await pc.setLocalDescription(offer);

    const SkyEnvelope = getSkyEnvelopeType();
    const message = SkyEnvelope.create({
      magic: SKY_MAGIC_V1,
      version: SKY_VERSION_V1,
      ts_ms: Date.now(),
      swarm_id: 'default',
      source_peer_id: PEER_ID,
      target_peer_id: TARGET_PEER_ID,
      signal: {
        type: SignalType.OFFER,
        sdp: offer.sdp,
      },
    });
    const encoded = encodeSkyEnvelope(message.toJSON());
    ws.send(encoded);
  }
});

ws.on('message', async (data) => {
  const bytes = data instanceof Buffer ? new Uint8Array(data) : new Uint8Array(data as ArrayBuffer);
  const env = decodeSkyEnvelope(bytes) as any; // Cast to any to access properties

  if (env.signal) {
    if (env.signal.type === SignalType.ANSWER) {
      await pc.setRemoteDescription(new wrtc.RTCSessionDescription({ sdp: env.signal.sdp, type: 'answer' }));
    } else if (env.signal.type === SignalType.ICE_CANDIDATE) {
      await pc.addIceCandidate(new wrtc.RTCIceCandidate({
        candidate: env.signal.candidate,
        sdpMid: env.signal.sdp_mid,
        sdpMLineIndex: env.signal.sdp_mline_index,
      }));
    } else if (env.signal.type === SignalType.OFFER) {
        await pc.setRemoteDescription(new wrtc.RTCSessionDescription({ sdp: env.signal.sdp, type: 'offer' }));
        const answer = await pc.createAnswer();
        await pc.setLocalDescription(answer);

        const SkyEnvelope = getSkyEnvelopeType();
        const message = SkyEnvelope.create({
            magic: SKY_MAGIC_V1,
            version: SKY_VERSION_V1,
            ts_ms: Date.now(),
            swarm_id: 'default',
            source_peer_id: PEER_ID,
            target_peer_id: env.source_peer_id,
            signal: {
                type: SignalType.ANSWER,
                sdp: answer.sdp,
            },
        });
        const encoded = encodeSkyEnvelope(message.toJSON());
        ws.send(encoded);
    }
  }
});

ws.on('close', () => {
  console.log(`[${PEER_ID}] Disconnected from signaling server`);
});

ws.on('error', (err) => {
  console.error(`[${PEER_ID}] Error:`, err);
});
