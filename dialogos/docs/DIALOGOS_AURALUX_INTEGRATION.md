# Dialogos + Auralux Voice Integration

This document describes the integration of voice input capabilities into Dialogos using the Auralux voice pipeline.

## Overview

The integration provides:
- **Microphone access** with permission handling
- **Voice Activity Detection (VAD)** using Silero VAD or simple energy-based fallback
- **Speech-to-Text (STT)** using Web Speech API
- **Text-to-Speech (TTS)** for AI responses
- **Unified state machine** for voice interaction flow

## Architecture

```
                    +------------------+
                    |  DialogosWidget  |
                    +--------+---------+
                             |
                    +--------v---------+
                    |  VoiceNeonInput  |
                    +--------+---------+
                             |
                    +--------v---------+
                    |   useVoiceState  |  <-- Unified state machine
                    +--------+---------+
                             |
         +-------------------+-------------------+
         |                   |                   |
+--------v--------+  +-------v-------+  +-------v-------+
|  useMicrophone  |  |    useVAD     |  | useSpeechToText|
+-----------------+  +---------------+  +----------------+
                             |
                    +--------v---------+
                    |  VADManager      |  <-- Auralux
                    |  (Silero/Simple) |
                    +------------------+
```

## Files

### Hooks (`/src/lib/dialogos/voice/`)

| File | Description |
|------|-------------|
| `index.ts` | Re-exports all voice hooks |
| `useMicrophone.ts` | Microphone permission and capture |
| `useVAD.ts` | Voice Activity Detection |
| `useSpeechToText.ts` | Web Speech API STT |
| `useTextToSpeech.ts` | Web Speech API TTS |
| `useVoiceState.ts` | Unified state machine |
| `VoiceOverlay.tsx` | Visual feedback component |

### Components (`/src/components/dialogos/ui/`)

| File | Description |
|------|-------------|
| `VoiceNeonInput.tsx` | Voice-enabled input component |

## Usage

### Basic Voice Input

The `VoiceNeonInput` component replaces the standard `NeonInput` in DialogosWidget:

```tsx
import { VoiceNeonInput } from './ui/VoiceNeonInput';

// In DialogosWidget
<VoiceNeonInput
  value={state.inputDraft}
  mode={state.mode}
  isThinking={state.isThinking}
  onInput$={(val) => (state.inputDraft = val)}
  onSubmit$={() => submit$(state.inputDraft)}
  emitBus$={emitBus$}
/>
```

### Keyboard Shortcuts

| Key | Action |
|-----|--------|
| `V` | Toggle voice input (when not focused on input) |
| `Escape` | Stop voice input |

### Using Individual Hooks

#### useMicrophone

```tsx
import { useMicrophone } from '@/lib/dialogos/voice';

const mic = useMicrophone({
  sampleRate: 16000,
  echoCancellation: true,
});

// Request permission
await mic.requestPermission$();

// Start capture
await mic.startCapture$();

// Access audio level (0-1)
console.log(mic.state.audioLevel);

// Stop capture
mic.stopCapture$();
```

#### useVAD

```tsx
import { useVAD } from '@/lib/dialogos/voice';

const vad = useVAD({
  positiveSpeechThreshold: 0.5,
  preferNeural: true,
  onSpeechStart$: $(() => console.log('Speech started')),
  onSpeechEnd$: $(async (audio, duration) => {
    console.log(`Speech ended: ${duration}ms`);
  }),
});

// Start VAD (needs MediaStream)
const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
await vad.start$(stream);
```

#### useSpeechToText

```tsx
import { useSpeechToText } from '@/lib/dialogos/voice';

const stt = useSpeechToText({
  language: 'en-US',
  continuous: true,
  onFinal$: $(async (result) => {
    console.log(`Transcript: ${result.text} (${result.confidence})`);
  }),
});

await stt.start$();
```

#### useTextToSpeech

```tsx
import { useTextToSpeech } from '@/lib/dialogos/voice';

const tts = useTextToSpeech({
  language: 'en-US',
  rate: 1.0,
});

await tts.speak$('Hello, world!');
```

#### useVoiceState (Unified)

```tsx
import { useVoiceState } from '@/lib/dialogos/voice';

const voice = useVoiceState({
  language: 'en-US',
  vadThreshold: 0.5,
  onTranscript$: $(async (text) => {
    console.log(`Final transcript: ${text}`);
  }),
});

// Activate voice (requests permissions, starts all subsystems)
await voice.activate$();

// Check state
console.log(voice.state.mode);      // 'idle' | 'listening' | 'speaking' | etc.
console.log(voice.state.transcript); // Accumulated transcript

// Speak response
await voice.speak$('I understood you.');

// Deactivate
voice.deactivate$();
```

## State Machine

The `useVoiceState` hook implements this state machine:

```
idle
  |
  v (activate)
requesting_permission
  |
  v (granted)
initializing
  |
  v (ready)
listening <------+
  |              |
  v (VAD start)  |
speaking         |
  |              |
  v (VAD end)    |
processing       |
  |              |
  +-(transcript)-+
  |
  v (AI responds)
responding
  |
  v (TTS done)
listening (loop) or idle (deactivated)
```

## Browser Compatibility

| Feature | Chrome | Firefox | Safari | Edge |
|---------|--------|---------|--------|------|
| getUserMedia | Yes | Yes | Yes | Yes |
| SpeechRecognition | Yes* | No | Yes* | Yes* |
| speechSynthesis | Yes | Yes | Yes | Yes |
| Silero VAD (ONNX) | Yes | Yes | Yes | Yes |

*SpeechRecognition requires HTTPS in production

## Bus Events

The integration emits these bus events:

| Topic | Kind | Data |
|-------|------|------|
| `dialogos.microphone.permission` | metric | `{ granted: boolean }` |
| `dialogos.microphone.started` | metric | `{ sampleRate: number }` |
| `dialogos.microphone.error` | metric | `{ error: string }` |
| `dialogos.vad.speech_start` | metric | `{ timestamp: string }` |
| `dialogos.vad.speech_end` | metric | `{ duration_ms: number, samples: number }` |
| `dialogos.stt.started` | metric | `{ language: string }` |
| `dialogos.stt.final` | metric | `{ text: string, confidence: number }` |
| `dialogos.tts.started` | metric | `{ chars: number }` |
| `dialogos.tts.ended` | metric | `{}` |
| `dialogos.voice.activated` | metric | `{ backend: string, sttSupported: boolean }` |

## Fallback Behavior

1. **VAD Fallback**: If Silero VAD (ONNX) fails to load, falls back to simple energy-based VAD
2. **STT Fallback**: If Web Speech API unavailable, voice input is disabled (transcript shows error)
3. **TTS Fallback**: If speechSynthesis unavailable, silent mode (no spoken responses)

## Performance Considerations

- Microphone capture uses 16kHz sample rate for STT compatibility
- VAD polling runs at ~60fps for responsive detection
- Audio level monitoring uses requestAnimationFrame
- Cleanup handlers properly release MediaStream tracks

## Security

- Microphone access requires user permission (browser prompt)
- HTTPS required in production for SpeechRecognition
- No audio data leaves the browser (local processing only)
- Bus events contain metadata only, not audio data

## Future Improvements

1. **Neural TTS**: Replace browser TTS with Auralux Vocos vocoder
2. **Neural STT**: Replace Web Speech API with Whisper/Conformer
3. **Push-to-Talk**: Hold-to-speak mode for noisy environments
4. **Wake Word**: Hands-free activation with keyword detection
5. **Multi-language**: Dynamic language switching during conversation
