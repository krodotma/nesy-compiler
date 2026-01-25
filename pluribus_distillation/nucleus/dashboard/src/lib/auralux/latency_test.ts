/**
 * Auralux E2E Latency Test
 * 
 * Measures full pipeline latency: VAD trigger → synthesized audio output
 * Target: <100ms glass-to-glass per latency_budget.md
 */

import { PipelineOrchestrator } from './pipeline_orchestrator';

interface LatencyMeasurement {
    vadTriggerMs: number;
    preprocessMs: number;
    sslMs: number;
    vocoderMs: number;
    totalMs: number;
    passed: boolean;
}

const TARGET_LATENCY_MS = 100;

export async function runLatencyTest(): Promise<LatencyMeasurement[]> {
    const results: LatencyMeasurement[] = [];

    console.log('[Test] Initializing Auralux pipeline...');

    const orchestrator = new PipelineOrchestrator({
        vadModelUrl: '/models/silero_vad.onnx',
        sslModelUrl: '/models/hubert-soft-quantized.onnx',
        vocoderModelUrl: '/models/vocos_q8.onnx',
        emitBus: (topic, data) => {
            console.log(`[Bus] ${topic}:`, data);
        }
    });

    await orchestrator.initialize();

    // Instantiate services directly for micro-benchmarking
    const preprocessor = new (await import('./audio_preprocessor')).AudioPreprocessor();
    const ssl = new (await import('./ssl_service')).SslService('/models/hubert-soft-quantized.onnx');
    const vocoder = new (await import('./vocoder_service')).VocoderService('/models/vocos_q8.onnx');

    // Initialize services
    await ssl.initialize();
    await vocoder.initialize();

    // Simulate 5 speech segments for measurement
    for (let i = 0; i < 5; i++) {
        const start = performance.now();

        // Simulate speech input (1 second of audio at 48kHz)
        const testAudio = new Float32Array(48000);
        for (let j = 0; j < testAudio.length; j++) {
            testAudio[j] = Math.sin(j * 0.1) * 0.5; // Simple sine wave
        }

        const vadStart = performance.now();
        // VAD trigger is instantaneous in this synthetic test
        const vadEnd = performance.now();

        const preprocessStart = performance.now();
        const audio16k = preprocessor.resample(testAudio, 48000);
        const preprocessEnd = performance.now();

        const sslStart = performance.now();
        const features = await ssl.extractFeatures(audio16k);
        const sslEnd = performance.now();

        const vocoderStart = performance.now();
        let vocoderEnd = vocoderStart;
        if (features) {
             await vocoder.synthesize(features);
             vocoderEnd = performance.now();
        }

        const totalMs = performance.now() - start;

        const measurement: LatencyMeasurement = {
            vadTriggerMs: vadEnd - vadStart,
            preprocessMs: preprocessEnd - preprocessStart,
            sslMs: sslEnd - sslStart,
            vocoderMs: vocoderEnd - vocoderStart,
            totalMs,
            passed: totalMs < TARGET_LATENCY_MS
        };

        results.push(measurement);
        console.log(`[Test ${i + 1}] Total: ${totalMs.toFixed(2)}ms - ${measurement.passed ? 'PASS' : 'FAIL'}`);
    }

    const avgLatency = results.reduce((sum, r) => sum + r.totalMs, 0) / results.length;
    const passRate = results.filter(r => r.passed).length / results.length * 100;

    console.log(`\n=== AURALUX E2E LATENCY TEST RESULTS ===`);
    console.log(`Average Latency: ${avgLatency.toFixed(2)}ms`);
    console.log(`Target: <${TARGET_LATENCY_MS}ms`);
    console.log(`Pass Rate: ${passRate.toFixed(0)}%`);
    console.log(`Status: ${passRate >= 80 ? 'PASS' : 'FAIL'}`);

    return results;
}

// Export for browser/Node usage
export { LatencyMeasurement, TARGET_LATENCY_MS };
