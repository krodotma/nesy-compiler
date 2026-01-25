/**
 * Ring Buffer Unit Tests
 * Verifies the SharedArrayBuffer-based AudioRingBuffer implementation.
 */

import { AudioRingBuffer } from '../nucleus/auralux/ring_buffer';

function assert(condition: boolean, message: string) {
    if (!condition) {
        throw new Error(`Assertion failed: ${message}`);
    }
}

async function runTests() {
    console.log('🧪 Ring Buffer Unit Tests');
    console.log('='.repeat(40));

    // Test 1: Basic write and read
    {
        console.log('\n📋 Test 1: Basic Write/Read');
        const buffer = new AudioRingBuffer(1024);
        const writeData = new Float32Array([1.0, 2.0, 3.0, 4.0, 5.0]);
        const readData = new Float32Array(5);

        const written = buffer.write(writeData);
        assert(written === 5, `Expected 5 written, got ${written}`);

        const available = buffer.availableRead();
        assert(available === 5, `Expected 5 available, got ${available}`);

        const read = buffer.read(readData);
        assert(read === 5, `Expected 5 read, got ${read}`);

        for (let i = 0; i < 5; i++) {
            assert(readData[i] === writeData[i], `Mismatch at index ${i}`);
        }
        console.log('   ✅ PASSED');
    }

    // Test 2: Wrap-around
    {
        console.log('\n📋 Test 2: Wrap-around Behavior');
        const buffer = new AudioRingBuffer(8);

        // Fill most of buffer
        const data1 = new Float32Array([1, 2, 3, 4, 5, 6]);
        buffer.write(data1);

        // Read some
        const readBuf = new Float32Array(4);
        buffer.read(readBuf);

        // Write more (should wrap)
        const data2 = new Float32Array([7, 8, 9]);
        const written = buffer.write(data2);
        assert(written === 3, `Expected 3 written after wrap, got ${written}`);

        // Verify available
        const available = buffer.availableRead();
        assert(available === 5, `Expected 5 available after wrap, got ${available}`);
        console.log('   ✅ PASSED');
    }

    // Test 3: Full buffer behavior
    {
        console.log('\n📋 Test 3: Full Buffer Protection');
        const buffer = new AudioRingBuffer(4);
        const data = new Float32Array([1, 2, 3, 4, 5]);

        const written = buffer.write(data);
        // Ring buffer should protect against overflow
        // Either write partial or return error count
        assert(written <= 4, `Should not write more than capacity, got ${written}`);
        console.log('   ✅ PASSED');
    }

    // Test 4: Empty buffer read
    {
        console.log('\n📋 Test 4: Empty Buffer Read');
        const buffer = new AudioRingBuffer(16);
        const readBuf = new Float32Array(4);

        const read = buffer.read(readBuf);
        assert(read === 0, `Expected 0 read from empty buffer, got ${read}`);
        console.log('   ✅ PASSED');
    }

    console.log('\n' + '='.repeat(40));
    console.log('🎉 All Ring Buffer Tests PASSED');
}

// Run if executed directly
runTests().catch(console.error);

export { runTests };
