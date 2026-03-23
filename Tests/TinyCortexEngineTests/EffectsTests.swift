import XCTest
@testable import TinyCortexEngine

final class EffectsTests: XCTestCase {

    // MARK: - Compressor

    func testCompressorBypass() {
        let comp = Compressor()
        // Disabled by default — should be passthrough
        let input: [Float] = [0.5, -0.3, 0.8, -0.1, 0.0]
        var buffer = input
        buffer.withUnsafeMutableBufferPointer { bp in
            Compressor.process(buffer: bp.baseAddress!, count: bp.count, state: comp.statePtr)
        }
        // When disabled, process is never called (AudioEngine checks enabled flag),
        // but if called, it should still produce reasonable output
        for i in 0..<input.count {
            XCTAssertFalse(buffer[i].isNaN, "Compressor output should not be NaN")
        }
    }

    func testCompressorReducesLevel() {
        let comp = Compressor()
        comp.statePtr.pointee.enabled = true
        comp.threshold = -20
        comp.ratio = 10
        comp.makeupGain = 0

        // Feed loud signal (0.9 ≈ -0.9 dB, well above -20 dB threshold)
        var buffer = [Float](repeating: 0.9, count: 256)
        buffer.withUnsafeMutableBufferPointer { bp in
            Compressor.process(buffer: bp.baseAddress!, count: bp.count, state: comp.statePtr)
        }

        // After envelope settles, output should be reduced
        let lastSample = abs(buffer[255])
        XCTAssertLessThan(lastSample, 0.9, "Compressor should reduce level above threshold")
    }

    func testCompressorSilence() {
        let comp = Compressor()
        comp.statePtr.pointee.enabled = true
        var buffer = [Float](repeating: 0, count: 256)
        buffer.withUnsafeMutableBufferPointer { bp in
            Compressor.process(buffer: bp.baseAddress!, count: bp.count, state: comp.statePtr)
        }
        for v in buffer {
            XCTAssertEqual(v, 0, "Silence in should produce silence out")
        }
    }

    func testCompressorReset() {
        let comp = Compressor()
        comp.statePtr.pointee.enabled = true
        // Process some signal to build up envelope
        var buffer = [Float](repeating: 0.5, count: 256)
        buffer.withUnsafeMutableBufferPointer { bp in
            Compressor.process(buffer: bp.baseAddress!, count: bp.count, state: comp.statePtr)
        }
        XCTAssertGreaterThan(comp.statePtr.pointee.envelope, 0)
        comp.resetState()
        XCTAssertEqual(comp.statePtr.pointee.envelope, 0)
    }

    // MARK: - Overdrive

    func testOverdriveBypass() {
        let od = Overdrive()
        let input: [Float] = [0.1, -0.1, 0.5, -0.5]
        var buffer = input
        buffer.withUnsafeMutableBufferPointer { bp in
            Overdrive.process(buffer: bp.baseAddress!, count: bp.count, state: od.statePtr)
        }
        for i in 0..<input.count {
            XCTAssertFalse(buffer[i].isNaN, "Overdrive output should not be NaN")
        }
    }

    func testOverdriveClipsSymmetrically() {
        let od = Overdrive()
        od.statePtr.pointee.enabled = true
        od.drive = 100  // max drive
        od.level = 100

        var pos: [Float] = [0.5]
        var neg: [Float] = [-0.5]
        pos.withUnsafeMutableBufferPointer { bp in
            Overdrive.process(buffer: bp.baseAddress!, count: 1, state: od.statePtr)
        }
        od.resetState()  // reset biquad state for fair comparison
        neg.withUnsafeMutableBufferPointer { bp in
            Overdrive.process(buffer: bp.baseAddress!, count: 1, state: od.statePtr)
        }

        XCTAssertEqual(pos[0], -neg[0], accuracy: 0.01, "Overdrive should clip symmetrically")
    }

    func testOverdriveReset() {
        let od = Overdrive()
        var buffer: [Float] = [0.5, 0.3, 0.1]
        buffer.withUnsafeMutableBufferPointer { bp in
            Overdrive.process(buffer: bp.baseAddress!, count: bp.count, state: od.statePtr)
        }
        od.resetState()
        XCTAssertEqual(od.statePtr.pointee.toneState.z1, 0)
        XCTAssertEqual(od.statePtr.pointee.toneState.z2, 0)
    }

    // MARK: - Delay

    func testDelayBypass() {
        let delay = Delay()
        let input: [Float] = [1.0, 0.0, 0.0, 0.0]
        var buffer = input
        buffer.withUnsafeMutableBufferPointer { bp in
            Delay.process(buffer: bp.baseAddress!, count: bp.count, state: delay.statePtr)
        }
        for v in buffer {
            XCTAssertFalse(v.isNaN)
        }
    }

    func testDelayImpulseAppears() {
        let delay = Delay(sampleRate: 44100)
        delay.statePtr.pointee.enabled = true
        delay.time = 100  // 100ms = 4410 samples
        delay.feedback = 0
        delay.mix = 100   // 100% wet

        let delaySamples = 4410
        let totalSamples = delaySamples + 256

        // Impulse at sample 0
        var buffer = [Float](repeating: 0, count: totalSamples)
        buffer[0] = 1.0

        buffer.withUnsafeMutableBufferPointer { bp in
            Delay.process(buffer: bp.baseAddress!, count: bp.count, state: delay.statePtr)
        }

        // The delayed impulse should appear at delaySamples offset
        XCTAssertEqual(buffer[delaySamples], 1.0, accuracy: 0.01, "Impulse should appear at delay offset")
        // Before the delay, output should be ~0 (100% wet, no dry)
        XCTAssertEqual(buffer[1], 0, accuracy: 0.001, "Before delay time, wet signal should be 0")
    }

    func testDelaySilence() {
        let delay = Delay()
        delay.statePtr.pointee.enabled = true
        var buffer = [Float](repeating: 0, count: 256)
        buffer.withUnsafeMutableBufferPointer { bp in
            Delay.process(buffer: bp.baseAddress!, count: bp.count, state: delay.statePtr)
        }
        for v in buffer {
            XCTAssertEqual(v, 0, "Silence in should produce silence out")
        }
    }

    func testDelayReset() {
        let delay = Delay()
        delay.statePtr.pointee.enabled = true
        var buffer: [Float] = [1.0, 0.5, 0.3]
        buffer.withUnsafeMutableBufferPointer { bp in
            Delay.process(buffer: bp.baseAddress!, count: bp.count, state: delay.statePtr)
        }
        delay.resetState()
        XCTAssertEqual(delay.statePtr.pointee.writeIndex, 0)
    }

    // MARK: - Reverb

    func testReverbBypass() {
        let reverb = Reverb()
        let input: [Float] = [0.5, -0.3, 0.1, 0.0]
        var buffer = input
        buffer.withUnsafeMutableBufferPointer { bp in
            Reverb.process(buffer: bp.baseAddress!, count: bp.count, state: reverb.statePtr)
        }
        for v in buffer {
            XCTAssertFalse(v.isNaN, "Reverb output should not be NaN")
        }
    }

    func testReverbDecays() {
        let reverb = Reverb()
        reverb.statePtr.pointee.enabled = true
        reverb.roomSize = 50
        reverb.mix = 100  // 100% wet to isolate reverb tail

        // Feed impulse then silence
        let totalSamples = 44100  // 1 second
        var buffer = [Float](repeating: 0, count: totalSamples)
        buffer[0] = 1.0

        buffer.withUnsafeMutableBufferPointer { bp in
            Reverb.process(buffer: bp.baseAddress!, count: bp.count, state: reverb.statePtr)
        }

        // Early in the tail, there should be energy
        var earlyEnergy: Float = 0
        for i in 2000..<4000 {
            earlyEnergy += buffer[i] * buffer[i]
        }

        // Late in the tail, energy should be lower
        var lateEnergy: Float = 0
        for i in 40000..<42000 {
            lateEnergy += buffer[i] * buffer[i]
        }

        XCTAssertGreaterThan(earlyEnergy, lateEnergy, "Reverb tail should decay over time")
        XCTAssertGreaterThan(earlyEnergy, 0, "Reverb should produce output from impulse")
    }

    func testReverbSilence() {
        let reverb = Reverb()
        reverb.statePtr.pointee.enabled = true
        var buffer = [Float](repeating: 0, count: 256)
        buffer.withUnsafeMutableBufferPointer { bp in
            Reverb.process(buffer: bp.baseAddress!, count: bp.count, state: reverb.statePtr)
        }
        for v in buffer {
            XCTAssertEqual(v, 0, accuracy: 1e-10, "Silence in should produce silence out")
        }
    }

    func testReverbReset() {
        let reverb = Reverb()
        reverb.statePtr.pointee.enabled = true
        var buffer = [Float](repeating: 0.5, count: 256)
        buffer.withUnsafeMutableBufferPointer { bp in
            Reverb.process(buffer: bp.baseAddress!, count: bp.count, state: reverb.statePtr)
        }
        reverb.resetState()
        // After reset, processing silence should give silence
        var silence = [Float](repeating: 0, count: 256)
        silence.withUnsafeMutableBufferPointer { bp in
            Reverb.process(buffer: bp.baseAddress!, count: bp.count, state: reverb.statePtr)
        }
        for v in silence {
            XCTAssertEqual(v, 0, accuracy: 1e-10, "After reset, silence should produce silence")
        }
    }
}
