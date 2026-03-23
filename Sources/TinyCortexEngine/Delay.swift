import Foundation

// MARK: - Delay state

public struct DelayParams {
    var delaySamples: Int   // time in samples
    var feedback: Float     // 0–0.95
    var mix: Float          // 0–1 dry/wet
}

public struct DelayState {
    var params0: DelayParams
    var params1: DelayParams
    var activeSlot: Int32
    var buffer: UnsafeMutablePointer<Float>
    var bufferLength: Int
    var writeIndex: Int
    var enabled: Bool
}

// MARK: - Delay

public final class Delay {

    /// Delay time in ms (50–2000).
    public var time: Float = 350 {
        didSet { updateParams { $0.delaySamples = Self.msToSamples(time, sampleRate: sampleRate) } }
    }

    /// Feedback 0–95%. Hard-capped at 0.95.
    public var feedback: Float = 30 {
        didSet { updateParams { $0.feedback = min(feedback / 100.0, 0.95) } }
    }

    /// Dry/wet mix 0–100%.
    public var mix: Float = 25 {
        didSet { updateParams { $0.mix = mix / 100.0 } }
    }

    public var enabled: Bool {
        get { statePtr.pointee.enabled }
        set { statePtr.pointee.enabled = newValue }
    }

    public let statePtr: UnsafeMutablePointer<DelayState>
    private let sampleRate: Float
    private let delayBuffer: UnsafeMutablePointer<Float>
    private let maxDelaySamples: Int

    public init(sampleRate: Float = 44100) {
        self.sampleRate = sampleRate
        // Max 2 seconds at sample rate
        maxDelaySamples = Int(2.0 * sampleRate)
        delayBuffer = .allocate(capacity: maxDelaySamples)
        delayBuffer.initialize(repeating: 0, count: maxDelaySamples)

        let defaults = DelayParams(
            delaySamples: Self.msToSamples(350, sampleRate: sampleRate),
            feedback: 0.3,
            mix: 0.25
        )
        statePtr = .allocate(capacity: 1)
        statePtr.initialize(to: DelayState(
            params0: defaults,
            params1: defaults,
            activeSlot: 0,
            buffer: delayBuffer,
            bufferLength: maxDelaySamples,
            writeIndex: 0,
            enabled: false
        ))
    }

    deinit {
        statePtr.deinitialize(count: 1)
        statePtr.deallocate()
        delayBuffer.deinitialize(count: maxDelaySamples)
        delayBuffer.deallocate()
    }

    private static func msToSamples(_ ms: Float, sampleRate: Float) -> Int {
        max(1, Int(ms * 0.001 * sampleRate))
    }

    private func updateParams(_ update: (inout DelayParams) -> Void) {
        let active = statePtr.pointee.activeSlot
        let inactive = 1 - active
        if active == 0 {
            statePtr.pointee.params1 = statePtr.pointee.params0
            update(&statePtr.pointee.params1)
        } else {
            statePtr.pointee.params0 = statePtr.pointee.params1
            update(&statePtr.pointee.params0)
        }
        statePtr.pointee.activeSlot = inactive
    }

    public func resetState() {
        let len = statePtr.pointee.bufferLength
        statePtr.pointee.buffer.initialize(repeating: 0, count: len)
        statePtr.pointee.writeIndex = 0
    }

    /// Process buffer in-place. Real-time safe.
    public static func process(
        buffer: UnsafeMutablePointer<Float>,
        count: Int,
        state: UnsafeMutablePointer<DelayState>
    ) {
        let p = state.pointee.activeSlot == 0 ? state.pointee.params0 : state.pointee.params1
        let delayBuf = state.pointee.buffer
        let bufLen = state.pointee.bufferLength
        let delaySamples = min(p.delaySamples, bufLen - 1)

        for i in 0..<count {
            let input = buffer[i]
            let writeIdx = state.pointee.writeIndex

            // Read from circular buffer
            var readIdx = writeIdx - delaySamples
            if readIdx < 0 { readIdx += bufLen }
            let delayed = delayBuf[readIdx]

            // Write to circular buffer
            delayBuf[writeIdx] = input + delayed * p.feedback

            // Advance write index
            state.pointee.writeIndex = (writeIdx + 1) % bufLen

            // Mix dry/wet
            buffer[i] = input * (1.0 - p.mix) + delayed * p.mix
        }
    }
}
