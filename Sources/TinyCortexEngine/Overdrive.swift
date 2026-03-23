import Foundation

// MARK: - Overdrive state

public struct OverdriveParams {
    var driveGain: Float    // linear pre-gain (1x–50x)
    var level: Float        // output volume (0–1)
    var toneCoeffs: BiquadCoefficients
}

public struct OverdriveState {
    var params0: OverdriveParams
    var params1: OverdriveParams
    var activeSlot: Int32
    var toneState: BiquadState
    var enabled: Bool
}

// MARK: - Overdrive

public final class Overdrive {

    /// Drive amount 0–100%. Maps to 1x–50x pre-gain (exponential).
    public var drive: Float = 30 {
        didSet { updateParams { $0.driveGain = Self.driveToGain(drive) } }
    }

    /// Tone 0–100%. Low-pass cutoff 800–8000 Hz.
    public var tone: Float = 60 {
        didSet { updateParams { $0.toneCoeffs = Self.toneCoeffs(tone, sampleRate: sampleRate) } }
    }

    /// Output level 0–100%.
    public var level: Float = 70 {
        didSet { updateParams { $0.level = level / 100.0 } }
    }

    public var enabled: Bool {
        get { statePtr.pointee.enabled }
        set { statePtr.pointee.enabled = newValue }
    }

    public let statePtr: UnsafeMutablePointer<OverdriveState>
    private let sampleRate: Float

    public init(sampleRate: Float = 44100) {
        self.sampleRate = sampleRate
        let defaults = OverdriveParams(
            driveGain: Self.driveToGain(30),
            level: 0.7,
            toneCoeffs: Self.toneCoeffs(60, sampleRate: sampleRate)
        )
        statePtr = .allocate(capacity: 1)
        statePtr.initialize(to: OverdriveState(
            params0: defaults,
            params1: defaults,
            activeSlot: 0,
            toneState: BiquadState(),
            enabled: false
        ))
    }

    deinit {
        statePtr.deinitialize(count: 1)
        statePtr.deallocate()
    }

    /// Map drive percentage (0–100) to linear gain (1x–50x), exponential.
    private static func driveToGain(_ drive: Float) -> Float {
        let normalized = drive / 100.0 // 0–1
        return powf(50, normalized)     // 1x at 0%, 50x at 100%
    }

    /// Low-pass biquad coefficients for tone control. Cutoff: 800–8000 Hz.
    private static func toneCoeffs(_ tone: Float, sampleRate: Float) -> BiquadCoefficients {
        let normalized = tone / 100.0
        let freq = 800.0 * powf(10, normalized) // 800 Hz at 0%, 8000 Hz at 100%
        let w0 = 2.0 * Float.pi * freq / sampleRate
        let cosw0 = cosf(w0)
        let sinw0 = sinf(w0)
        let alpha = sinw0 / (2.0 * 0.707) // Q = 0.707 (Butterworth)

        let a0 = 1.0 + alpha
        return BiquadCoefficients(
            b0: ((1.0 - cosw0) / 2.0) / a0,
            b1: (1.0 - cosw0) / a0,
            b2: ((1.0 - cosw0) / 2.0) / a0,
            a1: (-2.0 * cosw0) / a0,
            a2: (1.0 - alpha) / a0
        )
    }

    private func updateParams(_ update: (inout OverdriveParams) -> Void) {
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
        statePtr.pointee.toneState = BiquadState()
    }

    /// Process buffer in-place. Real-time safe.
    public static func process(
        buffer: UnsafeMutablePointer<Float>,
        count: Int,
        state: UnsafeMutablePointer<OverdriveState>
    ) {
        let p = state.pointee.activeSlot == 0 ? state.pointee.params0 : state.pointee.params1

        for i in 0..<count {
            // Pre-gain
            var x = buffer[i] * p.driveGain

            // Soft saturation
            x = tanhf(x)

            // Tone low-pass biquad (Transposed Direct Form II)
            let y = p.toneCoeffs.b0 * x + state.pointee.toneState.z1
            state.pointee.toneState.z1 = p.toneCoeffs.b1 * x - p.toneCoeffs.a1 * y + state.pointee.toneState.z2
            state.pointee.toneState.z2 = p.toneCoeffs.b2 * x - p.toneCoeffs.a2 * y
            x = y

            // Output level
            buffer[i] = x * p.level
        }
    }
}
