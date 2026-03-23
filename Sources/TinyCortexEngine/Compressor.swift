import Foundation

// MARK: - Compressor state

public struct CompressorParams {
    var threshold: Float    // dB, -60 to 0
    var ratio: Float        // 1 to 20
    var attackCoeff: Float  // one-pole coefficient derived from attack ms
    var releaseCoeff: Float // one-pole coefficient derived from release ms
    var makeupGain: Float   // linear gain from dB
}

public struct CompressorState {
    var params0: CompressorParams
    var params1: CompressorParams
    var activeSlot: Int32
    var envelope: Float
    var enabled: Bool
}

// MARK: - Compressor

public final class Compressor {

    /// Threshold in dB (-60 to 0). Level where compression starts.
    public var threshold: Float = -35 {
        didSet { updateParams { $0.threshold = threshold } }
    }

    /// Compression ratio (1 to 20).
    public var ratio: Float = 4.0 {
        didSet { updateParams { $0.ratio = ratio } }
    }

    /// Attack time in ms (1–100). 20ms lets pick transients through.
    public var attack: Float = 20 {
        didSet { updateParams { $0.attackCoeff = Self.msToCoeff(attack, sampleRate: sampleRate) } }
    }

    /// Release time in ms (10–500). 200ms for smooth sustain.
    public var release: Float = 200 {
        didSet { updateParams { $0.releaseCoeff = Self.msToCoeff(release, sampleRate: sampleRate) } }
    }

    /// Makeup gain in dB (0–24). Compensates for compression.
    public var makeupGain: Float = 6 {
        didSet { updateParams { $0.makeupGain = powf(10, makeupGain / 20) } }
    }

    public var enabled: Bool {
        get { statePtr.pointee.enabled }
        set { statePtr.pointee.enabled = newValue }
    }

    public let statePtr: UnsafeMutablePointer<CompressorState>
    private let sampleRate: Float

    public init(sampleRate: Float = 44100) {
        self.sampleRate = sampleRate
        let defaults = CompressorParams(
            threshold: -35,
            ratio: 4.0,
            attackCoeff: Self.msToCoeff(20, sampleRate: sampleRate),
            releaseCoeff: Self.msToCoeff(200, sampleRate: sampleRate),
            makeupGain: powf(10, 6.0 / 20.0)
        )
        statePtr = .allocate(capacity: 1)
        statePtr.initialize(to: CompressorState(
            params0: defaults,
            params1: defaults,
            activeSlot: 0,
            envelope: 0,
            enabled: false
        ))
    }

    deinit {
        statePtr.deinitialize(count: 1)
        statePtr.deallocate()
    }

    private static func msToCoeff(_ ms: Float, sampleRate: Float) -> Float {
        expf(-1.0 / (ms * 0.001 * sampleRate))
    }

    private func updateParams(_ update: (inout CompressorParams) -> Void) {
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
        statePtr.pointee.envelope = 0
    }

    /// Process buffer in-place. Real-time safe.
    public static func process(
        buffer: UnsafeMutablePointer<Float>,
        count: Int,
        state: UnsafeMutablePointer<CompressorState>
    ) {
        let p = state.pointee.activeSlot == 0 ? state.pointee.params0 : state.pointee.params1

        for i in 0..<count {
            let input = buffer[i]
            let absInput = abs(input)

            // Envelope follower (peak)
            let coeff = absInput > state.pointee.envelope ? p.attackCoeff : p.releaseCoeff
            state.pointee.envelope = coeff * state.pointee.envelope + (1.0 - coeff) * absInput

            // Convert envelope to dB
            let envDB = 20.0 * log10f(max(state.pointee.envelope, 1e-10))

            // Compute gain reduction
            var gainDB: Float = 0
            if envDB > p.threshold {
                gainDB = -(envDB - p.threshold) * (1.0 - 1.0 / p.ratio)
            }

            // Apply gain reduction + makeup
            buffer[i] = input * powf(10, gainDB / 20.0) * p.makeupGain
        }
    }
}
