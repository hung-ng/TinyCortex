import Foundation

// MARK: - Biquad filter types

public struct BiquadCoefficients {
    var b0: Float = 1, b1: Float = 0, b2: Float = 0
    var a1: Float = 0, a2: Float = 0  // normalized by a0
}

public struct BiquadState {
    var z1: Float = 0, z2: Float = 0  // Transposed Direct Form II delays
}

public struct EQBandCoefficients {
    var bass: BiquadCoefficients
    var mid: BiquadCoefficients
    var treble: BiquadCoefficients
}

/// Double-buffered EQ state for lock-free main→audio thread coefficient updates.
/// Main thread writes new coefficients to the inactive slot, then flips `activeSlot`.
/// Audio thread reads `activeSlot` and uses that slot's coefficients.
/// `activeSlot` is Int32 (word-sized) — atomic on ARM64 for aligned loads/stores.
/// Apple Silicon enforces TSO, so coefficient stores are visible before the slot flip.
public struct EQState {
    var coeffs0: EQBandCoefficients   // slot 0
    var coeffs1: EQBandCoefficients   // slot 1
    var activeSlot: Int32             // 0 or 1 — written by main, read by audio
    var bassState: BiquadState
    var midState: BiquadState
    var trebleState: BiquadState
    var enabled: Bool
}

// MARK: - Coefficient calculation (Audio EQ Cookbook — Robert Bristow-Johnson)

private func lowShelfCoefficients(freq: Float, gain: Float, sampleRate: Float) -> BiquadCoefficients {
    if abs(gain) < 0.01 { return BiquadCoefficients() }  // bypass at ~0 dB

    let A = powf(10, gain / 40)  // gain is in dB
    let w0 = 2 * Float.pi * freq / sampleRate
    let cosw0 = cosf(w0)
    let sinw0 = sinf(w0)
    let alpha = sinw0 / 2 * sqrtf(2)  // S = 1.0, simplifies: (A+1/A)*(1/S-1) + 2 = 2
    let sqrtA2alpha = 2 * sqrtf(A) * alpha

    let a0 = (A + 1) + (A - 1) * cosw0 + sqrtA2alpha
    return BiquadCoefficients(
        b0: A * ((A + 1) - (A - 1) * cosw0 + sqrtA2alpha) / a0,
        b1: 2 * A * ((A - 1) - (A + 1) * cosw0) / a0,
        b2: A * ((A + 1) - (A - 1) * cosw0 - sqrtA2alpha) / a0,
        a1: -2 * ((A - 1) + (A + 1) * cosw0) / a0,
        a2: ((A + 1) + (A - 1) * cosw0 - sqrtA2alpha) / a0
    )
}

private func peakingEQCoefficients(freq: Float, gain: Float, Q: Float, sampleRate: Float) -> BiquadCoefficients {
    if abs(gain) < 0.01 { return BiquadCoefficients() }

    let A = powf(10, gain / 40)
    let w0 = 2 * Float.pi * freq / sampleRate
    let cosw0 = cosf(w0)
    let sinw0 = sinf(w0)
    let alpha = sinw0 / (2 * Q)

    let a0 = 1 + alpha / A
    return BiquadCoefficients(
        b0: (1 + alpha * A) / a0,
        b1: (-2 * cosw0) / a0,
        b2: (1 - alpha * A) / a0,
        a1: (-2 * cosw0) / a0,
        a2: (1 - alpha / A) / a0
    )
}

private func highShelfCoefficients(freq: Float, gain: Float, sampleRate: Float) -> BiquadCoefficients {
    if abs(gain) < 0.01 { return BiquadCoefficients() }

    let A = powf(10, gain / 40)
    let w0 = 2 * Float.pi * freq / sampleRate
    let cosw0 = cosf(w0)
    let sinw0 = sinf(w0)
    let alpha = sinw0 / 2 * sqrtf(2)  // S = 1.0
    let sqrtA2alpha = 2 * sqrtf(A) * alpha

    let a0 = (A + 1) - (A - 1) * cosw0 + sqrtA2alpha
    return BiquadCoefficients(
        b0: A * ((A + 1) + (A - 1) * cosw0 + sqrtA2alpha) / a0,
        b1: -2 * A * ((A - 1) + (A + 1) * cosw0) / a0,
        b2: A * ((A + 1) + (A - 1) * cosw0 - sqrtA2alpha) / a0,
        a1: 2 * ((A - 1) - (A + 1) * cosw0) / a0,
        a2: ((A + 1) - (A - 1) * cosw0 - sqrtA2alpha) / a0
    )
}

// MARK: - Equalizer

public final class Equalizer {

    /// Bass gain in dB (-12 to +12). Recalculates coefficients on set.
    public var bassGain: Float = 0 {
        didSet { updateCoefficients { $0.bass = lowShelfCoefficients(freq: 200, gain: bassGain, sampleRate: sampleRate) } }
    }

    /// Mid gain in dB (-12 to +12). Recalculates coefficients on set.
    public var midGain: Float = 0 {
        didSet { updateCoefficients { $0.mid = peakingEQCoefficients(freq: 1000, gain: midGain, Q: 1.0, sampleRate: sampleRate) } }
    }

    /// Treble gain in dB (-12 to +12). Recalculates coefficients on set.
    public var trebleGain: Float = 0 {
        didSet { updateCoefficients { $0.treble = highShelfCoefficients(freq: 4000, gain: trebleGain, sampleRate: sampleRate) } }
    }

    let statePtr: UnsafeMutablePointer<EQState>
    private let sampleRate: Float

    public init(sampleRate: Float = 44100) {
        self.sampleRate = sampleRate
        let unity = EQBandCoefficients(
            bass: BiquadCoefficients(),
            mid: BiquadCoefficients(),
            treble: BiquadCoefficients()
        )
        statePtr = .allocate(capacity: 1)
        statePtr.initialize(to: EQState(
            coeffs0: unity,
            coeffs1: unity,
            activeSlot: 0,
            bassState: BiquadState(),
            midState: BiquadState(),
            trebleState: BiquadState(),
            enabled: true
        ))
    }

    deinit {
        statePtr.deinitialize(count: 1)
        statePtr.deallocate()
    }

    /// Write new coefficients to the inactive slot, then flip activeSlot.
    /// Called from main thread only. The audio thread sees a consistent coefficient set
    /// because the entire inactive slot is written before the Int32 flip (TSO on Apple Silicon).
    private func updateCoefficients(_ update: (inout EQBandCoefficients) -> Void) {
        let active = statePtr.pointee.activeSlot
        let inactive = 1 - active
        // Copy current active coefficients to inactive slot, then apply the update
        if active == 0 {
            statePtr.pointee.coeffs1 = statePtr.pointee.coeffs0
            update(&statePtr.pointee.coeffs1)
        } else {
            statePtr.pointee.coeffs0 = statePtr.pointee.coeffs1
            update(&statePtr.pointee.coeffs0)
        }
        // Flip — Int32 store is atomic on ARM64. TSO ensures prior stores are visible first.
        statePtr.pointee.activeSlot = inactive
    }

    /// Reset filter state (clear delay lines). Call on stop/start to avoid clicks.
    public func resetState() {
        statePtr.pointee.bassState = BiquadState()
        statePtr.pointee.midState = BiquadState()
        statePtr.pointee.trebleState = BiquadState()
    }

    /// Process buffer through 3-band EQ in-place. Real-time safe — no allocations.
    /// Called from render callback via state pointer (matches GateState pattern).
    public static func process(
        buffer: UnsafeMutablePointer<Float>,
        count: Int,
        state: UnsafeMutablePointer<EQState>
    ) {
        // Read active slot (Int32 load is atomic on ARM64) and snapshot coefficients
        let coeffs = state.pointee.activeSlot == 0 ? state.pointee.coeffs0 : state.pointee.coeffs1

        // Skip if all bands are at unity (bypass coefficients)
        let bassActive = abs(coeffs.bass.b0 - 1) > 0.0001 || abs(coeffs.bass.b1) > 0.0001
        let midActive = abs(coeffs.mid.b0 - 1) > 0.0001 || abs(coeffs.mid.b1) > 0.0001
        let trebleActive = abs(coeffs.treble.b0 - 1) > 0.0001 || abs(coeffs.treble.b1) > 0.0001

        if !bassActive && !midActive && !trebleActive { return }

        // Apply cascaded biquads: bass → mid → treble
        for i in 0..<count {
            var x = buffer[i]

            if bassActive {
                let y = coeffs.bass.b0 * x + state.pointee.bassState.z1
                state.pointee.bassState.z1 = coeffs.bass.b1 * x - coeffs.bass.a1 * y + state.pointee.bassState.z2
                state.pointee.bassState.z2 = coeffs.bass.b2 * x - coeffs.bass.a2 * y
                x = y
            }

            if midActive {
                let y = coeffs.mid.b0 * x + state.pointee.midState.z1
                state.pointee.midState.z1 = coeffs.mid.b1 * x - coeffs.mid.a1 * y + state.pointee.midState.z2
                state.pointee.midState.z2 = coeffs.mid.b2 * x - coeffs.mid.a2 * y
                x = y
            }

            if trebleActive {
                let y = coeffs.treble.b0 * x + state.pointee.trebleState.z1
                state.pointee.trebleState.z1 = coeffs.treble.b1 * x - coeffs.treble.a1 * y + state.pointee.trebleState.z2
                state.pointee.trebleState.z2 = coeffs.treble.b2 * x - coeffs.treble.a2 * y
                x = y
            }

            buffer[i] = x
        }
    }
}
