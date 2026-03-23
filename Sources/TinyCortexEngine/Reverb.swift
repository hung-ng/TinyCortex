import Foundation

// MARK: - Reverb internal types

struct CombFilter {
    var buffer: UnsafeMutablePointer<Float>
    var bufferLength: Int
    var index: Int
    var filterStore: Float  // LP filter state for damping
}

struct AllpassFilter {
    var buffer: UnsafeMutablePointer<Float>
    var bufferLength: Int
    var index: Int
}

// MARK: - Reverb state

public struct ReverbParams {
    var feedback: Float     // comb feedback 0.7–0.99 (mapped from roomSize)
    var damping: Float      // LP coefficient 0–1
    var mix: Float          // dry/wet 0–1
}

public struct ReverbState {
    var params0: ReverbParams
    var params1: ReverbParams
    var activeSlot: Int32
    var combs: (CombFilter, CombFilter, CombFilter, CombFilter,
                CombFilter, CombFilter, CombFilter, CombFilter)
    var allpasses: (AllpassFilter, AllpassFilter, AllpassFilter, AllpassFilter)
    var enabled: Bool
}

// MARK: - Reverb

/// Freeverb (Schroeder-Moorer): 8 parallel comb filters → 4 serial allpass filters.
public final class Reverb {

    /// Room size 0–100%. Maps comb feedback to 0.7–0.99.
    public var roomSize: Float = 40 {
        didSet { updateParams { $0.feedback = 0.7 + (roomSize / 100.0) * 0.29 } }
    }

    /// Damping 0–100%. LP coefficient in comb feedback. Higher = darker tail.
    public var damping: Float = 60 {
        didSet { updateParams { $0.damping = damping / 100.0 } }
    }

    /// Dry/wet mix 0–100%.
    public var mix: Float = 15 {
        didSet { updateParams { $0.mix = mix / 100.0 } }
    }

    public var enabled: Bool {
        get { statePtr.pointee.enabled }
        set { statePtr.pointee.enabled = newValue }
    }

    public let statePtr: UnsafeMutablePointer<ReverbState>
    private var allocatedBuffers: [UnsafeMutablePointer<Float>] = []
    private var allocatedLengths: [Int] = []

    // Freeverb comb filter lengths (samples at 44100 Hz)
    private static let combLengths = [1116, 1188, 1277, 1356, 1422, 1491, 1557, 1617]
    // Freeverb allpass filter lengths
    private static let allpassLengths = [556, 441, 341, 225]

    public init(sampleRate: Float = 44100) {
        let scale = sampleRate / 44100.0

        // Allocate comb buffers
        var combs: [CombFilter] = []
        var allBufs: [UnsafeMutablePointer<Float>] = []
        var allLens: [Int] = []

        for len in Self.combLengths {
            let scaledLen = max(1, Int(Float(len) * scale))
            let buf: UnsafeMutablePointer<Float> = .allocate(capacity: scaledLen)
            buf.initialize(repeating: 0, count: scaledLen)
            allBufs.append(buf)
            allLens.append(scaledLen)
            combs.append(CombFilter(buffer: buf, bufferLength: scaledLen, index: 0, filterStore: 0))
        }

        // Allocate allpass buffers
        var allpasses: [AllpassFilter] = []
        for len in Self.allpassLengths {
            let scaledLen = max(1, Int(Float(len) * scale))
            let buf: UnsafeMutablePointer<Float> = .allocate(capacity: scaledLen)
            buf.initialize(repeating: 0, count: scaledLen)
            allBufs.append(buf)
            allLens.append(scaledLen)
            allpasses.append(AllpassFilter(buffer: buf, bufferLength: scaledLen, index: 0))
        }

        let defaults = ReverbParams(
            feedback: 0.7 + 0.4 * 0.29,  // roomSize 40%
            damping: 0.6,
            mix: 0.15
        )

        statePtr = .allocate(capacity: 1)
        statePtr.initialize(to: ReverbState(
            params0: defaults,
            params1: defaults,
            activeSlot: 0,
            combs: (combs[0], combs[1], combs[2], combs[3],
                    combs[4], combs[5], combs[6], combs[7]),
            allpasses: (allpasses[0], allpasses[1], allpasses[2], allpasses[3]),
            enabled: false
        ))

        allocatedBuffers = allBufs
        allocatedLengths = allLens
    }

    deinit {
        statePtr.deinitialize(count: 1)
        statePtr.deallocate()
        for (buf, len) in zip(allocatedBuffers, allocatedLengths) {
            buf.deinitialize(count: len)
            buf.deallocate()
        }
    }

    private func updateParams(_ update: (inout ReverbParams) -> Void) {
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
        // Clear all comb buffers and state
        withCombPointers(statePtr) { combPtrs in
            for i in 0..<8 {
                let len = combPtrs[i].bufferLength
                combPtrs[i].buffer.initialize(repeating: 0, count: len)
                combPtrs[i].index = 0
                combPtrs[i].filterStore = 0
            }
        }
        // Clear all allpass buffers and state
        withAllpassPointers(statePtr) { apPtrs in
            for i in 0..<4 {
                let len = apPtrs[i].bufferLength
                apPtrs[i].buffer.initialize(repeating: 0, count: len)
                apPtrs[i].index = 0
            }
        }
    }

    /// Process buffer in-place. Real-time safe.
    public static func process(
        buffer: UnsafeMutablePointer<Float>,
        count: Int,
        state: UnsafeMutablePointer<ReverbState>
    ) {
        let p = state.pointee.activeSlot == 0 ? state.pointee.params0 : state.pointee.params1
        let damp1 = p.damping
        let damp2 = 1.0 - damp1

        for i in 0..<count {
            let input = buffer[i]
            var wet: Float = 0

            // 8 parallel comb filters
            withCombPointers(state) { combPtrs in
                for c in 0..<8 {
                    let output = combPtrs[c].buffer[combPtrs[c].index]

                    // Low-pass filter in feedback path (damping)
                    combPtrs[c].filterStore = output * damp2 + combPtrs[c].filterStore * damp1

                    // Write new sample: input + filtered feedback
                    combPtrs[c].buffer[combPtrs[c].index] = input + combPtrs[c].filterStore * p.feedback

                    // Advance index
                    combPtrs[c].index += 1
                    if combPtrs[c].index >= combPtrs[c].bufferLength {
                        combPtrs[c].index = 0
                    }

                    wet += output
                }
            }

            // 4 serial allpass filters
            withAllpassPointers(state) { apPtrs in
                for a in 0..<4 {
                    let bufOut = apPtrs[a].buffer[apPtrs[a].index]
                    let allpassFeedback: Float = 0.5

                    apPtrs[a].buffer[apPtrs[a].index] = wet + bufOut * allpassFeedback
                    wet = bufOut - wet * allpassFeedback

                    apPtrs[a].index += 1
                    if apPtrs[a].index >= apPtrs[a].bufferLength {
                        apPtrs[a].index = 0
                    }
                }
            }

            // Mix dry/wet
            buffer[i] = input * (1.0 - p.mix) + wet * p.mix
        }
    }
}

// MARK: - Tuple access helpers (real-time safe, no allocations)

/// Provides mutable pointers to all 8 comb filters in the tuple.
private func withCombPointers(
    _ state: UnsafeMutablePointer<ReverbState>,
    body: (UnsafeMutableBufferPointer<CombFilter>) -> Void
) {
    withUnsafeMutablePointer(to: &state.pointee.combs) { tuplePtr in
        let raw = UnsafeMutableRawPointer(tuplePtr)
        let bound = raw.assumingMemoryBound(to: CombFilter.self)
        body(UnsafeMutableBufferPointer(start: bound, count: 8))
    }
}

/// Provides mutable pointers to all 4 allpass filters in the tuple.
private func withAllpassPointers(
    _ state: UnsafeMutablePointer<ReverbState>,
    body: (UnsafeMutableBufferPointer<AllpassFilter>) -> Void
) {
    withUnsafeMutablePointer(to: &state.pointee.allpasses) { tuplePtr in
        let raw = UnsafeMutableRawPointer(tuplePtr)
        let bound = raw.assumingMemoryBound(to: AllpassFilter.self)
        body(UnsafeMutableBufferPointer(start: bound, count: 4))
    }
}
