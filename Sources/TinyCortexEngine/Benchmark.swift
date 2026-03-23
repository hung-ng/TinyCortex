import Foundation

// MARK: - Statistics

public struct StatsSummary {
    public let mean: Double
    public let p50: Double
    public let p95: Double
    public let p99: Double
    public let min: Double
    public let max: Double
    public let count: Int
}

private func computeStats(_ values: [Double]) -> StatsSummary {
    guard !values.isEmpty else {
        return StatsSummary(mean: 0, p50: 0, p95: 0, p99: 0, min: 0, max: 0, count: 0)
    }
    let sorted = values.sorted()
    let n = sorted.count
    let sum = sorted.reduce(0, +)
    return StatsSummary(
        mean: sum / Double(n),
        p50: sorted[n / 2],
        p95: sorted[min(Int(Double(n) * 0.95), n - 1)],
        p99: sorted[min(Int(Double(n) * 0.99), n - 1)],
        min: sorted[0],
        max: sorted[n - 1],
        count: n
    )
}

// MARK: - Render callback timing (real-time safe)
//
// Single-producer (audio thread), single-consumer (main thread) ring buffer.
// No locks — relies on ARM64 word-aligned load/store atomicity and Apple Silicon TSO
// for correct ordering. The audio thread writes data then updates the index;
// the main thread snapshots the index then reads data. Worst case on a stale read:
// one sample is missed or duplicated — acceptable for display-only stats.

struct RenderTimingState {
    let timings: UnsafeMutablePointer<Double>
    var writeIndex: Int32 = 0  // Int32 for atomic word-sized access on ARM64
    var count: Int32 = 0
    let capacity: Int32
}

func createRenderTimingState(capacity: Int32 = 1024) -> UnsafeMutablePointer<RenderTimingState> {
    let timings = UnsafeMutablePointer<Double>.allocate(capacity: Int(capacity))
    timings.initialize(repeating: 0, count: Int(capacity))
    let state = UnsafeMutablePointer<RenderTimingState>.allocate(capacity: 1)
    state.initialize(to: RenderTimingState(timings: timings, capacity: capacity))
    return state
}

func destroyRenderTimingState(_ state: UnsafeMutablePointer<RenderTimingState>) {
    state.pointee.timings.deinitialize(count: Int(state.pointee.capacity))
    state.pointee.timings.deallocate()
    state.deinitialize(count: 1)
    state.deallocate()
}

/// Record a timing sample (call from render callback — single producer only). Real-time safe.
func recordRenderTiming(_ state: UnsafeMutablePointer<RenderTimingState>, _ milliseconds: Double) {
    let idx = state.pointee.writeIndex
    // Write data BEFORE updating index — on Apple Silicon TSO, the main thread
    // will see the data write before the index update.
    state.pointee.timings[Int(idx)] = milliseconds
    state.pointee.writeIndex = (idx + 1) % state.pointee.capacity
    if state.pointee.count < state.pointee.capacity {
        state.pointee.count += 1
    }
}

/// Read timing stats (call from main thread — single consumer only).
/// Snapshots the index first, then reads data. May include one stale/fresh sample
/// at the boundary — acceptable for display purposes.
func readRenderTimingStats(_ state: UnsafeMutablePointer<RenderTimingState>) -> StatsSummary {
    // Snapshot indices first (word-sized reads, atomic on ARM64)
    let count = state.pointee.count
    let writeIdx = state.pointee.writeIndex
    let capacity = state.pointee.capacity
    guard count > 0 else {
        return StatsSummary(mean: 0, p50: 0, p95: 0, p99: 0, min: 0, max: 0, count: 0)
    }
    var values = [Double]()
    values.reserveCapacity(Int(count))
    let start = (writeIdx - count + capacity) % capacity
    for i: Int32 in 0..<count {
        values.append(state.pointee.timings[Int((start + i) % capacity)])
    }
    return computeStats(values)
}

// MARK: - mach_absolute_time helpers

/// Cached timebase for converting mach ticks to milliseconds.
let machTimebaseMs: Double = {
    var info = mach_timebase_info_data_t()
    mach_timebase_info(&info)
    return Double(info.numer) / Double(info.denom) / 1_000_000
}()

// MARK: - Standalone inference benchmark

public struct MemoryReport {
    public let weightBytes: Int
    public let activationBytes: Int
    public let stateBytes: Int
    public let ioBytes: Int
    public let totalBytes: Int
}

public final class InferenceBenchmark {
    private let engine: InferenceEngine

    public init(engine: InferenceEngine) {
        self.engine = engine
    }

    /// Run N iterations of synchronous process(), return timing stats in milliseconds.
    public func run(iterations: Int = 1000, warmup: Int = 50) -> StatsSummary {
        let bufSize = engine.bufferSize
        let input = UnsafeMutablePointer<Float>.allocate(capacity: bufSize)
        let output = UnsafeMutablePointer<Float>.allocate(capacity: bufSize)
        defer {
            input.deallocate()
            output.deallocate()
        }

        // Fill with realistic signal (sine wave)
        for i in 0..<bufSize {
            input[i] = sinf(Float(i) * 0.1) * 0.5
        }

        // Warmup
        for _ in 0..<warmup {
            engine.process(input: input, output: output)
        }

        // Timed iterations
        var timings = [Double]()
        timings.reserveCapacity(iterations)

        for _ in 0..<iterations {
            engine.resetState()
            let t0 = mach_absolute_time()
            engine.process(input: input, output: output)
            let t1 = mach_absolute_time()
            timings.append(Double(t1 - t0) * machTimebaseMs)
        }

        return computeStats(timings)
    }

    /// Report memory allocated for Metal buffers.
    public func memoryReport() -> MemoryReport {
        return engine.memoryReport()
    }
}
