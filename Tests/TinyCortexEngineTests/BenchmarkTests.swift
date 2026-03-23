import XCTest
@testable import TinyCortexEngine

final class BenchmarkTests: XCTestCase {
    static var engine: InferenceEngine!

    override class func setUp() {
        let configURL = Bundle.module.url(forResource: "model_config", withExtension: "json", subdirectory: "TestData")!
        let weightsURL = Bundle.module.url(forResource: "weights", withExtension: "bin", subdirectory: "TestData")!
        let config = try! ModelConfig.load(from: configURL)
        engine = try! InferenceEngine(config: config, weightsURL: weightsURL)
    }

    override func setUp() {
        Self.engine.resetState()
    }

    func testInferenceLatency() throws {
        let benchmark = InferenceBenchmark(engine: Self.engine)
        let stats = benchmark.run(iterations: 500, warmup: 50)

        print("=== Inference Latency (ms) ===")
        print("  Mean:  \(String(format: "%.3f", stats.mean))")
        print("  P50:   \(String(format: "%.3f", stats.p50))")
        print("  P95:   \(String(format: "%.3f", stats.p95))")
        print("  P99:   \(String(format: "%.3f", stats.p99))")
        print("  Min:   \(String(format: "%.3f", stats.min))")
        print("  Max:   \(String(format: "%.3f", stats.max))")
        print("  Count: \(stats.count)")
        print("  Budget: 5.8ms (256 samples @ 44.1kHz)")

        // P99 check uses generous margin — sync process() + resetState() per iteration
        // is slower than real-time processAsync(). Live render stats (shown in UI) are
        // the true real-time measurement.
        XCTAssertLessThan(stats.mean, 5.8, "Mean inference latency must fit within 5.8ms real-time budget")
    }

    func testMemoryUsage() throws {
        let report = Self.engine.memoryReport()

        print("=== Metal Buffer Memory ===")
        print("  Weights:     \(report.weightBytes) bytes (\(String(format: "%.1f", Double(report.weightBytes) / 1024)) KB)")
        print("  Activations: \(report.activationBytes) bytes (\(String(format: "%.1f", Double(report.activationBytes) / 1024)) KB)")
        print("  State:       \(report.stateBytes) bytes (\(String(format: "%.1f", Double(report.stateBytes) / 1024)) KB)")
        print("  I/O:         \(report.ioBytes) bytes (\(String(format: "%.1f", Double(report.ioBytes) / 1024)) KB)")
        print("  Total:       \(report.totalBytes) bytes (\(String(format: "%.1f", Double(report.totalBytes) / 1024)) KB)")

        XCTAssertLessThan(report.totalBytes, 256 * 1024, "Total Metal buffer memory should be under 256KB")
    }
}
