import XCTest
@testable import TinyCortexEngine

final class InferenceTests: XCTestCase {
    static var engine: InferenceEngine!

    override class func setUp() {
        let configURL = Bundle.module.url(forResource: "model_config", withExtension: "json", subdirectory: "TestData")!
        let weightsURL = Bundle.module.url(forResource: "weights", withExtension: "bin", subdirectory: "TestData")!
        let config = try! ModelConfig.load(from: configURL)
        engine = try! InferenceEngine(config: config, weightsURL: weightsURL)
    }

    override func setUp() {
        // Reset state before each test
        Self.engine.resetState()
    }

    // MARK: - Model Config

    func testModelConfigParsing() throws {
        let url = Bundle.module.url(forResource: "model_config", withExtension: "json", subdirectory: "TestData")!
        let config = try ModelConfig.load(from: url)
        XCTAssertEqual(config.channels, 16)
        XCTAssertEqual(config.blocks, 10)
        XCTAssertEqual(config.kernelSize, 3)
        XCTAssertEqual(config.totalParameters, 10769)
        XCTAssertEqual(config.totalBytes, 43076)
        XCTAssertEqual(config.receptiveField, 2047)
    }

    // MARK: - Single Buffer Test

    func testSingleBuffer() throws {
        let input = try loadFloatBinary("test_input_single")
        let expected = try loadFloatBinary("test_output_single")

        XCTAssertEqual(input.count, 256)
        XCTAssertEqual(expected.count, 256)

        var output = [Float](repeating: 0, count: 256)
        Self.engine.process(input: input, output: &output)

        // Compare with tolerance
        var maxDiff: Float = 0
        for i in 0..<256 {
            let diff = abs(output[i] - expected[i])
            maxDiff = max(maxDiff, diff)
        }

        print("Single buffer max diff: \(maxDiff)")
        XCTAssertLessThan(maxDiff, 1e-4, "Single buffer output differs from PyTorch reference by \(maxDiff)")
    }

    // MARK: - Multi Buffer State Test (most critical)

    func testMultiBufferState() throws {
        let input = try loadFloatBinary("test_input_multi")
        let expected = try loadFloatBinary("test_output_multi")

        let chunkSize = 256
        let numChunks = input.count / chunkSize
        XCTAssertEqual(numChunks, 8, "Expected 8 chunks (2048 samples)")

        var allOutput = [Float]()

        for chunk in 0..<numChunks {
            let start = chunk * chunkSize
            var output = [Float](repeating: 0, count: chunkSize)

            input.withUnsafeBufferPointer { inputBuf in
                Self.engine.process(
                    input: inputBuf.baseAddress! + start,
                    output: &output
                )
            }

            allOutput.append(contentsOf: output)
        }

        XCTAssertEqual(allOutput.count, expected.count)

        var maxDiff: Float = 0
        var sumSqDiff: Float = 0
        for i in 0..<expected.count {
            let diff = abs(allOutput[i] - expected[i])
            maxDiff = max(maxDiff, diff)
            sumSqDiff += diff * diff
        }
        let rmse = sqrt(sumSqDiff / Float(expected.count))

        print("Multi buffer max diff: \(maxDiff), RMSE: \(rmse)")
        XCTAssertLessThan(maxDiff, 1e-4, "Multi-buffer output differs from PyTorch reference by \(maxDiff)")
    }

    // MARK: - Impulse Response Test

    func testImpulseResponse() throws {
        let input = try loadFloatBinary("test_input_impulse")
        let expected = try loadFloatBinary("test_output_impulse")

        let chunkSize = 256
        let numChunks = input.count / chunkSize

        var allOutput = [Float]()

        for chunk in 0..<numChunks {
            let start = chunk * chunkSize
            var output = [Float](repeating: 0, count: chunkSize)

            input.withUnsafeBufferPointer { inputBuf in
                Self.engine.process(
                    input: inputBuf.baseAddress! + start,
                    output: &output
                )
            }

            allOutput.append(contentsOf: output)
        }

        // Verify non-zero response extends across receptive field
        let nonZeroCount = allOutput.enumerated().filter { abs($0.element) > 1e-6 }.count
        print("Impulse response non-zero samples: \(nonZeroCount)")
        XCTAssertGreaterThan(nonZeroCount, 100, "Impulse response should be non-zero for many samples")

        // Compare to PyTorch reference
        var maxDiff: Float = 0
        for i in 0..<expected.count {
            let diff = abs(allOutput[i] - expected[i])
            maxDiff = max(maxDiff, diff)
        }

        print("Impulse response max diff: \(maxDiff)")
        XCTAssertLessThan(maxDiff, 1e-4, "Impulse response differs from PyTorch reference by \(maxDiff)")
    }

    // MARK: - Silence Test

    func testSilenceProducesNearSilence() throws {
        let chunkSize = 256
        let numChunks = 16
        let zeroInput = [Float](repeating: 0, count: chunkSize)

        for chunk in 0..<numChunks {
            var output = [Float](repeating: 0, count: chunkSize)
            Self.engine.process(input: zeroInput, output: &output)
            let maxAbs = output.map { abs($0) }.max() ?? 0
            let rms = sqrt(output.map { $0 * $0 }.reduce(0, +) / Float(chunkSize))
            print("Silence chunk \(chunk): max=\(maxAbs)  rms=\(rms)")
            XCTAssertLessThan(maxAbs, 0.01, "Silence should produce near-silence, got max \(maxAbs) at chunk \(chunk)")
        }
    }

    // MARK: - Determinism Test

    func testDeterminism() throws {
        let input = try loadFloatBinary("test_input_single")
        var output1 = [Float](repeating: 0, count: 256)
        var output2 = [Float](repeating: 0, count: 256)

        Self.engine.process(input: input, output: &output1)

        Self.engine.resetState()
        Self.engine.process(input: input, output: &output2)

        for i in 0..<256 {
            XCTAssertEqual(output1[i], output2[i], "Output should be deterministic")
        }
    }

    // MARK: - Helpers

    private func loadFloatBinary(_ name: String) throws -> [Float] {
        guard let url = Bundle.module.url(forResource: name, withExtension: "bin", subdirectory: "TestData") else {
            throw TestError.fileNotFound(name)
        }
        let data = try Data(contentsOf: url)
        let count = data.count / MemoryLayout<Float>.size
        return data.withUnsafeBytes { buf in
            Array(buf.bindMemory(to: Float.self).prefix(count))
        }
    }

    enum TestError: Error {
        case fileNotFound(String)
    }
}
