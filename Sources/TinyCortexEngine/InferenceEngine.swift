import Foundation
import Metal

/// Metal-accelerated TCN inference engine.
/// Processes audio in fixed-size buffers with persistent state across calls.
public final class InferenceEngine {
    public let device: MTLDevice
    public let commandQueue: MTLCommandQueue
    public let config: ModelConfig
    public let bufferSize: Int

    private let channels: Int
    private let numBlocks: Int

    // Pipeline states
    private let conv1x1Pipeline: MTLComputePipelineState
    private let causalDilatedConv3Pipeline: MTLComputePipelineState
    private let preluPipeline: MTLComputePipelineState
    private let conv1x1ResidualPipeline: MTLComputePipelineState
    private let updateStatePipeline: MTLComputePipelineState

    // Buffers
    private let weightBuffer: MTLBuffer
    private let inputBuffer: MTLBuffer
    private let outputBuffer: MTLBuffer
    private let activationA: MTLBuffer
    private let activationB: MTLBuffer
    private let stateBuffers: [MTLBuffer]
    private var headOffsets: [Int]  // ring buffer head per block
    private let stateLengths: [Int]
    private var pendingCommandBuffer: MTLCommandBuffer?

    /// Previous async command buffer for double-buffered processing.

    // Weight layout constants
    private let inputConvWeightOffset: Int
    private let inputConvBiasOffset: Int
    private let blockBaseOffset: Int
    private let blockStride: Int
    private let outputConvWeightOffset: Int
    private let outputConvBiasOffset: Int

    // Block-relative offsets
    private let dilatedConvWeightRel = 0
    private let dilatedConvBiasRel: Int
    private let preluAlphaRel: Int
    private let conv1x1WeightRel: Int
    private let conv1x1BiasRel: Int

    private let threadgroupSize = MTLSize(width: 256, height: 1, depth: 1)

    public init(
        config: ModelConfig,
        weightsURL: URL,
        device: MTLDevice? = nil,
        bufferSize: Int = 256
    ) throws {
        let dev = device ?? MTLCreateSystemDefaultDevice()!
        self.device = dev
        self.commandQueue = dev.makeCommandQueue()!
        self.config = config
        self.bufferSize = bufferSize
        self.channels = config.channels
        self.numBlocks = config.blocks

        // Load Metal library
        let library = try InferenceEngine.loadMetalLibrary(device: dev)

        // Create pipeline states
        conv1x1Pipeline = try dev.makeComputePipelineState(
            function: library.makeFunction(name: "conv1x1")!)
        causalDilatedConv3Pipeline = try dev.makeComputePipelineState(
            function: library.makeFunction(name: "causal_dilated_conv3")!)
        preluPipeline = try dev.makeComputePipelineState(
            function: library.makeFunction(name: "prelu")!)
        conv1x1ResidualPipeline = try dev.makeComputePipelineState(
            function: library.makeFunction(name: "conv1x1_residual")!)
        updateStatePipeline = try dev.makeComputePipelineState(
            function: library.makeFunction(name: "update_state")!)

        // Load weights
        weightBuffer = try WeightLoader.load(from: weightsURL, device: dev)

        // Pre-compute weight offsets
        inputConvWeightOffset = config.offset(for: "input_conv.weight")
        inputConvBiasOffset = config.offset(for: "input_conv.bias")
        blockBaseOffset = config.offset(for: "blocks.0.net.0.conv.weight")
        blockStride = config.blockByteStride
        outputConvWeightOffset = config.offset(for: "output_conv.weight")
        outputConvBiasOffset = config.offset(for: "output_conv.bias")

        // Block-relative offsets (consistent across all blocks)
        let b0base = blockBaseOffset
        dilatedConvBiasRel = config.offset(for: "blocks.0.net.0.conv.bias") - b0base
        preluAlphaRel = config.offset(for: "blocks.0.net.1.weight") - b0base
        conv1x1WeightRel = config.offset(for: "blocks.0.net.2.weight") - b0base
        conv1x1BiasRel = config.offset(for: "blocks.0.net.2.bias") - b0base

        // Allocate activation buffers
        let T = bufferSize
        let C = channels
        let shared = MTLResourceOptions.storageModeShared

        inputBuffer = dev.makeBuffer(length: T * MemoryLayout<Float>.size, options: shared)!
        inputBuffer.label = "input"
        outputBuffer = dev.makeBuffer(length: T * MemoryLayout<Float>.size, options: shared)!
        outputBuffer.label = "output"
        activationA = dev.makeBuffer(length: T * C * MemoryLayout<Float>.size, options: shared)!
        activationA.label = "activationA"
        activationB = dev.makeBuffer(length: T * C * MemoryLayout<Float>.size, options: shared)!
        activationB.label = "activationB"

        // Allocate state buffers (one per block, zero-initialized)
        var states: [MTLBuffer] = []
        var lengths: [Int] = []
        for i in 0..<numBlocks {
            let dilation = 1 << i
            let stateLen = 2 * dilation
            lengths.append(stateLen)
            let stateBytes = stateLen * C * MemoryLayout<Float>.size
            let buf = dev.makeBuffer(length: stateBytes, options: shared)!
            buf.label = "state_\(i)"
            memset(buf.contents(), 0, stateBytes)
            states.append(buf)
        }
        stateBuffers = states
        stateLengths = lengths
        headOffsets = [Int](repeating: 0, count: numBlocks)
    }

    private static func loadMetalLibrary(device: MTLDevice) throws -> MTLLibrary {
        // Try compiled default library first (Xcode builds)
        if let lib = try? device.makeDefaultLibrary(bundle: Bundle.module) {
            return lib
        }
        // Fall back to runtime compilation from source
        guard let url = Bundle.module.url(forResource: "TCNKernels", withExtension: "metal") else {
            throw InferenceError.metalSourceNotFound
        }
        let source = try String(contentsOf: url, encoding: .utf8)
        return try device.makeLibrary(source: source, options: nil)
    }

    /// Report memory allocated for Metal buffers.
    public func memoryReport() -> MemoryReport {
        let ioBytes = inputBuffer.length + outputBuffer.length
        let activationBytes = activationA.length + activationB.length
        let stateBytes = stateBuffers.reduce(0) { $0 + $1.length }
        return MemoryReport(
            weightBytes: weightBuffer.length,
            activationBytes: activationBytes,
            stateBytes: stateBytes,
            ioBytes: ioBytes,
            totalBytes: weightBuffer.length + ioBytes + activationBytes + stateBytes
        )
    }

    /// Reset all state buffers to zero (e.g., between songs).
    public func resetState() {
        pendingCommandBuffer?.waitUntilCompleted()
        pendingCommandBuffer = nil
        for i in 0..<numBlocks {
            let stateBytes = stateLengths[i] * channels * MemoryLayout<Float>.size
            memset(stateBuffers[i].contents(), 0, stateBytes)
            headOffsets[i] = 0
        }
    }

    /// Synchronous processing — blocks until GPU completes. For tests and offline use.
    public func process(input: UnsafePointer<Float>, output: UnsafeMutablePointer<Float>) {
        let bytes = bufferSize * MemoryLayout<Float>.size
        memcpy(inputBuffer.contents(), input, bytes)

        guard let cmdBuf = encodeInference() else { return }
        cmdBuf.commit()
        cmdBuf.waitUntilCompleted()
        memcpy(output, outputBuffer.contents(), bytes)
    }

    /// Async double-buffered processing — outputs previous frame's result while
    /// submitting new GPU work. Adds one buffer of latency. For real-time audio.
    public func processAsync(input: UnsafePointer<Float>, output: UnsafeMutablePointer<Float>) {
        let bytes = bufferSize * MemoryLayout<Float>.size
        memcpy(inputBuffer.contents(), input, bytes)

        // Output previous frame's result (GPU had ~5.8ms to complete)
        if let pending = pendingCommandBuffer {
            pending.waitUntilCompleted()
            memcpy(output, outputBuffer.contents(), bytes)
        } else {
            memset(output, 0, bytes)
        }

        guard let cmdBuf = encodeInference() else {
            pendingCommandBuffer = nil
            return
        }
        cmdBuf.commit()
        pendingCommandBuffer = cmdBuf
    }

    /// Encode the full 42-dispatch inference pipeline into a command buffer.
    private func encodeInference() -> MTLCommandBuffer? {
        let T = bufferSize
        let C = channels

        guard let cmdBuf = commandQueue.makeCommandBuffer(),
              let encoder = cmdBuf.makeComputeCommandEncoder() else {
            return nil
        }

        // --- Dispatch 1: Input conv (1 → C) ---
        encodeConv1x1(
            encoder: encoder,
            input: inputBuffer, inputOffset: 0,
            output: activationA, outputOffset: 0,
            weightOffset: inputConvWeightOffset,
            biasOffset: inputConvBiasOffset,
            T: T, inCh: 1, outCh: C
        )

        // Barrier: input conv must finish before blocks read activationA
        encoder.memoryBarrier(resources: [activationA])

        // --- Dispatches 2–41: 10 TCN blocks × 4 dispatches ---
        for i in 0..<numBlocks {
            let dilation = 1 << i
            let stateLen = stateLengths[i]
            let head = headOffsets[i]
            let base = blockBaseOffset + i * blockStride

            // (a) Causal dilated conv: activationA + state → activationB
            encodeCausalDilatedConv3(
                encoder: encoder,
                input: activationA, output: activationB,
                state: stateBuffers[i],
                weightOffset: base + dilatedConvWeightRel,
                biasOffset: base + dilatedConvBiasRel,
                T: T, C: C, dilation: dilation,
                stateLen: stateLen, head: head
            )

            // Barrier: conv3 writes activationB, reads state →
            // update_state writes state, PReLU reads activationB
            encoder.memoryBarrier(resources: [activationB, stateBuffers[i]])

            // (b) Update state: write activationA to ring buffer
            encodeUpdateState(
                encoder: encoder,
                activation: activationA,
                state: stateBuffers[i],
                T: T, C: C, stateLen: stateLen, head: head
            )

            // Barrier: update_state must finish writing state
            encoder.memoryBarrier(resources: [stateBuffers[i]])

            // Advance ring buffer head on CPU for next call
            let writeCount = min(T, stateLen)
            headOffsets[i] = (head + writeCount) % stateLen

            // (c) PReLU: activationB in-place
            encodePReLU(
                encoder: encoder,
                data: activationB,
                alphaOffset: base + preluAlphaRel,
                T: T, C: C
            )

            // Barrier: PReLU must finish before conv1x1_residual reads activationB
            encoder.memoryBarrier(resources: [activationB])

            // (d) Conv1x1 + residual: activationB + activationA → activationA
            encodeConv1x1Residual(
                encoder: encoder,
                input: activationB,
                residual: activationA,
                output: activationA,
                weightOffset: base + conv1x1WeightRel,
                biasOffset: base + conv1x1BiasRel,
                T: T, C: C
            )

            // Barrier: conv1x1_residual must finish writing activationA
            // before next block reads it
            encoder.memoryBarrier(resources: [activationA])
        }

        // --- Dispatch 42: Output conv (C → 1) ---
        encodeConv1x1(
            encoder: encoder,
            input: activationA, inputOffset: 0,
            output: outputBuffer, outputOffset: 0,
            weightOffset: outputConvWeightOffset,
            biasOffset: outputConvBiasOffset,
            T: T, inCh: C, outCh: 1
        )

        encoder.endEncoding()
        return cmdBuf
    }

    // MARK: - Encoding helpers

    private func gridSize(threads: Int) -> MTLSize {
        MTLSize(width: threads, height: 1, depth: 1)
    }

    private func encodeConv1x1(
        encoder: MTLComputeCommandEncoder,
        input: MTLBuffer, inputOffset: Int,
        output: MTLBuffer, outputOffset: Int,
        weightOffset: Int, biasOffset: Int,
        T: Int, inCh: Int, outCh: Int
    ) {
        encoder.setComputePipelineState(conv1x1Pipeline)
        encoder.setBuffer(input, offset: inputOffset, index: 0)
        encoder.setBuffer(output, offset: outputOffset, index: 1)
        encoder.setBuffer(weightBuffer, offset: weightOffset, index: 2)
        encoder.setBuffer(weightBuffer, offset: biasOffset, index: 3)
        var t = UInt32(T), ic = UInt32(inCh), oc = UInt32(outCh)
        encoder.setBytes(&t, length: 4, index: 4)
        encoder.setBytes(&ic, length: 4, index: 5)
        encoder.setBytes(&oc, length: 4, index: 6)
        encoder.dispatchThreads(gridSize(threads: T), threadsPerThreadgroup: threadgroupSize)
    }

    private func encodeCausalDilatedConv3(
        encoder: MTLComputeCommandEncoder,
        input: MTLBuffer, output: MTLBuffer,
        state: MTLBuffer,
        weightOffset: Int, biasOffset: Int,
        T: Int, C: Int, dilation: Int,
        stateLen: Int, head: Int
    ) {
        encoder.setComputePipelineState(causalDilatedConv3Pipeline)
        encoder.setBuffer(input, offset: 0, index: 0)
        encoder.setBuffer(output, offset: 0, index: 1)
        encoder.setBuffer(state, offset: 0, index: 2)
        encoder.setBuffer(weightBuffer, offset: weightOffset, index: 3)
        encoder.setBuffer(weightBuffer, offset: biasOffset, index: 4)
        var t = UInt32(T), c = UInt32(C), d = UInt32(dilation)
        var sl = UInt32(stateLen), h = UInt32(head)
        encoder.setBytes(&t, length: 4, index: 5)
        encoder.setBytes(&c, length: 4, index: 6)
        encoder.setBytes(&d, length: 4, index: 7)
        encoder.setBytes(&sl, length: 4, index: 8)
        encoder.setBytes(&h, length: 4, index: 9)
        encoder.dispatchThreads(gridSize(threads: T), threadsPerThreadgroup: threadgroupSize)
    }

    private func encodePReLU(
        encoder: MTLComputeCommandEncoder,
        data: MTLBuffer,
        alphaOffset: Int,
        T: Int, C: Int
    ) {
        encoder.setComputePipelineState(preluPipeline)
        encoder.setBuffer(data, offset: 0, index: 0)
        encoder.setBuffer(weightBuffer, offset: alphaOffset, index: 1)
        var count = UInt32(T * C), c = UInt32(C)
        encoder.setBytes(&count, length: 4, index: 2)
        encoder.setBytes(&c, length: 4, index: 3)
        encoder.dispatchThreads(gridSize(threads: T * C), threadsPerThreadgroup: threadgroupSize)
    }

    private func encodeConv1x1Residual(
        encoder: MTLComputeCommandEncoder,
        input: MTLBuffer,
        residual: MTLBuffer,
        output: MTLBuffer,
        weightOffset: Int, biasOffset: Int,
        T: Int, C: Int
    ) {
        encoder.setComputePipelineState(conv1x1ResidualPipeline)
        encoder.setBuffer(input, offset: 0, index: 0)
        encoder.setBuffer(residual, offset: 0, index: 1)
        encoder.setBuffer(output, offset: 0, index: 2)
        encoder.setBuffer(weightBuffer, offset: weightOffset, index: 3)
        encoder.setBuffer(weightBuffer, offset: biasOffset, index: 4)
        var t = UInt32(T), c = UInt32(C)
        encoder.setBytes(&t, length: 4, index: 5)
        encoder.setBytes(&c, length: 4, index: 6)
        encoder.dispatchThreads(gridSize(threads: T), threadsPerThreadgroup: threadgroupSize)
    }

    private func encodeUpdateState(
        encoder: MTLComputeCommandEncoder,
        activation: MTLBuffer,
        state: MTLBuffer,
        T: Int, C: Int, stateLen: Int, head: Int
    ) {
        encoder.setComputePipelineState(updateStatePipeline)
        encoder.setBuffer(activation, offset: 0, index: 0)
        encoder.setBuffer(state, offset: 0, index: 1)
        var t = UInt32(T), c = UInt32(C), sl = UInt32(stateLen), h = UInt32(head)
        let writeCount = min(T, stateLen)
        encoder.setBytes(&t, length: 4, index: 2)
        encoder.setBytes(&c, length: 4, index: 3)
        encoder.setBytes(&sl, length: 4, index: 4)
        encoder.setBytes(&h, length: 4, index: 5)
        encoder.dispatchThreads(gridSize(threads: writeCount * C), threadsPerThreadgroup: threadgroupSize)
    }

    public enum InferenceError: Error {
        case metalSourceNotFound
    }
}
