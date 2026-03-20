import AudioToolbox
import CoreAudio

/// Real-time audio engine using AUHAL (kAudioUnitSubType_HALOutput).
/// Single render callback with synchronized I/O, automatic format conversion,
/// true 1-buffer latency. This is the standard approach used by JUCE, PortAudio, etc.
public final class AudioEngine {
    private let inferenceEngine: InferenceEngine
    private let bufferSize: Int

    private var auHAL: AudioComponentInstance?
    private var contextPtr: UnsafeMutablePointer<RenderContext>?

    public private(set) var isRunning = false
    public private(set) var sampleRate: Double = 0
    public var lastRenderFrames: Int = 0

    /// Noise gate threshold (RMS). Input below this level fades output to silence.
    /// Range: 0.0 (gate disabled) to 1.0. Typical values: 0.005–0.05.
    public var gateThreshold: Float {
        get { gateThresholdPtr.pointee }
        set { gateThresholdPtr.pointee = newValue }
    }
    private let gateThresholdPtr = UnsafeMutablePointer<Float>.allocate(capacity: 1)
    private let gateGainPtr = UnsafeMutablePointer<Float>.allocate(capacity: 1)

    public init(inferenceEngine: InferenceEngine, gateThreshold: Float = 0.01) {
        self.inferenceEngine = inferenceEngine
        self.bufferSize = inferenceEngine.bufferSize
        gateThresholdPtr.initialize(to: gateThreshold)
        gateGainPtr.initialize(to: 0.0)
    }

    deinit {
        if isRunning { stop() }
        gateThresholdPtr.deallocate()
        gateGainPtr.deallocate()
    }

    public func start() throws {
        let bSize = bufferSize

        // --- Create AUHAL audio unit ---
        var desc = AudioComponentDescription(
            componentType: kAudioUnitType_Output,
            componentSubType: kAudioUnitSubType_HALOutput,
            componentManufacturer: kAudioUnitManufacturer_Apple,
            componentFlags: 0,
            componentFlagsMask: 0
        )
        guard let component = AudioComponentFindNext(nil, &desc) else {
            throw audioError("HALOutput component not found", -1)
        }
        var au: AudioComponentInstance?
        var status = AudioComponentInstanceNew(component, &au)
        guard status == noErr, let au else {
            throw audioError("Failed to create HALOutput instance", status)
        }
        auHAL = au

        // --- Enable input (element 1) ---
        var enableIO: UInt32 = 1
        status = AudioUnitSetProperty(
            au, kAudioOutputUnitProperty_EnableIO,
            kAudioUnitScope_Input, 1,
            &enableIO, UInt32(MemoryLayout<UInt32>.size)
        )
        guard status == noErr else {
            throw audioError("Failed to enable input", status)
        }

        // --- Set devices ---
        var inputDeviceID = AudioDeviceID()
        var outputDeviceID = AudioDeviceID()
        var propSize = UInt32(MemoryLayout<AudioDeviceID>.size)
        var addr = AudioObjectPropertyAddress(
            mSelector: kAudioHardwarePropertyDefaultInputDevice,
            mScope: kAudioObjectPropertyScopeGlobal,
            mElement: kAudioObjectPropertyElementMain
        )
        AudioObjectGetPropertyData(
            AudioObjectID(kAudioObjectSystemObject), &addr, 0, nil, &propSize, &inputDeviceID
        )
        addr.mSelector = kAudioHardwarePropertyDefaultOutputDevice
        AudioObjectGetPropertyData(
            AudioObjectID(kAudioObjectSystemObject), &addr, 0, nil, &propSize, &outputDeviceID
        )

        // Set input device on element 1
        status = AudioUnitSetProperty(
            au, kAudioOutputUnitProperty_CurrentDevice,
            kAudioUnitScope_Global, 1,
            &inputDeviceID, UInt32(MemoryLayout<AudioDeviceID>.size)
        )
        guard status == noErr else {
            throw audioError("Failed to set input device", status)
        }

        // Set output device on element 0
        status = AudioUnitSetProperty(
            au, kAudioOutputUnitProperty_CurrentDevice,
            kAudioUnitScope_Global, 0,
            &outputDeviceID, UInt32(MemoryLayout<AudioDeviceID>.size)
        )
        guard status == noErr else {
            throw audioError("Failed to set output device", status)
        }

        // --- Set buffer size on both devices ---
        setDeviceBufferSize(inputDeviceID, frames: UInt32(bSize))
        setDeviceBufferSize(outputDeviceID, frames: UInt32(bSize))

        // --- Query sample rate from input device ---
        var sr: Float64 = 0
        propSize = UInt32(MemoryLayout<Float64>.size)
        addr = AudioObjectPropertyAddress(
            mSelector: kAudioDevicePropertyNominalSampleRate,
            mScope: kAudioObjectPropertyScopeGlobal,
            mElement: kAudioObjectPropertyElementMain
        )
        AudioObjectGetPropertyData(inputDeviceID, &addr, 0, nil, &propSize, &sr)
        sampleRate = sr

        // --- Set our desired stream format: Float32 mono ---
        // AUHAL converts between device format and this format automatically.
        var streamFormat = AudioStreamBasicDescription(
            mSampleRate: sr,
            mFormatID: kAudioFormatLinearPCM,
            mFormatFlags: kAudioFormatFlagIsFloat | kAudioFormatFlagIsPacked | kAudioFormatFlagIsNonInterleaved,
            mBytesPerPacket: UInt32(MemoryLayout<Float>.size),
            mFramesPerPacket: 1,
            mBytesPerFrame: UInt32(MemoryLayout<Float>.size),
            mChannelsPerFrame: 1,
            mBitsPerChannel: 32,
            mReserved: 0
        )

        // Set format on input scope of element 1 (what we READ from the input)
        status = AudioUnitSetProperty(
            au, kAudioUnitProperty_StreamFormat,
            kAudioUnitScope_Output, 1,
            &streamFormat, UInt32(MemoryLayout<AudioStreamBasicDescription>.size)
        )
        guard status == noErr else {
            throw audioError("Failed to set input stream format", status)
        }

        // Set format on output scope of element 0 (what we WRITE to the output)
        status = AudioUnitSetProperty(
            au, kAudioUnitProperty_StreamFormat,
            kAudioUnitScope_Input, 0,
            &streamFormat, UInt32(MemoryLayout<AudioStreamBasicDescription>.size)
        )
        guard status == noErr else {
            throw audioError("Failed to set output stream format", status)
        }

        // --- Allocate render buffer for pulling input ---
        let renderBuf = UnsafeMutablePointer<Float>.allocate(capacity: bSize)
        renderBuf.initialize(repeating: 0, count: bSize)

        // AudioBufferList for input pull
        let pullABL = UnsafeMutablePointer<AudioBufferList>.allocate(capacity: 1)
        pullABL.pointee = AudioBufferList(
            mNumberBuffers: 1,
            mBuffers: AudioBuffer(
                mNumberChannels: 1,
                mDataByteSize: UInt32(bSize * MemoryLayout<Float>.size),
                mData: UnsafeMutableRawPointer(renderBuf)
            )
        )

        let outChunk = UnsafeMutablePointer<Float>.allocate(capacity: bSize)
        outChunk.initialize(repeating: 0, count: bSize)

        // --- Build render context ---
        let ctxPtr = UnsafeMutablePointer<RenderContext>.allocate(capacity: 1)
        ctxPtr.initialize(to: RenderContext(
            inference: inferenceEngine,
            auHAL: au,
            pullABL: pullABL,
            renderBuf: renderBuf,
            outputChunk: outChunk,
            bufferSize: bSize,
            gateGain: gateGainPtr,
            gateThreshold: gateThresholdPtr,
            engine: self
        ))
        contextPtr = ctxPtr

        // --- Set render callback on output element 0 ---
        var callbackStruct = AURenderCallbackStruct(
            inputProc: renderCallback,
            inputProcRefCon: UnsafeMutableRawPointer(ctxPtr)
        )
        status = AudioUnitSetProperty(
            au, kAudioUnitProperty_SetRenderCallback,
            kAudioUnitScope_Input, 0,
            &callbackStruct, UInt32(MemoryLayout<AURenderCallbackStruct>.size)
        )
        guard status == noErr else {
            throw audioError("Failed to set render callback", status)
        }

        // Enable async double-buffering for real-time safety
        inferenceEngine.asyncProcessing = true

        // --- Initialize and start ---
        status = AudioUnitInitialize(au)
        guard status == noErr else {
            throw audioError("AudioUnitInitialize failed", status)
        }

        status = AudioOutputUnitStart(au)
        guard status == noErr else {
            throw audioError("AudioOutputUnitStart failed", status)
        }

        isRunning = true
    }

    public func stop() {
        if let au = auHAL {
            AudioOutputUnitStop(au)
            AudioUnitUninitialize(au)
            AudioComponentInstanceDispose(au)
            auHAL = nil
        }

        if let ctx = contextPtr {
            ctx.pointee.pullABL.deallocate()
            ctx.pointee.renderBuf.deallocate()
            ctx.pointee.outputChunk.deallocate()
            ctx.deinitialize(count: 1)
            ctx.deallocate()
            contextPtr = nil
        }

        isRunning = false
        inferenceEngine.resetState()
    }

    // MARK: - Device buffer size

    private func setDeviceBufferSize(_ device: AudioDeviceID, frames: UInt32) {
        var address = AudioObjectPropertyAddress(
            mSelector: kAudioDevicePropertyBufferFrameSize,
            mScope: kAudioObjectPropertyScopeGlobal,
            mElement: kAudioObjectPropertyElementMain
        )
        var frameSize = frames
        AudioObjectSetPropertyData(
            device, &address, 0, nil,
            UInt32(MemoryLayout<UInt32>.size), &frameSize
        )
    }

    // MARK: - Error helper

    private func audioError(_ message: String, _ status: OSStatus) -> NSError {
        NSError(
            domain: "AudioEngine", code: Int(status),
            userInfo: [NSLocalizedDescriptionKey: "\(message) (OSStatus \(status))"]
        )
    }
}

// MARK: - Render context and callback

private struct RenderContext {
    let inference: InferenceEngine
    let auHAL: AudioComponentInstance
    let pullABL: UnsafeMutablePointer<AudioBufferList>
    let renderBuf: UnsafeMutablePointer<Float>
    let outputChunk: UnsafeMutablePointer<Float>
    let bufferSize: Int
    let gateGain: UnsafeMutablePointer<Float>
    let gateThreshold: UnsafeMutablePointer<Float>
    unowned let engine: AudioEngine
}

/// AUHAL render callback — pulls input via AudioUnitRender, processes, writes output.
private func renderCallback(
    _ inRefCon: UnsafeMutableRawPointer,
    _ ioActionFlags: UnsafeMutablePointer<AudioUnitRenderActionFlags>,
    _ inTimeStamp: UnsafePointer<AudioTimeStamp>,
    _ inBusNumber: UInt32,
    _ inNumberFrames: UInt32,
    _ ioData: UnsafeMutablePointer<AudioBufferList>?
) -> OSStatus {
    let ctx = inRefCon.assumingMemoryBound(to: RenderContext.self).pointee
    let bSize = ctx.bufferSize
    let frames = Int(inNumberFrames)

    guard let ioData else { return noErr }
    let outABL = UnsafeMutableAudioBufferListPointer(ioData)
    guard outABL.count > 0, let outData = outABL[0].mData?.assumingMemoryBound(to: Float.self) else {
        return noErr
    }

    ctx.engine.lastRenderFrames = frames

    // Guard: if frames exceed our buffer capacity, output silence
    if frames > bSize {
        memset(outData, 0, frames * MemoryLayout<Float>.size)
        return noErr
    }

    // Pull input from AUHAL element 1 (input device)
    // Format is already Float32 mono thanks to the stream format we set.
    ctx.pullABL.pointee.mBuffers.mDataByteSize = UInt32(frames * MemoryLayout<Float>.size)
    ctx.pullABL.pointee.mBuffers.mData = UnsafeMutableRawPointer(ctx.renderBuf)

    let pullStatus = AudioUnitRender(
        ctx.auHAL, ioActionFlags, inTimeStamp, 1, inNumberFrames, ctx.pullABL
    )

    if pullStatus != noErr {
        // Input not ready — feed silence
        memset(ctx.renderBuf, 0, frames * MemoryLayout<Float>.size)
    }

    // Process through inference engine
    if frames == bSize {
        ctx.inference.process(input: ctx.renderBuf, output: outData)
    } else {
        var offset = 0
        while offset + bSize <= frames {
            ctx.inference.process(input: ctx.renderBuf + offset, output: ctx.outputChunk)
            memcpy(outData + offset, ctx.outputChunk, bSize * MemoryLayout<Float>.size)
            offset += bSize
        }
        if offset < frames {
            memset(outData + offset, 0, (frames - offset) * MemoryLayout<Float>.size)
        }
    }

    // --- Noise gate ---
    let threshold = ctx.gateThreshold.pointee
    if threshold > 0 {
        // Measure input RMS
        var sumSq: Float = 0
        for i in 0..<frames {
            let v = ctx.renderBuf[i]
            sumSq += v * v
        }
        let rms = sqrtf(sumSq / Float(frames))

        // Smooth gate gain: fast attack (~1ms), slow release (~50ms)
        let attackCoeff: Float = 0.9     // open quickly
        let releaseCoeff: Float = 0.005  // close slowly
        var gain = ctx.gateGain.pointee
        let target: Float = rms >= threshold ? 1.0 : 0.0
        let coeff = target > gain ? attackCoeff : releaseCoeff
        gain = gain + coeff * (target - gain)
        ctx.gateGain.pointee = gain

        // Apply gate gain to output
        if gain < 0.999 {
            for i in 0..<frames {
                outData[i] *= gain
            }
        }
    }

    // Clamp output to [-1, 1]
    for i in 0..<frames {
        let v = outData[i]
        if v > 1.0 { outData[i] = 1.0 }
        else if v < -1.0 { outData[i] = -1.0 }
    }

    return noErr
}
