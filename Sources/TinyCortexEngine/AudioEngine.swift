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

    /// Noise gate open threshold (RMS). Input above this level opens the gate.
    /// Range: 0.0 (gate disabled) to 1.0. Typical values: 0.005–0.05.
    public var gateThreshold: Float {
        get { gateStatePtr.pointee.openThreshold }
        set { gateStatePtr.pointee.openThreshold = newValue }
    }
    private let gateStatePtr = UnsafeMutablePointer<GateState>.allocate(capacity: 1)

    /// True if the device sample rate doesn't match the model's training rate (44100 Hz).
    public private(set) var sampleRateMismatch = false

    /// Set when the audio device changes. Contains the event description (e.g., "Device changed").
    /// Cleared on next successful start.
    public private(set) var deviceChangeMessage: String?

    private var inputDeviceID: AudioDeviceID = 0
    private var outputDeviceID: AudioDeviceID = 0
    private var aggregateDeviceID: AudioDeviceID = 0

    /// 3-band EQ (bass/mid/treble). Adjust gains in dB from the main thread.
    public let equalizer: Equalizer

    /// Render callback timing state (ring buffer of timing samples).
    private let timingStatePtr: UnsafeMutablePointer<RenderTimingState>

    /// Read render callback timing statistics (call from main thread).
    public var renderStats: StatsSummary {
        readRenderTimingStats(timingStatePtr)
    }

    public init(inferenceEngine: InferenceEngine, gateThreshold: Float = 0.01) {
        self.inferenceEngine = inferenceEngine
        self.bufferSize = inferenceEngine.bufferSize
        gateStatePtr.initialize(to: GateState(openThreshold: gateThreshold))
        self.equalizer = Equalizer(sampleRate: 44100)
        self.timingStatePtr = createRenderTimingState()
    }

    deinit {
        if isRunning { stop() }
        gateStatePtr.deallocate()
        destroyRenderTimingState(timingStatePtr)
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
        var inputDeviceID: AudioDeviceID = 0
        var outputDeviceID: AudioDeviceID = 0
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

        print("[AudioEngine] Default input device: \(inputDeviceID), output device: \(outputDeviceID)")

        // Determine which device to use for AUHAL
        var deviceForAU = inputDeviceID
        if inputDeviceID != outputDeviceID {
            // Different input/output devices — create an aggregate device
            deviceForAU = try createAggregateDevice(input: inputDeviceID, output: outputDeviceID)
            aggregateDeviceID = deviceForAU
            print("[AudioEngine] Created aggregate device: \(deviceForAU)")
        }

        // Set the device on AUHAL (single device for both I/O)
        status = AudioUnitSetProperty(
            au, kAudioOutputUnitProperty_CurrentDevice,
            kAudioUnitScope_Global, 0,
            &deviceForAU, UInt32(MemoryLayout<AudioDeviceID>.size)
        )
        guard status == noErr else {
            throw audioError("Failed to set device", status)
        }

        // --- Set buffer size ---
        setDeviceBufferSize(deviceForAU, frames: UInt32(bSize))

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
        sampleRateMismatch = sr != 44100
        if sampleRateMismatch {
            print("WARNING: Device sample rate is \(sr)Hz, model trained at 44100Hz")
        }

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
            gateState: gateStatePtr,
            eqState: equalizer.statePtr,
            timingState: timingStatePtr,
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

        // Note: render callback uses processAsync() for double-buffered real-time processing

        // --- Set MaximumFramesPerSlice to prevent oversized buffer delivery ---
        var maxFrames = UInt32(bSize)
        AudioUnitSetProperty(
            au, kAudioUnitProperty_MaximumFramesPerSlice,
            kAudioUnitScope_Global, 0,
            &maxFrames, UInt32(MemoryLayout<UInt32>.size)
        )

        // --- Initialize and start ---
        status = AudioUnitInitialize(au)
        guard status == noErr else {
            throw audioError("AudioUnitInitialize failed", status)
        }

        status = AudioOutputUnitStart(au)
        guard status == noErr else {
            throw audioError("AudioOutputUnitStart failed", status)
        }

        self.inputDeviceID = inputDeviceID
        isRunning = true
        deviceChangeMessage = nil
        registerDeviceListeners()
    }

    public func stop() {
        removeDeviceListeners()

        if let au = auHAL {
            AudioOutputUnitStop(au)
            AudioUnitUninitialize(au)
            AudioComponentInstanceDispose(au)
            auHAL = nil
        }

        destroyAggregateDevice()

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
        equalizer.resetState()
    }

    // MARK: - Device-change handling

    private var deviceListenerBlocks: [(AudioObjectID, AudioObjectPropertyAddress, AudioObjectPropertyListenerBlock)] = []
    private var restartPending = false

    private func registerDeviceListeners() {
        let handler: AudioObjectPropertyListenerBlock = { [weak self] _, _ in
            DispatchQueue.main.async {
                guard let self, self.isRunning, !self.restartPending else { return }
                self.restartPending = true
                // Stop immediately to prevent rendering with stale device
                self.stop()
                self.deviceChangeMessage = "Device changed — restarting..."
                // Delay restart to let the system finalize new default devices
                DispatchQueue.main.asyncAfter(deadline: .now() + 1.0) {
                    self.restartPending = false
                    do {
                        try self.start()
                        self.deviceChangeMessage = "Device changed — restarted with new device"
                    } catch {
                        self.deviceChangeMessage = "Device changed — restart failed: \(error.localizedDescription)"
                    }
                }
            }
        }

        // Listen for default device changes
        var defaultInputAddr = AudioObjectPropertyAddress(
            mSelector: kAudioHardwarePropertyDefaultInputDevice,
            mScope: kAudioObjectPropertyScopeGlobal,
            mElement: kAudioObjectPropertyElementMain
        )
        var defaultOutputAddr = AudioObjectPropertyAddress(
            mSelector: kAudioHardwarePropertyDefaultOutputDevice,
            mScope: kAudioObjectPropertyScopeGlobal,
            mElement: kAudioObjectPropertyElementMain
        )

        let systemObj = AudioObjectID(kAudioObjectSystemObject)
        AudioObjectAddPropertyListenerBlock(systemObj, &defaultInputAddr, nil, handler)
        deviceListenerBlocks.append((systemObj, defaultInputAddr, handler))

        AudioObjectAddPropertyListenerBlock(systemObj, &defaultOutputAddr, nil, handler)
        deviceListenerBlocks.append((systemObj, defaultOutputAddr, handler))

        // Listen for device-is-alive on current devices
        var aliveAddr = AudioObjectPropertyAddress(
            mSelector: kAudioDevicePropertyDeviceIsAlive,
            mScope: kAudioObjectPropertyScopeGlobal,
            mElement: kAudioObjectPropertyElementMain
        )

        AudioObjectAddPropertyListenerBlock(inputDeviceID, &aliveAddr, nil, handler)
        deviceListenerBlocks.append((inputDeviceID, aliveAddr, handler))

        // Listen for sample rate changes on the input device
        var sampleRateAddr = AudioObjectPropertyAddress(
            mSelector: kAudioDevicePropertyNominalSampleRate,
            mScope: kAudioObjectPropertyScopeGlobal,
            mElement: kAudioObjectPropertyElementMain
        )
        AudioObjectAddPropertyListenerBlock(inputDeviceID, &sampleRateAddr, nil, handler)
        deviceListenerBlocks.append((inputDeviceID, sampleRateAddr, handler))
    }

    private func removeDeviceListeners() {
        for (objID, var addr, block) in deviceListenerBlocks {
            AudioObjectRemovePropertyListenerBlock(objID, &addr, nil, block)
        }
        deviceListenerBlocks.removeAll()
    }

    // MARK: - Aggregate device

    private func createAggregateDevice(input: AudioDeviceID, output: AudioDeviceID) throws -> AudioDeviceID {
        let inputUID = try deviceUID(for: input)
        let outputUID = try deviceUID(for: output)

        let desc: [String: Any] = [
            kAudioAggregateDeviceNameKey as String: "TinyCortex Aggregate",
            kAudioAggregateDeviceUIDKey as String: "com.tinycortex.aggregate.\(UUID().uuidString)",
            kAudioAggregateDeviceIsPrivateKey as String: true,
            kAudioAggregateDeviceSubDeviceListKey as String: [
                [kAudioSubDeviceUIDKey as String: inputUID],
                [kAudioSubDeviceUIDKey as String: outputUID],
            ],
            kAudioAggregateDeviceMasterSubDeviceKey as String: inputUID,
        ]

        var aggregateID: AudioDeviceID = 0
        let status = AudioHardwareCreateAggregateDevice(desc as CFDictionary, &aggregateID)
        guard status == noErr else {
            throw audioError("Failed to create aggregate device", status)
        }
        return aggregateID
    }

    private func destroyAggregateDevice() {
        if aggregateDeviceID != 0 {
            AudioHardwareDestroyAggregateDevice(aggregateDeviceID)
            aggregateDeviceID = 0
        }
    }

    private func deviceUID(for deviceID: AudioDeviceID) throws -> String {
        var addr = AudioObjectPropertyAddress(
            mSelector: kAudioDevicePropertyDeviceUID,
            mScope: kAudioObjectPropertyScopeGlobal,
            mElement: kAudioObjectPropertyElementMain
        )
        var propSize = UInt32(MemoryLayout<Unmanaged<CFString>>.size)
        var uid: Unmanaged<CFString>?
        let status = AudioObjectGetPropertyData(deviceID, &addr, 0, nil, &propSize, &uid)
        guard status == noErr, let cfStr = uid?.takeRetainedValue() else {
            throw audioError("Failed to get device UID for device \(deviceID)", status)
        }
        return cfStr as String
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

// MARK: - Noise gate state machine

private enum GatePhase {
    case closed
    case attack
    case open
    case hold
    case release
}

private struct GateState {
    var phase: GatePhase = .closed
    var gain: Float = 0.0
    var holdCounter: Int = 0       // samples remaining in hold phase

    // Configurable threshold (user-facing)
    var openThreshold: Float       // RMS to open gate (e.g., 0.015)

    // Fixed parameters
    let closeThreshold: Float = 0.008   // RMS to start closing (lower = hysteresis)
    let attackCoeff: Float = 0.9        // ~1ms attack
    let releaseCoeff: Float = 0.005     // ~50ms release
    let holdSamples: Int = 4410         // 100ms at 44100 Hz

    init(openThreshold: Float) {
        self.openThreshold = openThreshold
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
    let gateState: UnsafeMutablePointer<GateState>
    let eqState: UnsafeMutablePointer<EQState>
    let timingState: UnsafeMutablePointer<RenderTimingState>
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

    // Timing starts after AudioUnitRender (I/O pull) — measures only our compute
    // pipeline (inference + gate + EQ + clamp), not device I/O wait time.
    let t0 = mach_absolute_time()

    // Process through inference engine (async double-buffered)
    if frames == bSize {
        ctx.inference.processAsync(input: ctx.renderBuf, output: outData)
    } else {
        var offset = 0
        while offset + bSize <= frames {
            ctx.inference.processAsync(input: ctx.renderBuf + offset, output: ctx.outputChunk)
            memcpy(outData + offset, ctx.outputChunk, bSize * MemoryLayout<Float>.size)
            offset += bSize
        }
        if offset < frames {
            memset(outData + offset, 0, (frames - offset) * MemoryLayout<Float>.size)
        }
    }

    // --- Noise gate (state machine with hysteresis + hold) ---
    let gs = ctx.gateState
    if gs.pointee.openThreshold > 0 {
        // Measure input RMS
        var sumSq: Float = 0
        for i in 0..<frames {
            let v = ctx.renderBuf[i]
            sumSq += v * v
        }
        let rms = sqrtf(sumSq / Float(frames))

        // State machine transitions
        switch gs.pointee.phase {
        case .closed:
            if rms >= gs.pointee.openThreshold {
                gs.pointee.phase = .attack
            }
        case .attack:
            gs.pointee.gain += gs.pointee.attackCoeff * (1.0 - gs.pointee.gain)
            if gs.pointee.gain >= 0.999 {
                gs.pointee.gain = 1.0
                gs.pointee.phase = .open
            }
        case .open:
            if rms < gs.pointee.closeThreshold {
                gs.pointee.holdCounter = gs.pointee.holdSamples
                gs.pointee.phase = .hold
            }
        case .hold:
            gs.pointee.holdCounter -= frames
            if rms >= gs.pointee.closeThreshold {
                // Signal came back — return to open
                gs.pointee.phase = .open
            } else if gs.pointee.holdCounter <= 0 {
                gs.pointee.phase = .release
            }
        case .release:
            if rms >= gs.pointee.openThreshold {
                gs.pointee.phase = .attack
            } else {
                gs.pointee.gain += gs.pointee.releaseCoeff * (0.0 - gs.pointee.gain)
                if gs.pointee.gain < 0.001 {
                    gs.pointee.gain = 0.0
                    gs.pointee.phase = .closed
                }
            }
        }

        // Apply gate gain to output
        if gs.pointee.gain < 0.999 {
            for i in 0..<frames {
                outData[i] *= gs.pointee.gain
            }
        }
    }

    // --- 3-band EQ ---
    if ctx.eqState.pointee.enabled {
        Equalizer.process(buffer: outData, count: frames, state: ctx.eqState)
    }

    // Clamp output to [-1, 1]
    for i in 0..<frames {
        let v = outData[i]
        if v > 1.0 { outData[i] = 1.0 }
        else if v < -1.0 { outData[i] = -1.0 }
    }

    // Record render callback timing
    let t1 = mach_absolute_time()
    recordRenderTiming(ctx.timingState, Double(t1 - t0) * machTimebaseMs)

    return noErr
}
