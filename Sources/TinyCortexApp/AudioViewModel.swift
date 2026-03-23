import SwiftUI
import Combine
import TinyCortexEngine

final class AudioViewModel: ObservableObject {
    // State
    @Published var isRunning = false
    @Published var statusMessage = ""

    // Gate
    @Published var gateThreshold: Float = 0.015

    // Compressor
    @Published var compressorEnabled: Bool = false
    @Published var compressorThreshold: Float = -35
    @Published var compressorRatio: Float = 4.0
    @Published var compressorAttack: Float = 20
    @Published var compressorRelease: Float = 200
    @Published var compressorMakeupGain: Float = 6

    // Overdrive
    @Published var overdriveEnabled: Bool = false
    @Published var overdriveDrive: Float = 30
    @Published var overdriveTone: Float = 60
    @Published var overdriveLevel: Float = 70

    // EQ (dB, -12 to +12)
    @Published var bassGain: Float = 0
    @Published var midGain: Float = 0
    @Published var trebleGain: Float = 0

    // Delay
    @Published var delayEnabled: Bool = false
    @Published var delayTime: Float = 350
    @Published var delayFeedback: Float = 30
    @Published var delayMix: Float = 25

    // Reverb
    @Published var reverbEnabled: Bool = false
    @Published var reverbRoomSize: Float = 40
    @Published var reverbDamping: Float = 60
    @Published var reverbMix: Float = 15

    // Performance
    @Published var renderTimeMean: Double = 0
    @Published var renderTimeP95: Double = 0
    @Published var renderTimePeak: Double = 0

    // Device info
    @Published var sampleRate: Double = 0
    @Published var sampleRateMismatch = false
    @Published var deviceChangeMessage: String?
    @Published var modelParams: Int = 0

    private var audioEngine: AudioEngine?
    private var inferenceEngine: InferenceEngine?
    private var pollTimer: Timer?
    private var cancellables = Set<AnyCancellable>()

    deinit {
        stopAudio()
    }

    private func setupBindings() {
        $gateThreshold.sink { [weak self] val in
            self?.audioEngine?.gateThreshold = val
        }.store(in: &cancellables)

        // Compressor
        $compressorEnabled.sink { [weak self] val in
            self?.audioEngine?.compressor.enabled = val
        }.store(in: &cancellables)
        $compressorThreshold.sink { [weak self] val in
            self?.audioEngine?.compressor.threshold = val
        }.store(in: &cancellables)
        $compressorRatio.sink { [weak self] val in
            self?.audioEngine?.compressor.ratio = val
        }.store(in: &cancellables)
        $compressorAttack.sink { [weak self] val in
            self?.audioEngine?.compressor.attack = val
        }.store(in: &cancellables)
        $compressorRelease.sink { [weak self] val in
            self?.audioEngine?.compressor.release = val
        }.store(in: &cancellables)
        $compressorMakeupGain.sink { [weak self] val in
            self?.audioEngine?.compressor.makeupGain = val
        }.store(in: &cancellables)

        // Overdrive
        $overdriveEnabled.sink { [weak self] val in
            self?.audioEngine?.overdrive.enabled = val
        }.store(in: &cancellables)
        $overdriveDrive.sink { [weak self] val in
            self?.audioEngine?.overdrive.drive = val
        }.store(in: &cancellables)
        $overdriveTone.sink { [weak self] val in
            self?.audioEngine?.overdrive.tone = val
        }.store(in: &cancellables)
        $overdriveLevel.sink { [weak self] val in
            self?.audioEngine?.overdrive.level = val
        }.store(in: &cancellables)

        // EQ
        $bassGain.sink { [weak self] val in
            self?.audioEngine?.equalizer.bassGain = val
        }.store(in: &cancellables)
        $midGain.sink { [weak self] val in
            self?.audioEngine?.equalizer.midGain = val
        }.store(in: &cancellables)
        $trebleGain.sink { [weak self] val in
            self?.audioEngine?.equalizer.trebleGain = val
        }.store(in: &cancellables)

        // Delay
        $delayEnabled.sink { [weak self] val in
            self?.audioEngine?.delay.enabled = val
        }.store(in: &cancellables)
        $delayTime.sink { [weak self] val in
            self?.audioEngine?.delay.time = val
        }.store(in: &cancellables)
        $delayFeedback.sink { [weak self] val in
            self?.audioEngine?.delay.feedback = val
        }.store(in: &cancellables)
        $delayMix.sink { [weak self] val in
            self?.audioEngine?.delay.mix = val
        }.store(in: &cancellables)

        // Reverb
        $reverbEnabled.sink { [weak self] val in
            self?.audioEngine?.reverb.enabled = val
        }.store(in: &cancellables)
        $reverbRoomSize.sink { [weak self] val in
            self?.audioEngine?.reverb.roomSize = val
        }.store(in: &cancellables)
        $reverbDamping.sink { [weak self] val in
            self?.audioEngine?.reverb.damping = val
        }.store(in: &cancellables)
        $reverbMix.sink { [weak self] val in
            self?.audioEngine?.reverb.mix = val
        }.store(in: &cancellables)
    }

    func toggleAudio() {
        if isRunning {
            stopAudio()
            return
        }
        startAudio()
    }

    private func startAudio() {
        do {
            guard let configURL = Bundle.module.url(
                forResource: "model_config", withExtension: "json", subdirectory: "Resources"
            ) else {
                statusMessage = "model_config.json not found"
                return
            }
            guard let weightsURL = Bundle.module.url(
                forResource: "weights", withExtension: "bin", subdirectory: "Resources"
            ) else {
                statusMessage = "weights.bin not found"
                return
            }

            let config = try ModelConfig.load(from: configURL)
            let engine = try InferenceEngine(config: config, weightsURL: weightsURL)
            let audio = AudioEngine(inferenceEngine: engine, gateThreshold: gateThreshold)

            // Apply current effect settings
            audio.compressor.enabled = compressorEnabled
            audio.compressor.threshold = compressorThreshold
            audio.compressor.ratio = compressorRatio
            audio.compressor.attack = compressorAttack
            audio.compressor.release = compressorRelease
            audio.compressor.makeupGain = compressorMakeupGain

            audio.overdrive.enabled = overdriveEnabled
            audio.overdrive.drive = overdriveDrive
            audio.overdrive.tone = overdriveTone
            audio.overdrive.level = overdriveLevel

            audio.equalizer.bassGain = bassGain
            audio.equalizer.midGain = midGain
            audio.equalizer.trebleGain = trebleGain

            audio.delay.enabled = delayEnabled
            audio.delay.time = delayTime
            audio.delay.feedback = delayFeedback
            audio.delay.mix = delayMix

            audio.reverb.enabled = reverbEnabled
            audio.reverb.roomSize = reverbRoomSize
            audio.reverb.damping = reverbDamping
            audio.reverb.mix = reverbMix

            try audio.start()

            self.inferenceEngine = engine
            self.audioEngine = audio
            self.modelParams = config.totalParameters
            self.isRunning = true
            self.statusMessage = ""

            setupBindings()
            startPolling()
        } catch {
            statusMessage = error.localizedDescription
        }
    }

    private func stopAudio() {
        cancellables.removeAll()
        pollTimer?.invalidate()
        pollTimer = nil
        audioEngine?.stop()
        audioEngine = nil
        inferenceEngine = nil
        isRunning = false
        renderTimeMean = 0
        renderTimeP95 = 0
        renderTimePeak = 0
        modelParams = 0
        sampleRate = 0
        sampleRateMismatch = false
        deviceChangeMessage = nil
    }

    private func startPolling() {
        pollTimer = Timer.scheduledTimer(withTimeInterval: 0.5, repeats: true) { [weak self] timer in
            guard let self, let audio = self.audioEngine else {
                timer.invalidate()
                return
            }
            guard self.isRunning else {
                timer.invalidate()
                return
            }

            self.sampleRate = audio.sampleRate
            self.sampleRateMismatch = audio.sampleRateMismatch
            self.deviceChangeMessage = audio.deviceChangeMessage

            let stats = audio.renderStats
            if stats.count > 0 {
                self.renderTimeMean = stats.mean
                self.renderTimeP95 = stats.p95
                self.renderTimePeak = stats.max
            }
        }
    }
}
