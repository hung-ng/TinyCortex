import SwiftUI
import Combine
import TinyCortexEngine

final class AudioViewModel: ObservableObject {
    // State
    @Published var isRunning = false
    @Published var statusMessage = ""

    // Gate
    @Published var gateThreshold: Float = 0.015

    // EQ (dB, -12 to +12)
    @Published var bassGain: Float = 0
    @Published var midGain: Float = 0
    @Published var trebleGain: Float = 0

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

        $bassGain.sink { [weak self] val in
            self?.audioEngine?.equalizer.bassGain = val
        }.store(in: &cancellables)

        $midGain.sink { [weak self] val in
            self?.audioEngine?.equalizer.midGain = val
        }.store(in: &cancellables)

        $trebleGain.sink { [weak self] val in
            self?.audioEngine?.equalizer.trebleGain = val
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

            // Apply current EQ settings
            audio.equalizer.bassGain = bassGain
            audio.equalizer.midGain = midGain
            audio.equalizer.trebleGain = trebleGain

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
