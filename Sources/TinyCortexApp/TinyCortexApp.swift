import SwiftUI
import TinyCortexEngine

@main
struct TinyCortexApp: App {
    var body: some Scene {
        WindowGroup {
            ContentView()
        }
    }
}

struct ContentView: View {
    @State private var isRunning = false
    @State private var statusText = "Ready"
    @State private var audioEngine: AudioEngine?

    var body: some View {
        VStack(spacing: 20) {
            Text("TinyCortex")
                .font(.largeTitle.bold())

            Text(statusText)
                .foregroundColor(isRunning ? .green : .secondary)

            Button(isRunning ? "Stop" : "Start") {
                toggleAudio()
            }
            .buttonStyle(.borderedProminent)
            .controlSize(.large)
        }
        .padding(40)
        .frame(minWidth: 300, minHeight: 200)
    }

    private func toggleAudio() {
        if isRunning {
            audioEngine?.stop()
            audioEngine = nil
            isRunning = false
            statusText = "Stopped"
            return
        }

        do {
            guard let configURL = Bundle.module.url(
                forResource: "model_config", withExtension: "json", subdirectory: "Resources"
            ) else {
                statusText = "Error: model_config.json not found in bundle"
                return
            }
            guard let weightsURL = Bundle.module.url(
                forResource: "weights", withExtension: "bin", subdirectory: "Resources"
            ) else {
                statusText = "Error: weights.bin not found in bundle"
                return
            }

            let config = try ModelConfig.load(from: configURL)
            let engine = try InferenceEngine(config: config, weightsURL: weightsURL)
            let audio = AudioEngine(inferenceEngine: engine)
            try audio.start()

            audioEngine = audio
            isRunning = true
            statusText = "Processing (\(config.totalParameters) params)\nSample rate: \(audio.sampleRate) Hz | Render frames: \(audio.lastRenderFrames)"

            // Update debug info periodically
            Timer.scheduledTimer(withTimeInterval: 1, repeats: true) { timer in
                guard let audio = self.audioEngine, isRunning else {
                    timer.invalidate()
                    return
                }
                statusText = "Processing (\(config.totalParameters) params)\nSample rate: \(audio.sampleRate) Hz | Render frames: \(audio.lastRenderFrames)"
            }
        } catch {
            statusText = "Error: \(error.localizedDescription)"
        }
    }
}
