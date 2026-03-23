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
    @StateObject private var viewModel = AudioViewModel()

    var body: some View {
        ScrollView {
            VStack(alignment: .leading, spacing: 16) {
                // Header
                HStack {
                    Text("TinyCortex")
                        .font(.title.bold())
                    Spacer()
                    Button(viewModel.isRunning ? "Stop" : "Start") {
                        viewModel.toggleAudio()
                    }
                    .buttonStyle(.borderedProminent)
                    .tint(viewModel.isRunning ? .red : .accentColor)
                    Circle()
                        .fill(viewModel.isRunning ? .green : .gray)
                        .frame(width: 10, height: 10)
                }

                if !viewModel.statusMessage.isEmpty {
                    Text(viewModel.statusMessage)
                        .foregroundColor(.red)
                        .font(.caption)
                }

                Divider()

                // Input (Gate)
                Section {
                    sliderRow("Gate", value: $viewModel.gateThreshold, range: 0...0.1, format: "%.3f")
                } header: {
                    Text("Input").font(.headline)
                }

                // Compressor
                Section {
                    sliderRow("Thresh", value: $viewModel.compressorThreshold, range: -60...0, format: "%+.0f dB")
                    sliderRow("Ratio", value: $viewModel.compressorRatio, range: 1...20, format: "%.1f:1")
                    sliderRow("Attack", value: $viewModel.compressorAttack, range: 1...100, format: "%.0f ms")
                    sliderRow("Release", value: $viewModel.compressorRelease, range: 10...500, format: "%.0f ms")
                    sliderRow("Makeup", value: $viewModel.compressorMakeupGain, range: 0...24, format: "%+.1f dB")
                } header: {
                    effectHeader("Compressor", enabled: $viewModel.compressorEnabled)
                }

                // Overdrive
                Section {
                    sliderRow("Drive", value: $viewModel.overdriveDrive, range: 0...100, format: "%.0f%%")
                    sliderRow("Tone", value: $viewModel.overdriveTone, range: 0...100, format: "%.0f%%")
                    sliderRow("Level", value: $viewModel.overdriveLevel, range: 0...100, format: "%.0f%%")
                } header: {
                    effectHeader("Overdrive", enabled: $viewModel.overdriveEnabled)
                }

                // EQ
                Section {
                    sliderRow("Bass", value: $viewModel.bassGain, range: -12...12, format: "%+.1f dB")
                    sliderRow("Mid", value: $viewModel.midGain, range: -12...12, format: "%+.1f dB")
                    sliderRow("Treble", value: $viewModel.trebleGain, range: -12...12, format: "%+.1f dB")
                } header: {
                    HStack {
                        Text("EQ").font(.headline)
                        Spacer()
                        Button("Reset") {
                            viewModel.bassGain = 0
                            viewModel.midGain = 0
                            viewModel.trebleGain = 0
                        }
                        .font(.caption)
                        .buttonStyle(.plain)
                        .foregroundColor(.accentColor)
                    }
                }

                // Delay
                Section {
                    sliderRow("Time", value: $viewModel.delayTime, range: 50...2000, format: "%.0f ms")
                    sliderRow("Feedbk", value: $viewModel.delayFeedback, range: 0...95, format: "%.0f%%")
                    sliderRow("Mix", value: $viewModel.delayMix, range: 0...100, format: "%.0f%%")
                } header: {
                    effectHeader("Delay", enabled: $viewModel.delayEnabled)
                }

                // Reverb
                Section {
                    sliderRow("Room", value: $viewModel.reverbRoomSize, range: 0...100, format: "%.0f%%")
                    sliderRow("Damp", value: $viewModel.reverbDamping, range: 0...100, format: "%.0f%%")
                    sliderRow("Mix", value: $viewModel.reverbMix, range: 0...100, format: "%.0f%%")
                } header: {
                    effectHeader("Reverb", enabled: $viewModel.reverbEnabled)
                }

                Divider()

                // Performance
                Section {
                    if viewModel.isRunning {
                        infoRow("Render", String(format: "%.2f ms / 5.8 ms", viewModel.renderTimeMean))
                        infoRow("P95", String(format: "%.2f ms", viewModel.renderTimeP95))
                        infoRow("Peak", String(format: "%.2f ms", viewModel.renderTimePeak))
                    }
                    if viewModel.modelParams > 0 {
                        infoRow("Model", "\(viewModel.modelParams) params")
                    }
                    if viewModel.sampleRate > 0 {
                        infoRow("Buffer", String(format: "256 @ %.0f Hz", viewModel.sampleRate))
                    }
                    if viewModel.sampleRateMismatch {
                        Label("Model trained at 44100 Hz", systemImage: "exclamationmark.triangle")
                            .foregroundColor(.orange)
                            .font(.caption)
                    }
                    if let msg = viewModel.deviceChangeMessage {
                        Label(msg, systemImage: "exclamationmark.triangle")
                            .foregroundColor(.orange)
                            .font(.caption)
                    }
                } header: {
                    Text("Performance").font(.headline)
                }
            }
            .padding(24)
        }
        .frame(minWidth: 360, minHeight: 520)
    }

    private func effectHeader(_ title: String, enabled: Binding<Bool>) -> some View {
        HStack {
            Text(title).font(.headline)
            Spacer()
            Toggle("", isOn: enabled)
                .toggleStyle(.switch)
                .labelsHidden()
        }
    }

    private func sliderRow(_ label: String, value: Binding<Float>, range: ClosedRange<Float>, format: String) -> some View {
        HStack(spacing: 8) {
            Text(label)
                .frame(width: 50, alignment: .leading)
                .font(.body.monospacedDigit())
            Slider(value: value, in: range)
            Text(String(format: format, value.wrappedValue))
                .frame(width: 65, alignment: .trailing)
                .font(.caption.monospacedDigit())
        }
    }

    private func infoRow(_ label: String, _ value: String) -> some View {
        HStack {
            Text(label)
                .foregroundColor(.secondary)
                .frame(width: 50, alignment: .leading)
            Text(value)
                .font(.body.monospacedDigit())
        }
        .font(.callout)
    }
}
