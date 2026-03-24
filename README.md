# TinyCortex

Real-time neural guitar amp modeler running on Apple Silicon. A causal dilated TCN with 10,769 parameters, trained on amplifier recordings, runs inference through hand-written Metal compute shaders. It processes 256-sample buffers within a ~5.8 ms latency budget at 44.1 kHz.

## Features

- **Neural amp modeling** with a temporal convolutional network (TCN), running entirely on the GPU via Metal compute shaders
- **Full effects chain**: noise gate, compressor, overdrive, 3-band EQ, delay, and reverb
- **Low latency**: 256-sample buffer size (~5.8 ms at 44.1 kHz), 42 GPU dispatches per buffer
- **Zero runtime allocations**: all Metal buffers are pre-allocated; the audio callback path is allocation-free
- **SwiftUI controls**: live parameter sliders for every effect, with real-time render timing stats

## Requirements

- macOS 13 or later
- Apple Silicon (M1 or newer)
- Xcode 15+ (for building from Xcode) or Swift 5.9+ toolchain
- Python 3.13+ and [uv](https://github.com/astral-sh/uv) (for training)
- An audio interface with a guitar input (for live use)

## Quick Start

### 1. Clone the repository

```bash
git clone https://github.com/your-username/TinyCortex.git
cd TinyCortex
```

### 2. Build and run the app

The repository includes pre-exported model weights in `Sources/TinyCortexApp/Resources/`.

**From the command line:**

```bash
swift build
swift run TinyCortexApp
```

**From Xcode:**

Open `Package.swift` in Xcode, select the `TinyCortexApp` scheme, and press Cmd+R.

## Training Your Own Model

If you want to train the amp model on your own recordings, follow these steps.

### 1. Prepare your audio data

Place paired WAV files in `data/raw/`:

- Files must be 44.1 kHz, mono, 32-bit float
- Each pair consists of a clean (DI) recording and the corresponding amped recording
- Name them `input_clean.wav` / `output_amped.wav`, or use the pattern `<clip>_clean.wav` / `<clip>_amped.wav` for multiple clips

### 2. Install Python dependencies

TinyCortex uses [uv](https://github.com/astral-sh/uv) for Python dependency management. Install uv if you have not already:

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

Then sync the project dependencies:

```bash
uv sync
```

### 3. Segment and split the data

This script loads the WAV pairs, normalizes them, segments them into overlapping windows, and splits them into train/val/test sets:

```bash
uv run python training/prepare_data.py
```

Output goes to `data/processed/` as `.pt` tensor files.

### 4. Train the model

```bash
uv run python training/train.py
```

Training uses ESR + DC offset loss with the Adam optimizer, learning rate scheduling, and early stopping. Checkpoints are saved to `training/checkpoints/`. The default configuration trains a TCN with 16 hidden channels, 10 blocks, and kernel size 3.

### 5. Evaluate the model

```bash
uv run python training/evaluate.py --checkpoint training/checkpoints/tcn_c16_b10_k3
```

This reports test metrics (ESR, SNR, MSE), generates spectrograms, and writes predicted audio to `training/evaluation/`.

### 6. Export weights for Metal inference

```bash
uv run python training/export_weights.py --checkpoint training/checkpoints/tcn_c16_b10_k3
```

This produces a `weights.bin` (float32 binary) and `model_config.json` (layer shapes and byte offsets). Copy both files to `Sources/TinyCortexApp/Resources/` to use them in the app.

## Running Tests

Test vectors are generated from PyTorch and are not checked into git. Before running tests for the first time (or after model changes), generate them:

```bash
uv run python training/generate_test_vectors.py
```

Then run the test suite:

```bash
swift test
```

Or from Xcode, press Cmd+U.

The test suite includes:
- 6 inference correctness tests (single-buffer, multi-buffer state, impulse response, silence, determinism, config parsing)
- 2 benchmark tests (latency, memory)
- 16 DSP effects unit tests (compressor, overdrive, delay, reverb)

## Signal Chain

```
Guitar Input
  -> Noise Gate (RMS sensing)
  -> Compressor (optional, pre-amp)
  -> Overdrive (optional, pre-amp)
  -> Neural Amp Model (Metal GPU inference)
  -> Gate Gain (applies noise gate attenuation)
  -> 3-Band EQ (low shelf, peak, high shelf)
  -> Delay (optional, post-amp)
  -> Reverb (optional, post-amp)
  -> Output Clamp [-1, 1]
  -> Audio Output
```

## Project Structure

```
Sources/TinyCortexEngine/          Core inference library (Swift + Metal)
  ├── ModelConfig.swift            JSON manifest parsing
  ├── WeightLoader.swift           Load weights.bin into Metal buffer
  ├── InferenceEngine.swift        Metal pipeline, buffer management, process()/processAsync()
  ├── AudioEngine.swift            AUHAL real-time audio I/O, noise gate, device handling
  ├── Equalizer.swift              3-band biquad EQ
  ├── Compressor.swift             RMS envelope compressor
  ├── Overdrive.swift              Soft-clip overdrive with tone control
  ├── Delay.swift                  Circular buffer delay with feedback
  ├── Reverb.swift                 Freeverb (8 comb + 4 allpass filters)
  ├── Benchmark.swift              Render timing and inference benchmarking
  └── Shaders/TCNKernels.metal     5 Metal compute kernels

Sources/TinyCortexApp/             SwiftUI app
  ├── TinyCortexApp.swift          Main app and ContentView with effect sliders
  ├── AudioViewModel.swift         ObservableObject bridging AudioEngine to SwiftUI
  └── Resources/                   Bundled model files

Tests/TinyCortexEngineTests/       Test suite
  ├── InferenceTests.swift         Inference correctness tests
  ├── BenchmarkTests.swift         Latency and memory benchmarks
  ├── EffectsTests.swift           DSP effects unit tests
  └── TestData/                    Generated reference binaries (not in git)

training/                          Python training pipeline (PyTorch)
data/                              Audio data (raw WAVs and processed tensors)
```

## How It Works

The core of TinyCortex is a **causal dilated temporal convolutional network** (TCN). It has 10 blocks with exponentially increasing dilation factors (1, 2, 4, ..., 512), giving a receptive field of 2,047 samples (~46 ms). The model takes a mono audio buffer as input and outputs the "amped" version of that buffer.

Inference runs on the GPU through 5 Metal compute kernels:

1. `conv1x1` for input/output projections
2. `causal_dilated_conv3` for dilated causal convolutions
3. `prelu` for parametric ReLU activation
4. `conv1x1_residual` for fused 1x1 convolution + residual connection
5. `update_state` for writing activations to ring buffer state

All model weights are stored in a single contiguous 43 KB binary file, loaded as one Metal buffer. Ping-pong activation buffers eliminate extra allocations during inference.

## License

TBD
