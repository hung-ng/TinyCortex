"""
Generate test vectors for Metal inference verification.

Processes audio in 256-sample chunks with manual state management,
matching the Metal implementation exactly. Exports raw float32 binaries.

Usage:
    uv run python training/generate_test_vectors.py
"""

import json
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from models import TinyCortexTCN

CHECKPOINT_DIR = Path(__file__).parent / "checkpoints" / "tcn_c16_b10_k3"
EXPORT_DIR = CHECKPOINT_DIR / "export"
TEST_DATA_DIR = Path(__file__).parent.parent / "Tests" / "TinyCortexEngineTests" / "TestData"
CHUNK_SIZE = 256


def load_model() -> TinyCortexTCN:
    config = json.loads((CHECKPOINT_DIR / "config.json").read_text())
    model = TinyCortexTCN(
        n_channels=config["channels"],
        n_blocks=config["blocks"],
        kernel_size=config["kernel_size"],
    )
    state_dict = torch.load(
        str(CHECKPOINT_DIR / "best_model.pt"),
        map_location="cpu",
        weights_only=True,
    )
    model.load_state_dict(state_dict)
    model.eval()
    return model


def chunked_inference(
    model: TinyCortexTCN, input_signal: torch.Tensor, chunk_size: int = CHUNK_SIZE
) -> torch.Tensor:
    """Process audio in chunks with manual state, matching Metal implementation.

    Uses a ring buffer state per block, identical to the Metal kernel logic.

    Args:
        model: Trained TCN model.
        input_signal: [1, 1, N] tensor.
        chunk_size: Samples per chunk (must match Metal buffer size).

    Returns:
        [1, 1, N] output tensor (may be shorter if N not divisible by chunk_size).
    """
    C = model.n_channels
    n_blocks = model.n_blocks
    n_samples = input_signal.shape[-1]
    n_chunks = n_samples // chunk_size

    # Initialize state ring buffers: one per block, shape [1, C, 2*dilation]
    # head[i] = write position in the ring buffer
    states = []
    heads = []
    for i in range(n_blocks):
        dilation = 2**i
        state_len = 2 * dilation
        states.append(torch.zeros(1, C, state_len))
        heads.append(0)

    output_chunks = []

    with torch.no_grad():
        for chunk_idx in range(n_chunks):
            start = chunk_idx * chunk_size
            end = start + chunk_size
            chunk = input_signal[:, :, start:end]  # [1, 1, chunk_size]

            # Input conv (1 → C)
            x = model.input_conv(chunk)  # [1, C, chunk_size]

            # Process each TCN block
            for i, block in enumerate(model.blocks):
                dilation = 2**i
                state_len = 2 * dilation
                residual = x
                head = heads[i]

                # Reconstruct linear state from ring buffer for conv
                state_linear = torch.cat(
                    [states[i][:, :, head:], states[i][:, :, :head]], dim=2
                ) if head > 0 else states[i].clone()

                # Prepend linearized state to create full causal context
                x_with_state = torch.cat([state_linear, x], dim=2)

                # Update ring buffer: write chunk samples at head position
                write_count = min(chunk_size, state_len)
                # Source: last write_count samples of x
                src = x[:, :, chunk_size - write_count:]
                for w in range(write_count):
                    phys = (head + w) % state_len
                    states[i][:, :, phys] = src[:, :, w]
                heads[i] = (head + write_count) % state_len

                # Apply raw Conv1d (no extra padding — state provides causality)
                conv = block.net[0].conv
                x = F.conv1d(
                    x_with_state, conv.weight, conv.bias, dilation=dilation
                )

                # PReLU
                x = block.net[1](x)

                # 1×1 conv
                x = block.net[2](x)

                # Residual add
                x = residual + x

            # Output conv (C → 1)
            y = model.output_conv(x)  # [1, 1, chunk_size]
            output_chunks.append(y)

    return torch.cat(output_chunks, dim=2)


def generate_deterministic_input(n_samples: int) -> torch.Tensor:
    """Generate a deterministic test signal (mix of sinusoids)."""
    t = torch.linspace(0, 1, n_samples)
    signal = (
        0.5 * torch.sin(2 * torch.pi * 100 * t)
        + 0.3 * torch.sin(2 * torch.pi * 440 * t)
        + 0.2 * torch.sin(2 * torch.pi * 1000 * t)
    )
    return signal.unsqueeze(0).unsqueeze(0)  # [1, 1, N]


def main():
    TEST_DATA_DIR.mkdir(parents=True, exist_ok=True)
    model = load_model()
    print(f"Loaded model: {model.count_parameters()} params")

    # --- Single-buffer test (256 samples) ---
    print("\nGenerating single-buffer test vectors...")
    single_input = generate_deterministic_input(CHUNK_SIZE)
    single_output = chunked_inference(model, single_input, CHUNK_SIZE)

    single_input.squeeze().numpy().astype(np.float32).tofile(
        TEST_DATA_DIR / "test_input_single.bin"
    )
    single_output.squeeze().numpy().astype(np.float32).tofile(
        TEST_DATA_DIR / "test_output_single.bin"
    )
    print(f"  Input shape: {single_input.shape}, range: [{single_input.min():.4f}, {single_input.max():.4f}]")
    print(f"  Output shape: {single_output.shape}, range: [{single_output.min():.4f}, {single_output.max():.4f}]")

    # --- Multi-buffer state test (2048 = 8 × 256 samples) ---
    print("\nGenerating multi-buffer test vectors...")
    multi_n = CHUNK_SIZE * 8  # 2048 samples
    multi_input = generate_deterministic_input(multi_n)
    multi_output = chunked_inference(model, multi_input, CHUNK_SIZE)

    multi_input.squeeze().numpy().astype(np.float32).tofile(
        TEST_DATA_DIR / "test_input_multi.bin"
    )
    multi_output.squeeze().numpy().astype(np.float32).tofile(
        TEST_DATA_DIR / "test_output_multi.bin"
    )
    print(f"  Input shape: {multi_input.shape}, range: [{multi_input.min():.4f}, {multi_input.max():.4f}]")
    print(f"  Output shape: {multi_output.shape}, range: [{multi_output.min():.4f}, {multi_output.max():.4f}]")

    # Verify chunked output matches single-pass PyTorch output
    with torch.no_grad():
        single_pass_output = model(multi_input)
    diff = (multi_output - single_pass_output).abs().max().item()
    print(f"  Max diff (chunked vs single-pass): {diff:.2e}")
    if diff > 1e-5:
        print("  WARNING: Chunked and single-pass outputs differ significantly!")
    else:
        print("  OK: Chunked matches single-pass PyTorch output")

    # --- Impulse response test ---
    print("\nGenerating impulse response test vectors...")
    impulse_n = CHUNK_SIZE * 9  # > receptive field of 2047
    impulse_input = torch.zeros(1, 1, impulse_n)
    impulse_input[0, 0, 0] = 1.0
    impulse_output = chunked_inference(model, impulse_input, CHUNK_SIZE)

    impulse_input.squeeze().numpy().astype(np.float32).tofile(
        TEST_DATA_DIR / "test_input_impulse.bin"
    )
    impulse_output.squeeze().numpy().astype(np.float32).tofile(
        TEST_DATA_DIR / "test_output_impulse.bin"
    )
    nonzero_count = (impulse_output.abs() > 1e-6).sum().item()
    print(f"  Non-zero output samples: {nonzero_count} (receptive field: {model.receptive_field()})")

    # Copy model files for tests
    import shutil
    for fname in ["model_config.json", "weights.bin"]:
        src = EXPORT_DIR / fname
        dst = TEST_DATA_DIR / fname
        if src.exists():
            shutil.copy2(src, dst)
            print(f"  Copied {fname} to TestData/")

    print("\nDone! Test vectors saved to Tests/TinyCortexEngineTests/TestData/")


if __name__ == "__main__":
    main()
