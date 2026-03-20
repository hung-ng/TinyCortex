"""
TinyCortex — Data Preparation Script

Loads clean/amped WAV pairs from data/raw/, validates them,
splits into overlapping segments, and saves train/val/test tensors.

Supports two naming conventions:
  - Single pair:  input_clean.wav / output_amped.wav
  - Multi-clip:   clip1_clean.wav / clip1_amped.wav, clip2_clean.wav / clip2_amped.wav, ...
"""

import argparse
import re
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import torchaudio

EXPECTED_SR = 44100
PROJECT_ROOT = Path(__file__).resolve().parent.parent
RAW_DIR = PROJECT_ROOT / "data" / "raw"
OUT_DIR = PROJECT_ROOT / "data" / "processed"


def discover_pairs(raw_dir: Path) -> list[tuple[Path, Path]]:
    """Find matching clean/amped WAV pairs in the raw directory."""
    pairs = []

    # Check for single-pair convention
    single_clean = raw_dir / "input_clean.wav"
    single_amped = raw_dir / "output_amped.wav"
    if single_clean.exists() and single_amped.exists():
        pairs.append((single_clean, single_amped))
        return pairs

    # Check for multi-clip convention: <prefix>_clean.wav / <prefix>_amped.wav
    clean_files = sorted(raw_dir.glob("*_clean.wav"))
    for clean_path in clean_files:
        prefix = re.sub(r"_clean\.wav$", "", clean_path.name)
        amped_path = raw_dir / f"{prefix}_amped.wav"
        if amped_path.exists():
            pairs.append((clean_path, amped_path))
        else:
            print(f"WARNING: No matching amped file for {clean_path.name}, skipping")

    return pairs


def load_and_validate(path: Path) -> torch.Tensor:
    """Load a WAV file and validate format (mono, 44.1kHz)."""
    waveform, sr = torchaudio.load(str(path))

    if sr != EXPECTED_SR:
        raise ValueError(f"{path.name}: expected {EXPECTED_SR}Hz, got {sr}Hz")

    # Convert to mono if stereo
    if waveform.shape[0] > 1:
        print(f"  {path.name}: converting {waveform.shape[0]}ch to mono")
        waveform = waveform.mean(dim=0, keepdim=True)

    return waveform.squeeze(0)  # shape: (samples,)


def normalize(tensor: torch.Tensor) -> torch.Tensor:
    """Normalize signal to [-1, 1] range."""
    peak = tensor.abs().max()
    if peak > 0:
        return tensor / peak
    return tensor


def create_segments(
    clean: torch.Tensor,
    amped: torch.Tensor,
    segment_len: int,
    hop_len: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Slice paired signals into overlapping segments."""
    n_samples = clean.shape[0]
    segments_clean = []
    segments_amped = []

    start = 0
    while start + segment_len <= n_samples:
        segments_clean.append(clean[start : start + segment_len])
        segments_amped.append(amped[start : start + segment_len])
        start += hop_len

    return torch.stack(segments_clean), torch.stack(segments_amped)


def split_data(
    clean_segs: torch.Tensor,
    amped_segs: torch.Tensor,
    train_frac: float = 0.8,
    val_frac: float = 0.1,
) -> dict[str, tuple[torch.Tensor, torch.Tensor]]:
    """Split segments into train/val/test sets."""
    n = clean_segs.shape[0]
    indices = torch.randperm(n)

    n_train = int(n * train_frac)
    n_val = int(n * val_frac)

    train_idx = indices[:n_train]
    val_idx = indices[n_train : n_train + n_val]
    test_idx = indices[n_train + n_val :]

    return {
        "train": (clean_segs[train_idx], amped_segs[train_idx]),
        "val": (clean_segs[val_idx], amped_segs[val_idx]),
        "test": (clean_segs[test_idx], amped_segs[test_idx]),
    }


def plot_sample(clean: torch.Tensor, amped: torch.Tensor, out_path: Path):
    """Plot a single segment pair for visual sanity check."""
    fig, axes = plt.subplots(2, 1, figsize=(12, 4), sharex=True)
    t = np.arange(clean.shape[0]) / EXPECTED_SR * 1000  # ms

    axes[0].plot(t, clean.numpy(), linewidth=0.5)
    axes[0].set_ylabel("Clean (DI)")
    axes[0].set_ylim(-1, 1)

    axes[1].plot(t, amped.numpy(), linewidth=0.5, color="tab:orange")
    axes[1].set_ylabel("Amped")
    axes[1].set_xlabel("Time (ms)")
    axes[1].set_ylim(-1, 1)

    fig.suptitle("Sample Segment: Clean vs Amped")
    fig.tight_layout()
    fig.savefig(str(out_path), dpi=150)
    plt.close(fig)
    print(f"  Sample plot saved to {out_path}")


def main():
    parser = argparse.ArgumentParser(description="Prepare TinyCortex training data")
    parser.add_argument(
        "--raw-dir", type=Path, default=RAW_DIR, help="Directory with raw WAV pairs"
    )
    parser.add_argument(
        "--out-dir", type=Path, default=OUT_DIR, help="Output directory for .pt files"
    )
    parser.add_argument(
        "--segment-len",
        type=int,
        default=4096,
        help="Samples per segment (default: 4096)",
    )
    parser.add_argument(
        "--hop-len",
        type=int,
        default=2048,
        help="Hop between segments (default: 2048 = 50%% overlap)",
    )
    parser.add_argument(
        "--plot", action="store_true", help="Save a sample plot for sanity checking"
    )
    args = parser.parse_args()

    # Discover pairs
    print(f"Scanning {args.raw_dir} for WAV pairs...")
    pairs = discover_pairs(args.raw_dir)
    if not pairs:
        print("ERROR: No clean/amped WAV pairs found in", args.raw_dir)
        print("Expected either:")
        print("  input_clean.wav + output_amped.wav")
        print("  <clip>_clean.wav + <clip>_amped.wav")
        sys.exit(1)

    print(f"Found {len(pairs)} pair(s)\n")

    # Load and concatenate all pairs
    all_clean = []
    all_amped = []

    for clean_path, amped_path in pairs:
        print(f"Loading {clean_path.name} / {amped_path.name}")
        clean = load_and_validate(clean_path)
        amped = load_and_validate(amped_path)

        if clean.shape[0] != amped.shape[0]:
            raise ValueError(
                f"Length mismatch: {clean_path.name} has {clean.shape[0]} samples, "
                f"{amped_path.name} has {amped.shape[0]} samples"
            )

        print(f"  Samples: {clean.shape[0]:,} ({clean.shape[0] / EXPECTED_SR:.2f}s)")
        all_clean.append(clean)
        all_amped.append(amped)

    clean_cat = torch.cat(all_clean)
    amped_cat = torch.cat(all_amped)

    total_duration = clean_cat.shape[0] / EXPECTED_SR
    print(f"\nTotal: {clean_cat.shape[0]:,} samples ({total_duration:.2f}s)")

    # Normalize
    print("Normalizing to [-1, 1]...")
    clean_cat = normalize(clean_cat)
    amped_cat = normalize(amped_cat)

    # Signal stats
    clean_energy = (clean_cat**2).mean().sqrt().item()
    amped_energy = (amped_cat**2).mean().sqrt().item()
    print(f"  Clean RMS: {clean_energy:.4f}")
    print(f"  Amped RMS: {amped_energy:.4f}")

    # Create segments
    print(
        f"\nSegmenting: length={args.segment_len}, hop={args.hop_len} "
        f"({args.hop_len / args.segment_len * 100:.0f}% overlap)..."
    )
    clean_segs, amped_segs = create_segments(
        clean_cat, amped_cat, args.segment_len, args.hop_len
    )
    print(f"  Created {clean_segs.shape[0]:,} segments")

    # Split
    splits = split_data(clean_segs, amped_segs)
    for name, (c, a) in splits.items():
        print(f"  {name}: {c.shape[0]:,} segments")

    # Save
    args.out_dir.mkdir(parents=True, exist_ok=True)
    for name, (c, a) in splits.items():
        path = args.out_dir / f"{name}.pt"
        torch.save({"clean": c, "amped": a}, str(path))
        print(f"  Saved {path}")

    # Optional plot
    if args.plot and clean_segs.shape[0] > 0:
        plot_sample(
            clean_segs[0], amped_segs[0], args.out_dir / "sample_segment.png"
        )

    print("\nDone!")


if __name__ == "__main__":
    main()
