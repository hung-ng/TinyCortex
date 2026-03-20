"""
TinyCortex — Evaluation Script

Computes test metrics, generates full-audio inference with crossfade,
plots spectrograms and loss curves.

Usage:
    uv run python training/evaluate.py --checkpoint training/checkpoints/tcn_c16_b10_k3
"""

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import torchaudio

from models import TinyCortexTCN

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data" / "processed"
RAW_DIR = PROJECT_ROOT / "data" / "raw"
EVAL_DIR = Path(__file__).resolve().parent / "evaluation"
SAMPLE_RATE = 44100


def load_model(checkpoint_dir: Path, device: torch.device) -> TinyCortexTCN:
    """Load model from checkpoint directory."""
    config = json.loads((checkpoint_dir / "config.json").read_text())
    model = TinyCortexTCN(
        n_channels=config["channels"],
        n_blocks=config["blocks"],
        kernel_size=config["kernel_size"],
    ).to(device)
    state_dict = torch.load(
        str(checkpoint_dir / "best_model.pt"),
        map_location=device,
        weights_only=True,
    )
    model.load_state_dict(state_dict)
    model.eval()
    return model


def compute_esr(predicted: torch.Tensor, target: torch.Tensor, eps: float = 1e-8) -> float:
    error = predicted - target
    return ((error**2).sum() / ((target**2).sum() + eps)).item()


def compute_metrics(model: TinyCortexTCN, device: torch.device) -> dict:
    """Compute ESR, MSE, SNR on the test set."""
    data = torch.load(str(DATA_DIR / "test.pt"), weights_only=True)
    clean = data["clean"].unsqueeze(1).float().to(device)
    amped = data["amped"].unsqueeze(1).float().to(device)

    with torch.no_grad():
        predicted = model(clean)

    esr = compute_esr(predicted, amped)
    mse = ((predicted - amped) ** 2).mean().item()
    # SNR: signal power / error power
    signal_power = (amped**2).mean().item()
    noise_power = ((predicted - amped) ** 2).mean().item()
    snr = 10 * np.log10(signal_power / (noise_power + 1e-10))

    return {"esr": esr, "mse": mse, "snr_db": snr}


def full_audio_inference(
    model: TinyCortexTCN,
    clean_wav: torch.Tensor,
    device: torch.device,
    segment_len: int = 4096,
    hop_len: int = 2048,
) -> torch.Tensor:
    """Process full audio with overlapping windows and crossfade."""
    n_samples = clean_wav.shape[0]
    output = torch.zeros(n_samples)
    weight = torch.zeros(n_samples)

    # Crossfade window: raised cosine
    overlap = segment_len - hop_len
    window = torch.ones(segment_len)
    fade = torch.linspace(0, 1, overlap)
    window[:overlap] = fade
    window[-overlap:] = 1 - fade

    model.eval()
    with torch.no_grad():
        start = 0
        while start + segment_len <= n_samples:
            chunk = clean_wav[start : start + segment_len].unsqueeze(0).unsqueeze(0).to(device)
            pred = model(chunk).squeeze().cpu()
            output[start : start + segment_len] += pred * window
            weight[start : start + segment_len] += window
            start += hop_len

    # Normalize by overlap weight
    mask = weight > 0
    output[mask] /= weight[mask]
    return output


def plot_spectrograms(
    clean: torch.Tensor,
    predicted: torch.Tensor,
    target: torch.Tensor,
    out_path: Path,
):
    """Plot spectrograms of clean, predicted, and target signals."""
    fig, axes = plt.subplots(3, 1, figsize=(14, 8))
    for ax, signal, title in zip(
        axes,
        [clean, predicted, target],
        ["Clean (DI)", "Predicted", "Target (Amped)"],
    ):
        spec = torch.stft(
            signal, n_fft=2048, hop_length=512, return_complex=True, window=torch.hann_window(2048)
        )
        magnitude_db = 20 * torch.log10(spec.abs() + 1e-8)
        ax.imshow(
            magnitude_db.numpy(),
            aspect="auto",
            origin="lower",
            extent=[0, len(signal) / SAMPLE_RATE, 0, SAMPLE_RATE / 2],
        )
        ax.set_title(title)
        ax.set_ylabel("Freq (Hz)")
        ax.set_ylim(0, 8000)
    axes[-1].set_xlabel("Time (s)")
    fig.tight_layout()
    fig.savefig(str(out_path), dpi=150)
    plt.close(fig)


def plot_loss_curves(log_path: Path, out_path: Path):
    """Plot training/validation loss curves."""
    log = json.loads(log_path.read_text())
    epochs = [e["epoch"] for e in log]
    train_loss = [e["train_loss"] for e in log]
    val_loss = [e["val_loss"] for e in log]

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(epochs, train_loss, label="Train", alpha=0.8)
    ax.plot(epochs, val_loss, label="Validation", alpha=0.8)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss (ESR + DC)")
    ax.set_title("Training Loss Curves")
    ax.legend()
    ax.set_yscale("log")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(str(out_path), dpi=150)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description="Evaluate TinyCortex amp model")
    parser.add_argument(
        "--checkpoint",
        type=Path,
        required=True,
        help="Path to checkpoint directory",
    )
    args = parser.parse_args()

    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    print(f"Device: {device}")

    checkpoint_dir = args.checkpoint
    if not checkpoint_dir.is_absolute():
        checkpoint_dir = Path.cwd() / checkpoint_dir

    model = load_model(checkpoint_dir, device)
    print(f"Loaded model: {model.count_parameters():,} params")

    EVAL_DIR.mkdir(parents=True, exist_ok=True)

    # Test set metrics
    print("\nTest set metrics:")
    metrics = compute_metrics(model, device)
    for k, v in metrics.items():
        print(f"  {k}: {v:.6f}")
    (EVAL_DIR / "metrics.json").write_text(json.dumps(metrics, indent=2))

    # Full-audio inference
    clean_path = RAW_DIR / "input_clean.wav"
    amped_path = RAW_DIR / "output_amped.wav"

    if clean_path.exists() and amped_path.exists():
        print("\nRunning full-audio inference...")
        clean_wav, sr = torchaudio.load(str(clean_path))
        amped_wav, _ = torchaudio.load(str(amped_path))
        clean_wav = clean_wav.squeeze(0)
        amped_wav = amped_wav.squeeze(0)

        # Normalize same as training
        clean_wav = clean_wav / clean_wav.abs().max()
        amped_wav = amped_wav / amped_wav.abs().max()

        predicted_wav = full_audio_inference(model, clean_wav, device)

        # Save predicted audio
        pred_path = EVAL_DIR / "predicted.wav"
        torchaudio.save(str(pred_path), predicted_wav.unsqueeze(0), sr)
        print(f"  Saved {pred_path}")

        # Spectrograms (use first 5 seconds)
        n = min(sr * 5, len(clean_wav))
        plot_spectrograms(
            clean_wav[:n], predicted_wav[:n], amped_wav[:n],
            EVAL_DIR / "spectrograms.png",
        )
        print(f"  Saved spectrograms.png")

    # Loss curves
    log_path = checkpoint_dir / "training_log.json"
    if log_path.exists():
        plot_loss_curves(log_path, EVAL_DIR / "loss_curves.png")
        print(f"  Saved loss_curves.png")

    print("\nEvaluation complete!")


if __name__ == "__main__":
    main()
