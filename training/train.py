"""
TinyCortex — Training Script

Usage:
    uv run python training/train.py
    uv run python training/train.py --channels 24 --epochs 1000
"""

import argparse
import json
import time
from pathlib import Path

import torch
from torch.utils.data import DataLoader, Dataset

from models import TinyCortexTCN

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data" / "processed"
CHECKPOINT_DIR = Path(__file__).resolve().parent / "checkpoints"


class AmpDataset(Dataset):
    """Loads segmented clean/amped pairs from a .pt file."""

    def __init__(self, pt_path: Path):
        data = torch.load(str(pt_path), weights_only=True)
        # Shape: (N, 4096) → (N, 1, 4096)
        self.clean = data["clean"].unsqueeze(1).float()
        self.amped = data["amped"].unsqueeze(1).float()

    def __len__(self):
        return self.clean.shape[0]

    def __getitem__(self, idx):
        return self.clean[idx], self.amped[idx]


def esr_loss(predicted: torch.Tensor, target: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """Error-to-Signal Ratio: sum((pred-target)²) / (sum(target²) + eps)"""
    error = predicted - target
    return (error**2).sum(dim=-1) / ((target**2).sum(dim=-1) + eps)


def dc_loss(predicted: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """DC offset loss: penalizes difference in mean level."""
    return (predicted.mean(dim=-1) - target.mean(dim=-1)) ** 2


def combined_loss(predicted: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """ESR + DC offset loss, averaged over batch."""
    return (esr_loss(predicted, target) + dc_loss(predicted, target)).mean()


def get_device() -> torch.device:
    if torch.backends.mps.is_available():
        print("Using MPS (Apple Silicon GPU)")
        return torch.device("mps")
    print("Using CPU")
    return torch.device("cpu")


def train(args):
    torch.manual_seed(args.seed)
    device = get_device()

    # Load data
    train_ds = AmpDataset(DATA_DIR / "train.pt")
    val_ds = AmpDataset(DATA_DIR / "val.pt")
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False)
    print(f"Train: {len(train_ds)} segments, Val: {len(val_ds)} segments")

    # Model
    model = TinyCortexTCN(
        n_channels=args.channels,
        n_blocks=args.blocks,
        kernel_size=args.kernel_size,
    ).to(device)
    print(f"Model: {model.count_parameters():,} parameters, "
          f"receptive field: {model.receptive_field()} samples "
          f"({model.receptive_field() / 44100 * 1000:.1f}ms)")

    # Optimizer & scheduler
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, factor=0.5, patience=20,
    )

    # Checkpoint setup
    run_name = f"tcn_c{args.channels}_b{args.blocks}_k{args.kernel_size}"
    run_dir = CHECKPOINT_DIR / run_name
    run_dir.mkdir(parents=True, exist_ok=True)

    # Save config
    config = {
        "channels": args.channels,
        "blocks": args.blocks,
        "kernel_size": args.kernel_size,
        "batch_size": args.batch_size,
        "lr": args.lr,
        "seed": args.seed,
    }
    (run_dir / "config.json").write_text(json.dumps(config, indent=2))

    # Training loop
    best_val_loss = float("inf")
    patience_counter = 0
    log = []

    print(f"\nTraining for up to {args.epochs} epochs (patience={args.patience})...")
    print("-" * 60)

    for epoch in range(1, args.epochs + 1):
        t0 = time.time()

        # Train
        model.train()
        train_loss_sum = 0.0
        train_batches = 0
        for clean, amped in train_loader:
            clean, amped = clean.to(device), amped.to(device)
            predicted = model(clean)
            loss = combined_loss(predicted, amped)

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            train_loss_sum += loss.item()
            train_batches += 1

        train_loss = train_loss_sum / max(train_batches, 1)

        # Validate
        model.eval()
        val_loss_sum = 0.0
        val_batches = 0
        with torch.no_grad():
            for clean, amped in val_loader:
                clean, amped = clean.to(device), amped.to(device)
                predicted = model(clean)
                loss = combined_loss(predicted, amped)
                val_loss_sum += loss.item()
                val_batches += 1

        val_loss = val_loss_sum / max(val_batches, 1)
        scheduler.step(val_loss)
        elapsed = time.time() - t0

        lr = optimizer.param_groups[0]["lr"]
        log.append({
            "epoch": epoch,
            "train_loss": train_loss,
            "val_loss": val_loss,
            "lr": lr,
            "time": elapsed,
        })

        # Checkpointing
        improved = ""
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            torch.save(model.state_dict(), str(run_dir / "best_model.pt"))
            improved = " *"
        else:
            patience_counter += 1

        if epoch % 10 == 0 or epoch == 1 or improved:
            print(f"Epoch {epoch:4d} | train {train_loss:.6f} | val {val_loss:.6f} | "
                  f"lr {lr:.1e} | {elapsed:.1f}s{improved}")

        # Early stopping
        if patience_counter >= args.patience:
            print(f"\nEarly stopping at epoch {epoch} (patience={args.patience})")
            break

    # Save training log
    (run_dir / "training_log.json").write_text(json.dumps(log, indent=2))

    print("-" * 60)
    print(f"Best val loss: {best_val_loss:.6f}")
    print(f"Checkpoint saved to: {run_dir}")


def main():
    parser = argparse.ArgumentParser(description="Train TinyCortex amp model")
    parser.add_argument("--channels", type=int, default=16, help="TCN hidden channels")
    parser.add_argument("--blocks", type=int, default=10, help="Number of TCN blocks")
    parser.add_argument("--kernel-size", type=int, default=3, help="Convolution kernel size")
    parser.add_argument("--batch-size", type=int, default=32, help="Batch size")
    parser.add_argument("--lr", type=float, default=3e-3, help="Learning rate")
    parser.add_argument("--epochs", type=int, default=500, help="Max epochs")
    parser.add_argument("--patience", type=int, default=50, help="Early stopping patience")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    args = parser.parse_args()
    train(args)


if __name__ == "__main__":
    main()
