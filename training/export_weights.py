"""
TinyCortex — Weight Export for Metal Inference

Exports trained PyTorch weights as:
  1. A single contiguous float32 .bin file (loadable as one Metal buffer)
  2. A model_config.json manifest with architecture, layer shapes, and byte offsets

Usage:
    uv run python training/export_weights.py --checkpoint training/checkpoints/tcn_c16_b10_k3
"""

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from models import TinyCortexTCN

PROJECT_ROOT = Path(__file__).resolve().parent.parent


def load_model(checkpoint_dir: Path) -> tuple[TinyCortexTCN, dict]:
    """Load model and config from checkpoint directory."""
    config = json.loads((checkpoint_dir / "config.json").read_text())
    model = TinyCortexTCN(
        n_channels=config["channels"],
        n_blocks=config["blocks"],
        kernel_size=config["kernel_size"],
    )
    state_dict = torch.load(
        str(checkpoint_dir / "best_model.pt"),
        map_location="cpu",
        weights_only=True,
    )
    model.load_state_dict(state_dict)
    model.eval()
    return model, config


def export_weights(model: TinyCortexTCN, config: dict, output_dir: Path):
    """Export weights as flat binary + JSON manifest."""
    output_dir.mkdir(parents=True, exist_ok=True)

    layers = []
    all_data = []
    byte_offset = 0

    for name, param in model.named_parameters():
        flat = param.detach().cpu().float().numpy().flatten()
        shape = list(param.shape)
        n_bytes = flat.nbytes

        layers.append({
            "name": name,
            "shape": shape,
            "dtype": "float32",
            "byte_offset": byte_offset,
            "byte_length": n_bytes,
        })
        all_data.append(flat)
        byte_offset += n_bytes

    # Write binary
    bin_path = output_dir / "weights.bin"
    combined = np.concatenate(all_data)
    combined.tofile(str(bin_path))

    # Write manifest
    manifest = {
        "architecture": "TinyCortexTCN",
        "channels": config["channels"],
        "blocks": config["blocks"],
        "kernel_size": config["kernel_size"],
        "receptive_field": model.receptive_field(),
        "total_parameters": model.count_parameters(),
        "total_bytes": byte_offset,
        "layers": layers,
    }
    manifest_path = output_dir / "model_config.json"
    manifest_path.write_text(json.dumps(manifest, indent=2))

    print(f"Exported {model.count_parameters():,} parameters ({byte_offset:,} bytes)")
    print(f"  Binary:   {bin_path}")
    print(f"  Manifest: {manifest_path}")

    return bin_path, manifest_path


def verify_export(model: TinyCortexTCN, bin_path: Path, manifest_path: Path):
    """Verify by rebuilding model from exported weights and comparing outputs."""
    manifest = json.loads(manifest_path.read_text())
    raw = np.fromfile(str(bin_path), dtype=np.float32)

    # Rebuild state dict from binary
    rebuilt_state = {}
    for layer in manifest["layers"]:
        start = layer["byte_offset"] // 4  # float32 = 4 bytes
        end = start + (layer["byte_length"] // 4)
        tensor = torch.from_numpy(raw[start:end].copy()).reshape(layer["shape"])
        rebuilt_state[layer["name"]] = tensor

    # Create fresh model and load rebuilt weights
    rebuilt_model = TinyCortexTCN(
        n_channels=manifest["channels"],
        n_blocks=manifest["blocks"],
        kernel_size=manifest["kernel_size"],
    )
    rebuilt_model.load_state_dict(rebuilt_state)
    rebuilt_model.eval()

    # Compare outputs
    test_input = torch.randn(1, 1, 4096)
    with torch.no_grad():
        out_original = model(test_input)
        out_rebuilt = rebuilt_model(test_input)

    max_diff = (out_original - out_rebuilt).abs().max().item()
    print(f"\nVerification: max output difference = {max_diff:.2e}")
    if max_diff < 1e-5:
        print("  PASS — export is faithful")
    else:
        print("  WARN — difference exceeds threshold, check export")


def main():
    parser = argparse.ArgumentParser(description="Export TinyCortex weights for Metal")
    parser.add_argument(
        "--checkpoint",
        type=Path,
        required=True,
        help="Path to checkpoint directory",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Output directory (default: <checkpoint>/export/)",
    )
    args = parser.parse_args()

    checkpoint_dir = args.checkpoint
    if not checkpoint_dir.is_absolute():
        checkpoint_dir = Path.cwd() / checkpoint_dir

    output_dir = args.output or (checkpoint_dir / "export")

    model, config = load_model(checkpoint_dir)
    print(f"Loaded model: {model.count_parameters():,} params")

    bin_path, manifest_path = export_weights(model, config, output_dir)
    verify_export(model, bin_path, manifest_path)


if __name__ == "__main__":
    main()
