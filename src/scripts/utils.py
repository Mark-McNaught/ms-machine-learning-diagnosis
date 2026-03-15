import os
from pathlib import Path

import random
import numpy as np
import torch

import matplotlib.pyplot as plt


def set_seed(seed=42):
    """
    Set random seed for reproducibility across Python, NumPy, and PyTorch.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    
    # PyTorch GPU (if available)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)  # For multi-GPU
    
    # Make CuDNN deterministic
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    
    print(f"Random seed set to {seed} for Python, NumPy, and PyTorch")


def save_weights(model, save_path):
    """Save model state dictionary."""
    torch.save(model.state_dict(), save_path)
    print(f"save_weights()>>> Model weights saved to {save_path}")


def load_weights(model, save_path, device=torch.device("cuda" if torch.cuda.is_available() else "cpu")):
    """Load model state dictionary."""
    weights = torch.load(save_path)
    model.load_state_dict(weights)
    model.eval()
    model.to(device)
    print("load_weights()>>> Model loaded successfully and set to evaluation mode.")
    return model


def save_fig(fig, save_dir, stem, formats=("png", "svg"), dpi=150):
    """
    Save a matplotlib figure as PNG and/or SVG.

    Saves to `save_dir / stem.<ext>` for each format requested.
    SVGs embed fonts as paths so files are self-contained and render
    correctly in LaTeX, Inkscape, and browser viewers.

    Args:
        fig:       matplotlib Figure object to save.
        save_dir:  Directory to save into (str or Path). Created if missing.
        stem:      Filename without extension, e.g. "srq1_test_results".
        formats:   Tuple of format strings. Default ("png", "svg").
                   Any format accepted by matplotlib's savefig is valid.
        dpi:       Resolution for raster formats (PNG). Ignored for SVG.

    Returns:
        List of Path objects for the saved files.

    Example:
        fig, ax = plt.subplots()
        ax.plot([1, 2, 3])
        utils.save_fig(fig, PLOTS_DIR, "my_chart")
    """
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    saved = []
    for fmt in formats:
        out_path = save_dir / f"{stem}.{fmt}"
        fig.savefig(
            out_path,
            dpi=dpi,
            bbox_inches="tight",
            format=fmt,
        )
        saved.append(out_path)
        print(f"save_fig()>>> Saved → {out_path}")

    return saved


def plot(losses, accuracies, config_name, val_losses=None, val_accuracies=None,
         save_dir=None, model_name="model", show=True):
    """
    Plot training (and optionally validation) loss and accuracy over epochs.
    Saves as PNG and SVG when save_dir is provided.
    """
    epochs = range(1, len(losses) + 1)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle(f"{model_name} — {config_name}", fontsize=14, fontweight="bold")

    # ── Loss ──────────────────────────────────────────────────────────────────
    axes[0].plot(epochs, losses, "b-", marker="o", linewidth=2, label="Train Loss")
    if val_losses:
        axes[0].plot(epochs, val_losses, "r-", marker="s", linewidth=2, label="Val Loss")
    axes[0].set_title("Loss", fontweight="bold")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss")
    axes[0].legend()
    axes[0].grid(True, linestyle="--", alpha=0.5)

    # ── Accuracy ──────────────────────────────────────────────────────────────
    axes[1].plot(epochs, accuracies, "b-", marker="o", linewidth=2, label="Train Acc")
    if val_accuracies:
        axes[1].plot(epochs, val_accuracies, "r-", marker="s", linewidth=2, label="Val Acc")
    axes[1].set_title("Accuracy", fontweight="bold")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Accuracy")
    axes[1].legend()
    axes[1].grid(True, linestyle="--", alpha=0.5)

    plt.tight_layout()

    if save_dir:
        out_dir = Path(save_dir) / model_name
        save_fig(fig, out_dir, f"{config_name}_curves")

    if show:
        plt.show()
    plt.close(fig)