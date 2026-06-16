#!/usr/bin/env python3
"""
train.py
--------
Train a YOLOv8n model on the milk bottle dataset using Apple MPS (M5 GPU).
Produces training graphs and saves the best model weights.

Run:
    python train.py
"""

import os
import sys
import time
from pathlib import Path

# ── Make sure we import from the conda env ────────────────────────────────────
CONDA_ENV_BIN = "/opt/homebrew/Caskroom/miniforge/base/envs/milk-env/bin"
if CONDA_ENV_BIN not in os.environ.get("PATH", ""):
    os.environ["PATH"] = CONDA_ENV_BIN + ":" + os.environ.get("PATH", "")

from ultralytics import YOLO
import matplotlib
matplotlib.use("Agg")   # non-interactive backend for saving graphs
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

# ── Configuration ─────────────────────────────────────────────────────────────
BASE_DIR      = Path(__file__).parent.resolve()
DATASET_YAML  = BASE_DIR / "dataset" / "dataset.yaml"
MODEL_NAME    = "yolov8n.pt"   # nano — best for real-time on MPS
PROJECT_DIR   = BASE_DIR / "runs" / "detect"
RUN_NAME      = "milk_bottle"
EPOCHS        = 100
IMAGE_SIZE    = 640
BATCH_SIZE    = 16             # good default for M5 memory
DEVICE        = "mps"          # Apple Silicon GPU
WORKERS       = 4
PATIENCE      = 20             # early stop if no improvement for 20 epochs


def check_dataset():
    if not DATASET_YAML.exists():
        print("❌  dataset/dataset.yaml not found.")
        print("   Please run:  python prepare_dataset.py  first.")
        sys.exit(1)
    print(f"✅  Dataset found: {DATASET_YAML}")


def train_model() -> Path:
    print("\n" + "=" * 60)
    print("  Milk Bottle YOLOv8 Training")
    print("=" * 60)
    print(f"  Model    : {MODEL_NAME}")
    print(f"  Device   : {DEVICE} (Apple M5 GPU)")
    print(f"  Epochs   : {EPOCHS}  (early stop patience={PATIENCE})")
    print(f"  Img size : {IMAGE_SIZE}x{IMAGE_SIZE}")
    print(f"  Batch    : {BATCH_SIZE}")
    print(f"  Output   : {PROJECT_DIR / RUN_NAME}")
    print("=" * 60)
    print("\nStarting training... (this takes ~15–30 min on M5)\n")

    model = YOLO(MODEL_NAME)

    start = time.time()
    results = model.train(
        data      = str(DATASET_YAML),
        epochs    = EPOCHS,
        imgsz     = IMAGE_SIZE,
        batch     = BATCH_SIZE,
        device    = DEVICE,
        workers   = WORKERS,
        patience  = PATIENCE,
        project   = str(PROJECT_DIR),
        name      = RUN_NAME,
        exist_ok  = True,
        plots     = True,          # saves training graphs automatically
        save      = True,
        verbose   = True,
        # Augmentation tuned for conveyor belt / industrial setting
        hsv_h     = 0.015,
        hsv_s     = 0.4,
        hsv_v     = 0.3,
        degrees   = 5.0,
        translate = 0.1,
        scale     = 0.4,
        flipud    = 0.0,
        fliplr    = 0.5,
        mosaic    = 1.0,
    )

    elapsed = time.time() - start
    best_weights = PROJECT_DIR / RUN_NAME / "weights" / "best.pt"
    mins  = int(elapsed // 60)
    secs  = int(elapsed % 60)
    print(f"\n⏱  Training finished in {mins}m {secs}s")
    return best_weights, PROJECT_DIR / RUN_NAME, elapsed


def format_duration(seconds: float) -> str:
    """Return a human-readable duration string."""
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = int(seconds % 60)
    if h > 0:
        return f"{h}h {m}m {s}s"
    elif m > 0:
        return f"{m}m {s}s"
    return f"{s}s"


def run_validation(best_weights: Path, run_dir: Path):
    """Validate on test split and print metrics."""
    print("\n" + "=" * 60)
    print("  Running validation on test set…")
    print("=" * 60)
    model   = YOLO(str(best_weights))
    metrics = model.val(
        data    = str(DATASET_YAML),
        split   = "test",
        imgsz   = IMAGE_SIZE,
        device  = DEVICE,
        verbose = True,
    )
    print(f"\n✅  mAP50      : {metrics.box.map50:.4f}")
    print(f"✅  mAP50-95   : {metrics.box.map:.4f}")
    print(f"✅  Precision  : {metrics.box.mp:.4f}")
    print(f"✅  Recall     : {metrics.box.mr:.4f}")
    return metrics


def plot_training_graphs(run_dir: Path, elapsed_secs: float = 0.0):
    """
    Ultralytics saves results.csv inside the run dir.
    We parse it and produce a clean multi-panel figure saved to run_dir.
    """
    import pandas as pd

    csv_path = run_dir / "results.csv"
    if not csv_path.exists():
        print(f"[WARN] results.csv not found at {csv_path}")
        return

    df = pd.read_csv(csv_path)
    df.columns = [c.strip() for c in df.columns]   # remove leading spaces

    total_epochs = int(df["epoch"].max()) if "epoch" in df.columns else EPOCHS
    duration_str  = format_duration(elapsed_secs) if elapsed_secs else "—"

    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    fig.suptitle(
        f"Milk Bottle Detector — Training Results\n"
        f"Epochs: {total_epochs}  |  Training Time: {duration_str}  |  Device: Apple M5 (MPS)",
        fontsize=16, fontweight="bold", y=1.02
    )

    def plot(ax, col, title, color):
        if col in df.columns:
            ax.plot(df["epoch"], df[col], color=color, linewidth=2)
            ax.set_title(title, fontsize=13, fontweight="bold")
            ax.set_xlabel("Epoch")
            ax.grid(True, alpha=0.3)
            ax.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(False)
        else:
            ax.set_visible(False)

    plot(axes[0, 0], "train/box_loss",  "Train Box Loss",       "#e74c3c")
    plot(axes[0, 1], "train/cls_loss",  "Train Class Loss",     "#e67e22")
    plot(axes[0, 2], "train/dfl_loss",  "Train DFL Loss",       "#f39c12")
    plot(axes[1, 0], "metrics/mAP50(B)","mAP@50",               "#27ae60")
    plot(axes[1, 1], "metrics/mAP50-95(B)", "mAP@50-95",        "#2980b9")
    plot(axes[1, 2], "val/box_loss",    "Val Box Loss",         "#8e44ad")

    # Add a shared timing annotation box at the bottom of the figure
    if elapsed_secs:
        fig.text(
            0.5, -0.02,
            f"⏱  Total Training Time: {format_duration(elapsed_secs)}",
            ha="center", fontsize=12, color="#555555",
            bbox=dict(boxstyle="round,pad=0.4", fc="#f5f5f5", ec="#cccccc")
        )

    plt.tight_layout()
    out_path = run_dir / "training_graphs.png"
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"\n📊  Training graphs saved → {out_path}")
    return out_path


def main():
    check_dataset()
    best_weights, run_dir, elapsed = train_model()

    if best_weights.exists():
        print(f"\n🏆  Best model saved → {best_weights}")
        run_validation(best_weights, run_dir)
        plot_training_graphs(run_dir, elapsed_secs=elapsed)
    else:
        print("\n⚠️  best.pt not found — training may have failed.")

    mins = int(elapsed // 60)
    secs = int(elapsed % 60)
    print("\n" + "=" * 60)
    print("  ✅  All done!")
    print(f"  Model         : {best_weights}")
    print(f"  Training Time : {format_duration(elapsed)}")
    print(f"  Graphs        : {run_dir / 'training_graphs.png'}")
    print("\nNext → run: python detect_live.py")
    print("=" * 60)


if __name__ == "__main__":
    main()
