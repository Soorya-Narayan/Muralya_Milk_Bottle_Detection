#!/usr/bin/env python3
"""
prepare_dataset.py
------------------
Merges all 5 annotated batches from Label Studio (YOLO format) into a single
clean dataset with train / val / test splits, then writes dataset.yaml.

Run:
    python prepare_dataset.py
"""

import os
import shutil
import random
import json
from pathlib import Path

# ── Configuration ─────────────────────────────────────────────────────────────
BASE_DIR       = Path(__file__).parent.resolve()
ANNOTATED_DIR  = BASE_DIR / "Annotated"
OUTPUT_DIR     = BASE_DIR / "dataset"
TRAIN_RATIO    = 0.80
VAL_RATIO      = 0.10
TEST_RATIO     = 0.10
SEED           = 42
ANNOTATED_BATCHES = ["annotated1", "annotated2", "annotated3", "annotated4", "annotated5"]

# ── Helpers ───────────────────────────────────────────────────────────────────
def collect_samples(annotated_dir: Path, batches: list[str]) -> list[dict]:
    """Return list of {image, label} path dicts for every matched pair."""
    samples = []
    for batch in batches:
        img_dir   = annotated_dir / batch / "images"
        lbl_dir   = annotated_dir / batch / "labels"
        if not img_dir.exists() or not lbl_dir.exists():
            print(f"  [WARN] Skipping {batch}: missing images/ or labels/")
            continue

        for img_path in sorted(img_dir.glob("*.jpg")):
            stem      = img_path.stem          # e.g. "42b0ce04-IMG_20260609_110240"
            lbl_path  = lbl_dir / (stem + ".txt")
            if lbl_path.exists():
                samples.append({"image": img_path, "label": lbl_path})
            else:
                print(f"  [WARN] No label for {img_path.name} — skipping")
    return samples


def split_samples(samples: list[dict], train_r: float, val_r: float, seed: int):
    """Shuffle and split into train / val / test lists."""
    random.seed(seed)
    random.shuffle(samples)
    n        = len(samples)
    n_train  = int(n * train_r)
    n_val    = int(n * val_r)
    train    = samples[:n_train]
    val      = samples[n_train : n_train + n_val]
    test     = samples[n_train + n_val:]
    return train, val, test


def copy_split(split_samples: list[dict], split_name: str, out_dir: Path):
    """Copy images + labels into out_dir/{images,labels}/{split_name}/."""
    img_out = out_dir / "images" / split_name
    lbl_out = out_dir / "labels" / split_name
    img_out.mkdir(parents=True, exist_ok=True)
    lbl_out.mkdir(parents=True, exist_ok=True)

    for s in split_samples:
        shutil.copy2(s["image"], img_out / s["image"].name)
        shutil.copy2(s["label"], lbl_out / s["label"].name)


def write_yaml(out_dir: Path, class_names: list[str]):
    """Write YOLO dataset.yaml."""
    yaml_content = f"""# Milk Bottle Detection — YOLOv8 Dataset Config
path: {out_dir}
train: images/train
val:   images/val
test:  images/test

nc: {len(class_names)}
names: {class_names}
"""
    yaml_path = out_dir / "dataset.yaml"
    yaml_path.write_text(yaml_content)
    return yaml_path


# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    print("=" * 60)
    print("  Milk Bottle Dataset Preparation")
    print("=" * 60)

    # 1. Collect all samples
    print(f"\n[1/4] Scanning annotated batches in: {ANNOTATED_DIR}")
    samples = collect_samples(ANNOTATED_DIR, ANNOTATED_BATCHES)
    print(f"      Found {len(samples)} image-label pairs")
    if not samples:
        raise RuntimeError("No samples found! Check your Annotated/ directory.")

    # 2. Read class names from first batch
    classes_file = ANNOTATED_DIR / ANNOTATED_BATCHES[0] / "classes.txt"
    class_names  = [l.strip() for l in classes_file.read_text().splitlines() if l.strip()]
    print(f"      Classes: {class_names}")

    # 3. Split
    print(f"\n[2/4] Splitting ({TRAIN_RATIO*100:.0f}% train / {VAL_RATIO*100:.0f}% val / {TEST_RATIO*100:.0f}% test)")
    train, val, test = split_samples(samples, TRAIN_RATIO, VAL_RATIO, SEED)
    print(f"      Train: {len(train)}  Val: {len(val)}  Test: {len(test)}")

    # 4. Copy to output directory
    print(f"\n[3/4] Copying files to: {OUTPUT_DIR}")
    if OUTPUT_DIR.exists():
        shutil.rmtree(OUTPUT_DIR)
    copy_split(train, "train", OUTPUT_DIR)
    copy_split(val,   "val",   OUTPUT_DIR)
    copy_split(test,  "test",  OUTPUT_DIR)

    # 5. Write dataset.yaml
    print(f"\n[4/4] Writing dataset.yaml")
    yaml_path = write_yaml(OUTPUT_DIR, class_names)
    print(f"      Saved: {yaml_path}")

    # Summary
    print("\n" + "=" * 60)
    print("  ✅  Dataset Ready!")
    print("=" * 60)
    print(f"  Location : {OUTPUT_DIR}")
    print(f"  Train    : {len(train):>5} images")
    print(f"  Val      : {len(val):>5} images")
    print(f"  Test     : {len(test):>5} images")
    print(f"  Classes  : {class_names}")
    print(f"  YAML     : {yaml_path}")
    print("\nNext → run: python train.py")


if __name__ == "__main__":
    main()
