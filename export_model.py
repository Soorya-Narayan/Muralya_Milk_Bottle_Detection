#!/usr/bin/env python3
"""
export_model.py
---------------
Exports the trained milk bottle YOLOv8 model to ONNX format.

ONNX is the most universal format — it can be used with:
  • Python (onnxruntime)
  • C++ applications
  • Web (via onnxruntime-web)
  • Edge devices (Jetson, Raspberry Pi)
  • OpenCV DNN module
  • Any cloud inference service

Run:
    python export_model.py
"""

import sys
import time
from pathlib import Path

BASE_DIR     = Path(__file__).parent.resolve()
WEIGHTS_PATH = BASE_DIR / "runs" / "detect" / "milk_bottle" / "weights" / "best.pt"
EXPORT_DIR   = BASE_DIR / "exported_model"
IMAGE_SIZE   = 640


def check_weights():
    if not WEIGHTS_PATH.exists():
        print(f"❌  Weights not found at: {WEIGHTS_PATH}")
        print("   Run  python train.py  first.")
        sys.exit(1)
    print(f"✅  Weights found: {WEIGHTS_PATH}")


def export_onnx():
    from ultralytics import YOLO

    print("\n" + "=" * 60)
    print("  Exporting YOLOv8 → ONNX")
    print("=" * 60)
    print(f"  Source  : {WEIGHTS_PATH}")
    print(f"  Format  : ONNX (opset 11)")
    print(f"  Img size: {IMAGE_SIZE}x{IMAGE_SIZE}")
    print(f"  Dynamic : True  (accepts any batch size at runtime)")
    print(f"  Simplify: True  (optimises graph for faster inference)")
    print("=" * 60)

    model = YOLO(str(WEIGHTS_PATH))

    start = time.time()
    exported_path = model.export(
        format   = "onnx",
        imgsz    = IMAGE_SIZE,
        dynamic  = True,          # variable batch size
        simplify = True,          # clean up redundant ops
        opset    = 11,            # broadest compatibility
        half     = False,         # FP32 for max compatibility
    )
    elapsed = time.time() - start

    return Path(exported_path), elapsed


def copy_to_export_dir(onnx_path: Path):
    """Copy the .onnx file and metadata into a tidy exported_model/ folder."""
    import shutil

    EXPORT_DIR.mkdir(exist_ok=True)

    # Copy weights
    dest = EXPORT_DIR / "milk_bottle_detector.onnx"
    shutil.copy2(onnx_path, dest)

    # Write a README with usage instructions
    readme = EXPORT_DIR / "README.md"
    readme.write_text(f"""# Milk Bottle Detector — ONNX Model

## Model Info
- **Architecture**: YOLOv8n (nano)
- **Trained on**: 858 images (1,073 total, 80/10/10 split)
- **Classes**: `milk_bottle`
- **Input size**: {IMAGE_SIZE}x{IMAGE_SIZE} (dynamic batch)
- **mAP@50**: 98.4%
- **Precision**: 93.9%
- **Recall**: 97.0%

## Files
- `milk_bottle_detector.onnx` — exported model (this file)

## Quick Start — Python (onnxruntime)

```python
import onnxruntime as rt
import numpy as np
import cv2

# Load model
session = rt.InferenceSession("milk_bottle_detector.onnx")
input_name = session.get_inputs()[0].name

# Preprocess image
img = cv2.imread("image.jpg")
img = cv2.resize(img, ({IMAGE_SIZE}, {IMAGE_SIZE}))
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
img = img.astype(np.float32) / 255.0
img = img.transpose(2, 0, 1)[np.newaxis]   # BCHW

# Run inference
outputs = session.run(None, {{input_name: img}})
print(outputs)
```

## Quick Start — OpenCV DNN

```python
import cv2

net = cv2.dnn.readNetFromONNX("milk_bottle_detector.onnx")
img = cv2.imread("image.jpg")
blob = cv2.dnn.blobFromImage(img, 1/255.0, ({IMAGE_SIZE},{IMAGE_SIZE}), swapRB=True)
net.setInput(blob)
outputs = net.forward()
```

## Install Runtime

```bash
pip install onnxruntime   # CPU
# or
pip install onnxruntime-gpu  # NVIDIA GPU
```
""")
    return dest


def print_model_info(onnx_path: Path, elapsed: float):
    """Print file size and basic info."""
    size_mb = onnx_path.stat().st_size / (1024 * 1024)
    print(f"\n✅  Export complete in {elapsed:.1f}s")
    print(f"   File  : {onnx_path}")
    print(f"   Size  : {size_mb:.2f} MB")


def main():
    print("=" * 60)
    print("  🥛 Milk Bottle Detector — Model Export")
    print("=" * 60)

    check_weights()
    onnx_path, elapsed = export_onnx()
    print_model_info(onnx_path, elapsed)

    # Copy to tidy export folder
    dest = copy_to_export_dir(onnx_path)

    print("\n" + "=" * 60)
    print("  📦  Exported Model Package")
    print("=" * 60)
    print(f"  Folder  : {EXPORT_DIR}/")
    print(f"  Model   : {dest.name}")
    print(f"  README  : README.md  (usage instructions inside)")
    print("\n  Install runtime:   pip install onnxruntime")
    print("=" * 60)


if __name__ == "__main__":
    main()
