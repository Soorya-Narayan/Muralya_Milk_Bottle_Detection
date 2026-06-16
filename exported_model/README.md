# Milk Bottle Detector — ONNX Model

## Model Info
- **Architecture**: YOLOv8n (nano)
- **Trained on**: 858 images (1,073 total, 80/10/10 split)
- **Classes**: `milk_bottle`
- **Input size**: 640x640 (dynamic batch)
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
img = cv2.resize(img, (640, 640))
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
img = img.astype(np.float32) / 255.0
img = img.transpose(2, 0, 1)[np.newaxis]   # BCHW

# Run inference
outputs = session.run(None, {input_name: img})
print(outputs)
```

## Quick Start — OpenCV DNN

```python
import cv2

net = cv2.dnn.readNetFromONNX("milk_bottle_detector.onnx")
img = cv2.imread("image.jpg")
blob = cv2.dnn.blobFromImage(img, 1/255.0, (640,640), swapRB=True)
net.setInput(blob)
outputs = net.forward()
```

## Install Runtime

```bash
pip install onnxruntime   # CPU
# or
pip install onnxruntime-gpu  # NVIDIA GPU
```
