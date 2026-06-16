#!/usr/bin/env python3
"""
detect_live.py
--------------
Real-time milk bottle detection from MacBook webcam using a trained YOLOv8 model.
Displays:
  • Bounding boxes + confidence scores
  • Live bottle count overlay
  • FPS counter
  • Mini live-graph of bottle count over time (right panel)

Controls:
  q  — quit
  s  — save screenshot
  p  — pause / resume
  +  — increase confidence threshold
  -  — decrease confidence threshold

Run:
    python detect_live.py
"""

import cv2
import time
import sys
import os
import numpy as np
from pathlib import Path
from collections import deque
from datetime import datetime

# ── Configuration ─────────────────────────────────────────────────────────────
BASE_DIR       = Path(__file__).parent.resolve()
WEIGHTS_PATH   = BASE_DIR / "runs" / "detect" / "milk_bottle" / "weights" / "best.pt"
CAMERA_INDEX   = 0       # MacBook built-in webcam
CONF_THRESHOLD = 0.40    # detection confidence (adjustable with +/-)
IOU_THRESHOLD  = 0.45
GRAPH_HISTORY  = 150     # number of frames to show in the live graph
SCREENSHOT_DIR = BASE_DIR / "screenshots"
SCREENSHOT_DIR.mkdir(exist_ok=True)

# ── Colours & fonts ───────────────────────────────────────────────────────────
CLR_BOX      = (0,   200, 80)    # green bounding box
CLR_LABEL_BG = (0,   160, 60)
CLR_COUNT_BG = (20,  20,  20)
CLR_FPS      = (200, 200, 200)
CLR_GRAPH_BG = (18,  18,  18)
CLR_GRAPH_LN = (0,   210, 100)
CLR_GRAPH_AX = (80,  80,  80)
FONT         = cv2.FONT_HERSHEY_SIMPLEX

# ── Graph panel dimensions ────────────────────────────────────────────────────
GRAPH_W = 340
GRAPH_H = 220   # will be placed top-right of the frame


def load_model():
    """Load YOLOv8 model. Falls back to yolov8n.pt if custom weights absent."""
    from ultralytics import YOLO
    if WEIGHTS_PATH.exists():
        print(f"✅  Loading custom weights: {WEIGHTS_PATH}")
        return YOLO(str(WEIGHTS_PATH))
    else:
        print(f"⚠️  Custom weights not found at:\n    {WEIGHTS_PATH}")
        print("   Falling back to base YOLOv8n (not trained on your data).")
        print("   Run  python train.py  first for accurate results.\n")
        return YOLO("yolov8n.pt")


def draw_boxes(frame: np.ndarray, boxes, class_names: list, conf_thresh: float) -> int:
    """Draw bounding boxes and labels. Returns count of detected bottles."""
    count = 0
    if boxes is None:
        return count
    for box in boxes:
        conf = float(box.conf[0])
        cls  = int(box.cls[0])
        if conf < conf_thresh:
            continue
        count += 1
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        label = f"{class_names[cls]}  {conf:.0%}"

        # Box
        cv2.rectangle(frame, (x1, y1), (x2, y2), CLR_BOX, 2)

        # Label background
        (tw, th), _ = cv2.getTextSize(label, FONT, 0.55, 1)
        cv2.rectangle(frame, (x1, y1 - th - 8), (x1 + tw + 6, y1), CLR_LABEL_BG, -1)
        cv2.putText(frame, label, (x1 + 3, y1 - 4), FONT, 0.55, (255, 255, 255), 1, cv2.LINE_AA)
    return count


def draw_hud(frame: np.ndarray, count: int, fps: float, conf: float, paused: bool):
    """Overlay count, FPS, conf, and controls on the frame."""
    h, w = frame.shape[:2]

    # Semi-transparent count badge (bottom-left)
    badge = np.zeros((90, 260, 3), dtype=np.uint8)
    badge[:] = (15, 15, 15)
    cv2.putText(badge, "BOTTLES DETECTED", (10, 22), FONT, 0.5, (150, 150, 150), 1, cv2.LINE_AA)
    cv2.putText(badge, str(count), (10, 76), FONT, 2.8, CLR_GRAPH_LN, 3, cv2.LINE_AA)
    badge_alpha = 0.80
    frame[h-90:h, 0:260] = cv2.addWeighted(
        frame[h-90:h, 0:260], 1 - badge_alpha, badge, badge_alpha, 0
    )

    # FPS + conf + pause (top-left thin bar)
    status = f"FPS: {fps:.1f}  |  Conf: {conf:.0%}"
    if paused:
        status += "  |  ⏸ PAUSED"
    cv2.putText(frame, status, (10, 24), FONT, 0.6, CLR_FPS, 1, cv2.LINE_AA)

    # Controls hint (very bottom)
    cv2.putText(frame, "q=quit  s=screenshot  p=pause  +/-=confidence",
                (10, h - 8), FONT, 0.43, (110, 110, 110), 1, cv2.LINE_AA)


def draw_graph(frame: np.ndarray, history: deque, max_count: int):
    """Draw a mini line-graph of bottle count over time (top-right corner)."""
    h, w = frame.shape[:2]
    panel = np.full((GRAPH_H, GRAPH_W, 3), CLR_GRAPH_BG, dtype=np.uint8)

    # Title
    cv2.putText(panel, "Bottle Count Over Time", (8, 18), FONT, 0.48, (200, 200, 200), 1, cv2.LINE_AA)

    # Axes
    margin = (28, 12, 20, 10)  # top, right, bottom, left
    gx0, gy0 = margin[3], margin[0]
    gx1 = GRAPH_W - margin[1]
    gy1 = GRAPH_H - margin[2]
    cv2.rectangle(panel, (gx0, gy0), (gx1, gy1), CLR_GRAPH_AX, 1)

    # Y-axis label
    cap_val = max(max_count, 1)
    cv2.putText(panel, f"{cap_val}", (0, gy0 + 8), FONT, 0.38, (150, 150, 150), 1)
    cv2.putText(panel, "0", (0, gy1),              FONT, 0.38, (150, 150, 150), 1)

    # Plot line
    pts = list(history)
    n   = len(pts)
    if n >= 2:
        gw  = gx1 - gx0
        gh  = gy1 - gy0
        xs  = [int(gx0 + i * gw / (GRAPH_HISTORY - 1)) for i in range(n)]
        ys  = [int(gy1 - (v / cap_val) * gh) for v in pts]
        for i in range(1, n):
            cv2.line(panel, (xs[i-1], ys[i-1]), (xs[i], ys[i]), CLR_GRAPH_LN, 2, cv2.LINE_AA)
        # Latest value dot
        cv2.circle(panel, (xs[-1], ys[-1]), 4, (255, 255, 100), -1)
        cv2.putText(panel, str(pts[-1]), (xs[-1] + 6, ys[-1] + 4), FONT, 0.45, (255, 255, 100), 1)

    # Composite onto frame (top-right)
    px = w - GRAPH_W
    frame[0:GRAPH_H, px:w] = cv2.addWeighted(
        frame[0:GRAPH_H, px:w], 0.25, panel, 0.75, 0
    )


def run_detection():
    model        = load_model()
    class_names  = model.names if hasattr(model, "names") else {0: "milk_bottle"}

    cap = cv2.VideoCapture(CAMERA_INDEX)
    if not cap.isOpened():
        print(f"❌  Cannot open camera index {CAMERA_INDEX}")
        sys.exit(1)

    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    history    = deque([0] * GRAPH_HISTORY, maxlen=GRAPH_HISTORY)
    conf_th    = CONF_THRESHOLD
    paused     = False
    prev_time  = time.time()
    fps        = 0.0
    last_frame = None
    max_count  = 1

    print("\n📷  Camera opened. Starting live detection…")
    print("   Controls: q=quit  s=screenshot  p=pause  +/-=confidence\n")

    while True:
        if not paused:
            ret, frame = cap.read()
            if not ret:
                print("❌  Failed to grab frame.")
                break
            last_frame = frame.copy()

            # ── Inference ──────────────────────────────────────────────────
            results = model.predict(
                source          = frame,
                conf            = conf_th,
                iou             = IOU_THRESHOLD,
                device          = "mps",
                verbose         = False,
                stream          = False,
            )

            count = draw_boxes(frame, results[0].boxes, class_names, conf_th)
            history.append(count)
            max_count = max(max_count, count, 1)

            # ── FPS ────────────────────────────────────────────────────────
            now       = time.time()
            fps       = 0.9 * fps + 0.1 * (1.0 / max(now - prev_time, 1e-6))
            prev_time = now
        else:
            frame = last_frame.copy() if last_frame is not None else np.zeros((720, 1280, 3), dtype=np.uint8)

        draw_hud(frame, history[-1], fps, conf_th, paused)
        draw_graph(frame, history, max_count)

        cv2.imshow("🥛 Milk Bottle Detector — Press q to quit", frame)

        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break
        elif key == ord("s"):
            ts   = datetime.now().strftime("%Y%m%d_%H%M%S")
            path = SCREENSHOT_DIR / f"detection_{ts}.jpg"
            cv2.imwrite(str(path), frame)
            print(f"📸  Screenshot saved → {path}")
        elif key == ord("p"):
            paused = not paused
            print("⏸  Paused" if paused else "▶  Resumed")
        elif key == ord("+") or key == ord("="):
            conf_th = min(conf_th + 0.05, 0.95)
            print(f"🔼  Confidence threshold: {conf_th:.0%}")
        elif key == ord("-"):
            conf_th = max(conf_th - 0.05, 0.05)
            print(f"🔽  Confidence threshold: {conf_th:.0%}")

    cap.release()
    cv2.destroyAllWindows()
    print("\n✅  Detection stopped.")


def main():
    print("=" * 60)
    print("  🥛 Milk Bottle Live Detection")
    print("=" * 60)
    run_detection()


if __name__ == "__main__":
    main()
