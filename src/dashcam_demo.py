import argparse
import time
from collections import Counter, deque
from pathlib import Path

import cv2
import numpy as np
from ultralytics import YOLO

CLASS_NAMES  = ["pothole", "stray_animal", "fallen_debris", "waterlogging", "construction_barrier"]
CLASS_COLORS = {
    0: (33,  150, 243),
    1: (76,  175,  80),
    2: (255, 152,   0),
    3: (156,  39, 176),
    4: (244,  67,  54),
}
ALERT_CLASSES = {0, 3}   # pothole and waterlogging trigger alert border


def parse_args():
    parser = argparse.ArgumentParser(description="Live dashcam inference")
    parser.add_argument("--weights", type=str, required=True)
    parser.add_argument("--source",  type=str, default="0",
                        help="'0' for webcam, or path to video file")
    parser.add_argument("--conf",    type=float, default=0.30)
    parser.add_argument("--iou",     type=float, default=0.45)
    parser.add_argument("--imgsz",   type=int, default=640)
    parser.add_argument("--device",  type=str, default="0")
    parser.add_argument("--save",    action="store_true",
                        help="Save annotated output video")
    return parser.parse_args()


def draw_overlay(frame, fps, counts):
    """Draw semi-transparent HUD with FPS and detection counts."""
    h, w = frame.shape[:2]
    overlay = frame.copy()

    # HUD background
    cv2.rectangle(overlay, (0, 0), (220, 30 + 22 * len(CLASS_NAMES)), (0, 0, 0), -1)
    frame = cv2.addWeighted(overlay, 0.45, frame, 0.55, 0)

    # FPS
    cv2.putText(frame, f"FPS: {fps:.1f}", (8, 20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1, cv2.LINE_AA)

    # Per-class counts
    for i, name in enumerate(CLASS_NAMES):
        count = counts.get(i, 0)
        color = CLASS_COLORS[i]
        y = 40 + i * 22
        cv2.rectangle(frame, (8, y - 12), (16, y + 2), color, -1)
        cv2.putText(frame, f"{name}: {count}", (22, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.48, (220, 220, 220), 1, cv2.LINE_AA)

    return frame


def draw_alert_border(frame, alert):
    """Flash a red border when high-priority hazards detected."""
    if alert:
        h, w = frame.shape[:2]
        cv2.rectangle(frame, (2, 2), (w - 2, h - 2), (0, 0, 255), 5)
    return frame


def run_demo(args):
    model = YOLO(args.weights)
    source = int(args.source) if args.source == "0" else args.source
    cap = cv2.VideoCapture(source)

    if not cap.isOpened():
        raise RuntimeError(f"Cannot open source: {args.source}")

    fps_w, fps_h = int(cap.get(3)), int(cap.get(4))
    writer = None
    if args.save:
        out_path = "runs/dashcam_output.mp4"
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(out_path, fourcc, 25, (fps_w, fps_h))
        print(f"Saving output → {out_path}")

    fps_queue = deque(maxlen=30)
    prev_time = time.time()

    print("\n🎥 Dashcam demo running — press 'q' to quit\n")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        results = model(frame, conf=args.conf, iou=args.iou,
                        imgsz=args.imgsz, device=args.device, verbose=False)

        # Count detections per class
        counts = Counter()
        has_alert = False
        for box in results[0].boxes:
            cls_id = int(box.cls[0])
            counts[cls_id] += 1
            if cls_id in ALERT_CLASSES:
                has_alert = True

        # Draw boxes
        for box in results[0].boxes:
            conf   = float(box.conf[0])
            cls_id = int(box.cls[0])
            color  = CLASS_COLORS.get(cls_id, (200, 200, 200))
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            label = f"{CLASS_NAMES[cls_id]} {conf:.2f}"
            (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
            cv2.rectangle(frame, (x1, y1 - th - 5), (x1 + tw + 4, y1), color, -1)
            cv2.putText(frame, label, (x1 + 2, y1 - 3),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)

        # FPS calculation
        cur_time = time.time()
        fps_queue.append(1.0 / max(cur_time - prev_time, 1e-6))
        prev_time = cur_time
        fps = np.mean(fps_queue)

        frame = draw_overlay(frame, fps, counts)
        frame = draw_alert_border(frame, has_alert)

        cv2.imshow("Road Hazard Detection — Dashcam", frame)
        if writer:
            writer.write(frame)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    if writer:
        writer.release()
    cv2.destroyAllWindows()
    print("\nDemo ended.")


if __name__ == "__main__":
    args = parse_args()
    run_demo(args)
