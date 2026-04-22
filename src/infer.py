import argparse
from pathlib import Path

import cv2
from ultralytics import YOLO

CLASS_COLORS = {
    0: (33,  150, 243),   # pothole            — blue
    1: (76,  175,  80),   # stray_animal       — green
    2: (255, 152,   0),   # fallen_debris      — orange
    3: (156,  39, 176),   # waterlogging       — purple
    4: (244,  67,  54),   # construction_barrier — red
}


def parse_args():
    parser = argparse.ArgumentParser(description="Run inference with a trained YOLO model")
    parser.add_argument("--weights", type=str, required=True)
    parser.add_argument("--source",  type=str, required=True,
                        help="Image file or directory path")
    parser.add_argument("--conf",    type=float, default=0.25)
    parser.add_argument("--iou",     type=float, default=0.45)
    parser.add_argument("--imgsz",   type=int, default=640)
    parser.add_argument("--show",    action="store_true",
                        help="Display annotated images in a window")
    parser.add_argument("--save",    action="store_true", default=True,
                        help="Save annotated images to runs/infer/")
    parser.add_argument("--device",  type=str, default="0")
    return parser.parse_args()


def draw_detections(image, results, conf_thresh=0.25):
    """Draw bounding boxes and labels manually for full control."""
    for box in results[0].boxes:
        conf = float(box.conf[0])
        if conf < conf_thresh:
            continue
        cls_id = int(box.cls[0])
        label  = results[0].names[cls_id]
        color  = CLASS_COLORS.get(cls_id, (200, 200, 200))

        x1, y1, x2, y2 = map(int, box.xyxy[0])
        cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)

        text = f"{label} {conf:.2f}"
        (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 1)
        cv2.rectangle(image, (x1, y1 - th - 6), (x1 + tw + 4, y1), color, -1)
        cv2.putText(image, text, (x1 + 2, y1 - 4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1, cv2.LINE_AA)
    return image


def run_inference(args):
    model = YOLO(args.weights)
    source = Path(args.source)

    if source.is_dir():
        image_paths = sorted(source.glob("*.jpg")) + sorted(source.glob("*.png"))
    elif source.is_file():
        image_paths = [source]
    else:
        raise FileNotFoundError(f"Source not found: {source}")

    out_dir = Path("runs/infer/annotated")
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"\nRunning inference on {len(image_paths)} image(s)...\n")

    for img_path in image_paths:
        image = cv2.imread(str(img_path))
        if image is None:
            print(f"  ⚠ Could not read {img_path.name}, skipping.")
            continue

        results = model(image, conf=args.conf, iou=args.iou,
                        imgsz=args.imgsz, device=args.device, verbose=False)
        n_det = len(results[0].boxes)
        print(f"  {img_path.name:40s} — {n_det} detection(s)")

        annotated = draw_detections(image.copy(), results, args.conf)

        if args.save:
            out_path = out_dir / img_path.name
            cv2.imwrite(str(out_path), annotated)

        if args.show:
            cv2.imshow("Road Hazard Detection", annotated)
            key = cv2.waitKey(0)
            if key == ord("q"):
                break

    cv2.destroyAllWindows()
    if args.save:
        print(f"\n✅ Annotated images saved to {out_dir}/")


if __name__ == "__main__":
    args = parse_args()
    run_inference(args)
