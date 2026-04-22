"""
train.py — Training script for YOLOv8 Road Hazard Detection
CSE3240 Computer Vision | Vashisht Reddy | 23FE10CSE00256

Usage:
    python src/train.py --model yolov8s --epochs 100 --batch 16
    python src/train.py --model yolov5s  --epochs 100 --batch 16
"""

import argparse
import os
from pathlib import Path
from ultralytics import YOLO


SUPPORTED_MODELS = {
    "yolov8n": "yolov8n.pt",
    "yolov8s": "yolov8s.pt",
    "yolov8m": "yolov8m.pt",
    "yolov5s": "yolov5su.pt",   # ultralytics-format YOLOv5s
}


def parse_args():
    parser = argparse.ArgumentParser(description="Train YOLOv8 for road hazard detection")
    parser.add_argument("--model",   type=str, default="yolov8s",
                        choices=SUPPORTED_MODELS.keys(),
                        help="Model variant to train")
    parser.add_argument("--epochs",  type=int, default=100)
    parser.add_argument("--batch",   type=int, default=16)
    parser.add_argument("--imgsz",   type=int, default=640)
    parser.add_argument("--device",  type=str, default="0",
                        help="GPU index or 'cpu'")
    parser.add_argument("--data",    type=str,
                        default="configs/dataset.yaml",
                        help="Path to dataset YAML")
    parser.add_argument("--name",    type=str, default=None,
                        help="Run name (defaults to model name)")
    parser.add_argument("--resume",  type=str, default=None,
                        help="Path to checkpoint .pt to resume from")
    return parser.parse_args()


def train(args):
    run_name = args.name or args.model
    weights = args.resume if args.resume else SUPPORTED_MODELS[args.model]

    print(f"\n{'='*55}")
    print(f"  Road Hazard Detection — Training")
    print(f"  Model   : {args.model}  ({weights})")
    print(f"  Epochs  : {args.epochs} | Batch: {args.batch} | ImgSz: {args.imgsz}")
    print(f"  Device  : {args.device}")
    print(f"{'='*55}\n")

    model = YOLO(weights)

    results = model.train(
        data=args.data,
        epochs=args.epochs,
        batch=args.batch,
        imgsz=args.imgsz,
        device=args.device,
        lr0=0.01,
        lrf=0.0001,
        momentum=0.937,
        weight_decay=0.0005,
        warmup_epochs=3,
        warmup_momentum=0.8,
        hsv_h=0.015,
        hsv_s=0.7,
        hsv_v=0.4,
        translate=0.1,
        scale=0.5,
        fliplr=0.5,
        mosaic=0.5,
        patience=20,
        save=True,
        plots=True,
        verbose=True,
        project="runs/train",
        name=run_name,
        exist_ok=False,
    )

    best_ckpt = Path("runs/train") / run_name / "weights" / "best.pt"
    print(f"\n✅ Training complete.")
    print(f"   Best checkpoint : {best_ckpt}")
    print(f"   mAP@0.5         : {results.results_dict.get('metrics/mAP50(B)', 'N/A'):.4f}")
    return results


if __name__ == "__main__":
    args = parse_args()
    train(args)
