import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from ultralytics import YOLO

CLASS_NAMES = ["pothole", "stray_animal", "fallen_debris", "waterlogging", "construction_barrier"]


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate a trained YOLO model")
    parser.add_argument("--weights", type=str, required=True,
                        help="Path to model weights (.pt)")
    parser.add_argument("--data",    type=str, default="configs/dataset.yaml")
    parser.add_argument("--split",   type=str, default="val",
                        choices=["val", "test"],
                        help="Dataset split to evaluate on")
    parser.add_argument("--imgsz",   type=int, default=640)
    parser.add_argument("--conf",    type=float, default=0.25,
                        help="Confidence threshold")
    parser.add_argument("--iou",     type=float, default=0.5,
                        help="IoU threshold for NMS")
    parser.add_argument("--device",  type=str, default="0")
    return parser.parse_args()


def print_metrics_table(metrics):
    """Pretty-print per-class metrics."""
    header = f"{'Class':<22} {'Precision':>10} {'Recall':>10} {'F1':>8} {'mAP@0.5':>10}"
    print("\n" + "="*65)
    print(header)
    print("-"*65)

    for i, name in enumerate(CLASS_NAMES):
        p  = metrics.box.p[i]  if hasattr(metrics.box, 'p')  else 0.0
        r  = metrics.box.r[i]  if hasattr(metrics.box, 'r')  else 0.0
        f1 = 2 * p * r / (p + r + 1e-6)
        ap = metrics.box.ap50[i] if hasattr(metrics.box, 'ap50') else 0.0
        print(f"{name:<22} {p:>10.3f} {r:>10.3f} {f1:>8.3f} {ap:>10.3f}")

    print("-"*65)
    mp  = metrics.box.mp
    mr  = metrics.box.mr
    mf1 = 2 * mp * mr / (mp + mr + 1e-6)
    map50 = metrics.box.map50
    map5095 = metrics.box.map
    print(f"{'Overall (mean)':<22} {mp:>10.3f} {mr:>10.3f} {mf1:>8.3f} {map50:>10.3f}")
    print(f"{'mAP@0.5:0.95':<22} {'':>10} {'':>10} {'':>8} {map5095:>10.3f}")
    print("="*65 + "\n")


def plot_per_class_map(metrics, save_dir):
    """Bar chart of per-class mAP@0.5."""
    ap50_vals = metrics.box.ap50 if hasattr(metrics.box, 'ap50') else [0]*len(CLASS_NAMES)

    fig, ax = plt.subplots(figsize=(9, 4))
    colors = ["#2196F3", "#4CAF50", "#FF9800", "#9C27B0", "#F44336"]
    bars = ax.bar(CLASS_NAMES, ap50_vals, color=colors, width=0.55, edgecolor="white")

    for bar, val in zip(bars, ap50_vals):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                f"{val:.3f}", ha="center", va="bottom", fontsize=9, fontweight="bold")

    ax.set_ylim(0, 1.0)
    ax.set_ylabel("mAP@0.5", fontsize=11)
    ax.set_title("Per-Class mAP@0.5", fontsize=13, fontweight="bold")
    ax.tick_params(axis="x", rotation=15)
    ax.set_xlabel("Hazard Class", fontsize=11)
    plt.tight_layout()

    out_path = Path(save_dir) / "per_class_map50.png"
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f"   Saved chart → {out_path}")


def evaluate(args):
    model = YOLO(args.weights)
    save_dir = Path(args.weights).parent.parent / "eval"
    save_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n{'='*55}")
    print(f"  Evaluating: {args.weights}")
    print(f"  Split     : {args.split}")
    print(f"{'='*55}\n")

    metrics = model.val(
        data=args.data,
        split=args.split,
        imgsz=args.imgsz,
        conf=args.conf,
        iou=args.iou,
        device=args.device,
        plots=True,
        save_json=True,
        project=str(save_dir.parent),
        name="eval",
        exist_ok=True,
    )

    print_metrics_table(metrics)
    plot_per_class_map(metrics, save_dir)

    # Save summary JSON
    summary = {
        "weights": str(args.weights),
        "split": args.split,
        "mAP50": float(metrics.box.map50),
        "mAP50_95": float(metrics.box.map),
        "precision": float(metrics.box.mp),
        "recall": float(metrics.box.mr),
    }
    with open(save_dir / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    print(f"   Saved summary → {save_dir / 'summary.json'}\n")

    return metrics


if __name__ == "__main__":
    args = parse_args()
    evaluate(args)
