"""
split_dataset.py — Split raw annotated images into train/val/test sets
CSE3240 Computer Vision | Vashisht Reddy | 23FE10CSE00256

Expects:
    raw_images/  — all .jpg/.png images
    raw_labels/  — corresponding YOLO .txt annotation files

Outputs:
    data/images/train/ val/ test/
    data/labels/train/ val/ test/

Usage:
    python scripts/split_dataset.py --images raw_images/ --labels raw_labels/
    python scripts/split_dataset.py --images raw_images/ --labels raw_labels/ --ratios 0.8 0.12 0.08
"""

import argparse
import random
import shutil
from pathlib import Path


def parse_args():
    parser = argparse.ArgumentParser(description="Split dataset into train/val/test")
    parser.add_argument("--images",  type=str, required=True, help="Raw images folder")
    parser.add_argument("--labels",  type=str, required=True, help="Raw labels folder")
    parser.add_argument("--out",     type=str, default="data",  help="Output root directory")
    parser.add_argument("--ratios",  type=float, nargs=3, default=[0.80, 0.12, 0.08],
                        metavar=("TRAIN", "VAL", "TEST"))
    parser.add_argument("--seed",    type=int, default=42)
    return parser.parse_args()


def split(args):
    random.seed(args.seed)
    img_dir = Path(args.images)
    lbl_dir = Path(args.labels)
    out     = Path(args.out)

    assert abs(sum(args.ratios) - 1.0) < 1e-6, "Ratios must sum to 1.0"

    images = sorted(img_dir.glob("*.jpg")) + sorted(img_dir.glob("*.png"))
    random.shuffle(images)
    n = len(images)
    n_train = int(n * args.ratios[0])
    n_val   = int(n * args.ratios[1])

    splits = {
        "train": images[:n_train],
        "val":   images[n_train:n_train + n_val],
        "test":  images[n_train + n_val:],
    }

    for split_name, files in splits.items():
        img_out = out / "images" / split_name
        lbl_out = out / "labels" / split_name
        img_out.mkdir(parents=True, exist_ok=True)
        lbl_out.mkdir(parents=True, exist_ok=True)

        copied = skipped = 0
        for img_path in files:
            lbl_path = lbl_dir / (img_path.stem + ".txt")
            if not lbl_path.exists():
                print(f"  ⚠ Missing label for {img_path.name}, skipping.")
                skipped += 1
                continue
            shutil.copy2(img_path, img_out / img_path.name)
            shutil.copy2(lbl_path, lbl_out / lbl_path.name)
            copied += 1

        print(f"  {split_name:6s}: {copied:4d} images copied, {skipped} skipped")

    print(f"\n✅ Dataset split complete → {out}/")
    print(f"   Train : {len(splits['train'])} | Val : {len(splits['val'])} | Test : {len(splits['test'])}")


if __name__ == "__main__":
    args = parse_args()
    split(args)
