import argparse
from pathlib import Path


def parse_args():
    parser = argparse.ArgumentParser(description="Verify YOLO annotation integrity")
    parser.add_argument("--data", type=str, default="data")
    parser.add_argument("--nc",   type=int, default=5, help="Number of classes")
    return parser.parse_args()


def check_split(img_dir, lbl_dir, nc, split_name):
    img_files = set(p.stem for p in img_dir.glob("*.jpg")) | \
                set(p.stem for p in img_dir.glob("*.png"))
    lbl_files = set(p.stem for p in lbl_dir.glob("*.txt"))

    missing_labels = img_files - lbl_files
    orphan_labels  = lbl_files - img_files

    errors = 0
    coord_errors = 0

    for lbl_path in lbl_dir.glob("*.txt"):
        with open(lbl_path) as f:
            lines = f.readlines()
        for i, line in enumerate(lines):
            parts = line.strip().split()
            if len(parts) != 5:
                print(f"  ❌ {lbl_path.name} line {i+1}: expected 5 values, got {len(parts)}")
                errors += 1
                continue
            cls_id = int(parts[0])
            coords = list(map(float, parts[1:]))
            if cls_id < 0 or cls_id >= nc:
                print(f"  ❌ {lbl_path.name} line {i+1}: invalid class id {cls_id}")
                errors += 1
            if any(v < 0.0 or v > 1.0 for v in coords):
                print(f"  ❌ {lbl_path.name} line {i+1}: coord out of [0,1]: {coords}")
                coord_errors += 1

    status = "✅" if not (missing_labels or orphan_labels or errors or coord_errors) else "⚠"
    print(f"\n{status} [{split_name}]")
    print(f"   Images          : {len(img_files)}")
    print(f"   Labels          : {len(lbl_files)}")
    print(f"   Missing labels  : {len(missing_labels)}")
    print(f"   Orphan labels   : {len(orphan_labels)}")
    print(f"   Format errors   : {errors}")
    print(f"   Coord errors    : {coord_errors}")

    if missing_labels:
        for s in sorted(missing_labels)[:5]:
            print(f"   - no label for: {s}")
        if len(missing_labels) > 5:
            print(f"   ... and {len(missing_labels)-5} more")


def run_check(args):
    data = Path(args.data)
    print(f"\nAnnotation check — {data.resolve()}\n" + "="*45)
    for split in ("train", "val", "test"):
        img_dir = data / "images" / split
        lbl_dir = data / "labels" / split
        if img_dir.exists():
            check_split(img_dir, lbl_dir, args.nc, split)
        else:
            print(f"\n  [{split}] directory not found, skipping.")
    print()


if __name__ == "__main__":
    args = parse_args()
    run_check(args)
