"""Dataset class-balancer for robotics navigation object detection.

Solves:
1. Training class imbalance (16:1 ratio between stop signs and barriers).
2. Validation class imbalance (1,363 stop signs vs only 50 barriers).

Creates a balanced training set (~1,500 per class) AND a balanced validation set (~200 per class).
"""

import os
import shutil
import random
from typing import Dict, List, Tuple, Set
from collections import defaultdict


def balance_yolo_dataset(
    dataset_dir: str = "data/dataset",
    output_dir: str = "data/dataset_balanced",
    target_train_per_class: int = 1500,
    target_val_per_class: int = 200,
    barrier_oversample_factor: int = 3,
) -> None:
    """Create a balanced dataset with equal class representation in BOTH train and val."""
    random.seed(42)

    in_img_train = os.path.join(dataset_dir, "images", "train")
    in_lbl_train = os.path.join(dataset_dir, "labels", "train")
    in_img_val = os.path.join(dataset_dir, "images", "val")
    in_lbl_val = os.path.join(dataset_dir, "labels", "val")

    out_img_train = os.path.join(output_dir, "images", "train")
    out_lbl_train = os.path.join(output_dir, "labels", "train")
    out_img_val = os.path.join(output_dir, "images", "val")
    out_lbl_val = os.path.join(output_dir, "labels", "val")

    if os.path.exists(output_dir):
        shutil.rmtree(output_dir)

    for p in [out_img_train, out_lbl_train, out_img_val, out_lbl_val]:
        os.makedirs(p, exist_ok=True)

    print("=" * 65)
    print("BALANCING BOTH TRAIN AND VALIDATION SETS")
    print("=" * 65)

    def index_files(lbl_dir: str, img_dir: str) -> Dict[int, List[Tuple[str, str]]]:
        """Categorize image/label pairs by class."""
        class_to_files = defaultdict(list)
        label_files = [f for f in os.listdir(lbl_dir) if f.endswith(".txt")]

        for lf in label_files:
            lp = os.path.join(lbl_dir, lf)
            base_name = os.path.splitext(lf)[0]

            img_name = None
            for ext in [".jpg", ".jpeg", ".png", ".bmp", ".webp"]:
                candidate = f"{base_name}{ext}"
                if os.path.exists(os.path.join(img_dir, candidate)):
                    img_name = candidate
                    break

            if not img_name:
                continue

            classes_in_file: Set[int] = set()
            with open(lp, "r") as f:
                for line in f:
                    parts = line.strip().split()
                    if parts:
                        try:
                            classes_in_file.add(int(parts[0]))
                        except ValueError:
                            pass

            if 1 in classes_in_file:
                class_to_files[1].append((img_name, lf))
            elif 0 in classes_in_file:
                class_to_files[0].append((img_name, lf))
            elif 2 in classes_in_file:
                class_to_files[2].append((img_name, lf))

        return class_to_files

    train_files = index_files(in_lbl_train, in_img_train)
    val_files = index_files(in_lbl_val, in_img_val)

    print(f"Original Raw Counts:")
    print(f"  Train: Cone={len(train_files[0])}, Barrier={len(train_files[1])}, StopSign={len(train_files[2])}")
    print(f"  Val:   Cone={len(val_files[0])}, Barrier={len(val_files[1])}, StopSign={len(val_files[2])}")

    # ==========================================
    # 1. BALANCE VALIDATION SET
    # ==========================================
    # Cones in val: keep all (~212)
    val_cones = val_files[0]

    # Barriers in val: original only had 50!
    # Take 100 unique barrier images from original train pool and add them to val:
    random.shuffle(train_files[1])
    extra_val_barriers = train_files[1][:100]
    remaining_train_barriers = train_files[1][100:]

    val_barriers = val_files[1] + extra_val_barriers

    # Stop Signs in val: subsample from 1,363 to ~250
    val_stops = random.sample(val_files[2], min(target_val_per_class + 50, len(val_files[2])))

    print(f"\nBalanced Validation Plan:")
    print(f"  - Cones:      {len(val_cones)} images")
    print(f"  - Barriers:   {len(val_barriers)} images (50 original + 100 transferred from train pool)")
    print(f"  - Stop Signs: {len(val_stops)} images (subsampled from {len(val_files[2])})")

    # ==========================================
    # 2. BALANCE TRAINING SET
    # ==========================================
    # Cones in train: ~1,470
    train_cones = train_files[0][:target_train_per_class]

    # Stop signs in train: subsample to target_train_per_class (~1,500)
    train_stops = random.sample(train_files[2], min(target_train_per_class, len(train_files[2])))

    # Barriers in train: remaining unique barriers (~460) duplicated 3x to reach ~1,380
    train_barriers = []
    for rep in range(barrier_oversample_factor):
        for img_name, lbl_name in remaining_train_barriers:
            train_barriers.append((img_name, lbl_name, rep))

    print(f"\nBalanced Training Plan:")
    print(f"  - Cones:      {len(train_cones)} images")
    print(f"  - Barriers:   {len(train_barriers)} images ({len(remaining_train_barriers)} unique x {barrier_oversample_factor})")
    print(f"  - Stop Signs: {len(train_stops)} images (subsampled from {len(train_files[2])})")

    # ==========================================
    # 3. WRITE TO DESTINATION
    # ==========================================
    def copy_pair(src_img_dir, src_lbl_dir, img_name, lbl_name, dst_img_dir, dst_lbl_dir, prefix=""):
        dst_img_name = f"{prefix}{img_name}" if prefix else img_name
        dst_lbl_name = f"{prefix}{lbl_name}" if prefix else lbl_name
        shutil.copy2(os.path.join(src_img_dir, img_name), os.path.join(dst_img_dir, dst_img_name))
        shutil.copy2(os.path.join(src_lbl_dir, lbl_name), os.path.join(dst_lbl_dir, dst_lbl_name))

    # Write training files
    for img, lbl in train_cones:
        copy_pair(in_img_train, in_lbl_train, img, lbl, out_img_train, out_lbl_train)

    for img, lbl in train_stops:
        copy_pair(in_img_train, in_lbl_train, img, lbl, out_img_train, out_lbl_train)

    for img, lbl, rep in train_barriers:
        pfx = f"rep{rep}_" if rep > 0 else ""
        copy_pair(in_img_train, in_lbl_train, img, lbl, out_img_train, out_lbl_train, prefix=pfx)

    # Write validation files
    for img, lbl in val_cones:
        copy_pair(in_img_val, in_lbl_val, img, lbl, out_img_val, out_lbl_val)

    for img, lbl in val_stops:
        copy_pair(in_img_val, in_lbl_val, img, lbl, out_img_val, out_lbl_val)

    # Base val barriers (from in_img_val)
    for img, lbl in val_files[1]:
        copy_pair(in_img_val, in_lbl_val, img, lbl, out_img_val, out_lbl_val)

    # Extra val barriers (transferred from in_img_train)
    for img, lbl in extra_val_barriers:
        copy_pair(in_img_train, in_lbl_train, img, lbl, out_img_val, out_lbl_val, prefix="extraval_")

    # Write dataset.yaml
    yaml_path = os.path.join(output_dir, "dataset.yaml")
    abs_out = os.path.abspath(output_dir).replace("\\", "/")
    content = f"""path: {abs_out}
train: images/train
val: images/val
names:
  0: cone
  1: barrier
  2: stop_sign
"""
    with open(yaml_path, "w") as f:
        f.write(content)

    print(f"\n[DONE] Balanced dataset created in: {output_dir}")
    print(f"[YAML] Saved dataset configuration to: {yaml_path}")

    # Inspect distribution
    from src.dataset.dataset_manager import DatasetManager
    dm = DatasetManager(output_dir)
    dm.inspect_and_summarize()


def main():
    balance_yolo_dataset(
        dataset_dir="data/dataset",
        output_dir="data/dataset_balanced",
        target_train_per_class=1500,
        target_val_per_class=200,
        barrier_oversample_factor=3,
    )


if __name__ == "__main__":
    main()
