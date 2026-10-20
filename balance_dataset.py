"""Dataset class-balancer for robotics navigation object detection.

Solves the 16:1 class imbalance between stop signs (8,904) and barriers (560).
Creates a balanced dataset with ~1,500 images per class for optimal mAP across all classes.
"""

import os
import shutil
import random
from typing import Dict, List, Tuple
from collections import defaultdict


def balance_yolo_dataset(
    dataset_dir: str = "data/dataset",
    output_dir: str = "data/dataset_balanced",
    target_samples_per_class: int = 1500,
    barrier_oversample_factor: int = 3,
) -> None:
    """Create a balanced dataset with equal class representation."""
    random.seed(42)

    in_img_train = os.path.join(dataset_dir, "images", "train")
    in_lbl_train = os.path.join(dataset_dir, "labels", "train")
    in_img_val = os.path.join(dataset_dir, "images", "val")
    in_lbl_val = os.path.join(dataset_dir, "labels", "val")

    out_img_train = os.path.join(output_dir, "images", "train")
    out_lbl_train = os.path.join(output_dir, "labels", "train")
    out_img_val = os.path.join(output_dir, "images", "val")
    out_lbl_val = os.path.join(output_dir, "labels", "val")

    for p in [out_img_train, out_lbl_train, out_img_val, out_lbl_val]:
        os.makedirs(p, exist_ok=True)

    print("=" * 65)
    print("BALANCING DATASET CLASSES (CONE, BARRIER, STOP_SIGN)")
    print("=" * 65)

    # 1. Categorize training files by dominant class
    class_to_files = defaultdict(list)
    label_files = [f for f in os.listdir(in_lbl_train) if f.endswith(".txt")]

    for lf in label_files:
        lp = os.path.join(in_lbl_train, lf)
        base_name = os.path.splitext(lf)[0]

        # Find matching image
        img_name = None
        for ext in [".jpg", ".jpeg", ".png", ".bmp", ".webp"]:
            candidate = f"{base_name}{ext}"
            if os.path.exists(os.path.join(in_img_train, candidate)):
                img_name = candidate
                break

        if not img_name:
            continue

        # Inspect classes in label file
        classes_in_file = set()
        with open(lp, "r") as f:
            for line in f:
                parts = line.strip().split()
                if parts:
                    try:
                        classes_in_file.add(int(parts[0]))
                    except ValueError:
                        pass

        # Index file by class (priority: barrier > cone > stop_sign)
        if 1 in classes_in_file:
            class_to_files[1].append((img_name, lf))
        elif 0 in classes_in_file:
            class_to_files[0].append((img_name, lf))
        elif 2 in classes_in_file:
            class_to_files[2].append((img_name, lf))

    print(f"Original Training Files Indexed:")
    print(f"  - Class 0 (Cone):      {len(class_to_files[0])} files")
    print(f"  - Class 1 (Barrier):   {len(class_to_files[1])} files")
    print(f"  - Class 2 (Stop Sign): {len(class_to_files[2])} files")

    # 2. Select balanced subset
    # Cones: take all (~1,470)
    selected_cones = class_to_files[0][:target_samples_per_class]

    # Stop Signs: subsample to target_samples_per_class (~1,500 instead of 8,900)
    selected_stops = random.sample(class_to_files[2], min(target_samples_per_class, len(class_to_files[2])))

    # Barriers: take all, then duplicate with copy prefix to reach target
    selected_barriers = []
    base_barriers = class_to_files[1]
    for rep in range(barrier_oversample_factor):
        for img_name, lbl_name in base_barriers:
            selected_barriers.append((img_name, lbl_name, rep))

    print(f"\nBalanced Training Plan:")
    print(f"  - Cones:      {len(selected_cones)} images")
    print(f"  - Barriers:   {len(selected_barriers)} images ({len(base_barriers)} unique x {barrier_oversample_factor})")
    print(f"  - Stop Signs: {len(selected_stops)} images (subsampled from {len(class_to_files[2])})")

    # 3. Copy to destination
    def copy_file_pair(src_img_name, src_lbl_name, dst_img_dir, dst_lbl_dir, prefix=""):
        base_name, ext = os.path.splitext(src_img_name)
        dst_img_name = f"{prefix}{src_img_name}" if prefix else src_img_name
        dst_lbl_name = f"{prefix}{src_lbl_name}" if prefix else src_lbl_name

        shutil.copy2(os.path.join(in_img_train, src_img_name), os.path.join(dst_img_dir, dst_img_name))
        shutil.copy2(os.path.join(in_lbl_train, src_lbl_name), os.path.join(dst_lbl_dir, dst_lbl_name))

    for img, lbl in selected_cones:
        copy_file_pair(img, lbl, out_img_train, out_lbl_train)

    for img, lbl in selected_stops:
        copy_file_pair(img, lbl, out_img_train, out_lbl_train)

    for img, lbl, rep in selected_barriers:
        pfx = f"rep{rep}_" if rep > 0 else ""
        copy_file_pair(img, lbl, out_img_train, out_lbl_train, prefix=pfx)

    # 4. Copy validation set (preserve complete ground truth)
    val_imgs = os.listdir(in_img_val)
    for vf in val_imgs:
        shutil.copy2(os.path.join(in_img_val, vf), os.path.join(out_img_val, vf))
    val_lbls = os.listdir(in_lbl_val)
    for vl in val_lbls:
        shutil.copy2(os.path.join(in_lbl_val, vl), os.path.join(out_lbl_val, vl))

    # 5. Write dataset.yaml
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
        target_samples_per_class=1500,
        barrier_oversample_factor=3,
    )


if __name__ == "__main__":
    main()
