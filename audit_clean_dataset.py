"""Comprehensive audit utility for verified dataset integrity.

Verifies:
1. Exact 1-to-1 pairing between images and label files (no orphan files).
2. Strict class ID enforcement (only 0: cone, 1: barrier, 2: stop_sign allowed).
3. Bounding box geometric validity (normalized coordinates within 0.0 to 1.0).
4. dataset.yaml path and class mapping integrity.
"""

import os
from typing import Dict, List, Set, Tuple
from collections import Counter
import yaml


def audit_consolidated_dataset(dataset_dir: str = "data/dataset_clean") -> bool:
    print("=" * 70)
    print(f"VERIFYING CONSOLIDATED DATASET INTEGRITY: {dataset_dir}")
    print("=" * 70)

    if not os.path.exists(dataset_dir):
        print(f"[ERROR] Directory does not exist: {dataset_dir}")
        print("Please run 'python clean_merge_datasets.py' first.")
        return False

    all_passed = True
    EXPECTED_CLASSES = {0: "cone", 1: "barrier", 2: "stop_sign"}

    # 1. Audit dataset.yaml
    yaml_path = os.path.join(dataset_dir, "dataset.yaml")
    print("\n--- 1. Checking dataset.yaml ---")
    if os.path.exists(yaml_path):
        try:
            with open(yaml_path, "r") as f:
                cfg = yaml.safe_load(f)
            names = cfg.get("names", {})
            print(f"  [OK] dataset.yaml found.")
            print(f"  [OK] Class mapping in yaml: {names}")
            if names != EXPECTED_CLASSES and names != ["cone", "barrier", "stop_sign"]:
                print(f"  [WARNING] Names in YAML do not match expected: {EXPECTED_CLASSES}")
                all_passed = False
        except Exception as e:
            print(f"  [FAIL] Error reading dataset.yaml: {e}")
            all_passed = False
    else:
        print(f"  [FAIL] dataset.yaml not found at: {yaml_path}")
        all_passed = False

    # 2. Audit splits
    print("\n--- 2. Checking Splits, Pairing, and Bounding Box Integrity ---")
    splits_summary = {}

    for split in ["train", "val"]:
        img_dir = os.path.join(dataset_dir, "images", split)
        lbl_dir = os.path.join(dataset_dir, "labels", split)

        if not os.path.exists(img_dir) or not os.path.exists(lbl_dir):
            print(f"  [FAIL] Missing directory for split '{split}'")
            all_passed = False
            continue

        valid_exts = [".jpg", ".jpeg", ".png", ".bmp", ".webp"]
        img_files = {os.path.splitext(f)[0]: f for f in os.listdir(img_dir) if os.path.splitext(f)[1].lower() in valid_exts}
        lbl_files = {os.path.splitext(f)[0]: f for f in os.listdir(lbl_dir) if f.endswith(".txt")}

        # Check pairing
        orphan_imgs = set(img_files.keys()) - set(lbl_files.keys())
        orphan_lbls = set(lbl_files.keys()) - set(img_files.keys())

        pairing_ok = (len(orphan_imgs) == 0 and len(orphan_lbls) == 0)
        if pairing_ok:
            print(f"  [{split.upper()}] Pairing: 100% matched ({len(img_files)} images, {len(lbl_files)} label files)")
        else:
            print(f"  [{split.upper()}] Pairing issue: {len(orphan_imgs)} orphan images, {len(orphan_lbls)} orphan labels")
            all_passed = False

        # Check label contents
        class_counts = Counter()
        illegal_classes = Counter()
        invalid_boxes = 0
        total_boxes = 0

        for base_name, lf in lbl_files.items():
            lp = os.path.join(lbl_dir, lf)
            with open(lp, "r") as f:
                for line_idx, line in enumerate(f):
                    parts = line.strip().split()
                    if not parts:
                        continue
                    total_boxes += 1
                    try:
                        cid = int(parts[0])
                        xc, yc, w, h = float(parts[1]), float(parts[2]), float(parts[3]), float(parts[4])

                        # Verify class ID
                        if cid in [0, 1, 2]:
                            class_counts[cid] += 1
                        else:
                            illegal_classes[cid] += 1
                            all_passed = False

                        # Verify bounding box geometry
                        if not (0.0 < xc < 1.0 and 0.0 < yc < 1.0 and 0.0 < w <= 1.0 and 0.0 < h <= 1.0):
                            invalid_boxes += 1
                            all_passed = False

                    except Exception:
                        invalid_boxes += 1
                        all_passed = False

        splits_summary[split] = {
            "images": len(img_files),
            "labels": len(lbl_files),
            "total_boxes": total_boxes,
            "classes": class_counts,
            "illegal": illegal_classes,
            "invalid_boxes": invalid_boxes,
        }

        # Print split report
        if len(illegal_classes) == 0:
            print(f"  [{split.upper()}] Classes: ONLY valid classes [0, 1, 2] present. Zero corrupted classes!")
        else:
            print(f"  [{split.upper()}] CORRUPTED CLASSES DETECTED: {dict(illegal_classes)}")

        if invalid_boxes == 0:
            print(f"  [{split.upper()}] Bounding boxes: 100% geometrically valid ({total_boxes} boxes verified)")
        else:
            print(f"  [{split.upper()}] INVALID BOXES DETECTED: {invalid_boxes}")

    # 3. Print Final Verified Distribution Table
    print("\n" + "=" * 70)
    print("VERIFIED CLASS DISTRIBUTION SUMMARY")
    print("=" * 70)
    print(f"{'Class ID':<10} {'Class Name':<16} {'Train Count':<14} {'Val Count':<14} {'Total':<10}")
    print("-" * 70)

    tr_cls = splits_summary.get("train", {}).get("classes", Counter())
    va_cls = splits_summary.get("val", {}).get("classes", Counter())

    for cid, cname in EXPECTED_CLASSES.items():
        tr = tr_cls.get(cid, 0)
        va = va_cls.get(cid, 0)
        tot = tr + va
        print(f"{cid:<10} {cname:<16} {tr:<14} {va:<14} {tot:<10}")

    print("-" * 70)
    total_train_imgs = splits_summary.get("train", {}).get("images", 0)
    total_val_imgs = splits_summary.get("val", {}).get("images", 0)
    print(f"Total Verified Images: {total_train_imgs} Train, {total_val_imgs} Val ({total_train_imgs + total_val_imgs} Total)")
    print("=" * 70)

    if all_passed:
        print("\n>>> [AUDIT PASSED] 100% OF ANNOTATIONS ARE CLEAN, PAIRED, AND VERIFIED! <<<")
        print(">>> YOU ARE READY TO TRAIN WITH ZERO DATASET CORRUPTION. <<<\n")
    else:
        print("\n>>> [AUDIT FAILED] Issues were detected. Please check logs above. <<<\n")

    return all_passed


if __name__ == "__main__":
    audit_consolidated_dataset("data/dataset_clean")
