"""Audits the original Roboflow datasets to check class definitions and label mappings.

Inspects data.yaml and label files in each folder to detect if label corruption occurred.
"""

import os
import yaml
from collections import Counter


def audit_datasets(data_root: str = "data"):
    print("=" * 65)
    print("AUDITING RAW ROBOFLOW DATASETS: DATA.YAML & LABELS")
    print("=" * 65)

    candidates = [
        d for d in os.listdir(data_root)
        if os.path.isdir(os.path.join(data_root, d))
        and d not in ["dataset", "dataset_balanced", "dataset_clean", "output", "sample_videos", "sample_images", "bdd100k"]
    ]

    for folder in sorted(candidates):
        folder_path = os.path.join(data_root, folder)
        yaml_path = os.path.join(folder_path, "data.yaml")

        print(f"\n📂 Dataset Folder: {folder}")
        print("-" * 55)

        # 1. Inspect data.yaml
        if os.path.exists(yaml_path):
            try:
                with open(yaml_path, "r") as f:
                    content = yaml.safe_load(f)
                nc = content.get("nc", "Unknown")
                names = content.get("names", [])
                print(f"  [data.yaml] Number of classes (nc): {nc}")
                print(f"  [data.yaml] Class names list:       {names}")
            except Exception as e:
                print(f"  [data.yaml] Could not parse YAML: {e}")
        else:
            print("  [data.yaml] NOT FOUND!")

        # 2. Inspect label files in train/labels or train/
        possible_dirs = [
            os.path.join(folder_path, "train", "labels"),
            os.path.join(folder_path, "train"),
            os.path.join(folder_path, "training", "labels"),
            os.path.join(folder_path, "valid", "labels"),
        ]

        lbl_dir = None
        for pd in possible_dirs:
            if os.path.exists(pd):
                txt_files = [f for f in os.listdir(pd) if f.endswith(".txt")]
                if txt_files:
                    lbl_dir = pd
                    break

        if lbl_dir:
            class_counter = Counter()
            txt_files = [f for f in os.listdir(lbl_dir) if f.endswith(".txt")]
            for tf in txt_files:
                with open(os.path.join(lbl_dir, tf), "r") as f:
                    for line in f:
                        parts = line.strip().split()
                        if parts:
                            try:
                                class_counter[int(parts[0])] += 1
                            except ValueError:
                                pass

            print(f"  [Label Files in {os.path.basename(lbl_dir)}] Total files: {len(txt_files)}")
            print(f"  [Original Class ID Counts]:")
            for cid, count in sorted(class_counter.items()):
                # Match with names if available
                cname = ""
                if isinstance(names, list) and 0 <= cid < len(names):
                    cname = f"('{names[cid]}')"
                elif isinstance(names, dict) and cid in names:
                    cname = f"('{names[cid]}')"
                print(f"     Class ID {cid} {cname}: {count} bounding boxes")
        else:
            print("  [Label Files] No .txt labels found.")

    print("\n" + "=" * 65)
    print("AUDIT COMPLETE")
    print("=" * 65)


if __name__ == "__main__":
    audit_datasets("data")
