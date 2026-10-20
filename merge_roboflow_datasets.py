"""Merges separate Roboflow datasets into a unified 3-class YOLO dataset for robotics navigation.

Classes:
  0: cone
  1: barrier
  2: stop_sign
"""

import os
import shutil
import glob
from typing import Dict, List, Tuple, Optional
import yaml


class DatasetMerger:
    """Merges multiple individual Roboflow YOLOv8 datasets into a unified dataset."""

    UNIFIED_CLASSES = ["cone", "barrier", "stop_sign"]

    def __init__(
        self,
        data_root: str = "data",
        output_dir: str = "data/dataset",
    ):
        self.data_root = data_root
        self.output_dir = output_dir

        self.out_images_train = os.path.join(output_dir, "images", "train")
        self.out_images_val = os.path.join(output_dir, "images", "val")
        self.out_labels_train = os.path.join(output_dir, "labels", "train")
        self.out_labels_val = os.path.join(output_dir, "labels", "val")

    def _prepare_output_dirs(self) -> None:
        """Create fresh output directories."""
        for p in [self.out_images_train, self.out_images_val, self.out_labels_train, self.out_labels_val]:
            os.makedirs(p, exist_ok=True)

    def find_dataset_folders(self) -> Dict[str, str]:
        """Automatically detect the 3 Roboflow dataset directories inside data/."""
        candidates = [
            d for d in os.listdir(self.data_root)
            if os.path.isdir(os.path.join(self.data_root, d))
            and d not in ["dataset", "output", "sample_videos", "sample_images", "bdd100k"]
        ]

        detected = {}
        for folder in candidates:
            lower = folder.lower()
            if "cone" in lower:
                detected["cone"] = os.path.join(self.data_root, folder)
            elif "barrier" in lower:
                detected["barrier"] = os.path.join(self.data_root, folder)
            elif "stop" in lower or "sign" in lower:
                detected["stop_sign"] = os.path.join(self.data_root, folder)

        return detected

    def _get_split_dirs(self, dataset_path: str, split: str) -> Tuple[str, str]:
        """Resolve images and labels directories for a given split (train / valid / val)."""
        # Roboflow uses 'valid' or 'val'
        possible_split_names = [split]
        if split == "val":
            possible_split_names.extend(["valid", "validation"])
        elif split == "train":
            possible_split_names.append("training")

        split_dir = None
        for sname in possible_split_names:
            candidate = os.path.join(dataset_path, sname)
            if os.path.exists(candidate):
                split_dir = candidate
                break

        if not split_dir:
            return "", ""

        # Check if structured as split/images and split/labels
        img_dir = os.path.join(split_dir, "images")
        lbl_dir = os.path.join(split_dir, "labels")

        if os.path.exists(img_dir) and os.path.exists(lbl_dir):
            return img_dir, lbl_dir

        # Or files directly inside split directory
        return split_dir, split_dir

    def merge_split(
        self,
        dataset_path: str,
        class_name: str,
        target_class_id: int,
        source_split: str,
        target_split: str,
    ) -> int:
        """Merge a specific split from a source dataset into the unified destination."""
        img_src_dir, lbl_src_dir = self._get_split_dirs(dataset_path, source_split)
        if not img_src_dir or not os.path.exists(img_src_dir):
            print(f"  [SKIP] Split '{source_split}' not found in {os.path.basename(dataset_path)}")
            return 0

        target_img_dir = self.out_images_train if target_split == "train" else self.out_images_val
        target_lbl_dir = self.out_labels_train if target_split == "train" else self.out_labels_val

        # Collect image files
        valid_exts = [".jpg", ".jpeg", ".png", ".bmp", ".webp"]
        img_files = [f for f in os.listdir(img_src_dir) if os.path.splitext(f)[1].lower() in valid_exts]

        count = 0
        prefix = f"{class_name}_{os.path.basename(dataset_path)[:6]}"

        for img_name in img_files:
            base_name, ext = os.path.splitext(img_name)
            label_name = f"{base_name}.txt"
            lbl_path = os.path.join(lbl_src_dir, label_name)

            if not os.path.exists(lbl_path):
                continue

            # Read and remap all bounding box annotations to target_class_id
            remapped_lines = []
            with open(lbl_path, "r") as f:
                for line in f:
                    parts = line.strip().split()
                    if len(parts) >= 5:
                        # Remap to unified target_class_id: class_id xc yc w h
                        remapped_line = f"{target_class_id} {parts[1]} {parts[2]} {parts[3]} {parts[4]}"
                        remapped_lines.append(remapped_line)

            if not remapped_lines:
                continue

            # Copy image with unique prefix to avoid filename collisions
            unique_img_name = f"{prefix}_{img_name}"
            unique_lbl_name = f"{prefix}_{label_name}"

            src_img_file = os.path.join(img_src_dir, img_name)
            dst_img_file = os.path.join(target_img_dir, unique_img_name)
            dst_lbl_file = os.path.join(target_lbl_dir, unique_lbl_name)

            shutil.copy2(src_img_file, dst_img_file)
            with open(dst_lbl_file, "w") as f:
                f.write("\n".join(remapped_lines) + "\n")

            count += 1

        print(f"  Merged {count} images from {os.path.basename(dataset_path)} ({source_split}) -> {target_split}")
        return count

    def write_dataset_yaml(self) -> str:
        """Write standard YOLO dataset.yaml configuration."""
        yaml_path = os.path.join(self.output_dir, "dataset.yaml")
        abs_output_dir = os.path.abspath(self.output_dir).replace("\\", "/")

        data = {
            "path": abs_output_dir,
            "train": "images/train",
            "val": "images/val",
            "names": {
                0: "cone",
                1: "barrier",
                2: "stop_sign",
            },
        }

        with open(yaml_path, "w") as f:
            yaml.dump(data, f, sort_keys=False)

        print(f"\n[YAML] Saved unified dataset config to: {yaml_path}")
        return yaml_path

    def run_merge(self, clean_existing: bool = True) -> None:
        """Run the complete merge process."""
        detected = self.find_dataset_folders()

        print("=" * 65)
        print("ROBOTICS PERCEPTION: MERGING ROBOFLOW DATASETS")
        print("=" * 65)

        if not detected:
            print("[ERROR] No dataset folders found in data/.")
            print("Expected folders containing: 'cone', 'barrier', 'stop'")
            return

        print(f"Detected {len(detected)} datasets:")
        for cname, fpath in detected.items():
            cid = self.UNIFIED_CLASSES.index(cname)
            print(f"  - Class {cid} [{cname.upper()}]: {os.path.basename(fpath)}")

        if clean_existing and os.path.exists(self.output_dir):
            print(f"\nCleaning old dataset directory: {self.output_dir}...")
            shutil.rmtree(self.output_dir)

        self._prepare_output_dirs()

        class_id_map = {
            "cone": 0,
            "barrier": 1,
            "stop_sign": 2,
        }

        total_train = 0
        total_val = 0

        for class_name, dataset_path in detected.items():
            cid = class_id_map[class_name]
            print(f"\nProcessing [{class_name.upper()}] (Target Class ID: {cid})...")

            # Merge train split
            train_count = self.merge_split(dataset_path, class_name, cid, "train", "train")
            total_train += train_count

            # Merge val/valid split
            val_count = self.merge_split(dataset_path, class_name, cid, "val", "val")
            total_val += val_count

        self.write_dataset_yaml()

        # Display distribution summary
        from src.dataset.dataset_manager import DatasetManager
        dm = DatasetManager(self.output_dir)
        dm.inspect_and_summarize()


def main():
    merger = DatasetMerger(data_root="data", output_dir="data/dataset")
    merger.run_merge(clean_existing=True)


if __name__ == "__main__":
    main()
