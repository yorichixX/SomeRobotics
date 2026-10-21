"""Dataset manager for robotics navigation object detection.

Supports:
1. Converting BDD100K JSON annotations to YOLO format.
2. Downloading and formatting Roboflow and custom road perception datasets.
3. Writing YOLO-compatible dataset.yaml files and verifying class distributions.
"""

from typing import Dict, List, Tuple, Optional, Any
import os
import json
import random
import cv2
import numpy as np


class DatasetManager:
    """Manages preparation and formatting of robotics navigation datasets."""

    CLASSES: List[str] = ["cone", "barrier", "stop_sign"]
    CLASS_TO_ID: Dict[str, int] = {name: idx for idx, name in enumerate(CLASSES)}

    def __init__(self, output_dir: str = "data/dataset"):
        self.output_dir = output_dir
        self.images_train = os.path.join(output_dir, "images", "train")
        self.images_val = os.path.join(output_dir, "images", "val")
        self.labels_train = os.path.join(output_dir, "labels", "train")
        self.labels_val = os.path.join(output_dir, "labels", "val")

    def create_directory_structure(self) -> None:
        """Create standard YOLO directory structure."""
        for path in [self.images_train, self.images_val, self.labels_train, self.labels_val]:
            os.makedirs(path, exist_ok=True)

    def write_dataset_yaml(self, yaml_path: Optional[str] = None) -> str:
        """Write YOLO dataset.yaml configuration file."""
        if yaml_path is None:
            yaml_path = os.path.join(self.output_dir, "dataset.yaml")

        data = {
            "path": os.path.abspath(self.output_dir),
            "train": "images/train",
            "val": "images/val",
            "names": {i: name for i, name in enumerate(self.CLASSES)},
        }

        try:
            import yaml
            with open(yaml_path, "w") as f:
                yaml.dump(data, f, sort_keys=False)
        except ImportError:
            # Fallback formatting when PyYAML is not yet installed
            lines = [
                f"path: {os.path.abspath(self.output_dir)}",
                "train: images/train",
                "val: images/val",
                "names:",
            ]
            for i, name in enumerate(self.CLASSES):
                lines.append(f"  {i}: {name}")
            with open(yaml_path, "w") as f:
                f.write("\n".join(lines) + "\n")
        return yaml_path

    def convert_bdd100k_to_yolo(
        self,
        bdd_json_path: str,
        images_src_dir: str,
        split: str = "train",
        category_map: Optional[Dict[str, str]] = None,
        max_samples: Optional[int] = None,
    ) -> int:
        """Convert BDD100K detection labels to YOLO format."""
        import shutil

        if category_map is None:
            category_map = {
                "traffic sign": "stop_sign",
                "barrier": "barrier",
                "cone": "cone",
            }

        target_images = self.images_train if split == "train" else self.images_val
        target_labels = self.labels_train if split == "train" else self.labels_val
        os.makedirs(target_images, exist_ok=True)
        os.makedirs(target_labels, exist_ok=True)

        print(f"[BDD100K] Loading annotations from: {bdd_json_path}")
        with open(bdd_json_path, "r") as f:
            data = json.load(f)

        converted_count = 0
        total_items = len(data)
        print(f"[BDD100K] Parsing {total_items} entries for '{split}' split...")

        for idx, item in enumerate(data):
            if max_samples is not None and converted_count >= max_samples:
                break

            image_name = item.get("name")
            labels = item.get("labels", [])
            if not labels:
                continue

            img_path = os.path.join(images_src_dir, image_name)
            if not os.path.exists(img_path):
                continue

            # Standard BDD100K images are 1280x720
            w, h = 1280.0, 720.0
            yolo_lines = []

            for label in labels:
                cat = label.get("category", "").lower()
                target_cat = category_map.get(cat)
                if not target_cat or target_cat not in self.CLASS_TO_ID:
                    continue

                class_id = self.CLASS_TO_ID[target_cat]
                box = label.get("box2d")
                if not box:
                    continue

                x1, y1 = float(box["x1"]), float(box["y1"])
                x2, y2 = float(box["x2"]), float(box["y2"])

                # Normalize bounding box
                bw = (x2 - x1) / w
                bh = (y2 - y1) / h
                xc = ((x1 + x2) / 2.0) / w
                yc = ((y1 + y2) / 2.0) / h

                if bw > 0.005 and bh > 0.005:
                    yolo_lines.append(f"{class_id} {xc:.6f} {yc:.6f} {bw:.6f} {bh:.6f}")

            if yolo_lines:
                dest_img = os.path.join(target_images, image_name)
                shutil.copy2(img_path, dest_img)

                label_name = os.path.splitext(image_name)[0] + ".txt"
                dest_label = os.path.join(target_labels, label_name)
                with open(dest_label, "w") as lf:
                    lf.write("\n".join(yolo_lines) + "\n")

                converted_count += 1
                if converted_count % 100 == 0:
                    print(f"  Converted {converted_count} labeled road scenes...")

        print(f"[BDD100K] Completed! Extracted {converted_count} images into {target_images}")
        return converted_count

    def download_and_extract_zip(self, url: str, extract_to: Optional[str] = None) -> str:
        """Download a dataset zip archive and extract to target directory."""
        import urllib.request
        import zipfile
        import tempfile

        if extract_to is None:
            extract_to = self.output_dir
        os.makedirs(extract_to, exist_ok=True)

        print(f"[DATASET DOWNLOAD] Downloading from: {url}")
        with tempfile.NamedTemporaryFile(suffix=".zip", delete=False) as tmp_file:
            tmp_path = tmp_file.name

        def progress_hook(block_num, block_size, total_size):
            if total_size > 0:
                percent = min(100.0, (block_num * block_size / total_size) * 100.0)
                print(f"\rDownloading: {percent:.1f}% ({block_num * block_size / (1024*1024):.1f} MB / {total_size / (1024*1024):.1f} MB)", end="")

        urllib.request.urlretrieve(url, tmp_path, reporthook=progress_hook)
        print("\nDownload finished! Extracting archive...")

        with zipfile.ZipFile(tmp_path, "r") as zip_ref:
            zip_ref.extractall(extract_to)

        os.remove(tmp_path)
        self.write_dataset_yaml()
        print(f"[DATASET EXTRACTED] Dataset ready in: {extract_to}")
        return extract_to

    def download_roboflow_dataset(
        self,
        api_key: str,
        workspace: str,
        project: str,
        version: int = 1,
    ) -> str:
        """Download public dataset directly using the Roboflow Python SDK."""
        try:
            from roboflow import Roboflow
            rf = Roboflow(api_key=api_key)
            proj = rf.workspace(workspace).project(project)
            dataset = proj.version(version).download("yolov8", location=self.output_dir)
            self.write_dataset_yaml()
            return self.output_dir
        except ImportError:
            raise ImportError("Please install roboflow: pip install roboflow")

    def inspect_and_summarize(self) -> Dict[str, Any]:
        """Inspect and print class distribution across train and validation splits."""
        stats: Dict[str, Any] = {
            "train_images": 0,
            "val_images": 0,
            "classes": {cls_name: {"train": 0, "val": 0, "total": 0} for cls_name in self.CLASSES},
            "unmapped_classes": 0,
        }

        for split, img_dir, lbl_dir in [
            ("train", self.images_train, self.labels_train),
            ("val", self.images_val, self.labels_val),
        ]:
            if not os.path.exists(lbl_dir):
                continue

            label_files = [f for f in os.listdir(lbl_dir) if f.endswith(".txt")]
            stats[f"{split}_images"] = len(label_files)

            for lf in label_files:
                lp = os.path.join(lbl_dir, lf)
                with open(lp, "r") as f:
                    for line in f:
                        parts = line.strip().split()
                        if not parts:
                            continue
                        try:
                            cid = int(parts[0])
                            if 0 <= cid < len(self.CLASSES):
                                cname = self.CLASSES[cid]
                                stats["classes"][cname][split] += 1
                                stats["classes"][cname]["total"] += 1
                            else:
                                stats["unmapped_classes"] += 1
                        except ValueError:
                            continue

        print("\n" + "=" * 60)
        print("DATASET CLASS DISTRIBUTION SUMMARY")
        print("=" * 60)
        print(f"Train Images: {stats['train_images']} | Val Images: {stats['val_images']}")
        print("-" * 60)
        print(f"{'Class ID':<10} {'Class Name':<15} {'Train':<10} {'Val':<10} {'Total':<10}")
        print("-" * 60)
        for i, cname in enumerate(self.CLASSES):
            tr = stats["classes"][cname]["train"]
            va = stats["classes"][cname]["val"]
            tot = stats["classes"][cname]["total"]
            print(f"{i:<10} {cname:<15} {tr:<10} {va:<10} {tot:<10}")
        print("-" * 60)

        return stats

