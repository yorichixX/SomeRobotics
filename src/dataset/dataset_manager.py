"""Dataset manager for robotics navigation object detection.

Supports:
1. Converting BDD100K JSON annotations to YOLO format.
2. Generating realistic synthetic navigation datasets for verification and offline training.
3. Writing YOLO-compatible dataset.yaml files.
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

    def generate_synthetic_dataset(
        self,
        num_train: int = 40,
        num_val: int = 10,
        image_width: int = 1280,
        image_height: int = 720,
    ) -> None:
        """Generate high-quality synthetic road scenes with cones, barriers, and stop signs.
        
        This enables immediate training and verification on any machine without downloading 100GB.
        """
        self.create_directory_structure()

        total = num_train + num_val
        for i in range(total):
            split = "train" if i < num_train else "val"
            img, annotations = self._render_synthetic_scene(
                width=image_width, height=image_height, seed=1000 + i
            )

            img_name = f"synth_{i:04d}.jpg"
            lbl_name = f"synth_{i:04d}.txt"

            dest_img = os.path.join(self.images_train if split == "train" else self.images_val, img_name)
            dest_lbl = os.path.join(self.labels_train if split == "train" else self.labels_val, lbl_name)

            cv2.imwrite(dest_img, img)

            lines = []
            for class_id, xc, yc, bw, bh in annotations:
                lines.append(f"{class_id} {xc:.6f} {yc:.6f} {bw:.6f} {bh:.6f}")

            with open(dest_lbl, "w") as f:
                f.write("\n".join(lines) + "\n")

        self.write_dataset_yaml()

    def _render_synthetic_scene(
        self, width: int = 1280, height: int = 720, seed: int = 42
    ) -> Tuple[np.ndarray, List[Tuple[int, float, float, float, float]]]:
        """Render a synthetic road scene containing road perspective and target objects."""
        random.seed(seed)
        np.random.seed(seed)

        # 1. Sky and ground gradient
        img = np.zeros((height, width, 3), dtype=np.uint8)
        horizon_y = int(height * 0.45)

        # Sky gradient (light blue to haze)
        for y in range(horizon_y):
            ratio = y / horizon_y
            b = int(220 + 20 * ratio)
            g = int(180 + 30 * ratio)
            r = int(140 + 30 * ratio)
            img[y, :] = [b, g, r]

        # Asphalt road ground (dark gray)
        for y in range(horizon_y, height):
            ratio = (y - horizon_y) / (height - horizon_y)
            gray = int(50 + 25 * ratio)
            img[y, :] = [gray, gray, gray]

        # Road lanes / perspective trapezoid
        road_pts = np.array(
            [
                [int(width * 0.42), horizon_y],
                [int(width * 0.58), horizon_y],
                [int(width * 0.95), height],
                [int(width * 0.05), height],
            ],
            np.int32,
        )
        cv2.fillPoly(img, [road_pts], (75, 75, 75))

        # Lane markings
        cv2.line(
            img,
            (int(width * 0.5), horizon_y),
            (int(width * 0.5), height),
            (240, 240, 240),
            4,
            cv2.LINE_AA,
        )

        annotations = []
        num_objects = random.randint(1, 4)

        # Generate non-overlapping objects on the road
        for _ in range(num_objects):
            target_type = random.choice(self.CLASSES)
            class_id = self.CLASS_TO_ID[target_type]

            # Determine distance / perspective position
            # y position in bottom half corresponds to perspective depth
            y_base = random.randint(int(height * 0.55), int(height * 0.92))
            depth_ratio = (y_base - horizon_y) / (height - horizon_y)

            # Object size scales with perspective depth
            if target_type == "cone":
                obj_h = int(130 * depth_ratio)
                obj_w = int(obj_h * 0.55)
                # Keep cone on the road
                x_center = random.randint(int(width * 0.25), int(width * 0.75))
                self._draw_cone(img, x_center, y_base, obj_w, obj_h)
            elif target_type == "stop_sign":
                obj_h = int(120 * depth_ratio)
                obj_w = int(obj_h * 0.95)
                # Stop signs usually on road sides
                x_center = random.choice(
                    [
                        random.randint(int(width * 0.12), int(width * 0.28)),
                        random.randint(int(width * 0.72), int(width * 0.88)),
                    ]
                )
                self._draw_stop_sign(img, x_center, y_base, obj_w, obj_h)
            else:  # barrier
                obj_h = int(110 * depth_ratio)
                obj_w = int(obj_h * 1.6)
                x_center = random.randint(int(width * 0.20), int(width * 0.80))
                self._draw_barrier(img, x_center, y_base, obj_w, obj_h)

            x1 = max(0, x_center - obj_w // 2)
            y1 = max(0, y_base - obj_h)
            x2 = min(width - 1, x_center + obj_w // 2)
            y2 = min(height - 1, y_base)

            # Normalized YOLO coordinates
            bw = (x2 - x1) / width
            bh = (y2 - y1) / height
            xc = ((x1 + x2) / 2.0) / width
            yc = ((y1 + y2) / 2.0) / height

            if bw > 0.01 and bh > 0.01:
                annotations.append((class_id, xc, yc, bw, bh))

        return img, annotations

    def _draw_cone(self, img: np.ndarray, xc: int, y_base: int, w: int, h: int) -> None:
        """Draw an orange traffic cone with reflective white stripe."""
        pts = np.array(
            [[xc, y_base - h], [xc - w // 2, y_base], [xc + w // 2, y_base]], np.int32
        )
        # Orange body (BGR: 0, 128, 255)
        cv2.fillPoly(img, [pts], (0, 130, 245))

        # Reflective white stripe across middle
        stripe_top = y_base - int(h * 0.55)
        stripe_bot = y_base - int(h * 0.35)
        w_top = int(w * 0.35)
        w_bot = int(w * 0.55)
        stripe_pts = np.array(
            [
                [xc - w_top // 2, stripe_top],
                [xc + w_top // 2, stripe_top],
                [xc + w_bot // 2, stripe_bot],
                [xc - w_bot // 2, stripe_bot],
            ],
            np.int32,
        )
        cv2.fillPoly(img, [stripe_pts], (245, 245, 245))

        # Black base
        cv2.rectangle(
            img,
            (xc - int(w * 0.6), y_base - int(h * 0.08)),
            (xc + int(w * 0.6), y_base),
            (30, 30, 30),
            -1,
        )

    def _draw_stop_sign(self, img: np.ndarray, xc: int, y_base: int, w: int, h: int) -> None:
        """Draw an octagonal red stop sign with post and white STOP text."""
        sign_h = int(h * 0.65)
        post_h = h - sign_h
        sign_center_y = y_base - post_h - sign_h // 2

        # Silver pole
        cv2.rectangle(
            img,
            (xc - 3, y_base - post_h),
            (xc + 3, y_base),
            (180, 180, 180),
            -1,
        )

        # Regular Octagon
        r = sign_h // 2
        pts = []
        for angle_deg in range(22, 382, 45):
            rad = np.radians(angle_deg)
            px = int(xc + r * np.cos(rad))
            py = int(sign_center_y + r * np.sin(rad))
            pts.append([px, py])
        pts_np = np.array(pts, np.int32)

        # Red octagon fill
        cv2.fillPoly(img, [pts_np], (30, 30, 210))
        # White border
        cv2.polylines(img, [pts_np], isClosed=True, color=(250, 250, 250), thickness=2)

        # Text "STOP"
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = max(0.3, r / 40.0)
        cv2.putText(
            img,
            "STOP",
            (xc - int(r * 0.65), sign_center_y + int(r * 0.3)),
            font,
            font_scale,
            (255, 255, 255),
            max(1, int(font_scale * 2)),
            cv2.LINE_AA,
        )

    def _draw_barrier(self, img: np.ndarray, xc: int, y_base: int, w: int, h: int) -> None:
        """Draw a traffic barrier (Jersey / safety barrier)."""
        x1, y1 = xc - w // 2, y_base - h
        x2, y2 = xc + w // 2, y_base

        # Barrier body (striped orange and white or concrete)
        cv2.rectangle(img, (x1, y1), (x2, y2), (230, 230, 230), -1)

        # Diagonal safety stripes
        stripe_width = max(8, w // 8)
        for sx in range(x1 - h, x2 + h, stripe_width * 2):
            pts = np.array(
                [
                    [sx, y1],
                    [min(x2, sx + stripe_width), y1],
                    [min(x2, sx + stripe_width + h), y2],
                    [max(x1, sx + h), y2],
                ],
                np.int32,
            )
            # Clip pts to barrier rectangle
            pts[:, 0] = np.clip(pts[:, 0], x1, x2)
            cv2.fillPoly(img, [pts], (20, 110, 240))

        # Dark border
        cv2.rectangle(img, (x1, y1), (x2, y2), (40, 40, 40), 2)

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

