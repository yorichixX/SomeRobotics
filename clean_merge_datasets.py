"""Clean and verified multi-dataset merger for robotics navigation.

Fixes the label corruption discovered in Traffic Stop Sign.v2i.yolov8:
- Traffic Stop Sign dataset had 11 classes (including speed signs '3', traffic lights, and Roboflow metadata text).
- Filters ONLY genuine stop signs (classes containing 'stop' or 'Stop Sign') and discards non-stop-sign objects.
- Maps cleanly:
    0: cone
    1: barrier
    2: stop_sign
- Balances training and validation sets so all classes have fair representation.
"""

import os
import shutil
import random
from typing import Dict, List, Tuple, Set
import yaml
from collections import defaultdict


class CleanDatasetMerger:
    def __init__(
        self,
        data_root: str = "data",
        output_dir: str = "data/dataset_clean",
    ):
        self.data_root = data_root
        self.output_dir = output_dir

        self.out_img_train = os.path.join(output_dir, "images", "train")
        self.out_lbl_train = os.path.join(output_dir, "labels", "train")
        self.out_img_val = os.path.join(output_dir, "images", "val")
        self.out_lbl_val = os.path.join(output_dir, "labels", "val")

    def _prepare_dirs(self):
        if os.path.exists(self.output_dir):
            shutil.rmtree(self.output_dir)
        for p in [self.out_img_train, self.out_lbl_train, self.out_img_val, self.out_lbl_val]:
            os.makedirs(p, exist_ok=True)

    def find_folders(self) -> Dict[str, str]:
        candidates = [
            d for d in os.listdir(self.data_root)
            if os.path.isdir(os.path.join(self.data_root, d))
            and d not in ["dataset", "dataset_balanced", "dataset_clean", "output", "sample_videos", "sample_images", "bdd100k"]
        ]
        detected = {}
        for folder in candidates:
            low = folder.lower()
            if "cone" in low:
                detected["cone"] = os.path.join(self.data_root, folder)
            elif "barrier" in low:
                detected["barrier"] = os.path.join(self.data_root, folder)
            elif "stop" in low or "sign" in low:
                detected["stop_sign"] = os.path.join(self.data_root, folder)
        return detected

    def process_cones(self, folder_path: str):
        """Process traffic cones (Class 0). Clean 1-class dataset."""
        print("\n[1/3] Processing Traffic Cones (Class 0)...")
        self._copy_split(folder_path, "train", target_class_id=0, prefix="cone_")
        self._copy_split(folder_path, "valid", target_class_id=0, prefix="cone_", target_split="val")

    def process_barriers(self, folder_path: str, oversample_factor: int = 3, val_transfer_count: int = 80):
        """Process barriers (Class 1). Oversamples and balances validation."""
        print("\n[2/3] Processing Road Barriers (Class 1)...")
        # Collect train files
        train_img_dir = os.path.join(folder_path, "train")
        lbl_dir = os.path.join(folder_path, "train", "labels") if os.path.exists(os.path.join(folder_path, "train", "labels")) else train_img_dir
        img_dir = os.path.join(folder_path, "train", "images") if os.path.exists(os.path.join(folder_path, "train", "images")) else train_img_dir

        valid_exts = [".jpg", ".jpeg", ".png", ".bmp", ".webp"]
        all_train_imgs = [f for f in os.listdir(img_dir) if os.path.splitext(f)[1].lower() in valid_exts]
        random.seed(42)
        random.shuffle(all_train_imgs)

        # Transfer val_transfer_count to validation to boost barrier val sample size
        extra_val_imgs = all_train_imgs[:val_transfer_count]
        remaining_train_imgs = all_train_imgs[val_transfer_count:]

        # Copy original valid split to val
        self._copy_split(folder_path, "valid", target_class_id=1, prefix="barrier_", target_split="val")

        # Copy extra val barriers
        for img_name in extra_val_imgs:
            base = os.path.splitext(img_name)[0]
            lbl_name = f"{base}.txt"
            src_lbl = os.path.join(lbl_dir, lbl_name)
            if not os.path.exists(src_lbl):
                continue
            # Remap to class 1
            remapped = self._remap_lines(src_lbl, allowed_classes=None, new_class_id=1)
            if remapped:
                shutil.copy2(os.path.join(img_dir, img_name), os.path.join(self.out_img_val, f"extraval_barrier_{img_name}"))
                with open(os.path.join(self.out_lbl_val, f"extraval_barrier_{lbl_name}"), "w") as f:
                    f.write("\n".join(remapped) + "\n")

        # Copy remaining train barriers with oversampling
        for rep in range(oversample_factor):
            pfx = f"rep{rep}_barrier_"
            for img_name in remaining_train_imgs:
                base = os.path.splitext(img_name)[0]
                lbl_name = f"{base}.txt"
                src_lbl = os.path.join(lbl_dir, lbl_name)
                if not os.path.exists(src_lbl):
                    continue
                remapped = self._remap_lines(src_lbl, allowed_classes=None, new_class_id=1)
                if remapped:
                    shutil.copy2(os.path.join(img_dir, img_name), os.path.join(self.out_img_train, f"{pfx}{img_name}"))
                    with open(os.path.join(self.out_lbl_train, f"{pfx}{lbl_name}"), "w") as f:
                        f.write("\n".join(remapped) + "\n")

        print(f"  Barriers processed: {len(remaining_train_imgs)} unique x {oversample_factor} train, {val_transfer_count} added to val.")

    def process_stop_signs(self, folder_path: str, max_train_samples: int = 1500, max_val_samples: int = 250):
        """Process Stop Signs (Class 2).
        
        CRITICAL FIX: Filters ONLY true stop signs (classes 4, 8, 9, 10).
        Discards corrupted classes (speed limit numbers, dashes, traffic lights).
        """
        print("\n[3/3] Processing Stop Signs (Class 2) - FILTERING CORRUPTED CLASSES...")
        yaml_path = os.path.join(folder_path, "data.yaml")
        valid_stop_class_ids = set()

        if os.path.exists(yaml_path):
            with open(yaml_path, "r") as f:
                content = yaml.safe_load(f)
            names = content.get("names", [])
            for cid, name in enumerate(names) if isinstance(names, list) else names.items():
                low_name = str(name).lower()
                # Keep only classes that explicitly refer to stop signs
                if "stop" in low_name:
                    valid_stop_class_ids.add(int(cid))

        if not valid_stop_class_ids:
            # Fallback to detected stop sign indices from audit: 4, 8, 9, 10
            valid_stop_class_ids = {4, 8, 9, 10}

        print(f"  Identified genuine stop sign class IDs: {valid_stop_class_ids}")
        print("  Discarding non-stop classes (e.g. speed signs '3', traffic lights, metadata)...")

        # Process train and valid
        self._filter_and_copy_stops(folder_path, "train", valid_stop_class_ids, max_train_samples, target_split="train")
        self._filter_and_copy_stops(folder_path, "valid", valid_stop_class_ids, max_val_samples, target_split="val")

    def _filter_and_copy_stops(self, folder_path: str, split_name: str, valid_stop_ids: Set[int], max_samples: int, target_split: str):
        split_dir = os.path.join(folder_path, split_name)
        if not os.path.exists(split_dir):
            return

        img_dir = os.path.join(split_dir, "images") if os.path.exists(os.path.join(split_dir, "images")) else split_dir
        lbl_dir = os.path.join(split_dir, "labels") if os.path.exists(os.path.join(split_dir, "labels")) else split_dir

        valid_exts = [".jpg", ".jpeg", ".png", ".bmp", ".webp"]
        all_imgs = [f for f in os.listdir(img_dir) if os.path.splitext(f)[1].lower() in valid_exts]
        random.seed(42)
        random.shuffle(all_imgs)

        out_img_dir = self.out_img_train if target_split == "train" else self.out_img_val
        out_lbl_dir = self.out_lbl_train if target_split == "train" else self.out_lbl_val

        saved_count = 0
        for img_name in all_imgs:
            if saved_count >= max_samples:
                break

            base = os.path.splitext(img_name)[0]
            lbl_name = f"{base}.txt"
            src_lbl = os.path.join(lbl_dir, lbl_name)
            if not os.path.exists(src_lbl):
                continue

            # Read and keep ONLY valid stop sign lines, remap to class 2, and sanitize geometry
            remapped_lines = []
            with open(src_lbl, "r") as f:
                for line in f:
                    parts = line.strip().split()
                    if len(parts) >= 5:
                        try:
                            cid = int(parts[0])
                            if cid in valid_stop_ids:
                                xc, yc, w, h = float(parts[1]), float(parts[2]), float(parts[3]), float(parts[4])
                                cleaned = self._sanitize_bbox(xc, yc, w, h)
                                if cleaned:
                                    c_xc, c_yc, c_w, c_h = cleaned
                                    remapped_lines.append(f"2 {c_xc:.6f} {c_yc:.6f} {c_w:.6f} {c_h:.6f}")
                        except ValueError:
                            pass

            # Only keep the image if it actually contains a real stop sign
            if remapped_lines:
                unique_name = f"stop_{img_name}"
                unique_lbl = f"stop_{lbl_name}"
                shutil.copy2(os.path.join(img_dir, img_name), os.path.join(out_img_dir, unique_name))
                with open(os.path.join(out_lbl_dir, unique_lbl), "w") as f:
                    f.write("\n".join(remapped_lines) + "\n")
                saved_count += 1

        print(f"  Extracted {saved_count} verified stop sign images -> {target_split}")

    def _copy_split(self, folder_path: str, split_name: str, target_class_id: int, prefix: str, target_split: str = "train"):
        split_dir = os.path.join(folder_path, split_name)
        if not os.path.exists(split_dir):
            return

        img_dir = os.path.join(split_dir, "images") if os.path.exists(os.path.join(split_dir, "images")) else split_dir
        lbl_dir = os.path.join(split_dir, "labels") if os.path.exists(os.path.join(split_dir, "labels")) else split_dir

        valid_exts = [".jpg", ".jpeg", ".png", ".bmp", ".webp"]
        all_imgs = [f for f in os.listdir(img_dir) if os.path.splitext(f)[1].lower() in valid_exts]

        out_img_dir = self.out_img_train if target_split == "train" else self.out_img_val
        out_lbl_dir = self.out_lbl_train if target_split == "train" else self.out_lbl_val

        count = 0
        for img_name in all_imgs:
            base = os.path.splitext(img_name)[0]
            lbl_name = f"{base}.txt"
            src_lbl = os.path.join(lbl_dir, lbl_name)
            if not os.path.exists(src_lbl):
                continue

            remapped = self._remap_lines(src_lbl, allowed_classes=None, new_class_id=target_class_id)
            if remapped:
                shutil.copy2(os.path.join(img_dir, img_name), os.path.join(out_img_dir, f"{prefix}{img_name}"))
                with open(os.path.join(out_lbl_dir, f"{prefix}{lbl_name}"), "w") as f:
                    f.write("\n".join(remapped) + "\n")
                count += 1

        print(f"  Extracted {count} images from {os.path.basename(folder_path)} ({split_name}) -> {target_split}")

    def _sanitize_bbox(self, xc: float, yc: float, w: float, h: float, min_dim: float = 0.005):
        """Clamps coordinates to [0.0, 1.0] and filters degenerate zero-area boxes."""
        xmin = max(0.0, min(1.0, xc - w / 2.0))
        ymin = max(0.0, min(1.0, yc - h / 2.0))
        xmax = max(0.0, min(1.0, xc + w / 2.0))
        ymax = max(0.0, min(1.0, yc + h / 2.0))

        new_w = xmax - xmin
        new_h = ymax - ymin
        if new_w < min_dim or new_h < min_dim:
            return None

        new_xc = max(0.0001, min(0.9999, xmin + new_w / 2.0))
        new_yc = max(0.0001, min(0.9999, ymin + new_h / 2.0))
        new_w = max(0.0001, min(1.0, new_w))
        new_h = max(0.0001, min(1.0, new_h))
        return new_xc, new_yc, new_w, new_h

    def _remap_lines(self, lbl_path: str, allowed_classes: Set[int] = None, new_class_id: int = 0) -> List[str]:
        out = []
        with open(lbl_path, "r") as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) >= 5:
                    try:
                        cid = int(parts[0])
                        if allowed_classes is None or cid in allowed_classes:
                            xc, yc, w, h = float(parts[1]), float(parts[2]), float(parts[3]), float(parts[4])
                            cleaned = self._sanitize_bbox(xc, yc, w, h)
                            if cleaned:
                                c_xc, c_yc, c_w, c_h = cleaned
                                out.append(f"{new_class_id} {c_xc:.6f} {c_yc:.6f} {c_w:.6f} {c_h:.6f}")
                    except ValueError:
                        pass
        return out

    def write_yaml(self):
        yaml_path = os.path.join(self.output_dir, "dataset.yaml")
        abs_out = os.path.abspath(self.output_dir).replace("\\", "/")
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
        print(f"\n[YAML] Saved clean dataset config to: {yaml_path}")

    def run(self):
        self._prepare_dirs()
        folders = self.find_folders()

        print("=" * 65)
        print("BUILDING CLEAN, VERIFIED DATASET (FILTERING CORRUPTED LABELS)")
        print("=" * 65)

        if "cone" in folders:
            self.process_cones(folders["cone"])
        if "barrier" in folders:
            self.process_barriers(folders["barrier"])
        if "stop_sign" in folders:
            self.process_stop_signs(folders["stop_sign"])

        self.write_yaml()

        from src.dataset.dataset_manager import DatasetManager
        dm = DatasetManager(self.output_dir)
        dm.inspect_and_summarize()


def main():
    merger = CleanDatasetMerger(data_root="data", output_dir="data/dataset_clean")
    merger.run()


if __name__ == "__main__":
    main()
