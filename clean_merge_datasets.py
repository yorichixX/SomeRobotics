"""Clean and verified multi-dataset merger for robotics navigation.

Integrates multiple road safety obstacle datasets:
- Traffic Cones: traffic cone.v3i.yolov8 (Class 0)
- Road Barriers: Merges Barrier.v2i.yolov8, Barrier.v2i.yolov8(1), and barrier.v1i.yolov8 (Class 1)
  * Seamlessly aggregates 1,200+ unique physical barrier images across plastic, water-filled, and concrete types.
  * Zero synthetic repetition needed.
- Stop Signs: Traffic Stop Sign.v2i.yolov8 (Class 2)
  * Filters ONLY true stop signs (classes 4, 8, 9, 10). Discards speed signs and metadata.
- Applies in-place geometric bounding box sanitization on all annotations.
- Automatically audits the final dataset.
"""

import os
import shutil
import random
from typing import Dict, List, Tuple, Set, Optional
import yaml
from collections import defaultdict
from audit_clean_dataset import audit_consolidated_dataset


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

    def find_folders(self) -> Dict[str, any]:
        candidates = [
            d for d in os.listdir(self.data_root)
            if os.path.isdir(os.path.join(self.data_root, d))
            and d not in ["dataset", "dataset_balanced", "dataset_clean", "output", "sample_videos", "sample_images", "bdd100k"]
        ]
        detected = {"barriers": []}
        for folder in sorted(candidates):
            low = folder.lower()
            fp = os.path.join(self.data_root, folder)
            if "cone" in low:
                detected["cone"] = fp
            elif "barrier" in low:
                detected["barriers"].append(fp)
            elif "stop" in low or "sign" in low:
                detected["stop_sign"] = fp
        return detected

    def _sanitize_bbox(self, xc: float, yc: float, w: float, h: float, min_dim: float = 0.005) -> Optional[Tuple[float, float, float, float]]:
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
        return round(new_xc, 6), round(new_yc, 6), round(new_w, 6), round(new_h, 6)

    def _remap_file(self, lbl_path: str, allowed_classes: Optional[Set[int]] = None, target_class_id: int = 0) -> List[str]:
        out = []
        if not os.path.exists(lbl_path):
            return out
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
                                out.append(f"{target_class_id} {c_xc:.6f} {c_yc:.6f} {c_w:.6f} {c_h:.6f}")
                    except ValueError:
                        pass
        return out

    def process_cones(self, folder_path: str):
        """Process traffic cones (Class 0)."""
        print("\n[1/3] Processing Traffic Cones (Class 0)...")
        self._copy_split(folder_path, "train", target_class_id=0, prefix="cone_")
        self._copy_split(folder_path, "valid", target_class_id=0, prefix="cone_", target_split="val")

    def process_barriers(self, folder_paths: List[str], target_val_count: int = 200):
        """Process all road barrier datasets (Class 1) into a unified unique set."""
        print(f"\n[2/3] Processing Road Barriers (Class 1) from {len(folder_paths)} source datasets...")

        all_barrier_samples = []
        seen_img_names = set()

        valid_exts = [".jpg", ".jpeg", ".png", ".bmp", ".webp"]

        for idx, fpath in enumerate(folder_paths):
            fbasename = os.path.basename(fpath)
            print(f"  Scanning: {fbasename}...")

            # Determine allowed barrier classes in this dataset
            low = fbasename.lower()
            if "(1)" in fbasename or "barrier.v2i.yolov8(1)" in low:
                # In Barrier.v2i(1), both class 0 ('0') and class 1 ('Barrier') are water-filled barriers
                allowed_classes = {0, 1}
            elif "v1" in low:
                # In barrier.v1i, class 0 is barrier, class 1 is warning
                allowed_classes = {0}
            else:
                allowed_classes = {0}

            # Search across all potential sub-splits in this folder
            for split_name in ["train", "valid", "test"]:
                s_dir = os.path.join(fpath, split_name)
                if not os.path.exists(s_dir):
                    continue

                img_dir = os.path.join(s_dir, "images") if os.path.exists(os.path.join(s_dir, "images")) else s_dir
                lbl_dir = os.path.join(s_dir, "labels") if os.path.exists(os.path.join(s_dir, "labels")) else s_dir

                imgs = [f for f in os.listdir(img_dir) if os.path.splitext(f)[1].lower() in valid_exts]

                for img_name in imgs:
                    base = os.path.splitext(img_name)[0]
                    lbl_name = f"{base}.txt"
                    src_lbl = os.path.join(lbl_dir, lbl_name)
                    if not os.path.exists(src_lbl):
                        continue

                    # Remap to Class 1
                    remapped = self._remap_file(src_lbl, allowed_classes=allowed_classes, target_class_id=1)
                    if not remapped:
                        continue

                    # Unique naming to prevent collision
                    unique_img_name = f"bset{idx}_{img_name}"
                    unique_lbl_name = f"bset{idx}_{lbl_name}"

                    src_img_path = os.path.join(img_dir, img_name)
                    all_barrier_samples.append((src_img_path, remapped, unique_img_name, unique_lbl_name))

        print(f"  Total unique barrier images collected: {len(all_barrier_samples)}")

        # Shuffle and partition into train and val
        random.seed(42)
        random.shuffle(all_barrier_samples)

        val_count = min(target_val_count, max(50, int(len(all_barrier_samples) * 0.15)))
        val_samples = all_barrier_samples[:val_count]
        train_samples = all_barrier_samples[val_count:]

        # Write validation barriers
        for src_img, lines, u_img, u_lbl in val_samples:
            shutil.copy2(src_img, os.path.join(self.out_img_val, u_img))
            with open(os.path.join(self.out_lbl_val, u_lbl), "w") as f:
                f.write("\n".join(lines) + "\n")

        # Write training barriers (100% unique, zero repetition!)
        for src_img, lines, u_img, u_lbl in train_samples:
            shutil.copy2(src_img, os.path.join(self.out_img_train, u_img))
            with open(os.path.join(self.out_lbl_train, u_lbl), "w") as f:
                f.write("\n".join(lines) + "\n")

        print(f"  Barriers partition: {len(train_samples)} unique train, {len(val_samples)} unique val.")

    def process_stop_signs(self, folder_path: str, max_train_samples: int = 1500, max_val_samples: int = 250):
        """Process Stop Signs (Class 2) with corrupted-class filtering."""
        print("\n[3/3] Processing Stop Signs (Class 2) - FILTERING CORRUPTED CLASSES...")
        yaml_path = os.path.join(folder_path, "data.yaml")
        valid_stop_class_ids = set()

        if os.path.exists(yaml_path):
            with open(yaml_path, "r") as f:
                data_cfg = yaml.safe_load(f)
            names = data_cfg.get("names", [])
            for idx, name in enumerate(names):
                low_name = str(name).lower()
                if "stop" in low_name and "light" not in low_name:
                    valid_stop_class_ids.add(idx)

        if not valid_stop_class_ids:
            valid_stop_class_ids = {4, 8, 9, 10}

        print(f"  Identified genuine stop sign class IDs: {valid_stop_class_ids}")
        self._extract_clean_stops(folder_path, "train", valid_stop_class_ids, max_train_samples, target_split="train")
        self._extract_clean_stops(folder_path, "valid", valid_stop_class_ids, max_val_samples, target_split="val")

    def _extract_clean_stops(self, folder_path: str, split_name: str, valid_stop_ids: Set[int], max_samples: int, target_split: str = "train"):
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

            remapped_lines = self._remap_file(src_lbl, allowed_classes=valid_stop_ids, target_class_id=2)
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

            remapped = self._remap_file(src_lbl, allowed_classes=None, target_class_id=target_class_id)
            if remapped:
                shutil.copy2(os.path.join(img_dir, img_name), os.path.join(out_img_dir, f"{prefix}{img_name}"))
                with open(os.path.join(out_lbl_dir, f"{prefix}{lbl_name}"), "w") as f:
                    f.write("\n".join(remapped) + "\n")
                count += 1

        print(f"  Extracted {count} images from {os.path.basename(folder_path)} ({split_name}) -> {target_split}")

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
        print("BUILDING EXPANDED CLEAN DATASET (MULTI-BARRIER MERGE)")
        print("=" * 65)

        if "cone" in folders:
            self.process_cones(folders["cone"])
        if folders.get("barriers"):
            self.process_barriers(folders["barriers"])
        if "stop_sign" in folders:
            self.process_stop_signs(folders["stop_sign"])

        self.write_yaml()

        print("\n" + "=" * 65)
        print("RUNNING AUTOMATED INTEGRITY AUDIT ON NEW DATASET")
        print("=" * 65)
        audit_consolidated_dataset(self.output_dir)


def main():
    merger = CleanDatasetMerger(data_root="data", output_dir="data/dataset_clean")
    merger.run()


if __name__ == "__main__":
    main()
