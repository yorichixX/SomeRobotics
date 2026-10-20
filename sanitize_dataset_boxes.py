"""In-place bounding box sanitizer for consolidated YOLO datasets.

Fixes Roboflow edge clipping and degenerate boxes:
1. Converts (xc, yc, w, h) to (xmin, ymin, xmax, ymax).
2. Clamps coordinates strictly to [0.0, 1.0].
3. Discards degenerate zero-width or zero-height boxes (e.g. w=0, h=0).
4. Recomputes valid normalized (xc, yc, w, h) with strict floating point margins.
5. Runs the full audit automatically to confirm 100% dataset compliance.
"""

import os
from typing import Optional, Tuple
from audit_clean_dataset import audit_consolidated_dataset


def sanitize_bbox(
    xc: float, yc: float, w: float, h: float, min_dim: float = 0.005
) -> Optional[Tuple[float, float, float, float]]:
    """Clamps bounding box to [0.0, 1.0] and filters zero-area boxes."""
    # Convert to corners
    xmin = max(0.0, min(1.0, xc - w / 2.0))
    ymin = max(0.0, min(1.0, yc - h / 2.0))
    xmax = max(0.0, min(1.0, xc + w / 2.0))
    ymax = max(0.0, min(1.0, yc + h / 2.0))

    new_w = xmax - xmin
    new_h = ymax - ymin

    # Drop degenerate or sub-pixel boxes (< 0.5% image dimension)
    if new_w < min_dim or new_h < min_dim:
        return None

    # Recompute center
    new_xc = xmin + new_w / 2.0
    new_yc = ymin + new_h / 2.0

    # Ensure coordinates stay strictly inside (0.0, 1.0)
    new_xc = max(0.0001, min(0.9999, new_xc))
    new_yc = max(0.0001, min(0.9999, new_yc))
    new_w = max(0.0001, min(1.0, new_w))
    new_h = max(0.0001, min(1.0, new_h))

    return round(new_xc, 6), round(new_yc, 6), round(new_w, 6), round(new_h, 6)


def sanitize_dataset(dataset_dir: str = "data/dataset_clean") -> None:
    print("=" * 70)
    print(f"SANITIZING BOUNDING BOXES IN-PLACE: {dataset_dir}")
    print("=" * 70)

    total_modified_files = 0
    total_repaired_boxes = 0
    total_dropped_degenerate_boxes = 0

    for split in ["train", "val"]:
        lbl_dir = os.path.join(dataset_dir, "labels", split)
        if not os.path.exists(lbl_dir):
            continue

        split_repaired = 0
        split_dropped = 0
        split_files_modified = 0

        for fname in os.listdir(lbl_dir):
            if not fname.endswith(".txt"):
                continue
            path = os.path.join(lbl_dir, fname)

            with open(path, "r") as f:
                lines = f.readlines()

            new_lines = []
            file_changed = False

            for line in lines:
                parts = line.strip().split()
                if not parts:
                    continue
                try:
                    cid = int(parts[0])
                    xc, yc, w, h = float(parts[1]), float(parts[2]), float(parts[3]), float(parts[4])

                    cleaned = sanitize_bbox(xc, yc, w, h)
                    if cleaned is None:
                        # Degenerate box dropped
                        file_changed = True
                        split_dropped += 1
                        continue

                    c_xc, c_yc, c_w, c_h = cleaned
                    # Check if modified
                    if (c_xc, c_yc, c_w, c_h) != (round(xc, 6), round(yc, 6), round(w, 6), round(h, 6)):
                        file_changed = True
                        split_repaired += 1

                    new_lines.append(f"{cid} {c_xc:.6f} {c_yc:.6f} {c_w:.6f} {c_h:.6f}\n")
                except Exception:
                    file_changed = True
                    split_dropped += 1

            if file_changed:
                split_files_modified += 1
                with open(path, "w") as f:
                    f.writelines(new_lines)

        print(f"[{split.upper()}] Sanitization complete:")
        print(f"  - Repaired edge-boundary boxes: {split_repaired}")
        print(f"  - Dropped degenerate boxes:    {split_dropped}")
        print(f"  - Files updated:               {split_files_modified}")

        total_repaired_boxes += split_repaired
        total_dropped_degenerate_boxes += split_dropped
        total_modified_files += split_files_modified

    print(f"\n[SUMMARY] Total boxes repaired: {total_repaired_boxes}")
    print(f"[SUMMARY] Total degenerate boxes removed: {total_dropped_degenerate_boxes}")
    print(f"[SUMMARY] Total label files updated: {total_modified_files}\n")

    # Run verification audit
    print("=" * 70)
    print("RUNNING AUTOMATED VERIFICATION AUDIT")
    print("=" * 70)
    audit_consolidated_dataset(dataset_dir)


if __name__ == "__main__":
    sanitize_dataset("data/dataset_clean")
