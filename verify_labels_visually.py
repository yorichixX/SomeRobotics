"""Visual Label Verification Tool for Eric Robots.

Renders bounding boxes and class names onto real dataset images from data/dataset_clean
so you can visually inspect and confirm 100% that:
- Class 0 corresponds to Traffic Cones
- Class 1 corresponds to Road Barriers
- Class 2 corresponds to Stop Signs

Saves preview images to: data/visual_verification/
"""

import os
import random
import cv2
from typing import Dict, List, Tuple
from collections import defaultdict


def verify_labels_visually(
    dataset_dir: str = "data/dataset_clean",
    output_dir: str = "data/visual_verification",
    samples_per_class: int = 5,
):
    print("=" * 70)
    print("GENERATING VISUAL LABEL VERIFICATION IMAGES")
    print(f"Target Directory: {output_dir}")
    print("=" * 70)

    os.makedirs(output_dir, exist_ok=True)

    CLASS_META = {
        0: {"name": "CONE", "color": (0, 255, 0)},       # Green
        1: {"name": "BARRIER", "color": (0, 140, 255)},   # Orange
        2: {"name": "STOP_SIGN", "color": (255, 0, 0)},   # Blue
    }

    # Index images by class presence across train and val
    class_to_images = defaultdict(list)

    for split in ["train", "val"]:
        img_dir = os.path.join(dataset_dir, "images", split)
        lbl_dir = os.path.join(dataset_dir, "labels", split)

        if not os.path.exists(lbl_dir):
            continue

        for lbl_file in os.listdir(lbl_dir):
            if not lbl_file.endswith(".txt"):
                continue
            lbl_path = os.path.join(lbl_dir, lbl_file)

            with open(lbl_path, "r") as f:
                lines = [line.strip().split() for line in f if line.strip()]

            cids = set()
            for parts in lines:
                if len(parts) >= 5:
                    try:
                        cids.add(int(parts[0]))
                    except ValueError:
                        pass

            base = os.path.splitext(lbl_file)[0]
            # Find matching image
            for ext in [".jpg", ".jpeg", ".png", ".bmp", ".webp"]:
                candidate = os.path.join(img_dir, f"{base}{ext}")
                if os.path.exists(candidate):
                    for cid in cids:
                        class_to_images[cid].append((candidate, lbl_path, split))
                    break

    random.seed(42)
    saved_samples = []

    for cid, meta in CLASS_META.items():
        pool = class_to_images.get(cid, [])
        print(f"\n--- Class {cid}: {meta['name']} (Available images: {len(pool)}) ---")
        if not pool:
            print(f"  [WARNING] No images found for Class {cid}!")
            continue

        # Sample diverse images
        selected = random.sample(pool, min(samples_per_class, len(pool)))

        for idx, (img_path, lbl_path, split) in enumerate(selected):
            img = cv2.imread(img_path)
            if img is None:
                continue

            h, w = img.shape[:2]

            with open(lbl_path, "r") as f:
                boxes = [line.strip().split() for line in f if line.strip()]

            box_counts = defaultdict(int)

            for b in boxes:
                if len(b) < 5:
                    continue
                box_cid = int(b[0])
                box_counts[box_cid] += 1

                b_meta = CLASS_META.get(box_cid, {"name": f"UNKNOWN_{box_cid}", "color": (0, 0, 255)})
                xc, yc, bw, bh = float(b[1]), float(b[2]), float(b[3]), float(b[4])

                x1 = int((xc - bw / 2.0) * w)
                y1 = int((yc - bh / 2.0) * h)
                x2 = int((xc + bw / 2.0) * w)
                y2 = int((yc + bh / 2.0) * h)

                # Clamp to image frame
                x1, y1 = max(0, x1), max(0, y1)
                x2, y2 = min(w - 1, x2), min(h - 1, y2)

                # Draw bounding box
                cv2.rectangle(img, (x1, y1), (x2, y2), b_meta["color"], 2)

                # Label text
                label_txt = f"{b_meta['name']} (ID {box_cid})"
                cv2.putText(
                    img,
                    label_txt,
                    (x1, max(20, y1 - 6)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    b_meta["color"],
                    2,
                )

            # Top overlay header
            header_txt = f"[{split.upper()}] Target: {meta['name']} | Boxes: {dict(box_counts)}"
            cv2.putText(img, header_txt, (15, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

            fname = f"sample_class{cid}_{meta['name']}_{split}_{idx + 1}_{os.path.basename(img_path)}"
            out_path = os.path.join(output_dir, fname)
            cv2.imwrite(out_path, img)

            saved_samples.append((cid, meta["name"], fname))
            print(f"  [{idx + 1}/{len(selected)}] Saved: {fname}")

    print("\n" + "=" * 70)
    print(f"VERIFICATION COMPLETE: {len(saved_samples)} sample images generated!")
    print(f"Open folder '{output_dir}' to inspect bounding box and class correctness.")
    print("=" * 70)


if __name__ == "__main__":
    verify_labels_visually("data/dataset_clean", "data/visual_verification", samples_per_class=5)
