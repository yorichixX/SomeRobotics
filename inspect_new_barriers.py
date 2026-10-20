"""Inspection script to verify the contents of the new barrier datasets.

Checks:
1. File overlap between Barrier.v2i.yolov8 and Barrier.v2i.yolov8(1).
2. What class ID 0 ('0') and class ID 1 ('Barrier') represent in Barrier.v2i.yolov8(1).
3. Contents of barrier.v1i.yolov8.
4. Generates sample annotated preview images saved to data/output/audit_samples/
"""

import os
import cv2
from typing import List, Dict

def inspect_datasets(data_root: str = "data"):
    print("=" * 70)
    print("INSPECTING NEW BARRIER DATASETS")
    print("=" * 70)

    f_v2 = os.path.join(data_root, "Barrier.v2i.yolov8")
    f_v2_1 = os.path.join(data_root, "Barrier.v2i.yolov8(1)")
    f_v1 = os.path.join(data_root, "barrier.v1i.yolov8")

    # 1. Compare file overlap between Barrier.v2i.yolov8 and Barrier.v2i.yolov8(1)
    if os.path.exists(f_v2) and os.path.exists(f_v2_1):
        def get_all_images(folder):
            imgs = set()
            for root, _, files in os.walk(folder):
                for f in files:
                    if f.lower().endswith((".jpg", ".jpeg", ".png")):
                        imgs.add(f)
            return imgs

        imgs_v2 = get_all_images(f_v2)
        imgs_v2_1 = get_all_images(f_v2_1)

        intersection = imgs_v2.intersection(imgs_v2_1)
        only_in_new = imgs_v2_1 - imgs_v2
        only_in_old = imgs_v2 - imgs_v2_1

        print("\n--- Overlap Analysis: Barrier.v2i.yolov8 vs Barrier.v2i.yolov8(1) ---")
        print(f"  Old Barrier.v2i images:       {len(imgs_v2)}")
        print(f"  New Barrier.v2i(1) images:    {len(imgs_v2_1)}")
        print(f"  Shared (duplicate) images:    {len(intersection)}")
        print(f"  Brand new unique images in (1): {len(only_in_new)}")

    # 2. Inspect barrier.v1i.yolov8
    if os.path.exists(f_v1):
        imgs_v1 = get_all_images(f_v1)
        print("\n--- barrier.v1i.yolov8 Analysis ---")
        print(f"  Total images in barrier.v1i:  {len(imgs_v1)}")
        if os.path.exists(f_v2):
            v1_shared_with_old = imgs_v1.intersection(imgs_v2)
            v1_unique = imgs_v1 - imgs_v2
            print(f"  Shared with old Barrier.v2:   {len(v1_shared_with_old)}")
            print(f"  Brand new unique images in v1:{len(v1_unique)}")

    # 3. Create visual preview samples for Class 0 ('0') and Class 1 ('Barrier') in Barrier.v2i(1)
    out_dir = os.path.join(data_root, "output", "audit_samples")
    os.makedirs(out_dir, exist_ok=True)

    if os.path.exists(f_v2_1):
        print(f"\n--- Sampling Annotations from Barrier.v2i(1) to {out_dir} ---")
        train_img_dir = os.path.join(f_v2_1, "train", "images") if os.path.exists(os.path.join(f_v2_1, "train", "images")) else os.path.join(f_v2_1, "train")
        train_lbl_dir = os.path.join(f_v2_1, "train", "labels") if os.path.exists(os.path.join(f_v2_1, "train", "labels")) else os.path.join(f_v2_1, "train")

        sampled_0 = 0
        sampled_1 = 0

        for f in sorted(os.listdir(train_lbl_dir)):
            if not f.endswith(".txt"):
                continue
            lbl_p = os.path.join(train_lbl_dir, f)
            with open(lbl_p, "r") as fp:
                lines = [l.strip().split() for l in fp if l.strip()]

            cids = [int(p[0]) for p in lines if len(p) >= 5]
            has_0 = 0 in cids
            has_1 = 1 in cids

            if (has_0 and sampled_0 < 3) or (has_1 and sampled_1 < 3):
                base = os.path.splitext(f)[0]
                img_candidates = [
                    os.path.join(train_img_dir, f"{base}.jpg"),
                    os.path.join(train_img_dir, f"{base}.jpeg"),
                    os.path.join(train_img_dir, f"{base}.png"),
                ]
                img_p = next((p for p in img_candidates if os.path.exists(p)), None)
                if not img_p:
                    continue

                img = cv2.imread(img_p)
                if img is None:
                    continue
                h, w = img.shape[:2]

                for p in lines:
                    cid = int(p[0])
                    xc, yc, bw, bh = float(p[1]), float(p[2]), float(p[3]), float(p[4])
                    x1 = int((xc - bw / 2) * w)
                    y1 = int((yc - bh / 2) * h)
                    x2 = int((xc + bw / 2) * w)
                    y2 = int((yc + bh / 2) * h)

                    color = (0, 0, 255) if cid == 0 else (0, 255, 0)
                    tag = f"Class {cid}"
                    cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
                    cv2.putText(img, tag, (x1, max(15, y1 - 5)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

                out_p = os.path.join(out_dir, f"sample_cid{'_0' if has_0 else ''}{'_1' if has_1 else ''}_{base}.jpg")
                cv2.imwrite(out_p, img)

                if has_0: sampled_0 += 1
                if has_1: sampled_1 += 1

                print(f"  Saved sample preview: {os.path.basename(out_p)}")

    print("\n[COMPLETE] Run complete. Check overlap numbers and sample images.")

if __name__ == "__main__":
    inspect_datasets("data")
