"""Check all stop sign images in val and test non-GTSRB stop signs."""

import os
import glob
from collections import defaultdict
from ultralytics import YOLO

def analyze_val_stops():
    val_dir = "data/dataset_clean/images/val"
    all_stops = sorted(glob.glob(os.path.join(val_dir, "*stop*.*")))
    print(f"Total stop images in val: {len(all_stops)}")

    gtsrb = [f for f in all_stops if "00014" in f]
    natural = [f for f in all_stops if "00014" not in f]

    print(f"GTSRB upscaled 32x32 crops (00014_...): {len(gtsrb)}")
    print(f"Natural road scene stop signs:         {len(natural)}")

    print("\nSample natural stop signs:")
    for f in natural[:10]:
        print(f"  - {os.path.basename(f)}")

    # Test detection on 5 natural stop signs
    model_path = "weights/best.pt"
    if os.path.exists(model_path):
        model = YOLO(model_path)
        print("\n--- Testing Detection on Natural Road Stop Signs ---")
        for f in natural[:10]:
            res = model.predict(f, conf=0.15, verbose=False)[0]
            boxes = res.boxes
            print(f"Image: {os.path.basename(f)}")
            if len(boxes) == 0:
                print("  No detection at conf=0.15")
            else:
                for b in boxes:
                    cid = int(b.cls[0].cpu().numpy())
                    cname = model.names.get(cid, str(cid))
                    score = float(b.conf[0].cpu().numpy())
                    box = [int(v) for v in b.xyxy[0].cpu().numpy()]
                    print(f"  -> [{cname}] Conf: {score:.3f} | Box: {box}")

if __name__ == "__main__":
    analyze_val_stops()
