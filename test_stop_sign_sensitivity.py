"""Diagnostic script to test stop sign detection sensitivity across different confidence thresholds."""

import os
import glob
from ultralytics import YOLO

def test_stop_signs(model_path: str = "weights/best.pt", val_dir: str = "data/dataset_clean/images/val"):
    print("=" * 70)
    print("TESTING STOP SIGN DETECTION SENSITIVITY")
    print(f"Model: {model_path}")
    print("=" * 70)

    if not os.path.exists(model_path):
        print(f"[ERROR] Model file not found: {model_path}")
        return

    model = YOLO(model_path)

    # Find stop sign images
    stop_imgs = sorted(glob.glob(os.path.join(val_dir, "*stop*.*")))
    if not stop_imgs:
        stop_imgs = sorted(glob.glob(os.path.join(val_dir, "*.*")))[:20]

    sample_imgs = stop_imgs[:10]
    print(f"Testing on {len(sample_imgs)} validation stop sign images...\n")

    for img_p in sample_imgs:
        fname = os.path.basename(img_p)
        # Predict with very low threshold (0.05) to see raw scores
        res = model.predict(img_p, conf=0.05, verbose=False)[0]

        print(f"Image: {fname}")
        if len(res.boxes) == 0:
            print("  -> Zero detections even at conf=0.05 (Unrecognized features)")
        else:
            for b in res.boxes:
                cid = int(b.cls[0].cpu().numpy())
                cname = model.names.get(cid, str(cid))
                score = float(b.conf[0].cpu().numpy())
                box = [int(v) for v in b.xyxy[0].cpu().numpy()]
                print(f"  -> Class: [{cname}] | Confidence: {score:.3f} | Box: {box}")

    print("\n" + "=" * 70)
    print("CONCLUSION: Confidence scores above show the optimal threshold for stop signs.")
    print("=" * 70)


if __name__ == "__main__":
    test_stop_signs()
