"""Forensic diagnosis of stop sign detection in the pipeline."""

import os
import glob
import cv2
from ultralytics import YOLO
from src.models.detector import ObjectDetector
from src.geometry.camera_model import PinholeCameraModel
from src.geometry.distance_estimator import GeometricDistanceEstimator

def diagnose():
    print("=" * 70)
    print("FORENSIC DIAGNOSIS: STOP SIGN INFERENCE")
    print("=" * 70)

    model_path = "weights/best.pt"
    if not os.path.exists(model_path):
        print(f"[ERROR] {model_path} not found!")
        return

    # 1. Inspect Model Metadata
    model = YOLO(model_path)
    print(f"Model Names in weights/best.pt: {model.names}")

    # 2. Inspect Camera and Estimator
    cam = PinholeCameraModel()
    estimator = GeometricDistanceEstimator(camera=cam)
    print(f"Estimator Priors keys: {list(estimator.priors.keys())}")

    # 3. Test ObjectDetector directly
    detector = ObjectDetector(
        model_path=model_path,
        conf_threshold=0.10,  # Lower threshold to see what is detected
        device="cuda" if os.environ.get("CUDA_VISIBLE_DEVICES") else "cpu",
        camera=cam,
        estimator=estimator,
    )

    # 4. Find all stop sign images in val
    val_dir = "data/dataset_clean/images/val"
    stop_imgs = sorted(glob.glob(os.path.join(val_dir, "*stop*.*")))
    print(f"\nTotal stop sign images in {val_dir}: {len(stop_imgs)}")

    if not stop_imgs:
        print("[WARNING] No stop sign images found matching '*stop*'!")
        return

    # Test first 10 stop images
    sample_imgs = stop_imgs[:10]
    for idx, img_p in enumerate(sample_imgs):
        fname = os.path.basename(img_p)
        img = cv2.imread(img_p)
        if img is None:
            continue

        h, w = img.shape[:2]
        print(f"\n[{idx + 1}/10] Image: {fname} (Resolution: {w}x{h})")

        # Test raw YOLO predict
        raw_res = model.predict(img, conf=0.01, verbose=False)[0]
        raw_boxes = raw_res.boxes
        print(f"  Raw YOLO (conf=0.01) found {len(raw_boxes)} boxes:")
        for b in raw_boxes:
            cid = int(b.cls[0].cpu().numpy())
            cname = model.names.get(cid, str(cid))
            score = float(b.conf[0].cpu().numpy())
            print(f"    - Class: {cid} ({cname}) | Score: {score:.4f} | Box: {[int(v) for v in b.xyxy[0].cpu().numpy()]}")

        # Test ObjectDetector pipeline
        detections = detector.detect(img)
        print(f"  ObjectDetector (conf=0.10) found {len(detections)} detections:")
        for d in detections:
            print(f"    - Det Class: {d.class_name} (ID {d.class_id}) | Score: {d.score:.4f} | Distance: {d.distance_m}m")

        # Check label ground truth
        lbl_p = os.path.join("data/dataset_clean/labels/val", f"{os.path.splitext(fname)[0]}.txt")
        if os.path.exists(lbl_p):
            with open(lbl_p, "r") as fp:
                print(f"  Ground Truth in label file: {[l.strip() for l in fp if l.strip()]}")


if __name__ == "__main__":
    diagnose()
