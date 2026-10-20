"""Main entry point: Run Object Detection + Distance Estimation on images or video."""

from typing import Optional, List
import argparse
import os
import glob
import cv2
import numpy as np

from src.geometry.camera_model import PinholeCameraModel
from src.geometry.distance_estimator import GeometricDistanceEstimator
from src.models.detector import ObjectDetector
from src.extra.bev_homography import BirdEyeViewMapper


def process_image(
    image_path: str,
    detector: ObjectDetector,
    bev_mapper: Optional[BirdEyeViewMapper] = None,
    output_path: Optional[str] = None,
) -> np.ndarray:
    """Process a single image frame: detect, estimate distance, and annotate."""
    img = cv2.imread(image_path)
    if img is None:
        raise ValueError(f"Could not load image: {image_path}")

    # 1. Detect objects and estimate distances
    detections = detector.detect(img)

    # 2. Annotate bounding boxes and distance text
    annotated = detector.annotate_frame(img, detections)

    # 3. Optional Bird's-Eye View composite
    if bev_mapper is not None:
        bev_grid = bev_mapper.render_bev_grid(detections)
        annotated = bev_mapper.create_composite_display(annotated, bev_grid)

    # Print detection log to console
    print(f"\n[IMAGE] {os.path.basename(image_path)}")
    print(f"Detected {len(detections)} navigation objects:")
    for det in detections:
        dist_str = f"{det.distance_m:.1f}m" if det.distance_m is not None else "N/A"
        print(f"  - [{det.class_name.upper()}] Conf: {det.score:.2f} | Distance: {dist_str} | Box: {[int(x) for x in det.box]}")

    if output_path is not None:
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        cv2.imwrite(output_path, annotated)
        print(f"Saved annotated result to: {output_path}")

    return annotated


def process_video(
    video_path: str,
    detector: ObjectDetector,
    bev_mapper: Optional[BirdEyeViewMapper] = None,
    output_path: Optional[str] = None,
) -> None:
    """Process a video file stream."""
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"Could not open video file: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    fw = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    fh = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    out_writer = None
    if output_path is not None:
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        # If BEV is enabled, horizontal width increases
        out_w = fw + int(bev_mapper.bev_width * (fh / bev_mapper.bev_height)) if bev_mapper else fw
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        out_writer = cv2.VideoWriter(output_path, fourcc, fps, (out_w, fh))

    frame_idx = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        detections = detector.detect(frame)
        annotated = detector.annotate_frame(frame, detections)

        if bev_mapper is not None:
            bev_grid = bev_mapper.render_bev_grid(detections)
            annotated = bev_mapper.create_composite_display(annotated, bev_grid)

        if out_writer is not None:
            out_writer.write(annotated)

        frame_idx += 1
        if frame_idx % 30 == 0:
            print(f"Processed {frame_idx} video frames...")

    cap.release()
    if out_writer is not None:
        out_writer.release()
        print(f"\n[VIDEO COMPLETE] Saved output video to: {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Eric Robots: Detection + Distance Estimation Pipeline")
    parser.add_argument("--source", type=str, default="data/dataset/images/val", help="Path to image, video, or folder")
    parser.add_argument("--model", type=str, default=None, help="Path to .pt or .onnx weights (optional)")
    parser.add_argument("--camera-config", type=str, default="configs/camera_config.yaml", help="Camera calibration YAML")
    parser.add_argument("--priors-config", type=str, default="configs/object_priors.yaml", help="Object priors YAML")
    parser.add_argument("--device", type=str, default="cpu", help="Device: 'cpu' or 'cuda'")
    parser.add_argument("--conf", type=float, default=0.35, help="Confidence threshold")
    parser.add_argument("--bev", action="store_true", help="Enable Bird's-Eye View radar overlay")
    parser.add_argument("--output-dir", type=str, default="data/output", help="Directory to save output")

    parser.add_argument("--max-images", type=int, default=15, help="Max images to process if source is a folder (0 for all)")

    args = parser.parse_args()

    # Initialize camera & distance estimator
    cam = PinholeCameraModel.from_yaml(args.camera_config)
    estimator = GeometricDistanceEstimator(camera=cam, priors_yaml_path=args.priors_config)

    # Initialize detector
    detector = ObjectDetector(
        model_path=args.model,
        conf_threshold=args.conf,
        device=args.device,
        camera=cam,
        estimator=estimator,
    )

    bev_mapper = BirdEyeViewMapper() if args.bev else None

    # Handle image, directory, or video
    source = args.source
    if os.path.isdir(source):
        images = sorted(glob.glob(os.path.join(source, "*.[jJ][pP][gG]")) + glob.glob(os.path.join(source, "*.[pP][nN][gG]")))
        if args.max_images > 0:
            images = images[:args.max_images]
        for img_p in images:
            out_p = os.path.join(args.output_dir, f"annotated_{os.path.basename(img_p)}")
            process_image(img_p, detector, bev_mapper, out_p)
    elif source.lower().endswith((".mp4", ".avi", ".mov", ".mkv")):
        out_p = os.path.join(args.output_dir, f"annotated_{os.path.basename(source)}")
        process_video(source, detector, bev_mapper, out_p)
    elif os.path.isfile(source):
        out_p = os.path.join(args.output_dir, f"annotated_{os.path.basename(source)}")
        process_image(source, detector, bev_mapper, out_p)
    else:
        print(f"Source not found: {source}")


if __name__ == "__main__":
    main()
