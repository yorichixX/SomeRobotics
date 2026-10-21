"""Demo script for Optical Flow tracking of moving navigation objects (cones, barriers)."""

import argparse
import os
import cv2
import numpy as np

from src.models.detector import ObjectDetector, Detection
from src.extra.optical_flow_tracker import OpticalFlowTracker


def download_real_sample_video(output_path: str = "data/sample_videos/real_driving_sample.mp4") -> str:
    """Download a real open-source driving video benchmark."""
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    if os.path.exists(output_path) and os.path.getsize(output_path) > 100000:
        return output_path
    url = "https://github.com/intel-iot-devkit/sample-videos/raw/master/car-detection.mp4"
    print(f"[INFO] Fetching real open-source driving video benchmark from: {url}...")
    import urllib.request
    urllib.request.urlretrieve(url, output_path)
    print(f"[COMPLETE] Real driving sample video saved to: {output_path}")
    return output_path


def main():
    parser = argparse.ArgumentParser(description="Run Optical Flow Tracking on Navigation Objects")
    parser.add_argument("--video", type=str, default=None, help="Input video file path")
    parser.add_argument("--model", type=str, default="weights/best.pt", help="Path to object detector model weights")
    parser.add_argument("--conf", type=float, default=0.25, help="Detection confidence threshold")
    parser.add_argument("--max-frames", type=int, default=None, help="Maximum number of frames to process")
    parser.add_argument("--output", type=str, default="data/output/tracked_flow_demo.mp4", help="Output video path")

    args = parser.parse_args()

    video_path = args.video
    if video_path is None or not os.path.exists(video_path):
        if args.video is not None:
            print(f"[ERROR] Video file not found: {args.video}")
            return
        print("[INFO] No input video specified. Downloading real driving video sample...")
        video_path = download_real_sample_video()

    tracker = OpticalFlowTracker()
    model_to_use = args.model if (args.model and os.path.exists(args.model)) else None
    detector = ObjectDetector(model_path=model_to_use, conf_threshold=args.conf)

    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    fw = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    fh = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(args.output, fourcc, fps, (fw, fh))

    frame_count = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        raw_detections = detector.detect(frame)
        if len(raw_detections) == 0 and args.video is None:
            raw_detections = detector._heuristic_fallback_detect(frame)

        # Physical sanity filter: reject false positive boxes covering the entire road canvas
        clean_detections = []
        for det in raw_detections:
            bw = det.box[2] - det.box[0]
            bh = det.box[3] - det.box[1]
            box_area = bw * bh
            img_area = frame.shape[0] * frame.shape[1]

            # Reject boxes covering > 30% of entire frame
            if box_area > 0.30 * img_area:
                continue
            # Cones are upright: reject boxes wider than 300px or with aspect ratio < 0.70
            if det.class_name == "cone":
                if bw > 300 or (bh / max(1.0, bw)) < 0.70:
                    continue
            clean_detections.append(det)

        annotated_frame = detector.annotate_frame(frame, clean_detections)
        tracked_frame, stats = tracker.update(annotated_frame, clean_detections)

        out.write(tracked_frame)
        frame_count += 1
        if frame_count % 30 == 0:
            print(f"Processed {frame_count} frames...")
        if args.max_frames is not None and frame_count >= args.max_frames:
            print(f"[INFO] Reached requested limit of {args.max_frames} frames.")
            break

    cap.release()
    out.release()
    print(f"\n[COMPLETE] Successfully processed {frame_count} frames.")
    print(f"[OUTPUT] Optical flow tracked video saved to: {args.output}")


if __name__ == "__main__":
    main()
