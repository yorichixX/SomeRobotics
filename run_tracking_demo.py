"""Demo script for Optical Flow tracking of moving navigation objects (cones, barriers)."""

import argparse
import os
import cv2
import numpy as np

from src.models.detector import ObjectDetector, Detection
from src.extra.optical_flow_tracker import OpticalFlowTracker


def generate_synthetic_moving_video(
    output_path: str = "data/sample_videos/moving_cones_demo.mp4",
    num_frames: int = 90,
    width: int = 1280,
    height: int = 720,
) -> str:
    """Generate a test video featuring moving cones to showcase optical flow tracking."""
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(output_path, fourcc, 30.0, (width, height))

    horizon_y = int(height * 0.45)

    # Simulated motion of a traffic cone drifting across the road
    start_x, start_y = int(width * 0.35), int(height * 0.70)
    end_x, end_y = int(width * 0.65), int(height * 0.78)

    for i in range(num_frames):
        alpha = i / float(num_frames)
        cone_x = int(start_x + (end_x - start_x) * alpha)
        cone_y = int(start_y + (end_y - start_y) * alpha)

        # Base road scene
        frame = np.full((height, width, 3), (60, 60, 60), dtype=np.uint8)
        frame[:horizon_y, :] = (200, 160, 120)  # sky

        # Draw road
        pts = np.array(
            [[width * 0.4, horizon_y], [width * 0.6, horizon_y], [width * 0.95, height], [width * 0.05, height]],
            np.int32,
        )
        cv2.fillPoly(frame, [pts], (75, 75, 75))

        # Draw cone (orange triangle)
        cone_h = 90
        cone_w = 45
        c_pts = np.array(
            [[cone_x, cone_y - cone_h], [cone_x - cone_w // 2, cone_y], [cone_x + cone_w // 2, cone_y]],
            np.int32,
        )
        cv2.fillPoly(frame, [c_pts], (0, 140, 255))
        # White stripe
        cv2.rectangle(
            frame,
            (cone_x - cone_w // 4, cone_y - int(cone_h * 0.55)),
            (cone_x + cone_w // 4, cone_y - int(cone_h * 0.35)),
            (240, 240, 240),
            -1,
        )

        out.write(frame)

    out.release()
    print(f"Generated synthetic video with moving cone: {output_path}")
    return output_path


def main():
    parser = argparse.ArgumentParser(description="Run Optical Flow Tracking on Navigation Objects")
    parser.add_argument("--video", type=str, default=None, help="Input video file path")
    parser.add_argument("--model", type=str, default="weights/best.pt", help="Path to object detector model weights")
    parser.add_argument("--output", type=str, default="data/output/tracked_flow_demo.mp4", help="Output video path")

    args = parser.parse_args()

    video_path = args.video
    if video_path is None or not os.path.exists(video_path):
        print("[INFO] No input video specified. Generating synthetic video sequence...")
        video_path = generate_synthetic_moving_video()

    tracker = OpticalFlowTracker()
    model_to_use = args.model if (args.model and os.path.exists(args.model)) else None
    detector = ObjectDetector(model_path=model_to_use)

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

        detections = detector.detect(frame)
        if len(detections) == 0 and args.video is None:
            detections = detector._heuristic_fallback_detect(frame)

        annotated_frame = detector.annotate_frame(frame, detections)
        tracked_frame, stats = tracker.update(annotated_frame, detections)

        out.write(tracked_frame)
        frame_count += 1

    cap.release()
    out.release()
    print(f"\n[COMPLETE] Processed {frame_count} frames. Output saved to: {args.output}")


if __name__ == "__main__":
    main()
