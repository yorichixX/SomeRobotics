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


def generate_realistic_moving_video(
    output_path: str = "data/sample_videos/moving_cones_demo.mp4",
    num_frames: int = 90,
    width: int = 1280,
    height: int = 720,
) -> str:
    """Generate a realistic video featuring moving cones to showcase optical flow tracking."""
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(output_path, fourcc, 30.0, (width, height))

    # Attempt to load a real street/road background image from dataset
    bg_frame = None
    for cand_dir in [
        "data/dataset_clean/images/val",
        "data/dataset/images/val",
    ]:
        if os.path.exists(cand_dir):
            cands = [
                os.path.join(cand_dir, f)
                for f in os.listdir(cand_dir)
                if f.lower().endswith((".jpg", ".png", ".jpeg"))
            ]
            if cands:
                bg = cv2.imread(cands[0])
                if bg is not None:
                    bg_frame = cv2.resize(bg, (width, height))
                    break

    # If no local real background image found, generate a photorealistic textured road canvas
    if bg_frame is None:
        bg_frame = np.full((height, width, 3), (72, 70, 68), dtype=np.uint8)
        # Add asphalt surface texture
        noise = np.random.randint(-8, 8, (height, width, 3), dtype=np.int16)
        bg_frame = np.clip(bg_frame.astype(np.int16) + noise, 0, 255).astype(np.uint8)
        horizon_y = int(height * 0.40)
        bg_frame[:horizon_y, :] = (215, 185, 145)  # Sky

        # Center dashed yellow line
        for y in range(horizon_y + 20, height, 40):
            cv2.line(bg_frame, (width // 2, y), (width // 2, min(y + 20, height)), (0, 215, 255), 4)
        # Road shoulder solid lines
        cv2.line(bg_frame, (int(width * 0.45), horizon_y), (int(width * 0.15), height), (240, 240, 240), 3)
        cv2.line(bg_frame, (int(width * 0.55), horizon_y), (int(width * 0.85), height), (240, 240, 240), 3)

    # Simulated motion: obstacle drifts smoothly across the road lane
    start_x, start_y = int(width * 0.35), int(height * 0.65)
    end_x, end_y = int(width * 0.68), int(height * 0.72)

    for i in range(num_frames):
        frame = bg_frame.copy()
        alpha = i / float(num_frames)
        cone_x = int(start_x + (end_x - start_x) * alpha)
        cone_y = int(start_y + (end_y - start_y) * alpha)

        # Scale cone with perspective (larger as it gets closer to camera)
        scale = 1.0 + 0.3 * alpha
        cone_h = int(110 * scale)
        cone_w = int(55 * scale)

        # 1. Black rubber base
        base_w = int(cone_w * 1.3)
        base_h = int(cone_h * 0.15)
        cv2.rectangle(
            frame,
            (cone_x - base_w // 2, cone_y - base_h),
            (cone_x + base_w // 2, cone_y),
            (25, 25, 25),
            -1,
        )

        # 2. Orange cone body (tapered polygon)
        cone_pts = np.array([
            [cone_x, cone_y - cone_h],
            [cone_x - cone_w // 2, cone_y - base_h],
            [cone_x + cone_w // 2, cone_y - base_h],
        ], np.int32)
        cv2.fillPoly(frame, [cone_pts], (0, 130, 255))

        # 3. 3D shadow shading on right half of cone
        shadow_pts = np.array([
            [cone_x, cone_y - cone_h],
            [cone_x, cone_y - base_h],
            [cone_x + cone_w // 2, cone_y - base_h],
        ], np.int32)
        cv2.fillPoly(frame, [shadow_pts], (0, 105, 210))

        # 4. White retroreflective stripe
        stripe_y1 = cone_y - int(cone_h * 0.60)
        stripe_y2 = cone_y - int(cone_h * 0.35)
        w1 = int(cone_w * 0.35)
        w2 = int(cone_w * 0.65)
        stripe_pts = np.array([
            [cone_x - w1 // 2, stripe_y1],
            [cone_x + w1 // 2, stripe_y1],
            [cone_x + w2 // 2, stripe_y2],
            [cone_x - w2 // 2, stripe_y2],
        ], np.int32)
        cv2.fillPoly(frame, [stripe_pts], (240, 240, 240))

        out.write(frame)

    out.release()
    print(f"Generated realistic test video sequence: {output_path}")
    return output_path


def main():
    parser = argparse.ArgumentParser(description="Run Optical Flow Tracking on Navigation Objects")
    parser.add_argument("--video", type=str, default=None, help="Input video file path")
    parser.add_argument("--model", type=str, default="weights/best.pt", help="Path to object detector model weights")
    parser.add_argument("--use-real-video", action="store_true", help="Download and evaluate on real autonomous driving road footage")
    parser.add_argument("--output", type=str, default="data/output/tracked_flow_demo.mp4", help="Output video path")

    args = parser.parse_args()

    video_path = args.video
    if args.use_real_video:
        video_path = download_real_sample_video()
    elif video_path is None or not os.path.exists(video_path):
        print("[INFO] No input video specified. Generating realistic test video sequence...")
        video_path = generate_realistic_moving_video()

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

    cap.release()
    out.release()
    print(f"\n[COMPLETE] Successfully processed {frame_count} frames.")
    print(f"[OUTPUT] Optical flow tracked video saved to: {args.output}")


if __name__ == "__main__":
    main()
