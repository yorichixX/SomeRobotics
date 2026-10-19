"""Optical flow tracking for moving navigation objects (cones, barriers)."""

from typing import List, Tuple, Dict, Optional, Any
import cv2
import numpy as np

from src.models.detector import Detection


class OpticalFlowTracker:
    """Tracks moving objects across video frames using Lucas-Kanade Optical Flow."""

    def __init__(
        self,
        max_corners: int = 15,
        quality_level: float = 0.3,
        min_distance: float = 7,
        lk_win_size: Tuple[int, int] = (15, 15),
        max_level: int = 2,
    ):
        self.feature_params = dict(
            maxCorners=max_corners,
            qualityLevel=quality_level,
            minDistance=min_distance,
            blockSize=7,
        )
        self.lk_params = dict(
            winSize=lk_win_size,
            maxLevel=max_level,
            criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03),
        )

        self.prev_gray: Optional[np.ndarray] = None
        self.tracked_tracks: List[List[Tuple[int, int]]] = []
        self.track_colors: List[Tuple[int, int, int]] = []

    def reset(self) -> None:
        """Reset tracking state."""
        self.prev_gray = None
        self.tracked_tracks.clear()
        self.track_colors.clear()

    def update(
        self,
        frame: np.ndarray,
        detections: List[Detection],
    ) -> Tuple[np.ndarray, List[Dict[str, Any]]]:
        """Track objects and draw optical flow vectors.
        
        Args:
            frame: Current BGR image frame
            detections: Current frame detections
            
        Returns:
            Tuple of (annotated_frame_with_flow, motion_statistics)
        """
        annotated = frame.copy()
        current_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        motion_stats: List[Dict[str, Any]] = []

        if self.prev_gray is None:
            self.prev_gray = current_gray
            return annotated, motion_stats

        # Extract features within current detection bounding boxes
        new_points = []
        for det in detections:
            x1, y1, x2, y2 = [int(v) for v in det.box]
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(frame.shape[1], x2), min(frame.shape[0], y2)

            roi_gray = self.prev_gray[y1:y2, x1:x2]
            if roi_gray.size == 0:
                continue

            corners = cv2.goodFeaturesToTrack(roi_gray, mask=None, **self.feature_params)
            if corners is not None:
                for pt in corners:
                    px, py = pt.ravel()
                    new_points.append([[float(px + x1), float(py + y1)]])

        if len(new_points) > 0:
            p0 = np.array(new_points, dtype=np.float32)

            # Calculate optical flow using Lucas-Kanade
            p1, st, err = cv2.calcOpticalFlowPyrLK(
                self.prev_gray, current_gray, p0, None, **self.lk_params
            )

            if p1 is not None:
                good_new = p1[st == 1]
                good_old = p0[st == 1]

                # Draw flow vectors and compute motion statistics
                velocities_x = []
                velocities_y = []

                for new_pt, old_pt in zip(good_new, good_old):
                    nx, ny = new_pt.ravel()
                    ox, oy = old_pt.ravel()

                    dx = nx - ox
                    dy = ny - oy
                    velocities_x.append(dx)
                    velocities_y.append(dy)

                    # Draw velocity vector arrow
                    cv2.arrowedLine(
                        annotated,
                        (int(ox), int(oy)),
                        (int(nx + dx * 2), int(ny + dy * 2)),
                        (0, 255, 255),
                        2,
                        tipLength=0.3,
                    )
                    cv2.circle(annotated, (int(nx), int(ny)), 3, (0, 255, 0), -1)

                if len(velocities_x) > 0:
                    mean_dx = float(np.mean(velocities_x))
                    mean_dy = float(np.mean(velocities_y))
                    speed = float(np.sqrt(mean_dx ** 2 + mean_dy ** 2))

                    motion_stats.append({
                        "tracked_points": len(velocities_x),
                        "mean_velocity_x": round(mean_dx, 2),
                        "mean_velocity_y": round(mean_dy, 2),
                        "speed_px_per_frame": round(speed, 2),
                    })

                    # Display optical flow HUD text
                    cv2.putText(
                        annotated,
                        f"Optical Flow: {len(velocities_x)} pts | Speed: {speed:.1f} px/f",
                        (20, 40),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.65,
                        (0, 255, 255),
                        2,
                        cv2.LINE_AA,
                    )

        self.prev_gray = current_gray
        return annotated, motion_stats
