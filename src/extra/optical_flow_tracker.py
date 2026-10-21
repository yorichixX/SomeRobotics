"""Optical flow tracking for moving navigation objects (cones, barriers)."""

from typing import List, Tuple, Dict, Optional, Any
from collections import deque
import cv2
import numpy as np

from src.models.detector import Detection


class OpticalFlowTracker:
    """Tracks moving objects across video frames using Lucas-Kanade Optical Flow."""

    def __init__(
        self,
        max_corners: int = 20,
        quality_level: float = 0.15,
        min_distance: float = 6,
        lk_win_size: Tuple[int, int] = (15, 15),
        max_level: int = 2,
        trail_length: int = 25,
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
        self.tracked_pts: Optional[np.ndarray] = None
        self.trajectory: deque = deque(maxlen=trail_length)

    def reset(self) -> None:
        """Reset tracking state."""
        self.prev_gray = None
        self.tracked_pts = None
        self.trajectory.clear()

    def update(
        self,
        frame: np.ndarray,
        detections: List[Detection],
    ) -> Tuple[np.ndarray, List[Dict[str, Any]]]:
        """Track objects and draw optical flow vectors with forward-backward validation.
        
        Args:
            frame: Current BGR image frame
            detections: Current frame detections
            
        Returns:
            Tuple of (annotated_frame_with_flow, motion_statistics)
        """
        annotated = frame.copy()
        current_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        motion_stats: List[Dict[str, Any]] = []

        # First frame: initialize points inside detected bounding boxes
        if self.prev_gray is None:
            self.prev_gray = current_gray
            self._reseed_features(current_gray, detections)
            return annotated, motion_stats

        # Track existing points from prev_gray to current_gray using Lucas-Kanade
        if self.tracked_pts is not None and len(self.tracked_pts) > 0:
            p1, st, _ = cv2.calcOpticalFlowPyrLK(
                self.prev_gray, current_gray, self.tracked_pts, None, **self.lk_params
            )

            # Forward-backward error validation (rejection of occlusions and false matches)
            p0_back, st_back, _ = cv2.calcOpticalFlowPyrLK(
                current_gray, self.prev_gray, p1, None, **self.lk_params
            )
            fb_err = np.linalg.norm(self.tracked_pts - p0_back, axis=2).ravel()

            # Filter valid points
            valid = (st.ravel() == 1) & (st_back.ravel() == 1) & (fb_err < 1.8)

            p1_valid = p1[valid]
            p0_valid = self.tracked_pts[valid]

            if len(p1_valid) > 0:
                dxs = p1_valid[:, 0, 0] - p0_valid[:, 0, 0]
                dys = p1_valid[:, 0, 1] - p0_valid[:, 0, 1]
                speeds = np.sqrt(dxs ** 2 + dys ** 2)

                # Filter out velocity outliers (e.g. speed > 35 px/frame)
                speed_mask = speeds < 35.0
                p1_clean = p1_valid[speed_mask]
                p0_clean = p0_valid[speed_mask]
                dxs_clean = dxs[speed_mask]
                dys_clean = dys[speed_mask]

                if len(p1_clean) > 0:
                    med_dx = float(np.median(dxs_clean))
                    med_dy = float(np.median(dys_clean))
                    med_speed = float(np.sqrt(med_dx ** 2 + med_dy ** 2))

                    motion_stats.append({
                        "tracked_points": len(p1_clean),
                        "mean_velocity_x": round(med_dx, 2),
                        "mean_velocity_y": round(med_dy, 2),
                        "speed_px_per_frame": round(med_speed, 2),
                    })

                    # Draw point-wise motion vectors
                    for (new_pt, old_pt) in zip(p1_clean, p0_clean):
                        nx, ny = int(new_pt[0, 0]), int(new_pt[0, 1])
                        ox, oy = int(old_pt[0, 0]), int(old_pt[0, 1])
                        cv2.circle(annotated, (nx, ny), 3, (0, 255, 0), -1)
                        # Clamped vector arrow
                        arrow_end_x = int(ox + (nx - ox) * 2.0)
                        arrow_end_y = int(oy + (ny - oy) * 2.0)
                        cv2.arrowedLine(annotated, (ox, oy), (arrow_end_x, arrow_end_y), (0, 255, 255), 1, tipLength=0.3)

                    # Draw prominent object centroid motion vector & trajectory trail
                    for det in detections:
                        cx = int((det.box[0] + det.box[2]) / 2.0)
                        cy = int((det.box[1] + det.box[3]) / 2.0)
                        self.trajectory.append((cx, cy))

                        # Draw fading trajectory trail
                        pts_list = list(self.trajectory)
                        for t_idx in range(1, len(pts_list)):
                            alpha = t_idx / float(len(pts_list))
                            color_val = int(255 * alpha)
                            cv2.line(annotated, pts_list[t_idx - 1], pts_list[t_idx], (color_val, color_val, 0), 2)

                        # Draw bold centroid vector arrow
                        if med_speed > 0.5:
                            vec_end_x = int(cx + med_dx * 5.0)
                            vec_end_y = int(cy + med_dy * 5.0)
                            cv2.arrowedLine(annotated, (cx, cy), (vec_end_x, vec_end_y), (0, 255, 255), 3, tipLength=0.3)
                            cv2.circle(annotated, (cx, cy), 5, (0, 255, 255), -1)

                    # High-visibility HUD status overlay
                    status_text = "MOVING" if med_speed >= 0.8 else "STATIC"
                    dir_text = ""
                    if med_speed >= 0.8:
                        if abs(med_dx) > abs(med_dy):
                            dir_text = " RIGHT" if med_dx > 0 else " LEFT"
                        else:
                            dir_text = " FORWARD" if med_dy > 0 else " RETREATING"

                    hud_str = (
                        f"[LUCAS-KANADE FLOW] Pts: {len(p1_clean)} | "
                        f"Vel: ({med_dx:+.1f}, {med_dy:+.1f}) px/f | "
                        f"Speed: {med_speed:.1f} px/f | {status_text}{dir_text}"
                    )
                    # Semi-transparent background banner
                    cv2.rectangle(annotated, (10, 12), (760, 48), (20, 20, 20), -1)
                    cv2.putText(
                        annotated,
                        hud_str,
                        (20, 36),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.58,
                        (0, 255, 255),
                        2,
                        cv2.LINE_AA,
                    )

                    # Update persistent points: keep those still inside active detections
                    self.tracked_pts = self._filter_and_reseed_points(current_gray, p1_clean, detections)
            else:
                self._reseed_features(current_gray, detections)
        else:
            self._reseed_features(current_gray, detections)

        self.prev_gray = current_gray
        return annotated, motion_stats

    def _filter_and_reseed_points(
        self,
        gray: np.ndarray,
        current_pts: np.ndarray,
        detections: List[Detection],
    ) -> Optional[np.ndarray]:
        """Keep points inside active boxes or road corridor and reseed if point density is low."""
        if not detections:
            if len(current_pts) < 15:
                # Reseed road corridor
                h, w = gray.shape[:2]
                mask = np.zeros_like(gray)
                mask[int(h * 0.40):int(h * 0.95), int(w * 0.10):int(w * 0.90)] = 255
                corners = cv2.goodFeaturesToTrack(gray, mask=mask, maxCorners=40, qualityLevel=0.1, minDistance=12)
                if corners is not None:
                    return corners
            return current_pts.reshape(-1, 1, 2)

        surviving = []
        for det in detections:
            x1, y1, x2, y2 = [int(v) for v in det.box]
            pts_in_box = []
            for pt in current_pts:
                px, py = pt[0, 0], pt[0, 1]
                if x1 <= px <= x2 and y1 <= py <= y2:
                    pts_in_box.append(pt)

            if len(pts_in_box) < 6:
                # Point count low: extract fresh Shi-Tomasi corners
                roi = gray[y1:y2, x1:x2]
                if roi.size > 0:
                    corners = cv2.goodFeaturesToTrack(roi, mask=None, **self.feature_params)
                    if corners is not None:
                        for c in corners:
                            surviving.append([[c[0, 0] + x1, c[0, 1] + y1]])
            else:
                surviving.extend(pts_in_box)

        if surviving:
            return np.array(surviving, dtype=np.float32)
        return None

    def _reseed_features(self, gray: np.ndarray, detections: List[Detection]) -> None:
        """Seed fresh Shi-Tomasi corners inside all active detections, or road corridor if empty."""
        pts = []
        if detections:
            for det in detections:
                x1, y1, x2, y2 = [int(v) for v in det.box]
                x1, y1 = max(0, x1), max(0, y1)
                x2, y2 = min(gray.shape[1], x2), min(gray.shape[0], y2)
                roi = gray[y1:y2, x1:x2]
                if roi.size == 0:
                    continue
                corners = cv2.goodFeaturesToTrack(roi, mask=None, **self.feature_params)
                if corners is not None:
                    for c in corners:
                        pts.append([[c[0, 0] + x1, c[0, 1] + y1]])
        else:
            # Fallback: track road corridor motion (vehicles, obstacles, lane markings)
            h, w = gray.shape[:2]
            mask = np.zeros_like(gray)
            mask[int(h * 0.40):int(h * 0.95), int(w * 0.10):int(w * 0.90)] = 255
            corners = cv2.goodFeaturesToTrack(gray, mask=mask, maxCorners=40, qualityLevel=0.1, minDistance=12)
            if corners is not None:
                for c in corners:
                    pts.append([[c[0, 0], c[0, 1]]])

        if pts:
            self.tracked_pts = np.array(pts, dtype=np.float32)
        else:
            self.tracked_pts = None
