"""Bird's-Eye View (BEV) Homography / Inverse Perspective Mapping (IPM)."""

from typing import List, Tuple, Optional
import cv2
import numpy as np

from src.models.detector import Detection


class BirdEyeViewMapper:
    """Transforms front-facing perspective camera images into top-down Bird's-Eye View.
    
    Uses Homography / Inverse Perspective Mapping (IPM) to project detected
    obstacles onto a top-down metric coordinate plane for robotics path planning.
    """

    def __init__(
        self,
        bev_width: int = 400,
        bev_height: int = 600,
        max_distance_m: float = 15.0,
        lateral_range_m: float = 5.0,
    ):
        self.bev_width = bev_width
        self.bev_height = bev_height
        self.max_distance_m = max_distance_m
        self.lateral_range_m = lateral_range_m

        # Scale factors (pixels per meter)
        self.px_per_m_y = bev_height / max_distance_m
        self.px_per_m_x = bev_width / (2.0 * lateral_range_m)

        self.H: Optional[np.ndarray] = None
        self.H_inv: Optional[np.ndarray] = None

    def compute_homography_from_points(
        self,
        src_points: np.ndarray,
        dst_points: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """Compute 3x3 Homography matrix H using 4 corresponding ground plane points."""
        if dst_points is None:
            # Default destination rectangle in top-down BEV image
            pad_x = int(self.bev_width * 0.25)
            dst_points = np.array(
                [
                    [pad_x, int(self.bev_height * 0.2)],                     # Top-Left
                    [self.bev_width - pad_x, int(self.bev_height * 0.2)],    # Top-Right
                    [self.bev_width - pad_x, int(self.bev_height * 0.9)],    # Bottom-Right
                    [pad_x, int(self.bev_height * 0.9)],                     # Bottom-Left
                ],
                dtype=np.float32,
            )

        self.H, _ = cv2.findHomography(src_points, dst_points)
        self.H_inv = np.linalg.inv(self.H) if self.H is not None else None
        return self.H

    def calibrate_default_trapezoid(self, img_w: int = 1280, img_h: int = 720) -> np.ndarray:
        """Calibrate default perspective ground trapezoid for standard front camera."""
        # 4 source points on ground plane in perspective image
        src_pts = np.array(
            [
                [img_w * 0.42, img_h * 0.55],  # Top-left road horizon
                [img_w * 0.58, img_h * 0.55],  # Top-right road horizon
                [img_w * 0.90, img_h * 0.95],  # Bottom-right near bumper
                [img_w * 0.10, img_h * 0.95],  # Bottom-left near bumper
            ],
            dtype=np.float32,
        )
        return self.compute_homography_from_points(src_pts)

    def warp_image(self, image: np.ndarray) -> np.ndarray:
        """Warp entire perspective camera frame to top-down bird's-eye view."""
        if self.H is None:
            self.calibrate_default_trapezoid(image.shape[1], image.shape[0])
        return cv2.warpPerspective(image, self.H, (self.bev_width, self.bev_height))

    def render_bev_grid(
        self,
        detections: List[Detection],
        background_image: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """Render a high-tech top-down radar navigation grid displaying detected obstacles."""
        if background_image is not None:
            bev_canvas = cv2.resize(background_image, (self.bev_width, self.bev_height))
        else:
            # Dark navy radar background
            bev_canvas = np.full((self.bev_height, self.bev_width, 3), (25, 20, 15), dtype=np.uint8)

        robot_x = self.bev_width // 2
        robot_y = int(self.bev_height * 0.95)

        # Draw distance range rings (every 2.5m)
        for dist_m in np.arange(2.5, self.max_distance_m + 0.1, 2.5):
            radius_px = int(dist_m * self.px_per_m_y)
            cv2.ellipse(
                bev_canvas,
                (robot_x, robot_y),
                (radius_px, radius_px),
                0,
                180,
                360,
                (60, 50, 45),
                1,
                cv2.LINE_AA,
            )
            # Distance text label
            cv2.putText(
                bev_canvas,
                f"{dist_m:.1f}m",
                (robot_x + 5, robot_y - radius_px + 12),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.35,
                (120, 110, 100),
                1,
                cv2.LINE_AA,
            )

        # Draw lateral guide lines
        cv2.line(bev_canvas, (robot_x, 0), (robot_x, robot_y), (60, 50, 45), 1, cv2.LINE_AA)

        # Draw robot marker (triangular arrow at bottom)
        robot_pts = np.array(
            [[robot_x, robot_y - 12], [robot_x - 10, robot_y + 8], [robot_x + 10, robot_y + 8]],
            np.int32,
        )
        cv2.fillPoly(bev_canvas, [robot_pts], (255, 200, 50))
        cv2.putText(bev_canvas, "ROBOT", (robot_x - 18, robot_y + 22), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (255, 200, 50), 1)

        # Project and draw detected objects on the BEV grid
        for det in detections:
            if det.distance_m is None or det.distance_m <= 0:
                continue

            # Calculate top-down position
            dist_y_px = int(det.distance_m * self.px_per_m_y)
            obj_y = robot_y - dist_y_px

            if obj_y < 0 or obj_y > self.bev_height:
                continue

            # Lateral offset calculated from bounding box horizontal position relative to center
            # X_m = Z * (u_center - cx) / fx
            x1, y1, x2, y2 = det.box
            u_center = (x1 + x2) / 2.0
            cx = 640.0  # Image center
            fx = 1108.5
            lateral_x_m = det.distance_m * (u_center - cx) / fx

            obj_x = int(robot_x + lateral_x_m * self.px_per_m_x)
            obj_x = np.clip(obj_x, 15, self.bev_width - 15)

            # Object color
            color = (0, 140, 255) if det.class_name == "cone" else ((30, 30, 220) if det.class_name == "stop_sign" else (20, 220, 240))

            # Draw obstacle dot with glow
            cv2.circle(bev_canvas, (obj_x, obj_y), 9, color, -1)
            cv2.circle(bev_canvas, (obj_x, obj_y), 13, (255, 255, 255), 1)

            # Draw annotation text
            label = f"{det.class_name[:4].upper()} {det.distance_m:.1f}m"
            cv2.putText(
                bev_canvas,
                label,
                (obj_x - 25, obj_y - 12),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.35,
                (255, 255, 255),
                1,
                cv2.LINE_AA,
            )

        return bev_canvas

    def create_composite_display(
        self,
        front_view: np.ndarray,
        bev_view: np.ndarray,
    ) -> np.ndarray:
        """Combine front perspective camera frame with side-by-side Bird's-Eye View map."""
        fh, fw = front_view.shape[:2]
        # Resize BEV canvas to match front view height
        bev_resized = cv2.resize(bev_view, (int(self.bev_width * (fh / self.bev_height)), fh))
        return np.hstack([front_view, bev_resized])
