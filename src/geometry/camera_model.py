"""Pinhole camera model for robotics perception."""

from typing import Tuple, Optional
import math
import numpy as np
import yaml


class PinholeCameraModel:
    """Represents a pinhole camera with intrinsic and extrinsic parameters.
    
    Standard coordinate frame (ROS / Computer Vision standard):
      - X: points right
      - Y: points down
      - Z: points forward (optical axis)
    """

    def __init__(
        self,
        image_width: int = 1280,
        image_height: int = 720,
        fov_horizontal_deg: Optional[float] = 60.0,
        focal_length_x: Optional[float] = None,
        focal_length_y: Optional[float] = None,
        principal_point_x: Optional[float] = None,
        principal_point_y: Optional[float] = None,
        camera_height_m: float = 1.20,
        camera_pitch_deg: float = 0.0,
    ):
        self.image_width = image_width
        self.image_height = image_height
        self.camera_height_m = camera_height_m
        self.camera_pitch_deg = camera_pitch_deg

        # If focal lengths are not explicitly provided, compute from FOV
        if focal_length_x is None:
            if fov_horizontal_deg is not None:
                self.focal_length_x = self.compute_focal_length_from_fov(
                    image_width, fov_horizontal_deg
                )
            else:
                self.focal_length_x = float(image_width)
        else:
            self.focal_length_x = float(focal_length_x)

        if focal_length_y is None:
            self.focal_length_y = self.focal_length_x
        else:
            self.focal_length_y = float(focal_length_y)

        # Optical center / principal point
        self.principal_point_x = (
            float(principal_point_x) if principal_point_x is not None else image_width / 2.0
        )
        self.principal_point_y = (
            float(principal_point_y) if principal_point_y is not None else image_height / 2.0
        )

        # Intrinsic Matrix K (3x3)
        self.K = np.array(
            [
                [self.focal_length_x, 0.0, self.principal_point_x],
                [0.0, self.focal_length_y, self.principal_point_y],
                [0.0, 0.0, 1.0],
            ],
            dtype=np.float64,
        )
        self.K_inv = np.linalg.inv(self.K)

    @classmethod
    def from_yaml(cls, yaml_path: str) -> "PinholeCameraModel":
        """Load camera model from a YAML configuration file."""
        with open(yaml_path, "r") as f:
            data = yaml.safe_load(f)
        return cls(
            image_width=data.get("image_width", 1280),
            image_height=data.get("image_height", 720),
            fov_horizontal_deg=data.get("fov_horizontal_deg", 60.0),
            focal_length_x=data.get("focal_length_x"),
            focal_length_y=data.get("focal_length_y"),
            principal_point_x=data.get("principal_point_x"),
            principal_point_y=data.get("principal_point_y"),
            camera_height_m=data.get("camera_height_m", 1.20),
            camera_pitch_deg=data.get("camera_pitch_deg", 0.0),
        )

    @staticmethod
    def compute_focal_length_from_fov(dimension_px: int, fov_deg: float) -> float:
        """Compute focal length in pixels from Field of View in degrees.
        
        Formula:
          f = (dim / 2) / tan(fov_rad / 2)
        """
        fov_rad = math.radians(fov_deg)
        return (dimension_px / 2.0) / math.tan(fov_rad / 2.0)

    def project_3d_to_2d(self, X: float, Y: float, Z: float) -> Tuple[float, float]:
        """Project a 3D point in camera frame (meters) to 2D image coordinates (pixels).
        
        Formula:
          u = fx * (X / Z) + cx
          v = fy * (Y / Z) + cy
        """
        if Z <= 1e-4:
            raise ValueError(f"Depth Z must be positive, got {Z}")
        u = self.focal_length_x * (X / Z) + self.principal_point_x
        v = self.focal_length_y * (Y / Z) + self.principal_point_y
        return u, v

    def backproject_pixel_to_ray(self, u: float, v: float) -> np.ndarray:
        """Back-project a 2D image pixel into a unit 3D direction vector in camera frame."""
        pixel_homo = np.array([u, v, 1.0], dtype=np.float64)
        ray_dir = self.K_inv @ pixel_homo
        norm = np.linalg.norm(ray_dir)
        if norm > 0:
            ray_dir = ray_dir / norm
        return ray_dir

    def estimate_ground_distance(self, v_bottom: float) -> Optional[float]:
        """Estimate forward distance Z using bottom edge of bounding box and ground plane.
        
        Assumes the object rests on the flat horizontal ground plane.
        Camera is at height h above ground with pitch angle theta.
        
        Formula:
          alpha = arctan((v_bottom - cy) / fy)
          Z = camera_height / tan(pitch_angle + alpha)
        """
        alpha_rad = math.atan((v_bottom - self.principal_point_y) / self.focal_length_y)
        pitch_rad = math.radians(self.camera_pitch_deg)
        total_angle = pitch_rad + alpha_rad

        # The bottom of the object must be below the horizon
        if total_angle <= 1e-4:
            return None  # At or above horizon
        z_ground = self.camera_height_m / math.tan(total_angle)
        return z_ground if z_ground > 0 else None
