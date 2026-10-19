"""Geometric monocular distance estimation using object dimension priors and camera intrinsics."""

from typing import Dict, Any, Optional, Tuple, List
import yaml
import numpy as np

from .camera_model import PinholeCameraModel


class GeometricDistanceEstimator:
    """Estimates 3D distance Z (meters) from 2D bounding boxes using pinhole camera geometry.
    
    Formula:
        Z = (focal_length * real_dimension) / pixel_dimension
    """

    # Default fallback priors if no YAML configuration is provided
    DEFAULT_PRIORS: Dict[str, Dict[str, float]] = {
        "cone": {
            "name": "Cone",
            "real_height_m": 0.70,
            "real_width_m": 0.38,
            "min_expected_distance_m": 0.3,
            "max_expected_distance_m": 50.0,
        },
        "barrier": {
            "name": "Barrier",
            "real_height_m": 0.85,
            "real_width_m": 1.50,
            "min_expected_distance_m": 0.5,
            "max_expected_distance_m": 80.0,
        },
        "stop_sign": {
            "name": "Stop Sign",
            "real_height_m": 0.75,
            "real_width_m": 0.75,
            "min_expected_distance_m": 0.5,
            "max_expected_distance_m": 100.0,
        },
    }

    # Class name alias mapping to handle different dataset label conventions
    CLASS_ALIASES: Dict[str, str] = {
        "cone": "cone",
        "traffic cone": "cone",
        "traffic_cone": "cone",
        "barrier": "barrier",
        "traffic barrier": "barrier",
        "traffic_barrier": "barrier",
        "jersey_barrier": "barrier",
        "stop sign": "stop_sign",
        "stop_sign": "stop_sign",
        "stopsign": "stop_sign",
    }

    def __init__(
        self,
        camera: Optional[PinholeCameraModel] = None,
        priors_yaml_path: Optional[str] = None,
        use_ground_fusion: bool = True,
    ):
        self.camera = camera if camera is not None else PinholeCameraModel()
        self.use_ground_fusion = use_ground_fusion
        self.priors = self._load_priors(priors_yaml_path)

    def _load_priors(self, yaml_path: Optional[str]) -> Dict[str, Dict[str, Any]]:
        """Load object physical dimensions from YAML file or return defaults."""
        if yaml_path is None:
            return self.DEFAULT_PRIORS.copy()
        try:
            with open(yaml_path, "r") as f:
                data = yaml.safe_load(f)
            return data if data else self.DEFAULT_PRIORS.copy()
        except Exception:
            return self.DEFAULT_PRIORS.copy()

    def normalize_class_name(self, raw_name: str) -> str:
        """Map raw detection class name to standardized key."""
        clean = raw_name.strip().lower().replace("-", "_")
        return self.CLASS_ALIASES.get(clean, clean)

    def estimate_distance(
        self,
        class_name: str,
        bbox: Tuple[float, float, float, float],
    ) -> Optional[float]:
        """Estimate distance Z in meters for a detected bounding box.
        
        Args:
            class_name: Target object class (e.g., 'cone', 'stop_sign', 'barrier')
            bbox: Bounding box in [xmin, ymin, xmax, ymax] pixel format
            
        Returns:
            Estimated distance Z in meters, or None if estimation fails.
        """
        key = self.normalize_class_name(class_name)
        if key not in self.priors:
            return None

        prior = self.priors[key]
        xmin, ymin, xmax, ymax = bbox
        bbox_w = max(1.0, float(xmax - xmin))
        bbox_h = max(1.0, float(ymax - ymin))

        # 1. Height-based geometric estimation (Primary):
        #    Z_h = (fy * H_real) / h_bbox
        real_h = prior.get("real_height_m", 0.70)
        z_height = (self.camera.focal_length_y * real_h) / bbox_h

        # 2. Width-based geometric estimation (Auxiliary):
        #    Z_w = (fx * W_real) / w_bbox
        real_w = prior.get("real_width_m")
        z_width = None
        if real_w is not None and real_w > 0:
            z_width = (self.camera.focal_length_x * real_w) / bbox_w

        # 3. Ground plane estimation (using bottom-edge contact point):
        z_ground = None
        if self.use_ground_fusion and ymax > self.camera.principal_point_y:
            z_ground = self.camera.estimate_ground_distance(ymax)

        # 4. Fusion / Selection Strategy:
        # Check if bbox is clipped by frame borders (partial visibility)
        is_clipped_top = ymin <= 2.0
        is_clipped_bottom = ymax >= (self.camera.image_height - 2.0)

        if is_clipped_top or is_clipped_bottom:
            # If height is clipped, prefer width or ground estimation
            if z_width is not None:
                final_z = z_width
            elif z_ground is not None:
                final_z = z_ground
            else:
                final_z = z_height
        else:
            # Height is unclipped and reliable
            if z_ground is not None and abs(z_ground - z_height) < 0.35 * z_height:
                # Weighted fusion: 70% pinhole height, 30% ground projection
                final_z = 0.70 * z_height + 0.30 * z_ground
            else:
                final_z = z_height

        # Clamp to realistic physical range
        min_d = prior.get("min_expected_distance_m", 0.2)
        max_d = prior.get("max_expected_distance_m", 120.0)
        final_z = float(np.clip(final_z, min_d, max_d))

        return round(final_z, 2)

    def format_annotation(self, class_name: str, distance_m: Optional[float]) -> str:
        """Format detection and distance into the exact required assignment string.
        
        Example outputs:
            "Cone, 1.5m"
            "Stop Sign, 3.2m"
            "Barrier, 4.0m"
        """
        key = self.normalize_class_name(class_name)
        prior = self.priors.get(key, {})
        display_name = prior.get("name", class_name.replace("_", " ").title())

        if distance_m is None:
            return display_name
        return f"{display_name}, {distance_m:.1f}m"
