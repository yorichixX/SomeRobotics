"""Unit tests for camera model and geometric distance estimator."""

import unittest
import math
import numpy as np

from src.geometry.camera_model import PinholeCameraModel
from src.geometry.distance_estimator import GeometricDistanceEstimator


def test_focal_length_from_fov():
    """Verify focal length computation from 60 degree FOV."""
    # For width=1280 and FOV=60 deg:
    # f = (1280 / 2) / tan(30 deg) = 640 / (1 / sqrt(3)) = 640 * 1.73205 = 1108.51
    f = PinholeCameraModel.compute_focal_length_from_fov(1280, 60.0)
    expected = 640.0 / math.tan(math.radians(30.0))
    assert math.isclose(f, expected, rel_tol=1e-3)


def test_project_and_distance_consistency():
    """Verify that a 3D object projected to 2D computes back to the exact original distance."""
    cam = PinholeCameraModel(image_width=1280, image_height=720, fov_horizontal_deg=60.0)
    estimator = GeometricDistanceEstimator(camera=cam, use_ground_fusion=False)

    target_distance = 5.0  # 5 meters
    cone_height = 0.70     # 70 cm

    # Projected pixel height: h_px = (fy * H_real) / Z
    h_px = (cam.focal_length_y * cone_height) / target_distance
    w_px = (cam.focal_length_x * 0.38) / target_distance

    # Create synthetic bounding box centered in image
    cx, cy = 640.0, 360.0
    bbox = (cx - w_px / 2, cy - h_px / 2, cx + w_px / 2, cy + h_px / 2)

    est_dist = estimator.estimate_distance("cone", bbox)
    assert est_dist is not None
    assert math.isclose(est_dist, target_distance, abs_tol=0.05)


def test_annotation_format():
    """Verify required assignment annotation format (e.g., 'Cone, 1.5m')."""
    cam = PinholeCameraModel()
    estimator = GeometricDistanceEstimator(camera=cam)

    label_cone = estimator.format_annotation("cone", 1.53)
    assert label_cone == "Cone, 1.5m"

    label_stop = estimator.format_annotation("stop_sign", 3.24)
    assert label_stop == "Stop Sign, 3.2m"

    label_barrier = estimator.format_annotation("barrier", 4.88)
    assert label_barrier == "Barrier, 4.9m"


def test_ground_distance_calculation():
    """Verify ground plane distance decreases as bounding box approaches bottom of frame."""
    cam = PinholeCameraModel(image_width=1280, image_height=720, camera_height_m=1.2, camera_pitch_deg=5.0)

    # Object closer to bottom of frame is physically closer to robot
    dist_far = cam.estimate_ground_distance(400.0)
    dist_near = cam.estimate_ground_distance(650.0)

    assert dist_far is not None
    assert dist_near is not None
    assert dist_near < dist_far
