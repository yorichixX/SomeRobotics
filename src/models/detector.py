"""Object detector module for robotics navigation with integrated distance estimation."""

from typing import List, Tuple, Optional, Dict, Any
from dataclasses import dataclass
import os
import cv2
import numpy as np

from src.geometry.camera_model import PinholeCameraModel
from src.geometry.distance_estimator import GeometricDistanceEstimator


@dataclass
class Detection:
    """Represents a single detected object in an image frame."""
    box: Tuple[float, float, float, float]  # [xmin, ymin, xmax, ymax] in pixels
    score: float                            # Confidence score [0.0, 1.0]
    class_id: int                           # Numeric class ID
    class_name: str                         # Class name (e.g. 'cone', 'barrier', 'stop_sign')
    distance_m: Optional[float] = None      # Estimated 3D distance in meters


class ObjectDetector:
    """Modular object detector supporting PyTorch (YOLO), ONNX Runtime, and distance estimation."""

    CLASS_NAMES: Dict[int, str] = {
        0: "cone",
        1: "barrier",
        2: "stop_sign",
    }

    # BGR color mapping for distinct visualization
    CLASS_COLORS: Dict[str, Tuple[int, int, int]] = {
        "cone": (0, 140, 255),       # Orange
        "barrier": (20, 220, 240),   # Yellow
        "stop_sign": (30, 30, 220),  # Red
    }

    def __init__(
        self,
        model_path: Optional[str] = None,
        conf_threshold: float = 0.35,
        iou_threshold: float = 0.45,
        device: str = "cpu",
        camera: Optional[PinholeCameraModel] = None,
        estimator: Optional[GeometricDistanceEstimator] = None,
    ):
        self.model_path = model_path
        self.conf_threshold = conf_threshold
        self.iou_threshold = iou_threshold
        self.device = device

        # Initialize camera and distance estimator
        self.camera = camera if camera is not None else PinholeCameraModel()
        self.estimator = (
            estimator
            if estimator is not None
            else GeometricDistanceEstimator(camera=self.camera)
        )

        self.model = None
        self.is_onnx = False

        if model_path is not None and os.path.exists(model_path):
            self._load_model(model_path)

    def _load_model(self, model_path: str) -> None:
        """Load YOLO model or ONNX session."""
        if model_path.endswith(".onnx"):
            self.is_onnx = True
            try:
                import onnxruntime as ort
                providers = (
                    ["CUDAExecutionProvider", "CPUExecutionProvider"]
                    if self.device == "cuda"
                    else ["CPUExecutionProvider"]
                )
                self.model = ort.InferenceSession(model_path, providers=providers)
            except ImportError:
                print("Warning: onnxruntime not installed, cannot load ONNX model.")
        else:
            try:
                from ultralytics import YOLO
                self.model = YOLO(model_path)
            except ImportError:
                print("Warning: ultralytics not installed.")

    def detect(self, image: np.ndarray) -> List[Detection]:
        """Detect navigation objects in image and estimate their distance.
        
        Args:
            image: BGR image array (H, W, C)
            
        Returns:
            List of Detection objects with boxes, scores, classes, and distances.
        """
        detections: List[Detection] = []

        if self.model is None:
            # If no trained weights loaded yet, fallback to heuristic color/contour detection
            # to verify pipeline end-to-end immediately without weights
            return self._heuristic_fallback_detect(image)

        if self.is_onnx:
            detections = self._infer_onnx(image)
        else:
            detections = self._infer_ultralytics(image)

        # Estimate distance for each detected object
        for det in detections:
            det.distance_m = self.estimator.estimate_distance(det.class_name, det.box)

        return detections

    def _infer_ultralytics(self, image: np.ndarray) -> List[Detection]:
        """Run inference using Ultralytics YOLO model."""
        results = self.model.predict(
            source=image,
            conf=self.conf_threshold,
            iou=self.iou_threshold,
            device=self.device,
            verbose=False,
        )
        detections = []
        for r in results:
            boxes = r.boxes
            for b in boxes:
                xyxy = b.xyxy[0].cpu().numpy().tolist()
                conf = float(b.conf[0].cpu().numpy())
                cls_id = int(b.cls[0].cpu().numpy())
                cls_name = self.CLASS_NAMES.get(cls_id, f"class_{cls_id}")

                detections.append(
                    Detection(
                        box=(xyxy[0], xyxy[1], xyxy[2], xyxy[3]),
                        score=conf,
                        class_id=cls_id,
                        class_name=cls_name,
                    )
                )
        return detections

    def _infer_onnx(self, image: np.ndarray) -> List[Detection]:
        """Run inference using ONNX Runtime session."""
        h, w = image.shape[:2]
        input_size = 640

        # Preprocessing: letterbox resize and normalize
        resized = cv2.resize(image, (input_size, input_size))
        rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
        tensor = rgb.astype(np.float32) / 255.0
        tensor = np.transpose(tensor, (2, 0, 1))[None, ...]

        input_name = self.model.get_inputs()[0].name
        outputs = self.model.run(None, {input_name: tensor})
        # Standard YOLOv8 ONNX output shape: (1, 4 + num_classes, num_anchors)
        out = outputs[0][0]
        if out.shape[0] < out.shape[1]:
            out = out.T

        boxes = out[:, :4]
        scores = out[:, 4:]

        class_ids = np.argmax(scores, axis=1)
        confidences = np.max(scores, axis=1)

        mask = confidences >= self.conf_threshold
        boxes = boxes[mask]
        class_ids = class_ids[mask]
        confidences = confidences[mask]

        if len(boxes) == 0:
            return []

        # Convert cx, cy, w, h to xmin, ymin, xmax, ymax and scale back
        scale_x = w / input_size
        scale_y = h / input_size

        detections = []
        for box, conf, cid in zip(boxes, confidences, class_ids):
            cx, cy, bw, bh = box
            x1 = (cx - bw / 2.0) * scale_x
            y1 = (cy - bh / 2.0) * scale_y
            x2 = (cx + bw / 2.0) * scale_x
            y2 = (cy + bh / 2.0) * scale_y

            cls_name = self.CLASS_NAMES.get(int(cid), f"class_{cid}")
            detections.append(
                Detection(
                    box=(x1, y1, x2, y2),
                    score=float(conf),
                    class_id=int(cid),
                    class_name=cls_name,
                )
            )

        return self._apply_nms(detections)

    def _apply_nms(self, detections: List[Detection]) -> List[Detection]:
        """Apply Non-Maximum Suppression to detections."""
        if not detections:
            return []
        boxes = [d.box for d in detections]
        scores = [d.score for d in detections]

        indices = cv2.dnn.NMSBoxes(
            bboxes=[[b[0], b[1], b[2] - b[0], b[3] - b[1]] for b in boxes],
            scores=scores,
            score_threshold=self.conf_threshold,
            nms_threshold=self.iou_threshold,
        )
        if len(indices) == 0:
            return []
        return [detections[i] for i in indices.flatten()]

    def _heuristic_fallback_detect(self, image: np.ndarray) -> List[Detection]:
        """Heuristic color segmentation fallback when deep model weights are not loaded.
        
        Ensures the pipeline is testable and verifiable immediately.
        """
        detections = []
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        h, w = image.shape[:2]

        # 1. Orange detection for cones
        mask_orange = cv2.inRange(hsv, (5, 120, 120), (22, 255, 255))
        contours, _ = cv2.findContours(mask_orange, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area > 150:
                x, y, bw, bh = cv2.boundingRect(cnt)
                aspect = bh / max(1, bw)
                if aspect >= 1.1:  # Cones are taller than wide
                    box = (float(x), float(y), float(x + bw), float(y + bh))
                    dist = self.estimator.estimate_distance("cone", box)
                    detections.append(
                        Detection(box=box, score=0.88, class_id=0, class_name="cone", distance_m=dist)
                    )

        # 2. Red detection for stop signs (wrap-around HSV)
        mask_red1 = cv2.inRange(hsv, (0, 120, 70), (10, 255, 255))
        mask_red2 = cv2.inRange(hsv, (170, 120, 70), (180, 255, 255))
        mask_red = cv2.bitwise_or(mask_red1, mask_red2)
        contours, _ = cv2.findContours(mask_red, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area > 200:
                x, y, bw, bh = cv2.boundingRect(cnt)
                aspect = bh / max(1, bw)
                if 0.7 <= aspect <= 1.4:  # Stop signs are approximately circular/octagonal
                    box = (float(x), float(y), float(x + bw), float(y + bh))
                    dist = self.estimator.estimate_distance("stop_sign", box)
                    detections.append(
                        Detection(box=box, score=0.92, class_id=2, class_name="stop_sign", distance_m=dist)
                    )

        return detections

    def annotate_frame(
        self,
        image: np.ndarray,
        detections: List[Detection],
        line_thickness: int = 2,
    ) -> np.ndarray:
        """Draw bounding boxes and formatted distance annotations on the frame.
        
        Required Assignment Format:
            "Cone, 1.5m"
            "Stop Sign, 3.2m"
        """
        annotated = image.copy()

        for det in detections:
            x1, y1, x2, y2 = [int(v) for v in det.box]
            color = self.CLASS_COLORS.get(det.class_name, (0, 255, 0))

            # Draw bounding box
            cv2.rectangle(annotated, (x1, y1), (x2, y2), color, line_thickness)

            # Draw corner accents for high-tech robotics HUD look
            corner_len = min(15, (x2 - x1) // 4, (y2 - y1) // 4)
            cv2.line(annotated, (x1, y1), (x1 + corner_len, y1), color, line_thickness + 1)
            cv2.line(annotated, (x1, y1), (x1, y1 + corner_len), color, line_thickness + 1)
            cv2.line(annotated, (x2, y1), (x2 - corner_len, y1), color, line_thickness + 1)
            cv2.line(annotated, (x2, y1), (x2, y1 + corner_len), color, line_thickness + 1)

            # Format label string: e.g. "Cone, 1.5m"
            label = self.estimator.format_annotation(det.class_name, det.distance_m)

            # Badge background for high legibility
            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 0.55
            thickness = 1
            (lbl_w, lbl_h), baseline = cv2.getTextSize(label, font, font_scale, thickness)

            badge_y1 = max(0, y1 - lbl_h - 8)
            badge_y2 = y1
            badge_x2 = min(annotated.shape[1], x1 + lbl_w + 10)

            # Solid badge background
            cv2.rectangle(
                annotated,
                (x1, badge_y1),
                (badge_x2, badge_y2),
                color,
                -1,
            )

            # White text
            cv2.putText(
                annotated,
                label,
                (x1 + 5, y1 - 4),
                font,
                font_scale,
                (255, 255, 255),
                thickness,
                cv2.LINE_AA,
            )

        return annotated
