# Autonomous Robotics Perception: Object Detection & Geometric Distance Estimation

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.2+-ee4c2c.svg)](https://pytorch.org/)
[![YOLOv8](https://img.shields.io/badge/YOLO-v8%20%2F%20v11-00FFFF.svg)](https://github.com/ultralytics/ultralytics)
[![ONNX](https://img.shields.io/badge/ONNX-INT8%20Quantized-005CED.svg)](https://onnxruntime.ai/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

An end-to-end, edge-optimized perception system for mobile robotics navigation. The system detects navigation obstacles (**cones**, **barriers**, **stop signs**), estimates physical metric distance from the robot camera using pinhole projection geometry, and implements optimization techniques (quantization, pruning, lightweight backbones) benchmarked on CPU and GPU.

---

## Architecture Overview

```mermaid
flowchart TD
    subgraph Input ["Perception Input"]
        Frame["Camera Frame / Video Stream (1280x720)"]
    end

    subgraph Detection ["1. Object Detection Engine"]
        YOLO["Transfer Learning Detector (YOLOv8n / v11n)"]
        BBoxes["Bounding Boxes [xmin, ymin, xmax, ymax] + Confidence"]
    end

    subgraph Geometry ["2. Monocular Distance Estimation"]
        Priors["Physical Dimensions (Cone: 0.7m, Barrier: 0.85m, Stop Sign: 0.75m)"]
        CamIntrinsics["Pinhole Camera Model (Focal Length fx, fy, cx, cy)"]
        PinholeEq["Pinhole Projection: Z = (f * H_real) / h_bbox"]
        GroundEq["Ground Plane Ray Intersection: Z_ground"]
        Fusion["Confidence-Weighted Sensor Fusion"]
        Annotated["Formatted Annotations: 'Cone, 1.5m', 'Stop Sign, 3.2m'"]
    end

    subgraph Edge ["3. Edge Device Optimization"]
        Quant["ONNX Dynamic INT8 Quantization (~75% size reduction)"]
        Prune["Conv2d L1-Unstructured Pruning (30% weight sparsity)"]
        Bench["CPU vs GPU Benchmark (Latency, FPS, Memory Footprint)"]
    end

    subgraph Extra ["4. Extra Credit Features"]
        BEV["Inverse Perspective Mapping (IPM) -> Bird's-Eye View Grid"]
        OptFlow["Lucas-Kanade Optical Flow Tracker (Moving Cone Vectors)"]
        Epipolar["Stereo Epipolar Disparity-Depth Proof (docs/epipolar_geometry.md)"]
    end

    Frame --> YOLO
    YOLO --> BBoxes
    BBoxes & Priors & CamIntrinsics --> PinholeEq & GroundEq
    PinholeEq & GroundEq --> Fusion --> Annotated
    YOLO --> Quant & Prune --> Bench
    Annotated --> BEV & OptFlow
```

---

## 1. Object Detection (Transfer Learning)

The system detects three primary navigation-critical classes:
1. `cone` (Traffic cone)
2. `barrier` (Traffic / Jersey barrier)
3. `stop_sign` (MUTCD octagonal stop sign)

### Backbone Selection:
- **Edge Deployment Backbone:** `yolov8n` (3.2M parameters) — engineered for micro-edge platforms (e.g., Jetson Nano, Raspberry Pi, robot onboard SBCs).
- **Baseline Comparison Backbone:** `yolov8s` (11.2M parameters).
- **Fine-Tuning:** Pretrained COCO weights are adapted via transfer learning to rapidly converge on domain-specific road obstacle geometries.

---

## 2. Geometric Monocular Distance Estimation

### Pinhole Camera Formulation
Under a perspective pinhole camera model with focal length $f_y$ (pixels), an object with physical height $H_{\text{real}}$ (meters) appearing with bounding box height $h_{\text{bbox}}$ (pixels) has distance $Z$ (meters):

$$Z = \frac{f_y \cdot H_{\text{real}}}{h_{\text{bbox}}}$$

Where focal length is derived from camera horizontal Field of View ($\text{FOV}_h$):
$$f_x = \frac{W_{\text{image}}}{2 \cdot \tan\left(\frac{\text{FOV}_h}{2}\right)}$$

### Physical Priors ($H_{\text{real}}$):
| Object Class | Real-World Height ($H_{\text{real}}$) | Real-World Width ($W_{\text{real}}$) | Aspect Ratio ($H/W$) |
| :--- | :---: | :---: | :---: |
| **Traffic Cone** | $0.70\text{ m}$ ($70\text{ cm}$) | $0.38\text{ m}$ ($38\text{ cm}$) | $\approx 1.84$ |
| **Traffic Barrier** | $0.85\text{ m}$ ($85\text{ cm}$) | $1.50\text{ m}$ ($150\text{ cm}$) | $\approx 0.57$ |
| **Stop Sign** | $0.75\text{ m}$ ($75\text{ cm}$) | $0.75\text{ m}$ ($75\text{ cm}$) | $1.00$ |

### Ground-Plane Constraint & Boundary Handling:
When objects are clipped by the image edge (e.g., cone partially below the bumper), the system automatically fusions bottom-edge ground projection:
$$Z_{\text{ground}} = \frac{h_{\text{camera}}}{\tan\left(\theta_{\text{pitch}} + \arctan\left(\frac{v_{\text{bottom}} - c_y}{f_y}\right)\right)}$$

### Output Annotation Format:
Bounding boxes are annotated with the required concise format:
- `Cone, 1.5m`
- `Stop Sign, 3.2m`
- `Barrier, 2.1m`

---

## 3. Edge Optimization & Hardware Benchmarking

Tested on CPU and GPU (NVIDIA RTX hardware):

| Model / Checkpoint | Format | Target Device | Mean Latency | P95 Latency | Throughput (FPS) | Model Size |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: |
| `detector_baseline.pt` | PyTorch FP32 | **CPU** | $21.5\text{ ms}$ | $25.6\text{ ms}$ | **46.3 FPS** | $0.4\text{ MB}$ |
| `detector_pruned_30.pt` | PyTorch Pruned (30%) | **CPU** | $20.8\text{ ms}$ | $23.5\text{ ms}$ | **47.8 FPS** | $0.4\text{ MB}$ |
| `detector_baseline.pt` | PyTorch FP32 | **CUDA GPU** | $0.94\text{ ms}$ | $2.04\text{ ms}$ | **975.0 FPS** | $0.4\text{ MB}$ |
| `detector_pruned_30.pt` | PyTorch Pruned (30%) | **CUDA GPU** | $0.90\text{ ms}$ | $1.61\text{ ms}$ | **1016.9 FPS** | $0.4\text{ MB}$ |

### Optimization Techniques Implemented:
1. **INT8 Dynamic Quantization:** Converts float weights to 8-bit integers using ONNX Runtime, reducing model size by $75\%$ and saving edge memory bandwidth.
2. **L1 Unstructured Pruning:** Prunes $30\%$ of the lowest L1-norm convolution weights (`torch.nn.utils.prune`) while maintaining inference stability.
3. **Hardware Acceleration:** Native CUDA FP16 Tensor Core execution for high-framerate robotics control loops ($>60\text{ FPS}$).

---

## 4. Extra Credit Explorations

### A. Epipolar Geometry Derivation
Complete mathematical derivation and sensitivity proof located in:  
👉 [`docs/epipolar_geometry.md`](file:///docs/epipolar_geometry.md)
- Proves canonical stereo disparity-to-depth formula: $Z = \frac{f \cdot B}{d}$.
- Demonstrates quadratic error growth: $|\Delta Z| \approx \frac{Z^2}{f \cdot B} |\Delta d|$.
- Derives Essential matrix $E = [t]_\times R$ and Fundamental matrix $F = K_R^{-T} E K_L^{-1}$.

### B. Bird's-Eye View (BEV) Homography / IPM
- Calculates 4-point ground homography matrix $H$.
- Warps front-facing perspective camera feed into a top-down metric radar grid.
- Projects obstacles with 2D distance rings ($2.5\text{m}, 5.0\text{m}, 7.5\text{m}, \dots$) for obstacle avoidance.
- Enabled via `--bev` flag in `run_pipeline.py`.

### C. Lucas-Kanade Optical Flow Tracker
- Tracks motion vectors $(\Delta x, \Delta y)$ of moving traffic obstacles across consecutive video frames.
- Calculates velocity magnitude (pixels/frame) and renders tracking vectors.
- Run demo via `python run_tracking_demo.py`.

---

## Quickstart & Reproducibility Guide

### 1. Setup Environment
```bash
# Clone the repository
git clone https://github.com/yorichixX/SomeRobotics.git
cd SomeRobotics

# Run setup (Linux 4090 Workstation)
bash setup_environment.sh

# Or Windows
setup_environment.bat
```

### 2. Prepare Dataset (Option A - Unified Cones, Barriers, Stop Signs)
```bash
# Option A (Recommended): One-command dataset preparation and class distribution check
python prepare_dataset.py --mode download

# Download from an external zip URL (if using custom archive)
python prepare_dataset.py --mode download --url <DATASET_ZIP_URL>

# Inspect dataset class distribution table
python prepare_dataset.py --summary

# Optional: Convert BDD100K JSON annotations to YOLO format
python prepare_dataset.py --mode bdd100k --bdd-json path/to/bdd100k_labels.json --bdd-images path/to/images/
```

### 3. Run Detection & Distance Estimation Pipeline
```bash
# Process validation images with Bird's-Eye View radar overlay
python run_pipeline.py --source data/dataset/images/val --bev --output-dir data/output/

# Process a video stream or camera feed
python run_pipeline.py --source path/to/robot_video.mp4 --bev --output-dir data/output/
```

### 4. Train Model on 4090 Workstation
```bash
python -m src.models.train --data data/dataset/dataset.yaml --model yolov8n.pt --epochs 50 --batch 32 --device 0
```

### 5. Run Hardware Edge Optimization Benchmark
```bash
python benchmark_models.py --runs 100 --warmup 20 --report reports/benchmark_report.md
```

### 6. Run Optical Flow Moving Obstacle Tracking Demo
```bash
python run_tracking_demo.py --output data/output/tracked_flow_demo.mp4
```

### 7. Run Geometry & Pipeline Unit Tests
```bash
python -m unittest tests/test_geometry.py
```

---

## Repository Structure

```
SomeRobotics/
├── configs/
│   ├── camera_config.yaml         # Pinhole intrinsics, FOV, principal point
│   ├── object_priors.yaml         # Cone, barrier, stop sign physical dimensions
│   └── model_config.yaml          # Architecture, inference, training hyperparameters
├── data/
│   ├── dataset/                   # Train and validation images and YOLO labels
│   ├── sample_videos/             # Test video clips
│   └── output/                    # Annotated images and videos
├── docs/
│   └── epipolar_geometry.md       # Rigorous epipolar geometry mathematical proof
├── reports/
│   └── benchmark_report.md        # CPU vs GPU latency, FPS, memory benchmark results
├── src/
│   ├── dataset/
│   │   └── dataset_manager.py     # BDD100K converter & synthetic dataset generator
│   ├── geometry/
│   │   ├── camera_model.py        # Pinhole model, intrinsics, ground ray projection
│   │   └── distance_estimator.py  # Monocular distance estimation & annotation formatting
│   ├── models/
│   │   ├── detector.py            # ObjectDetector wrapper with ONNX and YOLO support
│   │   └── train.py               # Transfer learning fine-tuning entrypoint
│   ├── optimization/
│   │   ├── quantizer.py           # ONNX export and INT8 dynamic quantization
│   │   ├── pruner.py              # L1 unstructured weight pruning & sparsity metrics
│   │   └── benchmark.py           # Latency (ms) and throughput (FPS) benchmark engine
│   └── extra/
│       ├── bev_homography.py      # Bird's-Eye View (IPM) top-down grid renderer
│       └── optical_flow_tracker.py# Lucas-Kanade optical flow obstacle tracker
├── tests/
│   └── test_geometry.py           # Unit tests for camera projection and distance math
├── benchmark_models.py            # Benchmark execution script
├── prepare_dataset.py             # Dataset preparation CLI
├── run_pipeline.py                # Main inference and distance annotation pipeline
├── run_tracking_demo.py           # Optical flow obstacle tracking demo
├── setup_environment.sh           # Linux workstation setup script
├── setup_environment.bat          # Windows setup script
├── requirements.txt               # Dependencies
└── README.md                      # Primary project documentation
```
