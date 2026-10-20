# Technical Decisions & Interview Preparation Guide: Eric Robots Assignment

Comprehensive reference guide written in **Simplified Technical English (~80% ASD-STE100)** for technical interviews at Eric Robots.

---

## 1. Architectural Decisions and Technical Justifications

### Decision 1: Model Architecture — Why YOLOv8-Nano (`yolov8n`)?
* **Question you will get in interview:** *"Why did you choose YOLOv8n instead of Faster R-CNN, SSD, or YOLOv8x?"*
* **Answer / Rationale:**
  1. **Latency Budget:** Autonomous robots require real-time control loops ($\ge 30\text{ FPS}$, latency $< 33\text{ ms}$). Two-stage detectors (like Faster R-CNN) execute an explicit Region Proposal Network (RPN) followed by RoI pooling, causing high latency ($> 70\text{ ms}$ on CPU). YOLOv8 is a single-stage detector.
  2. **Edge Hardware Constraints:** The assignment explicitly targets "limited hardware / edge devices" (e.g. Jetson Orin/Nano, Raspberry Pi). `yolov8n` has only **$3.01\text{M}$ parameters** ($6.5\text{ MB}$ weights) and requires only **$8.2\text{ GFLOPs}$**.
  3. **Anchor-Free Detection Head:** Older architectures (YOLOv3/v4/v5) use pre-defined anchor boxes. Real-world road obstacles have extreme aspect ratios (cones are tall and narrow; barriers are low and very wide). YOLOv8's anchor-free design predicts bounding box coordinates directly, providing superior localization for non-square objects.
  4. **Deployment Flexibility:** Ultralytics YOLOv8 provides native export pipelines to ONNX, INT8 TensorRT, and OpenVINO with zero architecture modifications.

---

### Decision 2: Distance Estimation — Why Geometric Pinhole over Deep Monocular Depth (MiDaS/DPT)?
* **Question you will get in interview:** *"Why not use a neural network like MiDaS or Depth-Anything to estimate distance?"*
* **Answer / Rationale:**
  1. **Scale Ambiguity:** Deep monocular depth networks output *relative depth* (values normalized between 0 and 1). They cannot output real physical meters without external ground truth depth sensors (LiDAR).
  2. **Compute Cost:** A secondary monocular depth transformer requires $50\text{M}+$ parameters and adds $\approx 100\text{ ms}$ inference latency, violating the edge constraint.
  3. **Physical Determinism:** Mobile robotics safety standards favor deterministic geometry over black-box depth estimation. Using known standard physical dimensions ($H_{\text{cone}} = 0.70\text{ m}$, $H_{\text{barrier}} = 0.85\text{ m}$, $H_{\text{stop}} = 0.75\text{ m}$) and camera focal length ($f_y$), distance is computed with exact pinhole optics:
     $$Z = \frac{f_y \cdot H_{\text{real}}}{h_{\text{bbox}}}$$
  4. **Ground-Plane Fallback:** When objects are partially clipped by the camera lens boundary (truncating $h_{\text{bbox}}$), we fused bottom-edge ground plane projection ($Z_{\text{ground}}$) to maintain reliable distance estimation.

---

### Decision 3: Edge Optimization Strategy — Quantization & Pruning
* **Question you will get in interview:** *"What optimization techniques did you choose, and what trade-offs did you observe?"*
* **Answer / Rationale:**
  1. **INT8 Dynamic Quantization:** Converts 32-bit floating-point weights to 8-bit integers. Reduces disk and memory bandwidth usage by $\approx 75\%$ with less than $1\%$ drop in precision.
  2. **L1-Unstructured Pruning:** Pruned $30\%$ of the lowest-magnitude convolution weights (`torch.nn.utils.prune`). Evaluates how much sparsity the network tolerates before bounding box regression degrades.
  3. **Hardware Benchmarking:** Measured latency and throughput across 200 iterations on both host CPU and CUDA GPU, recording P95 and mean latencies.

---

### Decision 4: Training Dynamics — Early Stopping & Cosine Annealing
* **Question you will get in interview:** *"How did you manage training convergence and prevent overfitting?"*
* **Answer / Rationale:**
  1. **Validation Fitness Monitoring:** Rather than training for fixed arbitrary epochs, we implemented early stopping with `--patience 20`. This monitors validation mAP (fitness), automatically halting training when test performance plateaus.
  2. **Cosine Learning Rate Decay (`cos_lr=True`):** Replaced step decay with a smooth cosine schedule. Cosine annealing avoids sharp gradient shocks, allowing the optimizer to settle smoothly into narrow loss basins.

---

### Decision 5: Data-Centric AI — Expanding Real Data vs. Synthetic Repetition
* **Question you will get in interview:** *"When barrier accuracy hit a bottleneck, why did you expand the dataset instead of tuning hyperparameters?"*
* **Answer / Rationale:**
  1. **Empirical Generalization Limit:** Training 75 epochs on 236 unique barrier images repeated $3\times$ caused the network to memorize training images. Training loss dropped by $34\%$, but validation mAP plateaued at $0.15$.
  2. **Data-Centric Solution:** Neural network capacity cannot invent missing physical features. Adding 2 diverse external barrier datasets expanded unique physical images from 236 to 1,284 ($>5\times$ increase). This eliminated the need for synthetic oversampling and exposed the detector to authentic intra-class variations (concrete, plastic, water-filled).

---

## 2. Roadblocks Faced and Solutions (STAR Format for Interviews)

### Roadblock 1: BDD100K Class Gap (Cones and Barriers Missing)
* **Situation:** The assignment instructions suggested using BDD100K to detect cones, barriers, and stop signs.
* **Task:** Verify class labels in BDD100K 2D detection.
* **Action:** Audited the official BDD100K class ontology. Discovered that BDD100K 2D detection contains only 10 classes (`car`, `truck`, `bus`, `traffic sign`, etc.). Cones and barriers are completely absent.
* **Result:** Designed a multi-source dataset merger (`clean_merge_datasets.py`) that combined real BDD100K road scenes with dedicated road safety obstacle datasets, producing a unified dataset with all 3 required classes.

---

### Roadblock 2: Extreme Class Imbalance ($16:1$ Ratio)
* **Situation:** Initial multi-dataset merge resulted in $8,904$ stop signs, $1,470$ cones, and only $560$ barriers.
* **Task:** Diagnose why barrier validation recall was low ($12\%$) and $\text{mAP}_{50}$ was $14.9\%$.
* **Action:** Diagnosed that in cross-entropy loss with batch size 32, stop signs accounted for $>80\%$ of every batch, drowning out the barrier gradient updates. Implemented balanced sampling to cap stop signs at $1,500$ instances and oversample the minority barrier class $3\times$.
* **Result:** Restored balanced gradient flow across all 3 classes, bringing the training distribution to near $1:1:1$.

---

### Roadblock 3: Dataset Label Poisoning & Corrupted Roboflow Class Mapping
* **Situation:** Even after initial balancing, barrier mAP remained low ($0.11 - 0.14$), while stop signs had an artificially high mAP of $0.916$.
* **Task:** Conduct a deep audit on raw label files and `data.yaml` across all downloaded datasets to verify label integrity.
* **Action:** Built `audit_raw_datasets.py` to inspect raw class IDs across all 8,353 label files. Discovered that the downloaded Roboflow "Traffic Stop Sign" export had **11 classes**, including over $6{,}000$ speed-limit digits (`'3'`, `'0'`), Roboflow metadata text strings, and traffic lights, all previously grouped under stop signs. Developed `clean_merge_datasets.py` with explicit whitelist filtering (`{4, 8, 9, 10}`) to isolate genuine octagonal stop signs and discard all non-stop-sign objects.
* **Result:** Eliminated $6{,}000+$ poisoned annotations. Re-balanced the dataset to $1{,}462$ cones, $1{,}293$ barriers, and $1{,}587$ stop signs. This boosted cone mAP@50 to $0.615$ (+18%) and cone mAP@50-95 to $0.332$ (+79%).

---

### Roadblock 4: Geometric Boundary Truncation & Zero-Area Boxes
* **Situation:** Running strict dataset validation (`audit_clean_dataset.py`) flagged $423$ invalid boxes in training and $40$ in validation.
* **Task:** Identify the exact mathematical cause of invalid bounding boxes and prevent gradient instability during training.
* **Action:** Developed `diagnose_invalid_boxes.py` to profile coordinate failures. Identified two root causes from Roboflow online augmentations:
  1. Edge boundary clipping: boxes touching image borders had centers rounded to $x_c \le 0.0$, $x_c \ge 1.0$, $y_c \le 0.0$, or $y_c \ge 1.0$.
  2. Degenerate zero-area boxes: 2 bounding boxes had width or height equal to zero ($w=0.0$).
  Developed `sanitize_dataset_boxes.py` to convert coordinates to $[x_{\min}, y_{\min}, x_{\max}, y_{\max}]$, clamp them strictly to $[0.0, 1.0]$, drop sub-pixel degenerate boxes ($< 0.005$), and recompute normalized center-width-height coordinates with floating-point margins.
* **Result:** $100\%$ of bounding box annotations passed geometric verification with zero gradient warnings or loss instabilities during YOLO training.

---

### Roadblock 5: Barrier Sample Scarcity & High Intra-Class Variance
* **Situation:** In the raw dataset, only 50 barrier instances existed in validation, and only 236 unique barrier images existed in training. Even with $3\times$ oversampling ($1{,}293$ train instances), barrier mAP@50 reached $0.150$ while recall doubled from $12\%$ to $27.3\%$.
* **Task:** Diagnose why barrier precision plateaued while cone precision reached $52.9\%$ and mAP reached $0.615$.
* **Action:** Conducted intra-class variance analysis. Discovered that the 236 barrier images contained 4 visually distinct subclasses: red/white plastic Jersey barriers, concrete blocks, yellow metal construction fences, and folding A-frame barriers. Cones, in contrast, have a uniform geometric silhouette and color scheme.
* **Result:** Identified that more training epochs on the same 236 images causes memorization (overfitting). Sourced 2 additional barrier datasets, expanding unique physical barrier images from 236 to 1,284 ($>5\times$ increase) and annotations to 2,299 without synthetic repetition.

---

### Roadblock 6: Multi-Dataset Class Schema Inconsistencies (`'0'` vs `'Barrier'`)
* **Situation:** When downloading the new barrier dataset (`Barrier.v2i(1)`), `data.yaml` listed two classes: `['0', 'Barrier']`. Class `0` contained $1{,}112$ bounding boxes, while `Barrier` contained only $75$.
* **Task:** Determine whether Class `0` contained corrupted labels (like digits or text) or genuine barriers.
* **Action:** Built `inspect_new_barriers.py` to calculate dataset overlap and generate visual preview crops. The visual audit proved that both Class `0` and Class `1` were genuine water-filled barriers whose original class name had defaulted to `'0'` during Roboflow upload.
* **Result:** Safely mapped both IDs into unified Class 1 (`barrier`), integrating 1,187 real barrier annotations with zero label pollution.

---

### Roadblock 7: Human-in-the-Loop Visual Verification Safeguard
* **Situation:** After merging multiple heterogeneous datasets, there was a risk of silent annotation misalignment (e.g., class index inversion or inverted normalized coordinates).
* **Task:** Establish an automated visual quality gate before committing to full model training.
* **Action:** Developed `verify_labels_visually.py` to sample random images from each class and render explicit color-coded bounding boxes (`CONE` in green, `BARRIER` in orange, `STOP_SIGN` in blue) into `data/visual_verification/`.
* **Result:** Enabled direct visual confirmation of ground-truth alignment across all three classes before launching training.

---

### Roadblock 8: Python 3.13 & CPU-Only PyTorch Build
* **Situation:** The remote workstation environment had a default Python 3.13 install with `torch-2.14.1+cpu`. Running `train.py --device 0` failed with `ValueError: Invalid CUDA device=0`.
* **Task:** Enable full CUDA acceleration on the NVIDIA GeForce RTX 4090 GPU.
* **Action:** 
  1. Isolated the project into a clean Python 3.11 Conda environment (`eric_robot`) where pre-compiled CUDA 12.1 PyTorch wheels are fully stable.
  2. Added defensive device auto-detection in `train.py` so the script gracefully warns and falls back to CPU if CUDA is ever unavailable.
* **Result:** Training completed on the RTX 4090 with $1.1\text{ ms}$ inference time and $18.5\text{ batches/sec}$.

---

### Roadblock 9: PyTorch 2.6 Weights-Only Default
* **Situation:** PyTorch 2.6 changed `torch.load` default to `weights_only=True`, breaking custom model loading in the pruning benchmark with `pickle.UnpicklingError`.
* **Task:** Maintain compatibility across modern PyTorch 2.6+ releases.
* **Action:** Refactored checkpoint loading in `src/optimization/pruner.py` and `src/optimization/benchmark.py` to explicitly pass `weights_only=False`.
* **Result:** Benchmarking runner runs flawlessly on both PyTorch 2.5 and 2.6+ installations.

---

## 3. High-Impact Questions You Can Expect in Your Interview

1. **"Why did you stop training when training loss was still steadily decreasing?"**
   * *Answer:* Between epoch 40 and 75, training classification loss continued to drop from 1.90 to 1.25, but validation mAP plateaued at 0.433–0.440. Training loss measures empirical risk on the training set, which will continue decreasing as the network memorizes background noise. Validation mAP measures true generalization risk. Halting training at the validation plateau (early stopping) avoids overfitting.
2. **"How does bounding box inaccuracy affect distance estimation?"**
   * *Answer:* Distance is inversely proportional to bounding box height ($Z \propto \frac{1}{h}$). A $1\text{ pixel}$ error at close range (where $h=200\text{ px}$) has negligible impact ($<1\%$). However, at long range (where $h=10\text{ px}$), a $1\text{ pixel}$ error causes a $10\%$ error in distance. This is why ground-plane fusion is critical for distant objects.
3. **"Why did you choose Homography for Bird's-Eye View instead of 3D point cloud reconstruction?"**
   * *Answer:* For flat roads, the ground plane fulfills the planar homography assumption. A $3\times 3$ Inverse Perspective Mapping (IPM) matrix $H$ warps front pixels to metric ground coordinates at $0.2\text{ ms}$ compute cost, making it ideal for 2D occupancy grid path planning.
4. **"How did you handle moving obstacles?"**
   * *Answer:* Implemented Lucas-Kanade optical flow (`calcOpticalFlowPyrLK`) on Shi-Tomasi corners extracted specifically inside detection bounding boxes. This calculates obstacle velocity vectors $(\Delta x, \Delta y)$ to determine whether a cone or barrier is static or moving across the robot's trajectory.
5. **"If given more time and compute, what would you improve next?"**
   * *Answer:* 
     1. Deploy TensorRT INT8 calibration directly on Jetson hardware for sub-1ms edge latency.
     2. Collect domain-specific video streams under night/rain conditions for dynamic lighting robustness.
     3. Implement a Kalman filter tracker (e.g. ByteTrack) to smooth distance estimates across consecutive video frames.
