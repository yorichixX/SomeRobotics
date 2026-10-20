"""Benchmarking script comparing edge optimization techniques on CPU and GPU."""

from typing import Optional, List
import argparse
import os
import torch
import torch.nn as nn

from src.optimization.quantizer import ModelQuantizer
from src.optimization.pruner import ModelPruner
from src.optimization.benchmark import EdgeBenchmark


class DummyDetectorModel(nn.Module):
    """Synthetic lightweight convolutional detector for benchmark demonstration when no weights file is passed."""
    def __init__(self, num_classes: int = 3):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, stride=2, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.act1 = nn.SiLU()
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.act2 = nn.SiLU()
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1)
        self.bn3 = nn.BatchNorm2d(128)
        self.act3 = nn.SiLU()
        self.conv4 = nn.Conv2d(128, 64, kernel_size=1)
        self.head = nn.Conv2d(64, num_classes + 4, kernel_size=1)

    def forward(self, x):
        x = self.act1(self.bn1(self.conv1(x)))
        x = self.act2(self.bn2(self.conv2(x)))
        x = self.act3(self.bn3(self.conv3(x)))
        x = self.conv4(x)
        return self.head(x)


def run_benchmarks(
    model_path: Optional[str] = None,
    devices: List[str] = ["cpu", "cuda"],
    runs: int = 100,
    warmup: int = 20,
    output_report: str = "reports/benchmark_report.md",
) -> None:
    """Run full benchmarking suite comparing unoptimized vs quantized vs pruned on CPU and GPU."""
    os.makedirs("weights", exist_ok=True)
    os.makedirs(os.path.dirname(output_report), exist_ok=True)

    benchmark_runner = EdgeBenchmark(num_warmup=warmup, num_runs=runs, imgsz=640)
    quantizer = ModelQuantizer("weights")
    pruner = ModelPruner("weights")

    # If no model provided, initialize lightweight model for benchmark
    if model_path is None or not os.path.exists(model_path):
        print("[INFO] No external weights specified. Initializing benchmark detector baseline...")
        model_path = "weights/detector_baseline.pt"
        dummy = DummyDetectorModel()
        torch.save({"model": dummy}, model_path)

    # Prepare pruned model
    pruned_path = "weights/detector_pruned_30.pt"
    try:
        pruner.prune_l1_unstructured(model_path, amount=0.30, output_pruned_path=pruned_path)
    except Exception as e:
        print(f"[WARNING] Pruning failed: {e}")
        pruned_path = None

    # Export ONNX FP32 model
    base_name = os.path.splitext(os.path.basename(model_path))[0]
    onnx_fp32_path = os.path.join("weights", f"{base_name}.onnx")
    onnx_int8_path = os.path.join("weights", f"{base_name}_int8.onnx")

    if not os.path.exists(onnx_fp32_path):
        print(f"\n[INFO] Exporting model to ONNX FP32 format: {onnx_fp32_path}...")
        try:
            onnx_fp32_path = quantizer.export_to_onnx(model_path, output_onnx_path=onnx_fp32_path, imgsz=640)
        except Exception as e:
            print(f"[WARNING] ONNX export failed: {e}")
            onnx_fp32_path = None

    # Quantize to INT8
    if onnx_fp32_path and os.path.exists(onnx_fp32_path) and not os.path.exists(onnx_int8_path):
        print(f"\n[INFO] Applying INT8 Dynamic Quantization: {onnx_int8_path}...")
        try:
            onnx_int8_path = quantizer.quantize_onnx_dynamic(onnx_fp32_path, output_quantized_path=onnx_int8_path)
        except Exception as e:
            print(f"[WARNING] INT8 quantization failed: {e}")
            onnx_int8_path = None

    results = []

    # Check CUDA availability
    available_devices = []
    for d in devices:
        if d.lower() == "cuda" and not torch.cuda.is_available():
            print("[WARNING] CUDA requested but not available. Skipping CUDA benchmarks.")
            continue
        available_devices.append(d)

    print("\n" + "=" * 60)
    print("RUNNING BENCHMARKS ACROSS DEVICES & OPTIMIZATIONS")
    print("=" * 60)

    for dev in available_devices:
        print(f"\n--- Benchmarking on device: {dev.upper()} ---")

        # 1. Baseline PyTorch FP32
        print(f"Testing Baseline FP32 on {dev.upper()}...")
        try:
            res_fp32 = benchmark_runner.benchmark_pytorch(model_path, device=dev, half=False)
            results.append(res_fp32)
            print(f"  Result: {res_fp32['mean_latency_ms']} ms ({res_fp32['fps']} FPS)")
        except Exception as e:
            print(f"  Skipped Baseline FP32 ({e})")

        # 2. PyTorch FP16 (GPU only)
        if dev == "cuda":
            print(f"Testing PyTorch FP16 on {dev.upper()}...")
            try:
                res_fp16 = benchmark_runner.benchmark_pytorch(model_path, device=dev, half=True)
                results.append(res_fp16)
                print(f"  Result: {res_fp16['mean_latency_ms']} ms ({res_fp16['fps']} FPS)")
            except Exception as e:
                print(f"  Skipped PyTorch FP16 ({e})")

        # 3. Pruned PyTorch model
        if pruned_path and os.path.exists(pruned_path):
            print(f"Testing Pruned Model on {dev.upper()}...")
            try:
                res_pruned = benchmark_runner.benchmark_pytorch(pruned_path, device=dev, half=False)
                res_pruned["format"] = "PyTorch Pruned (30%)"
                results.append(res_pruned)
                print(f"  Result: {res_pruned['mean_latency_ms']} ms ({res_pruned['fps']} FPS)")
            except Exception as e:
                print(f"  Skipped Pruned ({e})")

        # 4. ONNX FP32
        if onnx_fp32_path and os.path.exists(onnx_fp32_path):
            print(f"Testing ONNX FP32 on {dev.upper()}...")
            try:
                res_onnx_fp32 = benchmark_runner.benchmark_onnx(onnx_fp32_path, device=dev)
                results.append(res_onnx_fp32)
                print(f"  Result: {res_onnx_fp32['mean_latency_ms']} ms ({res_onnx_fp32['fps']} FPS)")
            except Exception as e:
                print(f"  Skipped ONNX FP32 on {dev.upper()} ({e})")

        # 5. ONNX INT8 (CPU edge optimization)
        if dev == "cpu" and onnx_int8_path and os.path.exists(onnx_int8_path):
            print(f"Testing ONNX INT8 Dynamic on CPU...")
            try:
                res_onnx_int8 = benchmark_runner.benchmark_onnx(onnx_int8_path, device="cpu")
                results.append(res_onnx_int8)
                print(f"  Result: {res_onnx_int8['mean_latency_ms']} ms ({res_onnx_int8['fps']} FPS)")
            except Exception as e:
                print(f"  Skipped ONNX INT8 ({e})")

    # Generate report
    report = benchmark_runner.generate_markdown_report(results, output_report_path=output_report)
    print("\n" + report)


def main():
    parser = argparse.ArgumentParser(description="Benchmark edge optimizations for Eric Robots")
    parser.add_argument("--model", type=str, default=None, help="Path to PyTorch or YOLO model")
    parser.add_argument("--runs", type=int, default=100, help="Number of benchmark iterations")
    parser.add_argument("--warmup", type=int, default=20, help="Number of warm-up iterations")
    parser.add_argument("--report", type=str, default="reports/benchmark_report.md", help="Output report markdown path")

    args = parser.parse_args()

    run_benchmarks(
        model_path=args.model,
        devices=["cpu", "cuda"],
        runs=args.runs,
        warmup=args.warmup,
        output_report=args.report,
    )


if __name__ == "__main__":
    main()
