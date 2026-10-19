"""One-command automated dependency installer for Eric Robots perception pipeline.

Supports Windows, Linux, and CUDA GPU detection (RTX 4090).
"""

import sys
import subprocess
import shutil


def run_cmd(cmd: list) -> bool:
    print(f"\n[RUNNING] {' '.join(cmd)}")
    try:
        subprocess.check_call(cmd)
        return True
    except subprocess.CalledProcessError as e:
        print(f"[ERROR] Command failed with exit code: {e.returncode}")
        return False


def main():
    print("=" * 65)
    print("ERIC ROBOTS: AUTOMATED DEPENDENCY INSTALLER")
    print("=" * 65)
    python_bin = sys.executable
    print(f"Current Python executable: {python_bin}")
    print(f"Python Version: {sys.version}")

    # 1. Upgrade pip
    print("\n--- 1. Upgrading pip ---")
    run_cmd([python_bin, "-m", "pip", "install", "--upgrade", "pip"])

    # 2. Check if PyTorch with CUDA is installed
    print("\n--- 2. Checking PyTorch & CUDA ---")
    has_cuda_torch = False
    try:
        import torch
        if torch.cuda.is_available():
            print(f"[FOUND] PyTorch {torch.__version__} with CUDA ({torch.cuda.get_device_name(0)})")
            has_cuda_torch = True
        else:
            print(f"[INFO] PyTorch is installed ({torch.__version__}), but CUDA is not active.")
    except ImportError:
        print("[INFO] PyTorch not yet installed.")

    # If CUDA PyTorch is not installed, install CUDA 12.1+ build
    if not has_cuda_torch:
        print("\nInstalling PyTorch with CUDA 12.1+ support for RTX 4090...")
        run_cmd([
            python_bin, "-m", "pip", "install",
            "torch", "torchvision",
            "--index-url", "https://download.pytorch.org/whl/cu121"
        ])

    # 3. Install requirements
    print("\n--- 3. Installing Project Dependencies ---")
    run_cmd([python_bin, "-m", "pip", "install", "-r", "requirements.txt"])

    # 4. Final verification
    print("\n" + "=" * 65)
    print("VERIFYING INSTALLATION")
    print("=" * 65)

    packages = [
        ("torch", "PyTorch"),
        ("torchvision", "TorchVision"),
        ("ultralytics", "Ultralytics YOLO"),
        ("cv2", "OpenCV"),
        ("yaml", "PyYAML"),
        ("numpy", "NumPy"),
        ("onnx", "ONNX"),
        ("onnxruntime", "ONNX Runtime"),
    ]

    all_passed = True
    for mod_name, label in packages:
        try:
            mod = __import__(mod_name)
            ver = getattr(mod, "__version__", "installed")
            print(f"  [OK] {label:<22} ({ver})")
        except ImportError:
            print(f"  [MISSING] {label:<18}")
            all_passed = False

    try:
        import torch
        cuda_status = f"Available ({torch.cuda.get_device_name(0)})" if torch.cuda.is_available() else "Not available (CPU only)"
        print(f"\nCUDA Acceleration: {cuda_status}")
    except Exception:
        pass

    if all_passed:
        print("\n>>> ALL DEPENDENCIES SUCCESSFULLY INSTALLED AND READY! <<<")
    else:
        print("\n>>> Some optional packages failed. Check the errors above. <<<")


if __name__ == "__main__":
    main()
