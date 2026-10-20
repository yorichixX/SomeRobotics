"""Training script for fine-tuning YOLO object detector on navigation objects."""

import argparse
import os
import shutil
import yaml


def train_model(
    data_yaml: str = "data/dataset/dataset.yaml",
    model_name: str = "yolov8n.pt",
    epochs: int = 50,
    batch_size: int = 32,
    imgsz: int = 640,
    device: str = "0",
    project: str = "runs/train",
    name: str = "exp",
    weights_export_dir: str = "weights",
    patience: int = 25,
    cos_lr: bool = True,
) -> str:
    """Fine-tune pretrained YOLO model on custom dataset.
    
    Args:
        data_yaml: Path to dataset.yaml
        model_name: Base model weights ('yolov8n.pt' or 'yolov8s.pt')
        epochs: Number of training epochs
        batch_size: Training batch size
        imgsz: Image input resolution
        device: Device identifier ('0' for GPU 0, or 'cpu')
        project: Save directory for training artifacts
        name: Experiment run name
        weights_export_dir: Target directory to copy best.pt
        
    Returns:
        Path to best.pt weights.
    """
    try:
        from ultralytics import YOLO
    except ImportError:
        raise ImportError(
            "ultralytics is not installed. Run 'pip install ultralytics' on the workstation."
        )

    import torch
    if device != "cpu" and not torch.cuda.is_available():
        print(f"\n[WARNING] CUDA device '{device}' requested, but PyTorch is CPU-only (version: {torch.__version__}).")
        print("[INFO] Falling back to device='cpu' so training can proceed.")
        print("[TIP] To enable your NVIDIA GPU, run:")
        print("      pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124 --force-reinstall\n")
        device = "cpu"

    print("=" * 60)
    print(f"Starting Transfer Learning for Robotics Navigation")
    print(f"Base Model: {model_name}")
    print(f"Dataset:    {data_yaml}")
    print(f"Epochs:     {epochs}")
    print(f"Batch Size: {batch_size}")
    print(f"Device:     {device}")
    print("=" * 60)

    # Initialize model with pretrained transfer learning weights
    model = YOLO(model_name)

    # Train
    results = model.train(
        data=os.path.abspath(data_yaml),
        epochs=epochs,
        batch=batch_size,
        imgsz=imgsz,
        device=device,
        project=project,
        name=name,
        exist_ok=True,
        save=True,
        plots=True,
        patience=patience,
        cos_lr=cos_lr,
    )

    # Locate best weights
    best_weights_path = None
    if hasattr(model, "trainer") and hasattr(model.trainer, "best") and model.trainer.best:
        best_weights_path = str(model.trainer.best)

    if not best_weights_path or not os.path.exists(best_weights_path):
        candidate_paths = [
            os.path.join("runs", "detect", project, name, "weights", "best.pt"),
            os.path.join(project, name, "weights", "best.pt"),
            os.path.join("runs", "detect", "train", "weights", "best.pt"),
        ]
        for p in candidate_paths:
            if os.path.exists(p):
                best_weights_path = p
                break

    os.makedirs(weights_export_dir, exist_ok=True)
    target_export_path = os.path.join(weights_export_dir, "best.pt")

    if best_weights_path and os.path.exists(best_weights_path):
        shutil.copy2(best_weights_path, target_export_path)
        print(f"\n[SUCCESS] Trained model copied to: {target_export_path}")
    else:
        print(f"\n[INFO] Best weights located at: {best_weights_path}")
        target_export_path = best_weights_path

    # Run validation
    print("\nRunning Validation Evaluation:")
    val_results = model.val()
    print(f"mAP50:    {val_results.box.map50:.4f}")
    print(f"mAP50-95: {val_results.box.map:.4f}")

    return target_export_path


def main():
    parser = argparse.ArgumentParser(description="Train YOLO model for Eric Robots assignment")
    parser.add_argument("--data", type=str, default="data/dataset/dataset.yaml", help="Path to dataset.yaml")
    parser.add_argument("--model", type=str, default="yolov8n.pt", help="Pretrained model (e.g. yolov8n.pt, yolov8s.pt)")
    parser.add_argument("--epochs", type=int, default=50, help="Number of training epochs")
    parser.add_argument("--batch", type=int, default=32, help="Batch size")
    parser.add_argument("--imgsz", type=int, default=640, help="Image resolution")
    parser.add_argument("--device", type=str, default="0", help="GPU device id ('0') or 'cpu'")
    parser.add_argument("--project", type=str, default="runs/train", help="Run directory")
    parser.add_argument("--name", type=str, default="exp", help="Experiment name")

    parser.add_argument("--patience", type=int, default=25, help="Early stopping patience (epochs without improvement)")
    parser.add_argument("--cos-lr", action="store_true", default=True, help="Use cosine learning rate scheduler")

    args = parser.parse_args()

    train_model(
        data_yaml=args.data,
        model_name=args.model,
        epochs=args.epochs,
        batch_size=args.batch,
        imgsz=args.imgsz,
        device=args.device,
        project=args.project,
        name=args.name,
        patience=args.patience,
        cos_lr=args.cos_lr,
    )


if __name__ == "__main__":
    main()
