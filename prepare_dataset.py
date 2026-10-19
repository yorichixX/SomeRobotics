"""Dataset preparation utility CLI for robotics navigation perception."""

import argparse
from src.dataset.dataset_manager import DatasetManager


def main():
    parser = argparse.ArgumentParser(description="Prepare dataset for Eric Robots navigation")
    parser.add_argument(
        "--mode",
        type=str,
        choices=["download", "roboflow", "synthetic", "bdd100k", "summary"],
        default="download",
        help="Dataset preparation mode",
    )
    parser.add_argument("--url", type=str, default=None, help="Direct URL to dataset zip archive")
    parser.add_argument("--output-dir", type=str, default="data/dataset", help="Output directory")
    parser.add_argument("--train-samples", type=int, default=100, help="Number of synthetic train samples")
    parser.add_argument("--val-samples", type=int, default=25, help="Number of synthetic val samples")
    parser.add_argument("--bdd-json", type=str, default=None, help="Path to BDD100K JSON annotations")
    parser.add_argument("--bdd-images", type=str, default=None, help="Path to BDD100K raw images folder")
    parser.add_argument("--roboflow-key", type=str, default=None, help="Roboflow API key")
    parser.add_argument("--roboflow-project", type=str, default=None, help="Roboflow project ID")
    parser.add_argument("--roboflow-workspace", type=str, default=None, help="Roboflow workspace name")
    parser.add_argument("--summary", action="store_true", help="Print dataset class distribution summary")

    args = parser.parse_args()
    dm = DatasetManager(output_dir=args.output_dir)

    if args.summary or args.mode == "summary":
        dm.inspect_and_summarize()
        return

    if args.mode == "download":
        if args.url:
            dm.download_and_extract_zip(args.url, extract_to=args.output_dir)
        else:
            print("[INFO] No external download URL passed. Generating high-resolution synthetic navigation dataset with Cones, Barriers, and Stop Signs...")
            dm.generate_synthetic_dataset(num_train=args.train_samples, num_val=args.val_samples)
        dm.inspect_and_summarize()

    elif args.mode == "roboflow":
        if not args.roboflow_key or not args.roboflow_project or not args.roboflow_workspace:
            print("Error: --roboflow-key, --roboflow-workspace, and --roboflow-project are required.")
            return
        dm.download_roboflow_dataset(
            api_key=args.roboflow_key,
            workspace=args.roboflow_workspace,
            project=args.roboflow_project,
        )
        dm.inspect_and_summarize()

    elif args.mode == "synthetic":
        print(f"Generating synthetic dataset ({args.train_samples} train, {args.val_samples} val)...")
        dm.generate_synthetic_dataset(num_train=args.train_samples, num_val=args.val_samples)
        dm.inspect_and_summarize()

    elif args.mode == "bdd100k":
        if not args.bdd_json or not args.bdd_images:
            print("Error: --bdd-json and --bdd-images are required for BDD100K conversion mode.")
            return
        print(f"Converting BDD100K dataset from {args.bdd_json}...")
        count = dm.convert_bdd100k_to_yolo(args.bdd_json, args.bdd_images, split="train")
        dm.write_dataset_yaml()
        print(f"Done! Converted {count} labeled images to YOLO format.")
        dm.inspect_and_summarize()


if __name__ == "__main__":
    main()
