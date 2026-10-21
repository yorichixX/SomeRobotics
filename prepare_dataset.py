"""Dataset preparation utility CLI for robotics navigation perception."""

import argparse
import sys
from src.dataset.dataset_manager import DatasetManager


def main():
    parser = argparse.ArgumentParser(description="Prepare and inspect real dataset for Eric Robots navigation")
    parser.add_argument(
        "--mode",
        type=str,
        choices=["summary", "download", "roboflow", "bdd100k"],
        default="summary",
        help="Dataset preparation mode (default: summary)",
    )
    parser.add_argument("--url", type=str, default=None, help="Direct URL to dataset zip archive")
    parser.add_argument("--output-dir", type=str, default="data/dataset", help="Output directory")
    parser.add_argument("--bdd-json", type=str, default=None, help="Path to BDD100K JSON annotations")
    parser.add_argument("--bdd-images", type=str, default=None, help="Path to BDD100K raw images folder")
    parser.add_argument("--roboflow-key", type=str, default=None, help="Roboflow API key")
    parser.add_argument("--roboflow-project", type=str, default=None, help="Roboflow project ID")
    parser.add_argument("--roboflow-workspace", type=str, default=None, help="Roboflow workspace name")
    parser.add_argument("--bdd-split", type=str, choices=["train", "val"], default="train", help="Dataset split: 'train' or 'val'")
    parser.add_argument("--max-samples", type=int, default=None, help="Maximum number of images to extract")
    parser.add_argument("--summary", action="store_true", help="Print dataset class distribution summary")

    args = parser.parse_args()
    dm = DatasetManager(output_dir=args.output_dir)

    if args.summary or args.mode == "summary":
        dm.inspect_and_summarize()
        return

    if args.mode == "download":
        if not args.url:
            print("Error: --url <ZIP_URL> is required for download mode.")
            sys.exit(1)
        dm.download_and_extract_zip(args.url, extract_to=args.output_dir)
        dm.inspect_and_summarize()

    elif args.mode == "roboflow":
        if not args.roboflow_key or not args.roboflow_project or not args.roboflow_workspace:
            print("Error: --roboflow-key, --roboflow-workspace, and --roboflow-project are required.")
            sys.exit(1)
        dm.download_roboflow_dataset(
            api_key=args.roboflow_key,
            workspace=args.roboflow_workspace,
            project=args.roboflow_project,
        )
        dm.inspect_and_summarize()

    elif args.mode == "bdd100k":
        if not args.bdd_json or not args.bdd_images:
            print("Error: --bdd-json and --bdd-images are required for BDD100K conversion mode.")
            sys.exit(1)
        print(f"Converting BDD100K dataset from {args.bdd_json} for '{args.bdd_split}' split...")
        count = dm.convert_bdd100k_to_yolo(
            bdd_json_path=args.bdd_json,
            images_src_dir=args.bdd_images,
            split=args.bdd_split,
            max_samples=args.max_samples,
        )
        dm.write_dataset_yaml()
        print(f"Done! Converted {count} labeled images to YOLO format.")
        dm.inspect_and_summarize()


if __name__ == "__main__":
    main()
