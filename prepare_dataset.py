"""Dataset preparation utility CLI."""

import argparse
from src.dataset.dataset_manager import DatasetManager


def main():
    parser = argparse.ArgumentParser(description="Prepare dataset for Eric Robots navigation")
    parser.add_argument("--mode", type=str, choices=["synthetic", "bdd100k"], default="synthetic",
                        help="Dataset source mode: 'synthetic' or 'bdd100k'")
    parser.add_argument("--output-dir", type=str, default="data/dataset", help="Output directory")
    parser.add_argument("--train-samples", type=int, default=50, help="Number of synthetic train samples")
    parser.add_argument("--val-samples", type=int, default=15, help="Number of synthetic val samples")
    parser.add_argument("--bdd-json", type=str, default=None, help="Path to BDD100K JSON annotations")
    parser.add_argument("--bdd-images", type=str, default=None, help="Path to BDD100K raw images folder")

    args = parser.parse_args()
    dm = DatasetManager(output_dir=args.output_dir)

    if args.mode == "synthetic":
        print(f"Generating synthetic dataset in {args.output_dir}...")
        dm.generate_synthetic_dataset(num_train=args.train_samples, num_val=args.val_samples)
        print("Done! Ready for training or evaluation.")
    elif args.mode == "bdd100k":
        if not args.bdd_json or not args.bdd_images:
            print("Error: --bdd-json and --bdd-images are required for BDD100K conversion mode.")
            return
        print(f"Converting BDD100K dataset from {args.bdd_json}...")
        count = dm.convert_bdd100k_to_yolo(args.bdd_json, args.bdd_images, split="train")
        dm.write_dataset_yaml()
        print(f"Done! Converted {count} labeled images to YOLO format.")


if __name__ == "__main__":
    main()
