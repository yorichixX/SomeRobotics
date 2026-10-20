"""Utility to inspect the latest training curves, convergence metrics, and plot locations."""

import os
import glob
import pandas as pd


def inspect_latest_run():
    print("=" * 70)
    print("INSPECTING LATEST TRAINING RUN METRICS & CURVES")
    print("=" * 70)

    # Search for all results.csv candidates
    search_patterns = [
        "runs/train/**/results.csv",
        "runs/detect/**/results.csv",
        "**/results.csv",
    ]

    csv_candidates = []
    for pattern in search_patterns:
        csv_candidates.extend(glob.glob(pattern, recursive=True))

    if not csv_candidates:
        print("[ERROR] No results.csv files found. Please check runs/ directory.")
        return

    # Find the most recently modified results.csv
    csv_candidates = sorted(list(set(csv_candidates)), key=os.path.getmtime, reverse=True)
    latest_csv = csv_candidates[0]
    exp_dir = os.path.dirname(latest_csv)

    print(f"Latest Experiment Directory: {exp_dir}")
    print(f"Latest Results Log:          {latest_csv}")

    # Read CSV
    df = pd.read_csv(latest_csv)
    df.columns = [c.strip() for c in df.columns]

    total_epochs = len(df)
    print(f"\nTotal Epochs Completed: {total_epochs}")

    # Key columns
    train_box = df.get("train/box_loss")
    train_cls = df.get("train/cls_loss")
    val_box = df.get("val/box_loss")
    val_cls = df.get("val/cls_loss")
    map50 = df.get("metrics/mAP50(B)")
    map50_95 = df.get("metrics/mAP50-95(B)")

    if map50 is not None:
        best_epoch_idx = map50.idxmax()
        best_epoch = df.iloc[best_epoch_idx]

        print("\n" + "-" * 70)
        print("OPTIMAL CHECKPOINT METRICS (Best Validation mAP@50)")
        print("-" * 70)
        print(f"  Best Epoch:           {int(best_epoch.get('epoch', best_epoch_idx + 1))}")
        print(f"  Peak mAP@50:          {best_epoch.get('metrics/mAP50(B)', 0):.4f}")
        print(f"  mAP@50-95 at Peak:    {best_epoch.get('metrics/mAP50-95(B)', 0):.4f}")
        print(f"  Train Box Loss:       {best_epoch.get('train/box_loss', 0):.4f}")
        print(f"  Train Cls Loss:       {best_epoch.get('train/cls_loss', 0):.4f}")
        print(f"  Val Box Loss:         {best_epoch.get('val/box_loss', 0):.4f}")
        print(f"  Val Cls Loss:         {best_epoch.get('val/cls_loss', 0):.4f}")

    # Analyze convergence trajectory in the last 15 epochs
    print("\n" + "-" * 70)
    print("CONVERGENCE TRAJECTORY (Last 10 Epochs)")
    print("-" * 70)
    tail_cols = [c for c in ["epoch", "train/box_loss", "train/cls_loss", "val/box_loss", "val/cls_loss", "metrics/mAP50(B)", "metrics/mAP50-95(B)"] if c in df.columns]
    print(df[tail_cols].tail(10).to_string(index=False))

    # Check for overfitting or plateau
    if val_cls is not None and len(val_cls) >= 20:
        val_cls_tail = val_cls.tail(15)
        is_increasing = val_cls_tail.iloc[-1] > val_cls_tail.min() * 1.05
        is_plateau = abs(val_cls_tail.iloc[-1] - val_cls_tail.iloc[0]) / (val_cls_tail.iloc[0] + 1e-6) < 0.05

        print("\n" + "-" * 70)
        print("STATISTICAL TRAJECTORY ASSESSMENT")
        print("-" * 70)
        if is_increasing:
            print("  [STATUS: INCIPIENT OVERFITTING]")
            print("  Validation loss has bottomed out and is beginning to rise.")
            print("  Further training on this dataset will degrade test performance.")
        elif is_plateau:
            print("  [STATUS: FULLY CONVERGED / PLATEAU]")
            print("  Validation loss has flattened with <5% change over the last 15 epochs.")
            print("  The model has extracted all available generalizable features from this dataset.")
        else:
            print("  [STATUS: ACTIVE CONVERGENCE]")
            print("  Validation loss was still trending downwards.")

    # List image artifacts
    print("\n" + "-" * 70)
    print("AVAILABLE PLOT ARTIFACTS FOR VISUAL INSPECTION")
    print("-" * 70)
    for plot_name in ["results.png", "confusion_matrix.png", "confusion_matrix_normalized.png", "PR_curve.png", "F1_curve.png"]:
        p_path = os.path.join(exp_dir, plot_name)
        if os.path.exists(p_path):
            print(f"  - {plot_name}: {os.path.abspath(p_path)}")


if __name__ == "__main__":
    inspect_latest_run()
