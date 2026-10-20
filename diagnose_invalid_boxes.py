"""Diagnostic script to inspect invalid boxes in dataset_clean."""

import os

def diagnose(dataset_dir: str = "data/dataset_clean"):
    print("=" * 70)
    print("DIAGNOSING INVALID BOUNDING BOXES IN DATASET_CLEAN")
    print("=" * 70)

    for split in ["train", "val"]:
        lbl_dir = os.path.join(dataset_dir, "labels", split)
        if not os.path.exists(lbl_dir):
            continue

        invalid_samples = []
        counts_by_reason = {
            "polygon_detected (>5 numbers)": 0,
            "xc <= 0.0": 0,
            "xc >= 1.0": 0,
            "yc <= 0.0": 0,
            "yc >= 1.0": 0,
            "w <= 0.0": 0,
            "w > 1.0": 0,
            "h <= 0.0": 0,
            "h > 1.0": 0,
            "parse_error": 0,
        }

        files_with_invalid = 0

        for f in sorted(os.listdir(lbl_dir)):
            if not f.endswith(".txt"):
                continue
            path = os.path.join(lbl_dir, f)
            has_invalid_in_file = False
            with open(path, "r") as fp:
                for line_idx, line in enumerate(fp):
                    parts = line.strip().split()
                    if not parts:
                        continue
                    try:
                        cid = int(parts[0])
                        num_parts = len(parts)
                        if num_parts > 5:
                            counts_by_reason["polygon_detected (>5 numbers)"] += 1
                            if len(invalid_samples) < 10:
                                invalid_samples.append((f, line_idx + 1, line.strip(), f"Polygon format ({num_parts} parts)"))
                            has_invalid_in_file = True
                            continue

                        xc, yc, w, h = float(parts[1]), float(parts[2]), float(parts[3]), float(parts[4])
                        reasons = []
                        if xc <= 0.0:
                            counts_by_reason["xc <= 0.0"] += 1
                            reasons.append(f"xc={xc} <= 0")
                        if xc >= 1.0:
                            counts_by_reason["xc >= 1.0"] += 1
                            reasons.append(f"xc={xc} >= 1")
                        if yc <= 0.0:
                            counts_by_reason["yc <= 0.0"] += 1
                            reasons.append(f"yc={yc} <= 0")
                        if yc >= 1.0:
                            counts_by_reason["yc >= 1.0"] += 1
                            reasons.append(f"yc={yc} >= 1")
                        if w <= 0.0:
                            counts_by_reason["w <= 0.0"] += 1
                            reasons.append(f"w={w} <= 0")
                        if w > 1.0:
                            counts_by_reason["w > 1.0"] += 1
                            reasons.append(f"w={w} > 1")
                        if h <= 0.0:
                            counts_by_reason["h <= 0.0"] += 1
                            reasons.append(f"h={h} <= 0")
                        if h > 1.0:
                            counts_by_reason["h > 1.0"] += 1
                            reasons.append(f"h={h} > 1")

                        if reasons:
                            has_invalid_in_file = True
                            if len(invalid_samples) < 10:
                                invalid_samples.append((f, line_idx + 1, line.strip(), ", ".join(reasons)))
                    except Exception as e:
                        counts_by_reason["parse_error"] += 1
                        has_invalid_in_file = True
                        if len(invalid_samples) < 10:
                            invalid_samples.append((f, line_idx + 1, line.strip(), f"Parse error: {e}"))

            if has_invalid_in_file:
                files_with_invalid += 1

        print(f"\n--- Split: {split.upper()} ---")
        print(f"Files with invalid annotations: {files_with_invalid}")
        print("Failure Breakdown:")
        for reason, count in counts_by_reason.items():
            if count > 0:
                print(f"  - {reason}: {count}")

        print("\nSample Invalid Lines (up to 10):")
        for fname, lnum, ltext, reason in invalid_samples:
            print(f"  File: {fname}:{lnum}")
            print(f"    Line:   '{ltext}'")
            print(f"    Reason: {reason}")


if __name__ == "__main__":
    diagnose("data/dataset_clean")
