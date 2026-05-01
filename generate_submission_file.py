import os
import argparse
import json
import pandas as pd


DATASET_ROOT = "/mnt/contest_release_data"
MANIFEST_PATH = os.path.join(DATASET_ROOT, "metadata", "contestant_manifest.json")
RESULTS_ROOT = (
    "/home/youssef_mohammad/projects/Aba-ViTrack/outputs/test/tracking_results"
)
TRACKER_NAME = "abavitrack"
OUTPUT_CSV = "submission.csv"
SPLIT = "public_lb"


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=str,
        default="abavit_patch16_224",
        help="YAML config name used during inference (determines checkpoint sub-dir).",
    )
    parser.add_argument(
        "--results_root",
        type=str,
        default=RESULTS_ROOT,
        help="Root directory where PyTracking stores results.",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=OUTPUT_CSV,
        help="Output CSV filename.",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    results_dir = os.path.join(args.results_root, TRACKER_NAME, args.config, "mtc_aic4")
    print(f"Looking for tracker results in: {results_dir}")

    with open(MANIFEST_PATH, "r") as f:
        manifest = json.load(f)[SPLIT]

    submission = []
    missing_seqs = []

    for seq_key, seq_info in manifest.items():
        safe_key = seq_key.replace("/", "_")
        res_file = os.path.join(results_dir, f"{safe_key}.txt")

        n_frames = seq_info["n_frames"]

        if not os.path.isfile(res_file):
            # Result file missing — fill with zeros so the CSV stays complete.
            missing_seqs.append(seq_key)
            for frame_idx in range(n_frames):
                submission.append(
                    {
                        "id": f"{seq_key}_{frame_idx}",
                        "x": 0,
                        "y": 0,
                        "w": 0,
                        "h": 0,
                    }
                )
            continue

        with open(res_file, "r") as f:
            lines = f.readlines()

        if len(lines) != n_frames:
            print(
                f"WARNING: {seq_key}: expected {n_frames} frames, "
                f"got {len(lines)} lines in result file."
            )

        for frame_idx, line in enumerate(lines):
            line = line.strip()
            if not line:
                x, y, w, h = 0, 0, 0, 0
            else:
                try:
                    parts = line.split("\t")
                    x, y, w, h = map(float, parts)
                except ValueError:
                    parts = line.split(",")
                    x, y, w, h = map(float, parts)

            if w <= 0 or h <= 0:
                x, y, w, h = 0, 0, 0, 0

            submission.append(
                {
                    "id": f"{seq_key}_{frame_idx}",
                    "x": x,
                    "y": y,
                    "w": w,
                    "h": h,
                }
            )

    if missing_seqs:
        print(
            f"\nWARNING: {len(missing_seqs)} sequences had no result file "
            f"and were filled with zeros:"
        )
        for s in missing_seqs:
            print(f"  {s}")

    df = pd.DataFrame(submission, columns=["id", "x", "y", "w", "h"])
    df.to_csv(args.output, index=False)
    print(f"\nSaved {len(df)} rows to {args.output}")


if __name__ == "__main__":
    main()
