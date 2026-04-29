"""
generate_submission_file.py
============================
Reads per-sequence tracker output files produced by PyTracking's running.py
and assembles them into the Kaggle submission CSV.

FIXES vs original
-----------------
1. results_dir now correctly points to:
       {results_root}/abavitrack/{CONFIG_NAME}/mtc_aic4/
   The original was missing the config-name sub-directory and the
   dataset-name ('mtc_aic4') sub-directory, so every res_file lookup
   would return a FileNotFoundError.

2. The safe_seq_key construction (replacing '/' with '_') already
   matches what the fixed mtcaic4dataset.py stores as seq.name.

3. Added --config argument so you can switch between
   abavit_patch16_224 (full model) and abavit_half_patch16_224 (++ model)
   without editing the file.

4. Graceful handling of missing result files (prints a warning and
   fills the rows with 0,0,0,0 so the CSV is always complete).

Usage
-----
    python generate_submission_file.py
    python generate_submission_file.py --config abavit_half_patch16_224
"""

import os
import argparse
import json
import pandas as pd

# ---------------------------------------------------------------------------
# Configurable paths
# ---------------------------------------------------------------------------
DATASET_ROOT   = '/mnt/contest_release_data'
MANIFEST_PATH  = os.path.join(DATASET_ROOT, 'metadata', 'contestant_manifest.json')
RESULTS_ROOT   = '/home/youssef_mohammad/projects/Aba-ViTrack/output/test/tracking_results'
TRACKER_NAME   = 'abavitrack'
OUTPUT_CSV     = 'submission.csv'
SPLIT          = 'public_lb'


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--config',
        type=str,
        default='abavit_patch16_224',
        help='YAML config name used during inference (determines checkpoint sub-dir).',
    )
    parser.add_argument(
        '--results_root',
        type=str,
        default=RESULTS_ROOT,
        help='Root directory where PyTracking stores results.',
    )
    parser.add_argument(
        '--output',
        type=str,
        default=OUTPUT_CSV,
        help='Output CSV filename.',
    )
    return parser.parse_args()


def main():
    args = parse_args()

    # -----------------------------------------------------------------------
    # Resolve the directory where individual sequence .txt files live.
    #
    # PyTracking's Tracker class sets:
    #   self.results_dir = f'{results_path}/{name}/{parameter_name}'
    # and _save_tracker_output writes:
    #   {results_dir}/{seq.dataset}/{seq.name}.txt
    #
    # With our fixes:
    #   seq.dataset = 'mtc_aic4'
    #   seq.name    = 'dataset2_basketball_player1'  (slashes replaced by '_')
    #
    # So the file lives at:
    #   {results_root}/{TRACKER_NAME}/{config}/mtc_aic4/{safe_seq_name}.txt
    # -----------------------------------------------------------------------
    results_dir = os.path.join(
        args.results_root, TRACKER_NAME, args.config, 'mtc_aic4'
    )
    print(f'Looking for tracker results in: {results_dir}')

    # -----------------------------------------------------------------------
    # Load manifest for the public leaderboard split
    # -----------------------------------------------------------------------
    with open(MANIFEST_PATH, 'r') as f:
        manifest = json.load(f)[SPLIT]

    submission = []
    missing_seqs = []

    for seq_key, seq_info in manifest.items():
        # seq_key  = 'dataset2/basketball_player1'  (original manifest key)
        # safe_key = 'dataset2_basketball_player1'  (matches seq.name in results)
        safe_key = seq_key.replace('/', '_')
        res_file = os.path.join(results_dir, f'{safe_key}.txt')

        n_frames = seq_info['n_frames']

        if not os.path.isfile(res_file):
            # Result file missing — fill with zeros so the CSV stays complete.
            missing_seqs.append(seq_key)
            for frame_idx in range(n_frames):
                submission.append({
                    'id': f'{seq_key}_{frame_idx}',
                    'x': 0, 'y': 0, 'w': 0, 'h': 0,
                })
            continue

        with open(res_file, 'r') as f:
            lines = f.readlines()

        if len(lines) != n_frames:
            print(
                f'WARNING: {seq_key}: expected {n_frames} frames, '
                f'got {len(lines)} lines in result file.'
            )

        for frame_idx, line in enumerate(lines):
            line = line.strip()
            if not line:
                x, y, w, h = 0, 0, 0, 0
            else:
                try:
                    # PyTracking saves with tab delimiter
                    parts = line.split('\t')
                    x, y, w, h = map(float, parts)
                except ValueError:
                    # Fall back to comma delimiter just in case
                    parts = line.split(',')
                    x, y, w, h = map(float, parts)

            # Replace non-visible / lost tracking with 0,0,0,0
            if w <= 0 or h <= 0:
                x, y, w, h = 0, 0, 0, 0

            # Submission ID format: <seq_id>_<frame_index>  (0-based)
            submission.append({
                'id': f'{seq_key}_{frame_idx}',
                'x': x, 'y': y, 'w': w, 'h': h,
            })

    # -----------------------------------------------------------------------
    # Write CSV
    # -----------------------------------------------------------------------
    if missing_seqs:
        print(
            f'\nWARNING: {len(missing_seqs)} sequences had no result file '
            f'and were filled with zeros:'
        )
        for s in missing_seqs:
            print(f'  {s}')

    df = pd.DataFrame(submission, columns=['id', 'x', 'y', 'w', 'h'])
    df.to_csv(args.output, index=False)
    print(f'\nSaved {len(df)} rows to {args.output}')


if __name__ == '__main__':
    main()
