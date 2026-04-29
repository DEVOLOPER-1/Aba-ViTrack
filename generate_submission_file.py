import os
import pandas as pd
import json

dataset_root = '/mnt/contest_release_data'
results_dir = '/home/youssef_mohammad/projects/Aba-ViTrack/output/test/tracking_results/abavitrack/'

manifest_path = '/mnt/contest_release_data/metadata/contestant_manifest.json'
with open(manifest_path, 'r') as f:
    manifest = json.load(f)['public_lb']

submission = []

for seq_key, seq_info in manifest.items():
    # PyTracking sanitizes sequence names by replacing slashes with underscores or hyphens
    # Adjust `safe_seq_key` based on exactly what Pytracking outputs in the results_dir
    safe_seq_key = seq_key.replace('/', '_') 
    res_file = os.path.join(results_dir, f"{safe_seq_key}.txt")
    
    with open(res_file, 'r') as f:
        lines = f.readlines()
        
    for frame_idx, line in enumerate(lines):
        x, y, w, h = map(float, line.strip().split('\t')) # Standard pytracking output separator
        # Replace non-visible/lost tracking with 0,0,0,0
        if w <= 0 or h <= 0:
            x, y, w, h = 0, 0, 0, 0
            
        submission.append({
            'id': f"{seq_key}_{frame_idx}",
            'x': x, 'y': y, 'w': w, 'h': h
        })

df = pd.DataFrame(submission)
df.to_csv('submission.csv', index=False)
print("Saved Kaggle submission to submission.csv")