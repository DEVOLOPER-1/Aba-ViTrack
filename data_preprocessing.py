import argparse
import os
import json
import cv2
import concurrent.futures


def process_video(seq_key, seq_info, dataset_root):
    try:
        vid_path = os.path.join(dataset_root, seq_info["video_path"])
        img_dir = os.path.join(
            dataset_root, seq_info["dataset"], seq_info["seq_name"], "img"
        )
        os.makedirs(img_dir, exist_ok=True)

        cap = cv2.VideoCapture(vid_path)
        frame_idx = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            frame_name = os.path.join(img_dir, f"{(frame_idx + 1):08d}.jpg")
            cv2.imwrite(frame_name, frame)
            frame_idx += 1
        cap.release()
        return f"Extracted {frame_idx} frames for {seq_key}"
    except Exception as e:
        return f"Error processing {seq_key}: {str(e)}"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_root", default="contest_release")
    parser.add_argument(
        "--manifest_path", default="contest_release/metadata/contestant_manifest.json"
    )
    parser.add_argument("--max_workers", type=int, default=8)
    args = parser.parse_args()

    with open(args.manifest_path, "r") as f:
        manifest = json.load(f)

    tasks = []
    for split, sequences in manifest.items():
        for seq_key, seq_info in sequences.items():
            tasks.append((seq_key, seq_info, args.dataset_root))

    max_workers = args.max_workers
    print(f"Starting parallel extraction using {max_workers} workers...")

    with concurrent.futures.ProcessPoolExecutor(max_workers=max_workers) as executor:
        futures = [executor.submit(process_video, *task) for task in tasks]

        for future in concurrent.futures.as_completed(futures):
            print(future.result())


if __name__ == "__main__":
    main()
