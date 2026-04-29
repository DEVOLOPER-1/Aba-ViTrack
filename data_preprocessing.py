import os
import json
import cv2
import concurrent.futures


def process_video(seq_key, seq_info, dataset_root):
    """Function to process a single video sequence."""
    try:
        vid_path = os.path.join(dataset_root, seq_info['video_path'])
        # Create an 'img' folder inside the sequence folder (standard for Aba-ViTrack)
        img_dir = os.path.join(dataset_root, seq_info['dataset'], seq_info['seq_name'], 'img')
        os.makedirs(img_dir, exist_ok=True)

        cap = cv2.VideoCapture(vid_path)
        frame_idx = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            # Save frames as 00000001.jpg, 00000002.jpg, etc.
            frame_name = os.path.join(img_dir, f"{(frame_idx + 1):08d}.jpg")
            cv2.imwrite(frame_name, frame)
            frame_idx += 1
        cap.release()
        return f"Extracted {frame_idx} frames for {seq_key}"
    except Exception as e:
        return f"Error processing {seq_key}: {str(e)}"


def main():
    dataset_root = '/mnt/contest_release_data'
    manifest_path = '/mnt/contest_release_data/metadata/contestant_manifest.json'

    with open(manifest_path, 'r') as f:
        manifest = json.load(f)

    # Flatten the tasks from the nested dictionary into a simple list of arguments
    tasks = []
    for split, sequences in manifest.items():
        for seq_key, seq_info in sequences.items():
            tasks.append((seq_key, seq_info, dataset_root))

    # Determine the number of workers (typically the number of CPU cores)
    # You can hardcode this to a specific number (e.g., max_workers=8) if you want to limit CPU usage
    max_workers = os.cpu_count()
    print(f"Starting parallel extraction using {max_workers} workers...")

    # Use ProcessPoolExecutor to run tasks in parallel
    with concurrent.futures.ProcessPoolExecutor(max_workers=16) as executor:
        # Submit all tasks to the executor
        futures = [executor.submit(process_video, *task) for task in tasks]

        # Process the results as they finish
        for future in concurrent.futures.as_completed(futures):
            print(future.result())


if __name__ == '__main__':
    # The if __name__ == '__main__': block is REQUIRED for multiprocessing in Python
    main()