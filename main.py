import subprocess
import os

num_gpus = os.environ.get("NUM_GPUS", None)


def ask(prompt, default=None):
    if default is not None:
        prompt = f"{prompt} [{default}]: "
    else:
        prompt = f"{prompt}: "
    val = input(prompt).strip()
    return val if val else default


print("Aba‑ViTrack Finetuned & enhanced Submission Pipeline by Team: Zerone")

dataset_root = ask("Dataset root path", "/dataset")
manifest_default = os.path.join(dataset_root, "metadata", "contestant_manifest.json")
manifest_path = ask("Manifest JSON path", manifest_default)
max_workers = ask("Number of CPU workers for extraction", "8")

run_preprocess = ask("Run data preprocessing (y/n)", "n").lower() == "y"

if num_gpus is None:
    num_gpus = ask("Number of GPUs", "0")

if run_preprocess:
    print(">>> Extracting frames...")
    subprocess.run(
        [
            "python",
            "data_preprocessing.py",
            "--dataset_root",
            dataset_root,
            "--manifest_path",
            manifest_path,
            "--max_workers",
            max_workers,
        ],
        check=True,
    )
else:
    print("Skipping preprocessing.")

config = "abavit_patch16_224"
test_epoch = "10"
threads = ask("Inference threads", "8")

results_root = ask("Results root directory", "/app/outputs/tracking_results")
output_csv = ask("Output CSV file", "/app/outputs/submission.csv")

print(">>> Tracking (Aba‑ViTrack inference)")
subprocess.run(
    [
        "python",
        "tracking/test.py",
        "abavitrack",
        config,
        "--dataset_name",
        "mtc_aic4",
        "--threads",
        threads,
        "--num_gpus",
        num_gpus,
        "--test_epoch",
        test_epoch,
    ],
    check=True,
)

print(">>> Generating submission CSV")
subprocess.run(
    [
        "python",
        "generate_submission_file.py",
        "--config",
        config,
        "--results_root",
        results_root,
        "--output",
        output_csv,
    ],
    check=True,
)

print(f"Done! Submission saved to {output_csv}")
