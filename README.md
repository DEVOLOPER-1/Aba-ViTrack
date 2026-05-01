# Aba-ViTrack Docker Execution Guide
>> Please note that GPU docker image is not tested yet. if it didn't work fallback to the CPU version and keep the threads low 1 or 2 maximum, and expect it would take time
## Quick Start

### Prerequisites
- Docker installed and configured
- Dataset directory prepared (see Dataset Preparation section below)
- For GPU: NVIDIA Docker runtime installed

### Building the Docker Image

**CPU Version:**
```bash
docker build -f Dockerfile.cpu -t abavitrack-cpu .
```

**GPU Version:**
```bash
docker build -f Dockerfile.gpu -t abavitrack-gpu .
```

---

## Running the Container

### CPU Version
```bash
docker run -it --rm \
    -v /path/to/dataset:/dataset \
    -v /path/to/output:/app/outputs \
    abavitrack-cpu
```

### GPU Version
```bash
docker run --gpus all -it --rm --gpus all \
    -v /path/to/dataset:/dataset \
    -v /path/to/output:/app/outputs \
    abavitrack-gpu
```

> **Important:** 
 - Replace `/path/to/dataset` with your dataset directory
 - Replace `/path/to/output` with where you want the submission CSV saved
 - **Do NOT modify the container-side paths** (`/dataset` and `/app/outputs`)

---

## Configuration & Prompts

Once the container starts, you will see the following interactive prompts:

```
Aba‑ViTrack Finetuned & enhanced Submission Pipeline by Team: Zerone
Dataset root path [contest_release]: 
Manifest JSON path [/dataset/metadata/contestant_manifest.json]: 
Number of CPU workers for extraction [8]: 
Run data preprocessing (y/n) [n]: 
Number of GPUs [0]: 
Inference threads [8]: 2
Results root directory [/app/outputs/tracking_results]: 
Output CSV file [/app/submission.csv]: 
```

### Prompt Details

| Prompt | Default | Notes |
|--------|---------|-------|
| Dataset root path | `contest_release` | Path relative to `/dataset` mount point |
| Manifest JSON path | `/dataset/metadata/contestant_manifest.json` | Path to the contest manifest file |
| CPU workers for extraction | `8` | Number of parallel workers for preprocessing |
| Run data preprocessing | `n` | **Set to `y` at least once** (required for first run) |
| Number of GPUs | `0` (CPU) / `1` (GPU) | Auto-set based on image; override if needed |
| Inference threads | `8` | Number of inference threads |
| Results root directory | `/app/outputs/tracking_results` | Where tracking results are saved |
| Output CSV file | `/app/submission.csv` | Final submission CSV filename |

---

## ⚠️ Hardcoded Defaults & Warnings

**The following parameters are now hardcoded and cannot be changed via prompts:**

| Parameter | Value | Source |
|-----------|-------|--------|
| **Config Name** | `abavit_gs_8` | `main.py` (line 46) |
| **Test Epoch** | `10` | `main.py` (line 47) |
| **Checkpoint File** | `AbaViTrack_ep0010.pth.tar` | `Dockerfile.cpu` (line 34) / `Dockerfile.gpu` (line 36) |
| **Checkpoint Path** | `/app/checkpoints/AbaViTrack_ep0010.pth.tar` | Built into Docker image |

**If you need to change these values, you must:**
1. Edit the corresponding source files
2. Rebuild the Docker image using `docker build`

---

## Dataset Preparation

### Directory Structure
The dataset should be organized as follows:
```
/path/to/dataset/
├── Dataset1
|   └── video1.mp4
├── metadata/
│   └── contestant_manifest.json
└── ...
```

### Preprocessing
When prompted, enter `y` to preprocess the data. This will:
1. Extract frames from video files
2. Create an `img/` subdirectory in each video folder
3. Store preprocessed frames at: `/dataset/video_name/img/`

Example result after preprocessing:
```
/path/to/dataset/
├── video1/
│   ├── img/
│   │   ├── 000001.jpg
│   │   ├── 000002.jpg
│   │   └── ...
│   ├── video1.mp4
|   └── ...
├── metadata/
│   └── contestant_manifest.json
└── ...
```

---

## Output Files

**Note:** This script is optimized for competition evaluators to generate submission CSVs with fast, consistent results.

Results are automatically saved to your mounted output directory:

| Output | Location | Purpose |
|--------|----------|---------|
| Tracking Results | `/app/outputs/tracking_results/` | Intermediate tracking predictions for all sequences |
| Submission CSV | `/app/submission.csv` | **Final submission file for evaluation** |

The generated CSV will be available at `/path/to/output/submission.csv` on your host machine (where you mounted `/app/outputs`).

---

## Troubleshooting

### Issue: "Checkpoint not found"
- Ensure `checkpoints/AbaViTrack_ep0010.pth.tar` exists in the project root before building
- Verify the checkpoint file is copied into the Docker image
- Rebuild the Docker image: `docker build -f Dockerfile.cpu -t abavitrack-cpu .`

### Issue: "Manifest file not found"
- Verify the manifest path is correct relative to the `/dataset` mount point
- Check that the file exists on your host machine at the mounted path

### Issue: Preprocessing fails
- Ensure sufficient disk space in the dataset directory
- Check that video files are valid and readable
- Verify the number of CPU workers doesn't exceed available CPU cores
- Check file permissions on the host machine

### Issue: Results not saved to host
- Ensure you mounted an output directory with `-v /path/to/output:/app/outputs`
- Verify write permissions on the host output directory
- Check disk space availability

---

## Example: Complete Workflow

```bash
# 1. Build the CPU image
docker build -f Dockerfile.gpu -t abavitrack-gpu .

# 2. Run with dataset and output mounts
docker run -it --rm \
    -v ~/datasets/mtc-aic:/dataset \
    -v ~/results:/app/outputs \
    abavitrack-cpu

# 3. When prompted, answer:
# Dataset root path [contest_release]: /dataset
# Manifest JSON path [/dataset/metadata/contestant_manifest.json]: /dataset/metadata/contestant_manifest.json
# Number of CPU workers for extraction [8]: 8
# Run data preprocessing (y/n) [n]: y
# Number of GPUs [0]: 0
# Inference threads [8]: 2
# Results root directory [/app/outputs/tracking_results]: /app/outputs/tracking_results
# Output CSV file [/app/submission.csv]: /app/submission.csv

# 4. Results will be available in ~/results/ on your host machine
```
