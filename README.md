# Aba-ViTrack Docker Execution Guide
**Submission Pipeline by Team: Zerone**

> **Performance Note:** The GPU Docker image is fully tested, optimized, and highly recommended (achieves ~81 FPS). If hardware constraints require falling back to the CPU version, expect significantly longer processing times. **For CPU execution, it is strongly advised to keep Inference Threads to 1 or 2 maximum** to prevent system overload.

## Table of Contents
1. [Prerequisites](#1-prerequisites)
2. [Dataset Preparation](#2-dataset-preparation)
3. [Building the Docker Image](#3-building-the-docker-image)
4. [Running the Pipeline](#4-running-the-pipeline)
5. [Interactive Configuration Prompts](#5-interactive-configuration-prompts)
6. [Outputs & Results](#6-outputs--results)
7. [Hardcoded Defaults & Warnings](#7-hardcoded-defaults--warnings)
8. [Troubleshooting](#8-troubleshooting)

---

## 1. Prerequisites

**A. Install Docker**
*   **Linux (Ubuntu/Debian):** 
    ```bash
    curl -fsSL [https://get.docker.com](https://get.docker.com) -o get-docker.sh
    sudo sh get-docker.sh
    ```
*   **Windows:** 
    Download and install [Docker Desktop](https://www.docker.com/products/docker-desktop/). Ensure the **WSL 2 backend** is enabled in the settings for optimal performance.

**B. Install NVIDIA Container Toolkit (GPU Version Only)**
*   To run the GPU version on Linux, you must install the NVIDIA Toolkit so Docker can interface with your graphics card:
    ```bash
    curl -fsSL [https://nvidia.github.io/libnvidia-container/gpgkey](https://nvidia.github.io/libnvidia-container/gpgkey) | sudo gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg \
      && curl -s -L [https://nvidia.github.io/libnvidia-container/stable/deb/nvidia-container-toolkit.list](https://nvidia.github.io/libnvidia-container/stable/deb/nvidia-container-toolkit.list) | \
        sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' | \
        sudo tee /etc/apt/sources.list.d/nvidia-container-toolkit.list
    sudo apt-get update
    sudo apt-get install -y nvidia-container-toolkit
    sudo systemctl restart docker
    ```

**C. Download the Model Checkpoint**
*   Our pre-trained checkpoint file (`AbaViTrack_ep0010.pth.tar`) MUST be placed in the project root under the `checkpoints/` directory before building the image.
    > Download link: [Google Drive Link](https://drive.google.com/file/d/1qLZT-t4KaD0L13T2DCN8XyB74PN4c-GM/view?usp=sharing)

---

## 2. Dataset Preparation

Before running the container, your raw dataset must be organized on your local machine as follows:
```text
/path/to/dataset/
├── Dataset1
|   └── video1.mp4
├── metadata/
│   └── contestant_manifest.json
└── ...
```

*Note: The container will handle the actual extraction of frames from these `.mp4` files during the execution phase.*

---

## 3. Building the Docker Image

Build the environment that matches your hardware.

**GPU Version (Recommended):**
```bash
docker build --network=host -f Dockerfile.gpu -t abavitrack-gpu .
```

**CPU Version:**
```bash
docker build -f Dockerfile.cpu -t abavitrack-cpu .
```

---

## 4. Running the Pipeline

Mount your prepared dataset and a local output folder, then start the container. 

> **Important Directory Mapping:** 
> *   Replace `/path/to/dataset` with your absolute local dataset directory.
> *   Replace `/path/to/output` with the local folder where you want the submission CSV saved.
> *   **Do NOT modify the container-side paths** (`:/dataset` and `:/app/outputs`). Our container utilizes a universal path fix that guarantees results are routed to these exact internal directories regardless of the host machine.

**GPU Version (Recommended):**
```bash
docker run --runtime=nvidia -it --rm \
    -v /path/to/dataset:/dataset \
    -v /path/to/output:/app/outputs \
    abavitrack-gpu
```

**CPU Version:**
```bash
docker run -it --rm \
    -v /path/to/dataset:/dataset \
    -v /path/to/output:/app/outputs \
    abavitrack-cpu
```

---

## 5. Interactive Configuration Prompts

Once the container starts, you will see an interactive menu. The pipeline is designed to use the default containerized paths, meaning you can simply press **Enter** for almost every prompt.

### Expected Prompts & Actions:

| Prompt | Default Container Value | Your Action |
| :--- | :--- | :--- |
| **Dataset root path** | `/dataset` | Press Enter |
| **Manifest JSON path** | `/dataset/metadata/contestant_manifest.json` | Press Enter |
| **CPU workers (extraction)** | `8` | Press Enter |
| **Run data preprocessing** | `n` | Type `y` on your first run to extract frames |
| **Inference threads** | `8` | **GPU:** Type up to 8 depending on VRAM. <br>**CPU:** Type 1 or 2 |
| **Results root directory** | `/app/outputs/tracking_results` | Press Enter |
| **Output CSV file** | `/app/outputs/submission.csv` | Press Enter |

---

## 6. Outputs & Results

This pipeline is optimized for competition evaluators to generate submission CSVs seamlessly. Results are automatically routed to your mounted local output directory:

| Output | Host Location | Purpose |
| :--- | :--- | :--- |
| **Tracking Results** | `/path/to/output/tracking_results/` | Intermediate predictions (`.txt` files) for all sequences. |
| **Submission CSV** | **`/path/to/output/submission.csv`** | **Final submission file for evaluation**. |

---

## 7. ⚠️ Hardcoded Defaults & Warnings

The following parameters are hardcoded to ensure reproducibility and cannot be changed via the interactive prompts:

| Parameter | Value |
| :--- | :--- |
| **Config Name** | `abavit_patch16_224` |
| **Test Epoch** | `10` |
| **Checkpoint File** | `AbaViTrack_ep0010.pth.tar` |
| **Checkpoint Path** | `/app/checkpoints/AbaViTrack_ep0010.pth.tar` |

*If you need to change these values, you must edit the `main.py` or `Dockerfile` source files and rebuild the image.*

---

## 8. Troubleshooting

*   **Issue: "Checkpoint not found"**
    *   Ensure `AbaViTrack_ep0010.pth.tar` exists in the `checkpoints/` folder before running the `docker build` command.
*   **Issue: "Manifest file not found" or "Contents of /dataset: []"**
    *   Verify your host dataset path is correct in the `-v` mount flag. If Docker cannot find your host folder, it will silently mount an empty directory.
*   **Issue: Tracker prints "FPS: -1" and finishes instantly**
    *   The framework has a built-in caching mechanism. If the target output directory already contains `.txt` results from a previous run, the tracker will skip those sequences. Clear your local output directory to force a fresh run.
*   **Issue: Results not saved to host**
    *   Ensure you did not change the default `/app/outputs/...` prompts during the script execution. The Docker volume strictly maps to that internal folderYou are totally right—I definitely dropped the ball on the ordering there. Putting the execution commands *before* explaining how to prepare the dataset is putting the cart way before the horse. I'm wide awake now!