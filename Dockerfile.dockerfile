FROM nvidia/cuda:11.8-cudnn8-devel-ubuntu20.04

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1

# Install Python 3.8 and system dependencies for OpenCV, lmdb, etc.
RUN apt-get update && apt-get install -y --no-install-recommends \
    git wget curl ca-certificates \
    build-essential \
    libglib2.0-0 libsm6 libxrender1 libxext6 libgl1-mesa-glx \
    libjpeg-turbo8-dev libpng-dev libtiff-dev \
    liblmdb-dev \
    python3.8 python3.8-dev python3.8-distutils python3-pip \
    && ln -s /usr/bin/python3.8 /usr/bin/python \
    && rm -rf /var/lib/apt/lists/*

# Upgrade pip and install CUDA-enabled PyTorch
RUN pip install --no-cache-dir \
    torch==2.4.1+cu118 \
    torchvision==0.19.1+cu118 \
    torchaudio==2.4.1+cu118 \
    --index-url https://download.pytorch.org/whl/cu118

# Copy application code (including requirements.txt)
COPY . /app
WORKDIR /app


# Install remaining Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# (Optional) Pre-download the model checkpoint – otherwise mount at runtime
# RUN mkdir -p /app/models && \
#     wget -O /app/models/abavit_gs_8.pth.tar <URL>

# Set environment variables (can be overridden)
ENV DATASET_ROOT=/dataset \
    CHECKPOINT_PATH=/app/checkpoints/AbaViTrack_ep0300.pth.tar \
    RESULTS_ROOT=/app/outputs/tracking_results \
    OUTPUT_CSV=/app/submission.csv

# Create output directories
RUN mkdir -p $RESULTS_ROOT

# Launch the interactive pipeline
ENTRYPOINT ["python", "main.py"]