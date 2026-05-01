# Build with:
#   docker build --build-arg USE_GPU=true -t abavitrack-gpu .
#   docker build --build-arg USE_GPU=false -t abavitrack-cpu .

ARG USE_GPU=true

# -----------------------------------------------------------------
# Stage 1: Select base image
# -----------------------------------------------------------------
FROM nvidia/cuda:11.8-cudnn8-devel-ubuntu20.04 as gpu
FROM python:3.8-slim as cpu
FROM ${USE_GPU:+gpu}${USE_GPU:-cpu}

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1

# -----------------------------------------------------------------
# System packages
# -----------------------------------------------------------------
RUN apt-get update && apt-get install -y --no-install-recommends \
    git wget curl ca-certificates build-essential \
    libglib2.0-0 libsm6 libxrender1 libxext6 libgl1-mesa-glx \
    libjpeg-turbo8-dev libpng-dev libtiff-dev liblmdb-dev \
    python3.8 python3.8-dev python3.8-distutils python3-pip \
    && ln -s /usr/bin/python3.8 /usr/bin/python \
    && rm -rf /var/lib/apt/lists/*

# -----------------------------------------------------------------
# PyTorch installation
# -----------------------------------------------------------------
ARG USE_GPU
RUN if [ "$USE_GPU" = "true" ] ; then \
        pip install --no-cache-dir torch==2.4.1+cu118 torchvision==0.19.1+cu118 torchaudio==2.4.1+cu118 \
            --index-url https://download.pytorch.org/whl/cu118 ; \
    else \
        pip install --no-cache-dir torch==2.4.1 torchvision==0.19.1 torchaudio==2.4.1 \
            --index-url https://download.pytorch.org/whl/cpu ; \
    fi

# -----------------------------------------------------------------
# Application
# -----------------------------------------------------------------
COPY . /app
WORKDIR /app



RUN pip install --no-cache-dir -r requirements.txt

ENV DATASET_ROOT=/dataset \
    CHECKPOINT_PATH=/app/models/abavit_gs_8.pth.tar \
    RESULTS_ROOT=/app/outputs/tracking_results \
    OUTPUT_CSV=/app/submission.csv

RUN mkdir -p $RESULTS_ROOT

# -----------------------------------------------------------------
# Auto‑detect GPU and set NUM_GPUS for the tracker
# -----------------------------------------------------------------
COPY entrypoint.sh /entrypoint.sh
RUN chmod +x /entrypoint.sh
ENTRYPOINT ["/entrypoint.sh"]

CMD ["python", "main.py"]



