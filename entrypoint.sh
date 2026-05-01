#!/bin/bash
# If NVIDIA GPU is available, set NUM_GPUS=1; otherwise 0
if python -c "import torch; assert torch.cuda.is_available()" &>/dev/null; then
    export NUM_GPUS=1
else
    export NUM_GPUS=0
fi

exec python main.py "$@"