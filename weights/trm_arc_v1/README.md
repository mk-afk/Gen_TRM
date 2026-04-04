---
license: mit
datasets:
- arcprize/arc_agi_v2_public_eval
- arcprize/arc_agi_v1_public_eval
---
This repository contains the TinyRecursiveModels checkpoints for arc v1 public eval and arc v2 public eval that were trained for the performance verification. They were trained using the code and recipe of the [official TRM repository](https://github.com/SamsungSAILMontreal/TinyRecursiveModels). We had to adapt the environment setup as detailed below. We provide these checkpoints for transparency and to facilitate further research. We did not contribute to the TRM reserach nor maintain the TRM code. For any questions, please reach out to the TRM maintainers.

TRM writes checkpoints as `torch state_dicts`. The subdirectories `arc_v1_public` and `arc_v2_public` contain the final checkpoints `step_<final-step>`, which can be loaded with the `load_checkpoint` or by providing the checkpoint path as `load_checkpoint=path/to/checkpoint`. For reference, see the `PretrainConfig` in `pretrain.py`.


## Replication Results


Tiny Recursion Model (TRM) results on ARC-AGI

 - ARC-AGI-1: 40%, $1.76/task
 - ARC-AGI-2: 6.2%, $2.10/task

Tweet: https://x.com/arcprize/status/1978872651180577060
Leaderboard: https://arcprize.org/leaderboard


## Environment Setup
```bash
# use uv for venv
sudo snap install astral-uv --classic
uv venv .venv -p 3.12
source .venv/bin/activate

# install python-dev for adam atan2
sudo apt install python3-dev -y
# install torch
PYTORCH_INDEX_URL=https://download.pytorch.org/whl/cu128
uv pip install torch torchvision torchaudio --index-url $PYTORCH_INDEX_URL
# install dependencies + adam atan
uv pip install packaging ninja wheel setuptools setuptools-scm
uv pip install --no-cache-dir --no-build-isolation adam-atan2 

# test torch, cuda and AdamAtan2
python
import torch
t = torch.tensor([0,1,2]).to('cuda')
from adam_atan2 import AdamATan2

# install remaining dependencies
uv pip install -r requirements.txt
```

## Dataset preprocessing
The repository already contains the raw data, but it needs to be preprocessed. Run the following commands to preprocess the v1 and v2 datasets to make predictions for the public eval datasets.

### ARC-AGI-1
```bash
python -m dataset.build_arc_dataset \
  --input-file-prefix kaggle/combined/arc-agi \
  --output-dir data/arc1concept-aug-1000 \
  --subsets training evaluation concept \
  --test-set-name evaluation
```

### ARC-AGI-2
```bash
python -m dataset.build_arc_dataset \
  --input-file-prefix kaggle/combined/arc-agi \
  --output-dir data/arc2concept-aug-1000 \
  --subsets training2 evaluation2 concept \
  --test-set-name evaluation2
```
## Training
To reproduce the checkpoints, run the following two training runs on a single 8:H100 node. Each run takes ~20-30h. To speed it up, instructions for multi-node training are below.

### ARC-AGI-2
```bash
run_name="trm_arc_v1_public"
torchrun --nproc-per-node 8 --rdzv_backend=c10d --rdzv_endpoint=localhost:0 --nnodes=1 pretrain.py \
arch=trm \
data_paths="[data/arc1concept-aug-1000]" \
arch.L_layers=2 \
arch.H_cycles=3 arch.L_cycles=4 \
+run_name=${run_name} ema=True
```

### ARC-AGI-2
```bash
run_name="trm_arc_v2_public"
torchrun --nproc-per-node 8 --rdzv_backend=c10d --rdzv_endpoint=localhost:0 --nnodes=1 pretrain.py \
arch=trm \
data_paths="[data/arc2concept-aug-1000]" \
arch.L_layers=2 \
arch.H_cycles=3 arch.L_cycles=4 \
+run_name=${run_name} ema=True
```

### For multi-node training:
```bash
export MAIN_ADDR=<MAIN_IP>
export MAIN_PORT=29500
export NNODES=2
export GPUS_PER_NODE=8          
export OMP_NUM_THREADS=8        
export NCCL_PORT_RANGE=40000-40050
run_name="arc_v1_public_2_nodes"
# on each node:
export NODE_RANK=0
torchrun \
  --nnodes $NNODES \
  --node_rank $NODE_RANK \
  --nproc_per_node $GPUS_PER_NODE \
  --rdzv_backend c10d \
  --rdzv_endpoint $MAIN_ADDR:$MAIN_PORT \
  pretrain.py \
arch=trm \
data_paths="[data/arc1concept-aug-1000]" \
arch.L_layers=2 \
arch.H_cycles=3 arch.L_cycles=4 \
+run_name=${run_name} ema=True \
eval_interval=50000
```