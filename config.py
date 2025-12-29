"""
Centralized configuration for:
- TRM language modeling
- TRM depth ablations
- RL-based text editing (PPO)
"""

# =================================================
# Reproducibility
# =================================================
SEED = 42


# =================================================
# Dataset
# =================================================
DATASET_NAME = "wikitext"
DATASET_CONFIG = "wikitext-2-v1"      # switch to "wikitext-103-v1" later
TEXT_FIELD = "text"


# =================================================
# Tokenization
# =================================================
TOKENIZER_NAME = "gpt2"
PAD_TOKEN = "eos"
MAX_LENGTH = None                     # None = no truncation


# =================================================
# Model: TRM (shared by LM + RL)
# =================================================
VOCAB_SIZE = None                     # filled after tokenizer load
D_MODEL = 128
TRM_STEPS = 4                         # {1, 2, 4, 8}
USE_LAYER_NORM = True


# =================================================
# Language Modeling Training
# =================================================
BATCH_SIZE = 1024
LEARNING_RATE = 3e-4
WEIGHT_DECAY = 0.01
GRAD_CLIP = 1.0

TRAIN_STEPS = 100_000
EVAL_EVERY = 5_000
EVAL_BATCHES = 500
REPORT_PERPLEXITY = True


# =================================================
# System
# =================================================
DEVICE = "cuda"                       # fallback handled in code
DTYPE = "float32"


# =================================================
# Logging
# =================================================
LOG_EVERY = 1_000


# =================================================
# Reinforcement Learning: Text Editing
# =================================================

# -------------------------
# Environment
# -------------------------
NUM_EDIT_ACTIONS = 4                  # ADD, REFINE, DELETE, STOP
MAX_BUFFER_LENGTH = 128               # max tokens in response buffer
EDIT_WINDOW = 32                      # context window around cursor


# -------------------------
# RL Training (PPO)
# -------------------------
RL_EPISODES = 3_000                   # 500 is too small for PPO
RL_MAX_STEPS = 40                     # edits per episode
RL_LEARNING_RATE = 3e-4
RL_GAMMA = 0.99
RL_LOG_EVERY = 10


# -------------------------
# Reward Shaping
# -------------------------
REWARD_CONFIG = {
    # LM likelihood improvement
    # ASSUMES: per-token Δ log-prob, normalized by length
    "lm_weight": 1.0,

    # per-step cost (discourage endless edits)
    "edit_penalty": 0.02,

    # length penalty (brevity bias)
    "length_penalty": 0.01,

    # reward for deciding to STOP
    # keep low initially to avoid early STOP collapse
    "stop_bonus": 0.1,
}


# =================================================
# TRM ↔ RL Interaction
# =================================================
FREEZE_TRM = True                     # freeze TRM during RL
UNFREEZE_TRM_AFTER = None             # set (e.g., 1000) only for ablations
TRM_CHECKPOINT = None                 # path to pretrained TRM LM
