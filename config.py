# config.py
# Centralized experiment configuration for TRM language experiments

# -------------------------
# Reproducibility
# -------------------------
SEED = 42

# -------------------------
# Dataset
# -------------------------
DATASET_NAME = "wikitext"
DATASET_CONFIG = "wikitext-2-v1"   # switch to wikitext-103-v1 later
TEXT_FIELD = "text"

# -------------------------
# Tokenization
# -------------------------
TOKENIZER_NAME = "gpt2"
MAX_LENGTH = None        # None = full text, token-level training
PAD_TOKEN = "eos"

# -------------------------
# Model (TRM)
# -------------------------
VOCAB_SIZE = None        # filled after tokenizer load
D_MODEL = 128
TRM_STEPS = 4            # CHANGE THIS: 1, 2, 4, 8
USE_LAYER_NORM = True

# -------------------------
# Training
# -------------------------
BATCH_SIZE = 1024
LEARNING_RATE = 3e-4
WEIGHT_DECAY = 0.01
TRAIN_STEPS = 100_000
EVAL_EVERY = 5_000
GRAD_CLIP = 1.0

# -------------------------
# Evaluation
# -------------------------
EVAL_BATCHES = 500       # number of random batches
REPORT_PERPLEXITY = True

# -------------------------
# System
# -------------------------
DEVICE = "cuda"          # fallback handled in code
DTYPE = "float32"

# -------------------------
# Logging
# -------------------------
LOG_EVERY = 1000



# =================================================
# Reinforcement Learning (Text Editing)
# =================================================

# -------------------------
# RL environment
# -------------------------
NUM_EDIT_ACTIONS = 6        # MOVE_LEFT, MOVE_RIGHT, DELETE, INSERT, REPLACE, STOP
MAX_BUFFER_LENGTH = 128     # max tokens in response buffer
EDIT_WINDOW = 32            # local context window around cursor

# -------------------------
# RL training
# -------------------------
RL_EPISODES = 500           # number of episodes
RL_MAX_STEPS = 40           # max edits per episode
RL_LEARNING_RATE = 3e-4
RL_GAMMA = 0.99             # discount factor
RL_LOG_EVERY = 10

# -------------------------
# RL reward shaping
# -------------------------
REWARD_CONFIG = {
    # language model likelihood improvement
    "lm_weight": 1.0,

    # penalty per edit step (encourages fewer edits)
    "edit_penalty": 0.01,

    # penalty per token in buffer (brevity)
    "length_penalty": 0.01,

    # bonus for STOP action when buffer is stable
    "stop_bonus": 0.5,
}

# -------------------------
# RL / TRM interaction
# -------------------------
FREEZE_TRM = True           # freeze TRM weights during RL
UNFREEZE_TRM_AFTER = None   # e.g. 200 episodes, or None
TRM_CHECKPOINT = None       # path to pretrained TRM LM (optional)
