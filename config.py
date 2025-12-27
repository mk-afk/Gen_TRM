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
