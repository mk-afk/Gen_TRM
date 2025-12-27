import math
import torch
import torch.nn as nn
import config

from LanguageTRM import LanguageTRM
from Dataset_loader import train_ids, valid_ids, tokenizer


# -------------------------
# Device
# -------------------------
device = (
    "cuda"
    if torch.cuda.is_available() and config.DEVICE == "cuda"
    else "cpu"
)

torch.manual_seed(config.SEED)


# -------------------------
# Sampling
# -------------------------
def sample_batch(ids):
    idx = torch.randint(0, len(ids) - 1, (config.BATCH_SIZE,))
    x = ids[idx]
    y = ids[idx + 1]
    return x.to(device), y.to(device)


# -------------------------
# Evaluation
# -------------------------
@torch.no_grad()
def eval_loss(model, ids, steps=config.EVAL_BATCHES):
    model.eval()
    losses = []

    for _ in range(steps):
        x, y = sample_batch(ids)
        logits = model(x)
        loss = nn.functional.cross_entropy(logits, y)
        losses.append(loss.item())

    model.train()
    return sum(losses) / len(losses)


# -------------------------
# Experiment
# -------------------------
def run_experiment():
    model = LanguageTRM(
        vocab_size=len(tokenizer),
        d_model=config.D_MODEL,
        steps=config.TRM_STEPS,
    ).to(device)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.LEARNING_RATE,
        weight_decay=config.WEIGHT_DECAY,
    )

    print(f"\n=== TRM steps = {config.TRM_STEPS} ===")

    model.train()

    for step in range(1, config.TRAIN_STEPS + 1):
        x, y = sample_batch(train_ids)

        logits = model(x)
        loss = nn.functional.cross_entropy(logits, y)

        optimizer.zero_grad()
        loss.backward()

        if config.GRAD_CLIP is not None:
            torch.nn.utils.clip_grad_norm_(
                model.parameters(),
                config.GRAD_CLIP,
            )

        optimizer.step()

        if step % config.EVAL_EVERY == 0:
            val_loss = eval_loss(model, valid_ids)
            ppl = math.exp(val_loss)

            print(
                f"step {step:6d} | "
                f"train loss {loss.item():.3f} | "
                f"val loss {val_loss:.3f} | "
                f"ppl {ppl:.1f}"
            )

    return model
