import torch
import torch.nn as nn
import math
import config


def sample_batch(ids, device):
    idx = torch.randint(0, len(ids) - 1, (config.BATCH_SIZE,))
    x = ids[idx]
    y = ids[idx + 1]
    return x.to(device), y.to(device)


@torch.no_grad()
def eval_loss(model, ids, device):
    model.eval()
    losses = []

    for _ in range(config.EVAL_BATCHES):
        x, y = sample_batch(ids, device)
        logits = model(x)
        loss = nn.functional.cross_entropy(logits, y)
        losses.append(loss.item())

    model.train()
    return sum(losses) / len(losses)


def train_loop(model, train_ids, valid_ids, device):
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.LEARNING_RATE,
        weight_decay=config.WEIGHT_DECAY,
    )

    for step in range(1, config.TRAIN_STEPS + 1):
        x, y = sample_batch(train_ids, device)
        logits = model(x)
        loss = nn.functional.cross_entropy(logits, y)

        optimizer.zero_grad()
        loss.backward()

        if config.GRAD_CLIP:
            torch.nn.utils.clip_grad_norm_(
                model.parameters(), config.GRAD_CLIP
            )

        optimizer.step()

        if step % config.EVAL_EVERY == 0:
            val_loss = eval_loss(model, valid_ids, device)
            ppl = math.exp(val_loss)
            print(
                f"step {step:6d} | "
                f"train loss {loss.item():.3f} | "
                f"val loss {val_loss:.3f} | "
                f"ppl {ppl:.1f}"
            )
