# train/bc.py
import torch
import torch.nn.functional as F
import torch.optim as optim


def train_bc(
    policy,
    expert_loader,
    device,
    max_steps=100_000,
    lr=3e-4,
    log_every=100,
):
    # Freeze TRM
    for p in policy.trm.parameters():
        p.requires_grad = False

    policy.train()

    optimizer = optim.AdamW(
        list(policy.action_head.parameters()) +
        list(policy.token_head.parameters()),
        lr=lr,
    )

    running_loss = 0.0
    step = 0

    while step < max_steps:
        for context, action_tgt, token_tgt in expert_loader:
            context = context.to(device)
            action_tgt = action_tgt.to(device)
            token_tgt = token_tgt.to(device)

            optimizer.zero_grad()
            action_logits, token_logits = policy(context)

            loss = (
                F.cross_entropy(action_logits, action_tgt)
                + F.cross_entropy(token_logits, token_tgt)
            )

            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            step += 1

            if step % log_every == 0:
                print(
                    f"[BC] step {step:6d} | "
                    f"loss {running_loss / log_every:.4f}"
                )
                running_loss = 0.0

            if step >= max_steps:
                break

    return policy
