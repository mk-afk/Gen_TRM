# rl/ppo.py
import torch
import torch.nn as nn


def ppo_update(
    policy,
    value_fn,
    optimizer,
    trajectory,
    old_log_probs,
    clip_eps=0.2,
    gamma=0.99,
):
    rewards = trajectory["rewards"]
    log_probs = torch.stack(trajectory["log_probs"])

    # compute returns
    returns = []
    R = 0.0
    for r in reversed(rewards):
        R = r + gamma * R
        returns.insert(0, R)
    returns = torch.tensor(returns, device=log_probs.device)

    # value baseline
    values = value_fn()
    advantages = returns - values.detach()
    advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

    # policy ratio
    ratio = torch.exp(log_probs - old_log_probs)

    clipped = torch.clamp(ratio, 1 - clip_eps, 1 + clip_eps)
    policy_loss = -torch.min(ratio * advantages, clipped * advantages).mean()

    value_loss = nn.functional.mse_loss(values, returns)

    loss = policy_loss + 0.5 * value_loss

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    return loss.item()
