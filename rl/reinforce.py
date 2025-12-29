# rl/reinforce.py
import torch
import torch.nn.functional as F


def compute_returns(rewards, gamma):
    """
    rewards: list[float]
    returns: tensor (T,)
    """
    R = 0.0
    returns = []

    for r in reversed(rewards):
        R = r + gamma * R
        returns.insert(0, R)

    return torch.tensor(returns, dtype=torch.float32)


def reinforce_update(
    policy,
    optimizer,
    trajectory,
    gamma,
):
    """
    Advantage-based REINFORCE (A2C-style)
    """

    rewards = trajectory["rewards"]
    log_probs = trajectory["log_probs"]
    values = trajectory["values"]

    returns = compute_returns(rewards, gamma).to(values.device)

    # Normalize returns for stability
    if returns.numel() > 1:
        returns = (returns - returns.mean()) / (returns.std() + 1e-8)

    # Advantage
    advantage = returns - values.detach()

    # Losses
    policy_loss = -(log_probs * advantage).sum()
    value_loss  = F.mse_loss(values, returns)

    loss = policy_loss + 0.5 * value_loss

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    return loss.item()
