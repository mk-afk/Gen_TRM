# rl/reinforce.py
import torch
import torch.nn.functional as F


def compute_returns(rewards, gamma):
    R = 0.0
    returns = []

    for r in reversed(rewards):
        R = r + gamma * R
        returns.insert(0, R)

    return torch.tensor(returns, dtype=torch.float32)


def reinforce_update(
    policy,
    value_fn,
    optimizer,
    trajectory,
    gamma,
    entropy_coef=0.01,
):
    rewards   = trajectory["rewards"]
    log_probs = trajectory["log_probs"]
    states    = trajectory["states"]
    entropies = trajectory.get("entropies", None)

    # ---- returns ----
    returns = compute_returns(rewards, gamma).to(log_probs.device)

    # ---- value estimates ----
    values = value_fn(states).view(-1)
    returns = returns.view(-1)

    # ---- advantage ----
    advantage = returns - values.detach()
    if advantage.numel() > 1:
        advantage = (advantage - advantage.mean()) / (advantage.std() + 1e-8)

    # ---- losses ----
    policy_loss = -(log_probs * advantage).sum()
    value_loss  = F.mse_loss(values, returns)

    loss = policy_loss + 0.5 * value_loss

    if entropies is not None:
        entropy_loss = -entropies.mean()
        loss = loss + entropy_coef * entropy_loss

    optimizer.zero_grad()
    loss.backward()
    torch.nn.utils.clip_grad_norm_(policy.parameters(), 1.0)
    optimizer.step()

    return {
        "loss": loss.item(),
        "policy": policy_loss.item(),
        "value": value_loss.item(),
    }
