# rl/ppo.py
import torch
import torch.nn.functional as F
from envs.actions import EditAction


def compute_returns(rewards, dones, gamma):
    returns = []
    R = 0.0
    for r, d in zip(reversed(rewards), reversed(dones)):
        if d:
            R = 0.0
        R = r + gamma * R
        returns.insert(0, R)
    return torch.stack(returns)


def ppo_update(
    policy,
    value_fn,
    optimizer,
    trajectory,
    clip_eps=0.2,
    gamma=0.99,
    value_coef=0.5,
    entropy_coef=0.01,
    latent_coef=0.0,
):
    """
    trajectory keys:
        states
        actions
        rewards
        dones
        old_log_probs
        latents (optional)
    """

    states = trajectory["states"]
    actions = trajectory["actions"]
    rewards = trajectory["rewards"]
    dones = trajectory["dones"]
    old_log_probs = trajectory["old_log_probs"].detach()
    stop_entropy_bonus = 0.5

    # ---- returns ----
    returns = compute_returns(rewards, dones, gamma)
    returns = (returns - returns.mean()) / (returns.std() + 1e-8)

    # ---- forward pass ----
    policy_out = policy(states, actions)
    log_probs = policy_out["log_prob"]
    entropy = policy_out["entropy"]

    # ---- early stopping KL ----
    with torch.no_grad():
        kl = (trajectory["old_log_probs"] - log_probs).mean()

    if kl > 0.02:
        return {
            "loss": 0.0,
            "policy": 0.0,
            "value": 0.0,
            "entropy": entropy.mean().item(),
            "kl": kl.item(),
        }
    
    # ---- value function ----
    values = value_fn(states).squeeze(-1)

    

    # ---- advantage ----
    advantages = returns - values.detach()
    advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

    # ---- PPO ratio ----
    ratio = torch.exp(log_probs - old_log_probs)

    clipped_ratio = torch.clamp(ratio, 1.0 - clip_eps, 1.0 + clip_eps)
    policy_loss = -torch.min(
        ratio * advantages,
        clipped_ratio * advantages
    ).mean()

    # ---- value loss (Huber) ----
    value_loss = F.smooth_l1_loss(values, returns)

    # ---- entropy bonus ----
    entropy_loss = -entropy.mean()

    if actions == EditAction.STOP:
        entropy_loss = entropy_loss + stop_entropy_bonus

    # ---- latent regularization (optional) ----
    latent_loss = 0.0
    if latent_coef > 0.0 and "latents" in trajectory:
        z = trajectory["latents"]
        latent_loss = ((z[1:] - z[:-1]) ** 2).mean()

    total_loss = (
        policy_loss
        + value_coef * value_loss
        + entropy_coef * entropy_loss
        + latent_coef * latent_loss
    )

    optimizer.zero_grad()
    total_loss.backward()
    torch.nn.utils.clip_grad_norm_(policy.parameters(), 1.0)
    optimizer.step()

    return {
        "loss": total_loss.item(),
        "policy": policy_loss.item(),
        "value": value_loss.item(),
        "entropy": entropy.mean().item(),
    }
