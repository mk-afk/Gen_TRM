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
):
    states = trajectory["states"]            # (T, D)
    actions = trajectory["actions"]          # list of dicts
    rewards = trajectory["rewards"]
    dones = trajectory["dones"]
    old_log_probs = trajectory["old_log_probs"]

    # ---- returns ----
    returns = compute_returns(rewards, dones, gamma)
    returns = (returns - returns.mean()) / (returns.std() + 1e-8)

    log_probs = []
    entropies = []

    # ---- recompute policy distributions ----
    for t, state in enumerate(states):
        out = policy(tokens_t, return_hidden=False)
        action_logits = policy.action_head(out).squeeze(0)
        token_logits  = policy.token_head(out).squeeze(0)

        action = actions[t]["action"]
        token  = actions[t]["token"]

        action_dist = torch.distributions.Categorical(logits=action_logits)
        logp = action_dist.log_prob(action)
        entropy = action_dist.entropy()

        # STOP entropy bonus
        if action.item() == EditAction.STOP:
            entropy = entropy + 0.5

        if action.item() in (EditAction.ADD, EditAction.REFINE):
            token_dist = torch.distributions.Categorical(logits=token_logits)
            logp = logp + token_dist.log_prob(token)
            entropy = entropy + token_dist.entropy()

        log_probs.append(logp)
        entropies.append(entropy)

    log_probs = torch.stack(log_probs)
    entropies = torch.stack(entropies)

    # ---- KL guard ----
    with torch.no_grad():
        kl = (old_log_probs - log_probs).mean()
    if kl > 0.02:
        return {
            "loss": 0.0,
            "policy": 0.0,
            "value": 0.0,
            "entropy": entropies.mean().item(),
            "kl": kl.item(),
        }

    # ---- value function ----
    values = value_fn(states)

    # ---- advantage ----
    advantages = returns - values.detach()
    advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

    # ---- PPO ratio ----
    ratio = torch.exp(log_probs - old_log_probs)
    clipped_ratio = torch.clamp(ratio, 1 - clip_eps, 1 + clip_eps)

    policy_loss = -torch.min(
        ratio * advantages,
        clipped_ratio * advantages
    ).mean()

    # ---- value loss (Huber) ----
    value_loss = F.smooth_l1_loss(values, returns)

    # ---- entropy bonus ----
    entropy_loss = -entropies.mean()

    total_loss = (
        policy_loss
        + value_coef * value_loss
        + entropy_coef * entropy_loss
    )

    optimizer.zero_grad()
    total_loss.backward()
    torch.nn.utils.clip_grad_norm_(policy.parameters(), 1.0)
    optimizer.step()

    return {
        "loss": total_loss.item(),
        "policy": policy_loss.item(),
        "value": value_loss.item(),
        "entropy": entropies.mean().item(),
        "kl": kl.item(),
    }
