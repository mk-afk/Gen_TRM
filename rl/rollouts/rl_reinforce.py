# rl/rollout_reinforce.py
import torch
from envs.actions import EditAction


def collect_rollout_reinforce(
    env,
    policy,
    tokenizer,
    device,
    max_steps=50,
):
    """
    Minimal REINFORCE rollout.
    No PPO assumptions. No fixed-length hacks.
    """

    trajectory = {
        "states": [],       # list[(D,)]
        "log_probs": [],    # list[scalar]
        "rewards": [],      # list[float]
        "entropies": [],    # list[scalar]
    }

    buffer = env.reset()

    for step in range(max_steps):
        state = env.encode_state(buffer)
        tokens = state["tokens"].to(device)

        # ---- policy forward ----
        out = policy(tokens, return_hidden=True)

        action_logits = out["action_logits"]
        token_logits  = out["token_logits"]
        h_last        = out["hidden_states"]    # (D,)

        # ---- sample action ----
        action_dist = torch.distributions.Categorical(logits=action_logits)
        action = action_dist.sample()

        logp = action_dist.log_prob(action)
        entropy = action_dist.entropy()

        token = None

        # ---- optional token ----
        if action.item() in (EditAction.ADD, EditAction.REFINE):
            token_dist = torch.distributions.Categorical(logits=token_logits)
            token = token_dist.sample()

            logp = logp + token_dist.log_prob(token)
            entropy = entropy + token_dist.entropy()

        # ---- env step ----
        buffer, reward, done = env.step(buffer, action.item(), token)

        # ---- record ----
        trajectory["states"].append(h_last.detach())
        trajectory["log_probs"].append(logp)
        trajectory["entropies"].append(entropy)
        trajectory["rewards"].append(float(reward))

        if done:
            break

    # ---- stack tensors ----
    trajectory["states"] = torch.stack(trajectory["states"])       # (T, D)
    trajectory["log_probs"] = torch.stack(trajectory["log_probs"]) # (T,)
    trajectory["entropies"] = torch.stack(trajectory["entropies"]) # (T,)

    return trajectory
