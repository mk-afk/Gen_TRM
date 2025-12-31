
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
    trajectory = {
        "states": [],      # (T, D)
        "log_probs": [],   # (T,)
        "rewards": [],     # list[float]
    }

    buffer = env.reset()

    for step in range(max_steps):
        state = env.encode_state(buffer)
        tokens = state["tokens"].to(device)

        # ---- policy forward ----
        out = policy(tokens, return_hidden=True)

        action_logits = out["action_logits"]
        token_logits  = out["token_logits"]
        h_last        = out["hidden_states"]   # (D,)

        # ---- sample action ----
        action_dist = torch.distributions.Categorical(logits=action_logits)
        action = action_dist.sample()
        logp = action_dist.log_prob(action)

        token = None
        if action.item() in (EditAction.ADD, EditAction.REFINE):
            token_dist = torch.distributions.Categorical(logits=token_logits)
            token = token_dist.sample()
            logp = logp + token_dist.log_prob(token)

        # ---- env step ----
        buffer, reward, done = env.step(buffer, action.item(), token)

        # ---- record ----
        trajectory["states"].append(h_last.detach())
        trajectory["log_probs"].append(logp)
        trajectory["rewards"].append(reward)

        if done:
            break

    # ---- stack tensors ----
    trajectory["states"] = torch.stack(trajectory["states"])
    trajectory["log_probs"] = torch.stack(trajectory["log_probs"])

    return trajectory
