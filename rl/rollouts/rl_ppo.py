# rl/rollout_ppo.py
import torch
from envs.actions import EditAction
from rl.rollouts.rollout import valid_action_mask


def collect_rollout_ppo(
    env,
    policy,
    tokenizer,
    device,
    max_steps=50,
):
    trajectory = {
        "states": [],          # (T, D)
        "actions": [],         # list[dict]
        "action_masks": [],    # (T, A)
        "old_log_probs": [],   # (T,)
        "rewards": [],         # (T,)
        "entropies": [],       # (T,)
        "dones": [],           # (T,)
    }

    buffer = env.reset()

    for step in range(max_steps):
        state = env.encode_state(buffer)
        tokens = state["tokens"].to(device)

        out = policy(tokens, return_hidden=True)

        action_logits = out["action_logits"]
        token_logits  = out["token_logits"]
        h_last        = out["hidden_states"]

        # ---- action mask ----
        mask = valid_action_mask(buffer, device, step)
        masked_logits = action_logits.clone()
        masked_logits[~mask] = -1e9

        action_dist = torch.distributions.Categorical(logits=masked_logits)
        action = action_dist.sample()
        logp = action_dist.log_prob(action)
        entropy = action_dist.entropy()

        token = None
        if action.item() in (EditAction.ADD, EditAction.REFINE):
            token_dist = torch.distributions.Categorical(logits=token_logits)
            token = token_dist.sample()
            logp = logp + token_dist.log_prob(token)
            entropy = entropy + token_dist.entropy()

        buffer, reward, done = env.step(buffer, action.item(), token)

        # ---- record ----
        trajectory["states"].append(h_last.detach())
        trajectory["actions"].append({"action": action, "token": token})
        trajectory["action_masks"].append(mask)
        trajectory["old_log_probs"].append(logp.detach())
        trajectory["rewards"].append(torch.tensor(reward, device=device))
        trajectory["entropies"].append(entropy.detach())
        trajectory["dones"].append(done)

        if done:
            break

    # ---- stack tensors ----
    trajectory["states"] = torch.stack(trajectory["states"])
    trajectory["old_log_probs"] = torch.stack(trajectory["old_log_probs"])
    trajectory["rewards"] = torch.stack(trajectory["rewards"])
    trajectory["entropies"] = torch.stack(trajectory["entropies"])
    trajectory["action_masks"] = torch.stack(trajectory["action_masks"])
    trajectory["dones"] = torch.tensor(trajectory["dones"], device=device)

    return trajectory