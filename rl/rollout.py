# rl/rollout.py
import torch
from envs.actions import EditAction

def valid_action_mask(buffer, device, step, min_edits=3):
    mask = torch.ones(4, device=device, dtype=torch.bool)

    # ---- PREVENT EARLY STOP ----
    if step < min_edits:
        mask[EditAction.STOP] = False

    # Never delete or refine BOS
    if buffer.cursor == 0:
        mask[EditAction.DELETE] = False
        mask[EditAction.REFINE] = False

    # Cannot refine beyond buffer
    if buffer.cursor >= len(buffer.tokens):
        mask[EditAction.REFINE] = False

    return mask



def collect_rollout(
    env,
    policy,
    tokenizer,
    device,
    max_steps=50,
):
    """
    PPO-compatible rollout collection.
    """

    trajectory = {
        "states": [],
        "actions": [],
        "old_log_probs": [],
        "rewards": [],
        "dones": [],
        "entropies": [],
    }

    buffer = env.reset()

    for step in range(max_steps):
        state = env.encode_state(buffer)
        state_tokens = state["tokens"].to(device)

        # ---- policy forward (sampling mode) ----
        with torch.no_grad():
            out = policy(state_tokens)

        action_logits = out["action_logits"]
        token_logits = out["token_logits"]

        # ---- ACTION MASKING ----
        mask = valid_action_mask(buffer, device, step)
        masked_action_logits = action_logits.clone()
        masked_action_logits[~mask] = -1e9

        action_dist = torch.distributions.Categorical(logits=masked_action_logits)
        action = action_dist.sample()
        action_id = action.item()

        log_prob = action_dist.log_prob(action)
        entropy = action_dist.entropy()

        # ---- EXTRA STOP ENTROPY BONUS ----
        if action_id == EditAction.STOP:
            entropy = entropy + 0.5


        token = None
        token_logprob = torch.tensor(0.0, device=device)

        if action_id in (EditAction.ADD, EditAction.REFINE):
            masked_token_logits = token_logits.clone()
            masked_token_logits[tokenizer.pad_token_id] = -1e9
            if tokenizer.bos_token_id is not None:
                masked_token_logits[tokenizer.bos_token_id] = -1e9

            token_dist = torch.distributions.Categorical(logits=masked_token_logits)
            token = token_dist.sample()
            token_logprob = token_dist.log_prob(token)

            entropy = entropy + token_dist.entropy()

        # ---- ENV STEP ----
        buffer, reward, done = env.step(buffer, action_id, token)

        # ---- RECORD (CRITICAL) ----
        trajectory["states"].append(state_tokens)
        trajectory["actions"].append({
            "action": action,
            "token": token,
        })
        trajectory["old_log_probs"].append((log_prob + token_logprob).detach())
        trajectory["rewards"].append(torch.tensor(reward, device=device))
        trajectory["dones"].append(done)
        trajectory["entropies"].append(entropy.detach())

        if done:
            break

    # ---- STACK ----
    for k in ["states", "old_log_probs", "rewards", "entropies"]:
        trajectory[k] = torch.stack(trajectory[k])

    trajectory["dones"] = torch.tensor(
        trajectory["dones"], device=device, dtype=torch.bool
    )

    return trajectory
