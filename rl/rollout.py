# rl/rollout.py
import torch
from envs.actions import EditAction


def valid_action_mask(buffer, device):
    mask = torch.ones(4, device=device, dtype=torch.bool)
    # DELETE, ADD, REFINE, STOP

    if len(buffer.tokens) <= 1:  # only BOS
        mask[EditAction.DELETE] = False
        mask[EditAction.REFINE] = False

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
    Runs one episode and returns a trajectory dict.
    """

    trajectory = {
        "log_probs": [],
        "rewards": [],
        "actions": [],
        "tokens": [],
        "entropies": [],
    }

    buffer = env.reset()

    for _ in range(max_steps):
        state = env.encode_state(buffer)
        state_tokens = state["tokens"].to(device)

        action_logits, token_logits = policy(state_tokens)

        # ---- ACTION MASKING ----
        mask = valid_action_mask(buffer, device)
        masked_action_logits = action_logits.clone()
        masked_action_logits[~mask] = -1e9

        action_dist = torch.distributions.Categorical(logits=masked_action_logits)
        action = action_dist.sample()
        action_id = action.item()

        log_prob = action_dist.log_prob(action)
        entropy = action_dist.entropy()

        token = None
        token_logprob = torch.tensor(0.0, device=device)

        # Sample token if needed
        if action_id in (EditAction.ADD, EditAction.REFINE):
            token_logits = token_logits.clone()
            token_logits[tokenizer.pad_token_id] = -1e9
            if tokenizer.bos_token_id is not None:
                token_logits[tokenizer.bos_token_id] = -1e9

            token_dist = torch.distributions.Categorical(logits=token_logits)
            token = token_dist.sample()
            token_logprob = token_dist.log_prob(token)

        buffer, reward, done = env.step(buffer, action_id, token)

        # record
        trajectory["log_probs"].append(log_prob + token_logprob)
        trajectory["rewards"].append(reward)
        trajectory["actions"].append(action_id)
        trajectory["tokens"].append(token.item() if token is not None else None)
        trajectory["entropies"].append(entropy)

        if done:
            break

    return trajectory
