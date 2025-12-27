# rl/rollout.py
import torch
from envs.actions import EditAction


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
    }

    buffer = env.reset()

    for _ in range(max_steps):
        state = env.encode_state(buffer)
        tokens = state["tokens"].to(device)

        action_logits, token_logits = policy(tokens)

        # sample action
        action_dist = torch.distributions.Categorical(logits=action_logits)
        action = action_dist.sample()
        action_logprob = action_dist.log_prob(action)

        # sample token if needed
        token = None
        token_logprob = torch.tensor(0.0, device=device)

        if action.item() in {EditAction.INSERT, EditAction.REPLACE}:
            token_dist = torch.distributions.Categorical(logits=token_logits)
            token = token_dist.sample()
            token_logprob = token_dist.log_prob(token)

        # environment step
        buffer, reward, done = env.step(buffer, action.item(), token)

        # record
        trajectory["log_probs"].append(action_logprob + token_logprob)
        trajectory["rewards"].append(reward)
        trajectory["actions"].append(action.item())
        trajectory["tokens"].append(token.item() if token is not None else None)

        if done:
            break

    return trajectory
