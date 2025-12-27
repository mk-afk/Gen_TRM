# rl/reinforce.py
import torch


def compute_returns(rewards, gamma=0.99):
    returns = []
    R = 0.0
    for r in reversed(rewards):
        R = r + gamma * R
        returns.insert(0, R)
    return torch.tensor(returns)


def reinforce_update(policy, optimizer, trajectory, gamma=0.99):
    """
    Performs one REINFORCE update.
    """

    returns = compute_returns(trajectory["rewards"], gamma)
    returns = (returns - returns.mean()) / (returns.std() + 1e-8)

    loss = 0.0
    for log_prob, R in zip(trajectory["log_probs"], returns):
        loss += -log_prob * R

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    return loss.item()
