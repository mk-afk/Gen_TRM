# train/rl_train_utils.py
import torch
from rl.rollout import collect_rollout
from rl.reinforce import reinforce_update


def rl_train_loop(
    env,
    policy,
    tokenizer,
    device,
    num_episodes=1000,
    max_steps_per_episode=50,
    lr=3e-4,
    gamma=0.99,
    log_every=10,
):
    """
    High-level RL training loop (REINFORCE).
    """

    policy.to(device)
    policy.train()

    optimizer = torch.optim.Adam(policy.parameters(), lr=lr)

    episode_rewards = []

    for episode in range(1, num_episodes + 1):
        # 1) Collect trajectory
        trajectory = collect_rollout(
            env=env,
            policy=policy,
            tokenizer=tokenizer,
            device=device,
            max_steps=max_steps_per_episode,
        )

        total_reward = sum(trajectory["rewards"])
        episode_rewards.append(total_reward)

        # 2) Policy gradient update
        loss = reinforce_update(
            policy=policy,
            optimizer=optimizer,
            trajectory=trajectory,
            gamma=gamma,
        )

        # 3) Logging
        if episode % log_every == 0:
            avg_reward = sum(episode_rewards[-log_every:]) / log_every
            print(
                f"[Episode {episode:5d}] "
                f"loss={loss:.4f} | "
                f"reward={total_reward:.3f} | "
                f"avg_reward={avg_reward:.3f}"
            )

    return episode_rewards
