# train/rl_train_utils.py
import torch
from rl.rollout import collect_rollout
from rl.ppo import ppo_update


def rl_train_loop(
    env,
    policy,
    value_fn,
    tokenizer,
    device,
    num_episodes=1000,
    max_steps_per_episode=50,
    lr=3e-4,
    gamma=0.99,
    ppo_epochs=4,
    clip_eps=0.2,
    log_every=10,
):
    """
    PPO training loop for EditPolicy.
    """

    policy.to(device).train()
    value_fn.to(device).train()

    optimizer = torch.optim.Adam(
        list(policy.parameters()) + list(value_fn.parameters()),
        lr=lr
    )

    episode_rewards = []

    for episode in range(1, num_episodes + 1):
        # ---- collect rollout ----
        trajectory = collect_rollout(
            env=env,
            policy=policy,
            tokenizer=tokenizer,
            device=device,
            max_steps=max_steps_per_episode,
        )

        total_reward = sum(trajectory["rewards"]).item()
        episode_rewards.append(total_reward)

        # ---- PPO updates (multiple epochs on same trajectory) ----
        for _ in range(ppo_epochs):
            stats = ppo_update(
                policy=policy,
                value_fn=value_fn,
                optimizer=optimizer,
                trajectory=trajectory,
                gamma=gamma,
                clip_eps=clip_eps,
            )

        # ---- logging ----
        if episode % log_every == 0:
            avg_reward = sum(episode_rewards[-log_every:]) / log_every
            print(
                f"[Episode {episode:5d}] "
                f"loss={stats['loss']:.4f} | "
                f"reward={total_reward:.3f} | "
                f"avg_reward={avg_reward:.3f} | "
                f"entropy={stats['entropy']:.3f}"
            )

    return episode_rewards
