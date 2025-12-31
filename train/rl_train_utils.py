# train/rl_train_utils.py
import torch

from rl.rollouts.rl_reinforce import collect_rollout_reinforce
from rl.rollouts.rl_ppo import collect_rollout_ppo

from rl.reinforce import reinforce_update
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

    # --- PPO-specific ---
    algo="ppo",              # "ppo" or "reinforce"
    ppo_epochs=4,
    clip_eps=0.2,

    log_every=10,
):
    """
    Unified RL training loop.

    algo:
        - "reinforce": REINFORCE / A2C-style
        - "ppo": PPO (clipped surrogate)
    """

    assert algo in {"ppo", "reinforce"}, f"Unknown algo {algo}"

    policy.to(device).train()
    if value_fn is not None:
        value_fn.to(device).train()

    # Optimizer
    if algo == "ppo":
        optimizer = torch.optim.Adam(
            list(policy.parameters()) + list(value_fn.parameters()),
            lr=lr,
        )
    else:
        optimizer = torch.optim.Adam(
            list(policy.parameters()),
            lr=lr,
        )

    episode_rewards = []

    for episode in range(1, num_episodes + 1):

        # ======================================================
        # 1. COLLECT ROLLOUT
        # ======================================================
        if algo == "ppo":
            trajectory = collect_rollout_ppo(
                env=env,
                policy=policy,
                tokenizer=tokenizer,
                device=device,
                max_steps=max_steps_per_episode,
            )
            total_reward = trajectory["rewards"].sum().item()

        else:  # reinforce
            trajectory = collect_rollout_reinforce(
                env=env,
                policy=policy,
                tokenizer=tokenizer,
                device=device,
                max_steps=max_steps_per_episode,
            )
            total_reward = sum(trajectory["rewards"])

        episode_rewards.append(total_reward)

        # ======================================================
        # 2. UPDATE
        # ======================================================
        if algo == "ppo":
            for _ in range(ppo_epochs):
                stats = ppo_update(
                    policy=policy,
                    value_fn=value_fn,
                    optimizer=optimizer,
                    trajectory=trajectory,
                    gamma=gamma,
                    clip_eps=clip_eps,
                )

        else:  # reinforce / A2C
            stats = reinforce_update(
                policy=policy,
                optimizer=optimizer,
                trajectory=trajectory,
                gamma=gamma,
            )

        # ======================================================
        # 3. LOGGING
        # ======================================================
        if episode % log_every == 0:
            avg_reward = sum(episode_rewards[-log_every:]) / log_every

            if algo == "ppo":
                print(
                    f"[Episode {episode:5d} | PPO] "
                    f"loss={stats['loss']:.4f} | "
                    f"reward={total_reward:.3f} | "
                    f"avg_reward={avg_reward:.3f} | "
                    f"entropy={stats['entropy']:.3f}"
                )
            else:
                print(
                    f"[Episode {episode:5d} | RL] "
                    f"loss={stats:.4f} | "
                    f"reward={total_reward:.3f} | "
                    f"avg_reward={avg_reward:.3f}"
                )

    return episode_rewards
