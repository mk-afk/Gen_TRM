# train/warmstart.py
import itertools
import torch
import torch.nn.functional as F
import torch.optim as optim

from rl.rollout import collect_rollout
from rl.reinforce import compute_returns


def warmstart_bc_to_rl(
    policy,
    env,
    expert_loader,
    tokenizer,
    device,
    episodes=100,
    lr=3e-4,
    gamma=0.99,
    bc_weight_start=1.0,
    bc_weight_end=0.0,
    max_steps=40,
    log_every=10,
):
    policy.train()
    optimizer = optim.AdamW(policy.parameters(), lr=lr)
    expert_iter = itertools.cycle(expert_loader)

    for ep in range(episodes):
        # ---- RL rollout ----
        traj = collect_rollout(
            env=env,
            policy=policy,
            tokenizer=tokenizer,
            device=device,
            max_steps=max_steps,
        )

        returns = compute_returns(traj["rewards"], gamma).to(device)
        if returns.numel() > 1:
            returns = (returns - returns.mean()) / (returns.std() + 1e-8)

        rl_loss = sum(-lp * R for lp, R in zip(traj["log_probs"], returns))

        # ---- BC minibatch ----
        bc_weight = bc_weight_start + (
            bc_weight_end - bc_weight_start
        ) * (ep / episodes)

        bc_loss = 0.0
        if bc_weight > 0:
            ctx, act, tok = next(expert_iter)
            ctx, act, tok = ctx.to(device), act.to(device), tok.to(device)
            a_logits, t_logits = policy(ctx)
            bc_loss = (
                F.cross_entropy(a_logits, act)
                + F.cross_entropy(t_logits, tok)
            )

        loss = rl_loss + bc_weight * bc_loss

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if ep % log_every == 0:
            print(
                f"[Warm] ep {ep:3d} | "
                f"reward {sum(traj['rewards']):6.2f} | "
                f"bc_w {bc_weight:.2f}"
            )

    return policy
