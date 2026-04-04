"""
Training pipeline for the SearchController.

Phase 1 (here): Controller trains against a frozen TRM.
Phase 2 (future): Co-training with asymmetric learning rates + anchor loss on
    L_total = L_deep_supervision + α·L_search_objective.

Rollout collection
------------------
Run one search episode with the controller selecting actions.
At each step t:
    obs      = (acting_branch_feats, best_branch_feats, budget_remaining)
    action   = EXPAND | BRANCH  (STOP ends the episode)
    reward   = Φ(s_{t+1}) – Φ(s_t) – λ·cost(action)
    next_obs = (next_acting_feats, next_best_feats, next_budget)
    done     = budget exhausted

Φ(s) = max branch value in the frontier  (potential-based shaping —
        the Φ differences telescope so the optimal policy is preserved).

Q-learning update
-----------------
Standard DQN:  Q_target = r + γ · max_a Q(s', a) · (1 – done)
Action selection uses counterfactual STOP advantage:  A(s, a) = Q(s, a) – Q(s, STOP).
"""

from __future__ import annotations

import random
from collections import deque
from dataclasses import dataclass, field
from typing import Iterator, List, Optional, Tuple

import torch
import torch.nn.functional as F
import torch.optim as optim

from .branch import Branch, default_clone_carry
from .branching import branch_carry
from .controller import Action, SearchController
from .frontier import Frontier
from .search_loop import _confidence, _initial_carry


# ---------------------------------------------------------------------------
# Transition & replay buffer
# ---------------------------------------------------------------------------

@dataclass
class Transition:
    """Compact (s, a, r, s', done) for Q-learning.

    State is represented as (acting_feats, best_feats, budget) rather than
    storing the full frontier, keeping the replay buffer GPU-memory-free.
    """
    # State t
    acting_feats: torch.Tensor      # [feature_dim]  — branch acted on
    best_feats: torch.Tensor        # [feature_dim]  — best frontier branch
    budget: float                   # remaining segments at time t

    action: int                     # Action.EXPAND or Action.BRANCH
    reward: float

    # State t+1
    next_acting_feats: torch.Tensor  # [feature_dim]
    next_best_feats: torch.Tensor    # [feature_dim]
    next_budget: float

    done: bool


class ReplayBuffer:
    def __init__(self, capacity: int = 10_000) -> None:
        self._buf: deque[Transition] = deque(maxlen=capacity)

    def push(self, t: Transition) -> None:
        self._buf.append(t)

    def sample(self, n: int) -> List[Transition]:
        return random.sample(self._buf, min(n, len(self._buf)))

    def __len__(self) -> int:
        return len(self._buf)


# ---------------------------------------------------------------------------
# Rollout collection
# ---------------------------------------------------------------------------

@torch.no_grad()
def collect_rollout(
    trm,
    controller: SearchController,
    batch: dict,
    budget_segments: int = 16,
    max_frontier: int = 8,
    branch_m: int = 2,
    cost_expand: float = 1.0,
    lambda_cost: float = 0.1,
    noise_scale: float = 0.1,
    epsilon: float = 0.1,
) -> Tuple[List[Transition], Optional[Branch]]:
    """Run one search episode; return transitions and the best branch found.

    TRM is fully frozen — all forward passes run inside @torch.no_grad().

    Args:
        trm:              TinyRecursiveReasoningModel_ACTV1_Inner, frozen.
        controller:       SearchController (parameters updated externally).
        batch:            Problem batch dict (inputs, puzzle_identifiers, …).
        budget_segments:  Total compute budget in segments.
        max_frontier:     Hard cap on frontier size.
        branch_m:         Children produced per BRANCH action.
        cost_expand:      Compute cost of a single EXPAND step.
        lambda_cost:      Cost penalty weight in the reward.
        noise_scale:      Entropy-guided noise magnitude for branching.
        epsilon:          Epsilon-greedy exploration probability.

    Returns:
        (transitions, best_branch)
    """
    frontier = Frontier(max_size=max_frontier)
    transitions: List[Transition] = []

    # ---- Seed frontier ----
    carry = _initial_carry(trm, batch)
    carry, *_, logits = trm.proposal_step(carry, batch)
    conf = _confidence(logits)
    root = Branch(
        batch=batch,
        output=logits.argmax(dim=-1),
        carry=carry.clone(),
        value=conf,
        logp=conf,
        steps=1,
        cost=cost_expand,
    )
    frontier.add(root)
    segments_used = 1
    phi_current = conf

    while segments_used < budget_segments:
        budget_remaining = float(budget_segments - segments_used)
        branches = list(frontier)
        N = len(branches)

        # ---- Per-branch features ----
        branch_feats = controller.get_branch_features(branches, trm)  # [N, D]
        best_idx = max(range(N), key=lambda i: branches[i].value)
        best_feats = branch_feats[best_idx]                            # [D]

        # ---- Action selection ----
        action, branch_idx = controller.select_action(
            branches, branch_feats, budget_remaining, epsilon
        )

        if action == Action.STOP:
            break

        acting_branch = branches[branch_idx]
        acting_feats_cpu = branch_feats[branch_idx].detach().cpu()
        best_feats_cpu = best_feats.detach().cpu()

        # ---- Execute action ----
        if action == Action.EXPAND:
            new_carry, *_, logits = trm.proposal_step(acting_branch.carry, batch)
            conf = _confidence(logits)
            expanded = Branch(
                batch=batch,
                output=logits.argmax(dim=-1),
                carry=new_carry.clone(),
                value=conf,
                logp=conf,
                steps=acting_branch.steps + 1,
                cost=acting_branch.cost + cost_expand,
            )
            frontier.add(expanded)
            # Advance the acting branch in-place so future steps continue from here.
            acting_branch.carry = new_carry.clone()
            acting_branch.output = expanded.output
            acting_branch.value = conf
            acting_branch.steps = expanded.steps
            acting_branch.cost = expanded.cost
            segments_used += 1
            step_cost = cost_expand
            next_acting_feats_cpu = trm.get_features(new_carry).squeeze(0).detach().cpu()

        else:  # BRANCH
            *_, logits_ent = trm.proposal_step(acting_branch.carry, batch)
            children_carries = branch_carry(
                carry=acting_branch.carry,
                logits=logits_ent,
                m=branch_m,
                noise_scale=noise_scale,
                puzzle_emb_len=trm.puzzle_emb_len,
            )
            best_child: Optional[Branch] = None
            n_spawned = 0
            for cc in children_carries:
                if segments_used >= budget_segments:
                    break
                nc, *_, logits = trm.proposal_step(cc, batch)
                c_conf = _confidence(logits)
                cb = Branch(
                    batch=batch,
                    output=logits.argmax(dim=-1),
                    carry=nc.clone(),
                    value=c_conf,
                    logp=c_conf,
                    steps=acting_branch.steps + 1,
                    cost=acting_branch.cost + cost_expand,
                )
                frontier.add(cb)
                if best_child is None or c_conf > best_child.value:
                    best_child = cb
                segments_used += 1
                n_spawned += 1

            step_cost = cost_expand * n_spawned
            if best_child is not None:
                next_acting_feats_cpu = (
                    trm.get_features(best_child.carry).squeeze(0).detach().cpu()
                )
            else:
                next_acting_feats_cpu = acting_feats_cpu

        frontier.prune()

        # ---- Shaped reward: Φ(s_{t+1}) – Φ(s_t) – λ·cost ----
        phi_next = max(b.value for b in frontier)
        reward = phi_next - phi_current - lambda_cost * step_cost
        phi_current = phi_next

        # ---- Next-state best features ----
        new_branches = list(frontier)
        new_feats = controller.get_branch_features(new_branches, trm)
        new_best_idx = max(range(len(new_branches)), key=lambda i: new_branches[i].value)
        next_best_feats_cpu = new_feats[new_best_idx].detach().cpu()

        done = segments_used >= budget_segments

        transitions.append(Transition(
            acting_feats=acting_feats_cpu,
            best_feats=best_feats_cpu,
            budget=budget_remaining,
            action=int(action),
            reward=reward,
            next_acting_feats=next_acting_feats_cpu,
            next_best_feats=next_best_feats_cpu,
            next_budget=budget_remaining - 1.0,
            done=done,
        ))

    return transitions, frontier.best()


# ---------------------------------------------------------------------------
# Q-learning update
# ---------------------------------------------------------------------------

def train_step(
    controller: SearchController,
    optimizer: torch.optim.Optimizer,
    transitions: List[Transition],
    gamma: float = 0.99,
    device: torch.device = torch.device("cpu"),
) -> float:
    """One vectorised Q-learning update on a batch of transitions.

    Returns the scalar MSE loss.
    """
    if not transitions:
        return 0.0

    B = len(transitions)
    D = controller.feature_dim

    # ---- Stack batch onto device ----
    acting  = torch.stack([t.acting_feats for t in transitions]).to(device)       # [B, D]
    best    = torch.stack([t.best_feats   for t in transitions]).to(device)       # [B, D]
    budgets = torch.tensor([t.budget      for t in transitions], device=device)   # [B]
    actions = torch.tensor([t.action      for t in transitions], device=device)   # [B]
    rewards = torch.tensor([t.reward      for t in transitions],
                           dtype=torch.float32, device=device)                     # [B]
    n_act   = torch.stack([t.next_acting_feats for t in transitions]).to(device)  # [B, D]
    n_best  = torch.stack([t.next_best_feats   for t in transitions]).to(device)  # [B, D]
    n_buds  = torch.tensor([t.next_budget      for t in transitions], device=device)
    dones   = torch.tensor([t.done             for t in transitions],
                           dtype=torch.float32, device=device)                     # [B]

    # ---- Predicted Q for taken action ----
    # Each item's "best_ref" is its own best_feats (already per-transition).
    q_expand, q_branch, q_stop = controller.q_net(acting, best, budgets)   # all [B]

    expand_mask = (actions == int(Action.EXPAND)).float()
    q_pred = expand_mask * q_expand + (1.0 - expand_mask) * q_branch       # [B]

    # ---- Target Q (next state, no gradient) ----
    with torch.no_grad():
        nq_expand, nq_branch, nq_stop = controller.q_net(n_act, n_best, n_buds)
        # Max over EXPAND, BRANCH on the same (next) branch, and STOP.
        q_next = torch.stack([nq_expand, nq_branch, nq_stop], dim=1).max(dim=1).values  # [B]

    q_target = rewards + gamma * q_next * (1.0 - dones)

    loss = F.mse_loss(q_pred, q_target.detach())

    optimizer.zero_grad()
    loss.backward()
    torch.nn.utils.clip_grad_norm_(controller.parameters(), max_norm=1.0)
    optimizer.step()

    return loss.item()


# ---------------------------------------------------------------------------
# Training config & main loop
# ---------------------------------------------------------------------------

@dataclass
class TrainConfig:
    # Search
    budget_segments: int = 16
    max_frontier: int = 8
    branch_m: int = 2
    cost_expand: float = 1.0
    lambda_cost: float = 0.1
    noise_scale: float = 0.1

    # Q-learning
    gamma: float = 0.99
    batch_size: int = 32
    replay_capacity: int = 10_000
    learning_rate: float = 3e-4
    train_every: int = 4        # update every N episodes

    # Exploration
    epsilon_start: float = 0.5
    epsilon_end: float = 0.05
    epsilon_decay_episodes: int = 500

    # Controller architecture
    trm_hidden_size: int = 512
    controller_hidden_dim: int = 256

    # Run
    n_episodes: int = 1_000
    log_every: int = 100


def train(
    trm,
    batches: List[dict],
    config: Optional[TrainConfig] = None,
    device: torch.device = torch.device("cpu"),
    controller: Optional[SearchController] = None,
) -> SearchController:
    """Train the SearchController against a frozen TRM.

    Args:
        trm:        TinyRecursiveReasoningModel_ACTV1_Inner, in eval mode.
        batches:    List of batch dicts (inputs, puzzle_identifiers, …).
        config:     Hyperparameters; uses defaults if None.
        device:     Torch device.
        controller: Optional existing controller to resume training from.

    Returns:
        Trained SearchController.
    """
    if config is None:
        config = TrainConfig()

    trm = trm.to(device).eval()
    for p in trm.parameters():
        p.requires_grad_(False)

    if controller is not None:
        controller = controller.to(device)
    else:
        controller = SearchController(
            trm_hidden_size=config.trm_hidden_size,
            hidden_dim=config.controller_hidden_dim,
        ).to(device)

    optimizer = optim.Adam(controller.parameters(), lr=config.learning_rate)
    replay = ReplayBuffer(capacity=config.replay_capacity)

    for episode in range(config.n_episodes):
        # Linear epsilon decay.
        frac = min(1.0, episode / max(1, config.epsilon_decay_episodes))
        epsilon = config.epsilon_start + frac * (config.epsilon_end - config.epsilon_start)

        batch = random.choice(batches)
        batch = {k: v.to(device) for k, v in batch.items()}

        transitions, best = collect_rollout(
            trm=trm,
            controller=controller,
            batch=batch,
            budget_segments=config.budget_segments,
            max_frontier=config.max_frontier,
            branch_m=config.branch_m,
            cost_expand=config.cost_expand,
            lambda_cost=config.lambda_cost,
            noise_scale=config.noise_scale,
            epsilon=epsilon,
        )

        for t in transitions:
            replay.push(t)

        loss = 0.0
        if (episode + 1) % config.train_every == 0 and len(replay) >= config.batch_size:
            batch_t = replay.sample(config.batch_size)
            loss = train_step(controller, optimizer, batch_t,
                              gamma=config.gamma, device=device)

        if (episode + 1) % config.log_every == 0:
            best_val = best.value if best is not None else float("nan")
            print(
                f"ep {episode + 1:>6}/{config.n_episodes} | "
                f"loss={loss:.4f} | best_val={best_val:.4f} | "
                f"ε={epsilon:.3f} | replay={len(replay)}"
            )

    return controller
