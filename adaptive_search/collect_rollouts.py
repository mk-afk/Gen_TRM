# adaptive_search/collect_rollouts.py

from typing import List, Tuple
import torch

from .branch import Branch, default_clone_carry
from .delta_features import delta_features
from .search_loop import frontier_search


@torch.no_grad()
def collect_delta_rollout(
    *,
    trm,
    branches: List[Branch],
    budget_segments: int,
    max_frontier: int,
    branch_m: int,
    device: torch.device,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Collect ONE Δ̂ training example from a snapshot of branches.

    Returns:
        features: FloatTensor [B, 3]
        targets:  FloatTensor [B]
    """

    # Snapshot current state (independent carries so the rollout doesn't
    # mutate the caller's branches).
    snapshot_branches = [
        b.clone(carry_copy_fn=default_clone_carry, strict=False)
        for b in branches
    ]

    # Baseline: best value if we stopped right now.
    baseline_value = max(b.value for b in snapshot_branches)

    # Run full frontier search forward from the best snapshot branch.
    best_snapshot = max(snapshot_branches, key=lambda b: b.value)
    best_branch = frontier_search(
        trm=trm,
        batch=best_snapshot.batch,
        budget_segments=budget_segments,
        max_frontier=max_frontier,
        branch_m=branch_m,
        initial_carry=best_snapshot.carry,
    )

    delta_target = best_branch.value - baseline_value

    # Build training tensors.
    budget_remaining = float(budget_segments)
    features = delta_features(snapshot_branches, budget_remaining, device=device)
    targets = torch.full(
        (features.shape[0],),
        delta_target,
        dtype=torch.float32,
        device=device,
    )

    return features, targets
