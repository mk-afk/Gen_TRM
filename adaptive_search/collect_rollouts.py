# adaptive_search/collect_rollouts.py

from typing import List, Tuple
import copy
import torch

from .branch import Branch
from .delta_features import delta_features
from .search_loop import adaptive_search


@torch.no_grad()
def collect_delta_rollout(
    *,
    trm,
    delta_net,
    branches: List[Branch],
    puzzle_identifier: torch.Tensor,
    max_len: int,
    compute_cost: float,
    branch_k: int,
    device: torch.device,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Collect ONE Δ̂ training example from a snapshot of branches.

    Returns:
        features: FloatTensor [B, D]
        targets:  FloatTensor [B]
    """

    # -------------------------------
    # Snapshot current state
    # -------------------------------

    snapshot_branches = [
        Branch(
            tokens=b.tokens.clone(),
            score=b.score,
            length=b.length,
        )
        for b in branches
    ]

    # Baseline: STOP now
    baseline_score = max(b.score for b in snapshot_branches)

    # -------------------------------
    # Run full adaptive search forward
    # -------------------------------

    best_branch = adaptive_search(
        trm=trm,
        delta_net=delta_net,
        initial_tokens=snapshot_branches[0].tokens,
        puzzle_identifier=puzzle_identifier,
        max_len=max_len,
        compute_cost=compute_cost,
        branch_k=branch_k,
        device=device,
    )

    final_score = best_branch.score

    # -------------------------------
    # Δ̂ target (same for all branches)
    # -------------------------------

    delta_target = final_score - baseline_score

    # -------------------------------
    # Build training tensors
    # -------------------------------

    features = delta_features(snapshot_branches, max_len, device=device)
    targets = torch.full(
        (features.shape[0],),
        delta_target,
        dtype=torch.float32,
        device=device,
    )

    return features, targets
