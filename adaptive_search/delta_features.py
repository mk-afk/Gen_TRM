# adaptive_search/delta_features.py

import torch
from typing import List
from .branch import Branch


def delta_features(
    branches: List[Branch],
    budget_remaining: float,
    device: torch.device | None = None,
) -> torch.Tensor:
    """
    Compute Δ̂ feature vectors for all branches.

    Args:
        branches:
            List of Branch objects
        budget_remaining:
            Remaining compute budget (segments), used for normalised length.
        device:
            Optional torch.device

    Returns:
        features: FloatTensor [B, 3]
            Columns:
              0: normalised steps (steps / (steps + budget_remaining))
              1: current value
              2: value gap vs best branch
    """

    assert len(branches) > 0, "No branches provided"

    best_value = max(b.value for b in branches)

    feats = []
    for b in branches:
        normalised_steps = b.steps / (b.steps + budget_remaining + 1e-8)
        feats.append([
            normalised_steps,
            b.value,
            b.value - best_value,
        ])

    features = torch.tensor(feats, dtype=torch.float32)

    if device is not None:
        features = features.to(device)

    return features
