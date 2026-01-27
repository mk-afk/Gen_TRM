# adaptive_search/delta_features.py

import torch
from typing import List
from .branch import Branch


def delta_features(
    branches: List[Branch],
    max_len: int,
    device: torch.device | None = None,
) -> torch.Tensor:
    """
    Compute Δ̂ feature vectors for all branches.

    Args:
        branches:
            List of Branch objects
        max_len:
            Maximum allowed sequence length
        device:
            Optional torch.device

    Returns:
        features: FloatTensor [B, 3]
            Columns:
              0: normalized length
              1: current score
              2: score gap vs best
    """

    assert len(branches) > 0, "No branches provided"

    best_score = max(b.score for b in branches)

    feats = []
    for b in branches:
        feats.append([
            b.length / max_len,
            b.score,
            b.score - best_score,
        ])

    features = torch.tensor(feats, dtype=torch.float32)

    if device is not None:
        features = features.to(device)

    return features
