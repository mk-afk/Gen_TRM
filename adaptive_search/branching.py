"""Entropy-guided carry perturbation for branching diversity.

High-entropy token positions get the most noise; low-entropy positions get
near-zero noise.  Puzzle-prefix positions are always kept clean (zero noise).
"""

from __future__ import annotations
from typing import List

import torch

from models.recursive_reasoning.trm import TinyRecursiveReasoningModel_ACTV1InnerCarry


def branch_carry(
    carry: TinyRecursiveReasoningModel_ACTV1InnerCarry,
    logits: torch.Tensor,
    m: int,
    noise_scale: float = 0.1,
    puzzle_emb_len: int = 0,
) -> List[TinyRecursiveReasoningModel_ACTV1InnerCarry]:
    """Create m child carries from a parent carry by perturbing z_L.

    Noise is scaled by per-position entropy: uncertain positions (high entropy)
    receive more perturbation so children diverge where the model is unsure,
    while confident positions remain anchored.

    Args:
        carry:          Parent carry.  Not modified.
        logits:         Output logits *after* the puzzle prefix has been
                        stripped — shape [B, seq_len, vocab_size].
        m:              Number of child carries to produce.
        noise_scale:    Base magnitude of the Gaussian noise.  Calibrate
                        empirically; start at 0.1.
        puzzle_emb_len: Number of puzzle-prefix positions in z_L that should
                        receive zero noise.

    Returns:
        List of m new TinyRecursiveReasoningModel_ACTV1InnerCarry objects,
        each with an independently perturbed z_L.
    """
    # Per-position entropy over vocab.  logits: [B, seq_len, vocab_size]
    probs = torch.softmax(logits.float(), dim=-1)
    entropy = -(probs * (probs + 1e-8).log()).sum(dim=-1)   # [B, seq_len]

    # Normalise entropy to [0, 1] per batch item.
    ent_min = entropy.amin(dim=-1, keepdim=True)            # [B, 1]
    ent_max = entropy.amax(dim=-1, keepdim=True)            # [B, 1]
    weights = (entropy - ent_min) / (ent_max - ent_min + 1e-8)  # [B, seq_len]

    # Pad puzzle prefix positions with zero weight so they are not perturbed.
    if puzzle_emb_len > 0:
        pad = torch.zeros(
            weights.shape[0], puzzle_emb_len, device=weights.device, dtype=weights.dtype
        )
        weights = torch.cat([pad, weights], dim=1)          # [B, seq_len + puzzle_emb_len]

    weights = weights.unsqueeze(-1)                         # [B, L, 1]

    children: List[TinyRecursiveReasoningModel_ACTV1InnerCarry] = []
    for _ in range(m):
        child = carry.clone()
        child.z_L = child.z_L + weights * noise_scale * torch.randn_like(child.z_L)
        children.append(child)

    return children
