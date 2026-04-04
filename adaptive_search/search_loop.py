"""Search loops built on top of the frozen TRM inner model.

Two levels of complexity:

1. greedy_search   — single branch, expand num_steps times, return best seen.
                     Validates the TRM fork interface end-to-end.

2. frontier_search — multi-branch with a trivial hard-coded policy:
                       * Always expand the best branch.
                       * Every branch_every steps, branch the best branch into
                         m children using entropy-guided perturbation.
                     Validates carry cloning, branch divergence, and the
                     Frontier machinery before plugging in a Q-controller.
"""

from __future__ import annotations
from typing import Dict, Optional

import torch

from models.recursive_reasoning.trm import (
    TinyRecursiveReasoningModel_ACTV1InnerCarry,
    TinyRecursiveReasoningModel_ACTV1_Inner,
)
from .branch import Branch
from .branching import branch_carry
from .frontier import Frontier


# ---------------------------------------------------------------------------
# Scoring helper
# ---------------------------------------------------------------------------

def _confidence(logits: torch.Tensor) -> float:
    """Mean of per-position max log-softmax — higher means more confident.

    logits: [B, seq_len, vocab_size]
    Returns a single float averaged over both batch and sequence dimensions.
    """
    log_probs = torch.log_softmax(logits.float(), dim=-1)
    return log_probs.max(dim=-1).values.mean().item()


# ---------------------------------------------------------------------------
# Initial carry helper
# ---------------------------------------------------------------------------

def _initial_carry(
    trm: TinyRecursiveReasoningModel_ACTV1_Inner,
    batch: Dict[str, torch.Tensor],
) -> TinyRecursiveReasoningModel_ACTV1InnerCarry:
    """Build a fresh carry by broadcasting H_init / L_init to batch size."""
    B = batch["inputs"].shape[0]
    L = trm.config.seq_len + trm.puzzle_emb_len
    D = trm.config.hidden_size
    device = trm.H_init.device

    z_H = trm.H_init.view(1, 1, D).expand(B, L, D).clone().to(device)
    z_L = trm.L_init.view(1, 1, D).expand(B, L, D).clone().to(device)
    return TinyRecursiveReasoningModel_ACTV1InnerCarry(z_H=z_H, z_L=z_L)


# ---------------------------------------------------------------------------
# 1. Greedy single-branch search
# ---------------------------------------------------------------------------

def greedy_search(
    trm: TinyRecursiveReasoningModel_ACTV1_Inner,
    batch: Dict[str, torch.Tensor],
    num_steps: int = 8,
    cost_per_step: float = 1.0,
    initial_carry: Optional[TinyRecursiveReasoningModel_ACTV1InnerCarry] = None,
) -> Branch:
    """Expand a single branch num_steps times; return the best output seen.

    Args:
        trm:            Frozen TRM inner model.
        batch:          Problem batch dict (inputs, puzzle_identifiers, …).
        num_steps:      Number of proposal_step calls.
        cost_per_step:  Cost charged per segment (for the cost field).
        initial_carry:  If None, H_init / L_init are used.

    Returns:
        The Branch with the highest confidence value observed across all steps.
    """
    carry = initial_carry if initial_carry is not None else _initial_carry(trm, batch)

    best: Optional[Branch] = None

    for step in range(num_steps):
        carry, *_, logits = trm.proposal_step(carry, batch)
        output = logits.argmax(dim=-1)
        conf = _confidence(logits)

        branch = Branch(
            batch=batch,
            output=output,
            carry=carry.clone(),
            value=conf,
            logp=conf,
            steps=step + 1,
            cost=(step + 1) * cost_per_step,
        )

        if best is None or conf > best.value:
            best = branch

    assert best is not None, "num_steps must be >= 1"
    return best


# ---------------------------------------------------------------------------
# 2. Frontier search with trivial hard-coded policy
# ---------------------------------------------------------------------------

def frontier_search(
    trm: TinyRecursiveReasoningModel_ACTV1_Inner,
    batch: Dict[str, torch.Tensor],
    budget_segments: int = 16,
    max_frontier: int = 8,
    branch_m: int = 2,
    branch_every: int = 4,
    noise_scale: float = 0.1,
    cost_per_step: float = 1.0,
    initial_carry: Optional[TinyRecursiveReasoningModel_ACTV1InnerCarry] = None,
) -> Branch:
    """Multi-branch search with a trivial always-expand-best policy.

    Policy rules (hard-coded, no Q-function):
      - Every step: expand the best branch one proposal_step further.
      - Every `branch_every` steps: also branch the best branch into `branch_m`
        entropy-perturbed children and add them to the frontier.

    Args:
        trm:              Frozen TRM inner model.
        batch:            Problem batch dict.
        budget_segments:  Total number of proposal_step calls allowed.
        max_frontier:     Maximum number of live branches at any time.
        branch_m:         Children produced when branching.
        branch_every:     Branch on steps that are multiples of this value.
        noise_scale:      Entropy-guided noise magnitude.
        cost_per_step:    Cost charged per segment.
        initial_carry:    If None, H_init / L_init are used.

    Returns:
        The highest-value branch on the frontier when the budget is exhausted.
    """
    frontier = Frontier(max_size=max_frontier)

    # Seed the frontier with a single root branch.
    root_carry = initial_carry if initial_carry is not None else _initial_carry(trm, batch)
    root_carry, *_, logits = trm.proposal_step(root_carry, batch)
    conf = _confidence(logits)
    root = Branch(
        batch=batch,
        output=logits.argmax(dim=-1),
        carry=root_carry.clone(),
        value=conf,
        logp=conf,
        steps=1,
        cost=cost_per_step,
    )
    frontier.add(root)
    segments_used = 1

    while segments_used < budget_segments:
        best = frontier.best()
        assert best is not None

        # --- Branch step (every branch_every segments) ---
        if segments_used % branch_every == 0:
            *_, logits_for_entropy = trm.proposal_step(best.carry, batch)
            children_carries = branch_carry(
                carry=best.carry,
                logits=logits_for_entropy,
                m=branch_m,
                noise_scale=noise_scale,
                puzzle_emb_len=trm.puzzle_emb_len,
            )
            for child_carry in children_carries:
                if segments_used >= budget_segments:
                    break
                new_carry, *_, logits = trm.proposal_step(child_carry, batch)
                conf = _confidence(logits)
                frontier.add(Branch(
                    batch=batch,
                    output=logits.argmax(dim=-1),
                    carry=new_carry.clone(),
                    value=conf,
                    logp=conf,
                    steps=best.steps + 1,
                    cost=best.cost + cost_per_step,
                ))
                segments_used += 1

            frontier.prune()
            continue

        # --- Expand best branch one step ---
        new_carry, *_, logits = trm.proposal_step(best.carry, batch)
        conf = _confidence(logits)
        expanded = Branch(
            batch=batch,
            output=logits.argmax(dim=-1),
            carry=new_carry.clone(),
            value=conf,
            logp=conf,
            steps=best.steps + 1,
            cost=best.cost + cost_per_step,
        )
        frontier.add(expanded)

        # Advance best in-place so the next iteration steps forward from here.
        best.carry = new_carry.clone()
        best.output = expanded.output
        best.value = conf
        best.logp = conf
        best.steps = expanded.steps
        best.cost = expanded.cost

        segments_used += 1
        frontier.prune()

    result = frontier.best()
    assert result is not None
    return result
