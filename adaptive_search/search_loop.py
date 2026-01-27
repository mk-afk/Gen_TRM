# adaptive_search/search_loop.py

from typing import List, Optional
import torch
import torch.nn.functional as F

from .branch import Branch
from .delta_features import delta_features
from .delta_net import DeltaNet


@torch.no_grad()
def adaptive_search(
    *,
    trm,                          # TinyRecursiveReasoningModel_ProposalOnly
    delta_net: DeltaNet,
    initial_tokens: torch.Tensor, # [T0]
    puzzle_identifier: torch.Tensor,  # scalar LongTensor
    max_len: int,
    compute_cost: float,
    branch_k: int = 2,
    device: Optional[torch.device] = None,
) -> Branch:
    """
    Δ̂-controlled adaptive generative search.

    Args:
        trm:
            Proposal-only TRM
        delta_net:
            Learned Δ̂ network
        initial_tokens:
            LongTensor [T0]
        puzzle_identifier:
            LongTensor scalar (global, constant)
        max_len:
            Maximum allowed sequence length
        compute_cost:
            Scalar threshold for STOP
        branch_k:
            Fixed branching factor (top-k)
        device:
            Optional torch.device

    Returns:
        Best Branch found
    """

    if device is None:
        device = initial_tokens.device

    # ------------------------------------------------------------------
    # Initialize
    # ------------------------------------------------------------------

    branches: List[Branch] = [
        Branch(
            tokens=initial_tokens.clone(),
            score=0.0,
            length=initial_tokens.shape[0],
        )
    ]

    carry = trm.initial_carry(batch_size=1)

    puzzle_id = puzzle_identifier.to(device).view(1)

    # ------------------------------------------------------------------
    # Main loop
    # ------------------------------------------------------------------

    while True:
        # ---- STOP if all branches are at max length ----
        if all(b.length >= max_len for b in branches):
            break

        # ---- Compute Δ̂ for all branches ----
        feats = delta_features(branches, max_len, device=device)
        delta_vals = delta_net(feats)  # [B]

        delta_max, best_idx = delta_vals.max(dim=0)

        # ---- Global STOP condition ----
        if delta_max.item() <= compute_cost:
            break

        # ---- Select incumbent branch ----
        incumbent = branches[best_idx.item()]

        # ---- If incumbent cannot advance, force STOP ----
        if incumbent.length >= max_len:
            break

        # ---- Proposal step (ONE TRM CALL) ----
        carry, logits = trm.proposal_step(
            carry,
            tokens=incumbent.tokens.unsqueeze(0),
            puzzle_identifiers=puzzle_id,
        )

        log_probs = F.log_softmax(logits[0], dim=-1)

        # ---- Optional branching ----
        if branch_k > 1:
            topk_logp, topk_tok = torch.topk(log_probs, k=branch_k)

            # Spawn new branches (excluding top-1)
            for tok, lp in zip(topk_tok[1:], topk_logp[1:]):
                new_tokens = torch.cat(
                    [incumbent.tokens, tok.view(1)], dim=0
                )

                branches.append(
                    Branch(
                        tokens=new_tokens,
                        score=incumbent.score + lp.item(),
                        length=incumbent.length + 1,
                    )
                )
        else:
            # Only greedy
            topk_logp, topk_tok = torch.topk(log_probs, k=1)

        # ---- Advance ONLY the incumbent ----
        next_tok = topk_tok[0]
        next_lp = topk_logp[0]

        incumbent.tokens = torch.cat(
            [incumbent.tokens, next_tok.view(1)], dim=0
        )
        incumbent.score += next_lp.item()
        incumbent.length += 1

        # ---- Deduplicate branches (keep best score per token sequence) ----
        dedup = {}
        for b in branches:
            k = b.key()
            if k not in dedup or b.score > dedup[k].score:
                dedup[k] = b
        branches = list(dedup.values())

    # ------------------------------------------------------------------
    # Return best branch
    # ------------------------------------------------------------------

    return max(branches, key=lambda b: b.score)
