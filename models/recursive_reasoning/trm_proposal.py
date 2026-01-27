# models/recursive_reasoning/trm_proposal.py

from typing import Tuple
import torch
import torch.nn.functional as F
from torch import nn

from .trm import (
    TinyRecursiveReasoningModel_ACTV1Config,
    TinyRecursiveReasoningModel_ACTV1Inner,
    TinyRecursiveReasoningModel_ACTV1InnerCarry,
)


class TinyRecursiveReasoningModel_ProposalOnly(nn.Module):
    """
    Proposal-only Tiny Recursive Reasoning Model.

    This class:
    - Reuses the ACTV1 inner TRM exactly
    - Exposes a deterministic (carry, tokens) -> next-token logits API
    - Does NOT implement ACT, halting, or Q-learning
    - Is safe for adaptive search with learned Δ̂
    """

    def __init__(self, config_dict: dict):
        super().__init__()

        # Use the same config schema as ACTV1
        self.config = TinyRecursiveReasoningModel_ACTV1Config(**config_dict)

        # Reuse the exact inner model (LM + recursion)
        self.inner = TinyRecursiveReasoningModel_ACTV1Inner(self.config)

        # Safety: proposal-only model should never be in training mode
        self.eval()

    # ---------------------------------------------------------------------
    # Carry handling
    # ---------------------------------------------------------------------

    def initial_carry(self, batch_size: int = 1) -> TinyRecursiveReasoningModel_ACTV1InnerCarry:
        """
        Create an empty inner carry for proposal search.

        No halted / steps / ACT state.
        """
        return self.inner.empty_carry(batch_size)

    # ---------------------------------------------------------------------
    # Proposal step
    # ---------------------------------------------------------------------

    @torch.no_grad()
    def proposal_step(
        self,
        carry: TinyRecursiveReasoningModel_ACTV1InnerCarry,
        tokens: torch.Tensor,
        puzzle_identifiers: torch.Tensor,
    ) -> Tuple[TinyRecursiveReasoningModel_ACTV1InnerCarry, torch.Tensor]:
        """
        Perform ONE deterministic proposal step.

        Args:
            carry:
                TinyRecursiveReasoningModel_ACTV1InnerCarry
            tokens:
                LongTensor [B, T]
            puzzle_identifiers:
                LongTensor [B] (REQUIRED, since puzzle_emb_ndim > 0)

        Returns:
            new_carry:
                Updated inner carry (detached, safe to reuse)
            next_token_logits:
                FloatTensor [B, vocab_size]
        """

        # Sanity checks (fail fast, helpful for debugging)
        assert tokens.dim() == 2, "tokens must be [B, T]"
        assert puzzle_identifiers.dim() == 1, "puzzle_identifiers must be [B]"
        assert tokens.shape[0] == puzzle_identifiers.shape[0], "batch size mismatch"

        batch = {
            "inputs": tokens,
            "puzzle_identifiers": puzzle_identifiers,
        }

        # Call the shared inner TRM
        new_carry, logits, _ = self.inner(carry, batch)

        # logits: [B, T, vocab]
        next_token_logits = logits[:, -1]

        return new_carry, next_token_logits

    # ---------------------------------------------------------------------
    # Convenience helper (optional)
    # ---------------------------------------------------------------------

    @torch.no_grad()
    def next_token_logprobs(
        self,
        carry: TinyRecursiveReasoningModel_ACTV1InnerCarry,
        tokens: torch.Tensor,
        puzzle_identifiers: torch.Tensor,
    ):
