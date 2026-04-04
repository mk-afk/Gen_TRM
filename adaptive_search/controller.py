"""
Search controller: learned Q-network that manages the frontier.

Architecture
------------
Input per branch:
    get_features(carry) = [B, 3*D]  (puzzle_feat | y_pooled | z_pooled)

The Q-network is called with:
    branch_feats  [N, 3D]  — one row per branch in the frontier
    best_feats    [N, 3D]  — best-branch features broadcast to every row
    budget        [N]      — remaining segment budget broadcast to every row

This makes the per-branch input [6D+1], shared trunk → EXPAND and BRANCH heads.
STOP is a separate head that reads only [best_feats | budget] → scalar per item.

Action selection
----------------
Compute advantage  A(s, a) = Q(s, a) – Q(s, STOP).
Pick (branch, action) with the highest positive advantage.
Return STOP when no action beats the baseline.
"""

from __future__ import annotations
from enum import IntEnum
from typing import List, Tuple

import torch
import torch.nn as nn

from .branch import Branch


class Action(IntEnum):
    STOP = 0
    EXPAND = 1
    BRANCH = 2


class QNetwork(nn.Module):
    """Vectorised Q-network over the frontier.

    All three inputs are fully expanded before entering the network so a
    single matrix-multiply covers the whole frontier (or training batch).

    Args:
        feature_dim: Dimension of get_features output (= 3 * trm_hidden_size).
        hidden_dim:  Width of hidden layers.
    """

    def __init__(self, feature_dim: int, hidden_dim: int = 256) -> None:
        super().__init__()
        self.feature_dim = feature_dim

        # Shared trunk: [branch | best_ref | budget] → hidden
        self.trunk = nn.Sequential(
            nn.Linear(feature_dim * 2 + 1, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )
        self.expand_head = nn.Linear(hidden_dim, 1)
        self.branch_head = nn.Linear(hidden_dim, 1)

        # STOP head: [best_feats | budget] → scalar per item
        self.stop_net = nn.Sequential(
            nn.Linear(feature_dim + 1, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

        # Conservative init: Q-values start near zero so the controller
        # doesn't branch wildly before any learning signal arrives.
        for head in (self.expand_head, self.branch_head, self.stop_net[-1]):
            nn.init.zeros_(head.weight)
            nn.init.zeros_(head.bias)

    def forward(
        self,
        branch_feats: torch.Tensor,  # [N, feature_dim]
        best_feats: torch.Tensor,    # [N, feature_dim]  — pre-expanded
        budget: torch.Tensor,        # [N]
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Returns:
            q_expand  [N]  — Q-value for EXPAND on each branch
            q_branch  [N]  — Q-value for BRANCH on each branch
            q_stop    [N]  — Q-value for STOP (from best-branch features)
        """
        x = torch.cat([branch_feats, best_feats, budget.unsqueeze(-1)], dim=-1)  # [N, 2D+1]
        h = self.trunk(x)
        q_expand = self.expand_head(h).squeeze(-1)  # [N]
        q_branch = self.branch_head(h).squeeze(-1)  # [N]

        stop_in = torch.cat([best_feats, budget.unsqueeze(-1)], dim=-1)  # [N, D+1]
        q_stop = self.stop_net(stop_in).squeeze(-1)                      # [N]

        return q_expand, q_branch, q_stop


class SearchController(nn.Module):
    """Wraps QNetwork and exposes action selection for the search loop.

    Args:
        trm_hidden_size: TRM hidden_size D.  feature_dim = 3*D.
        hidden_dim:      Width of hidden layers in the Q-network.
    """

    def __init__(self, trm_hidden_size: int, hidden_dim: int = 256) -> None:
        super().__init__()
        self.feature_dim = 3 * trm_hidden_size
        self.q_net = QNetwork(self.feature_dim, hidden_dim)

    # ------------------------------------------------------------------
    # Feature extraction
    # ------------------------------------------------------------------

    @torch.no_grad()
    def get_branch_features(self, branches: List[Branch], trm_inner) -> torch.Tensor:
        """Extract get_features for every branch carry.

        Assumes each branch carry has batch size B=1 (single problem).
        Returns [N, feature_dim].
        """
        return torch.cat([trm_inner.get_features(b.carry) for b in branches], dim=0)

    # ------------------------------------------------------------------
    # Action selection
    # ------------------------------------------------------------------

    def select_action(
        self,
        branches: List[Branch],
        branch_feats: torch.Tensor,  # [N, feature_dim]  precomputed
        budget_remaining: float,
        epsilon: float = 0.0,
    ) -> Tuple[Action, int]:
        """Select the best (action, branch_index) pair.

        Returns (Action.STOP, -1) when no action beats the STOP baseline.

        Args:
            branches:         Current frontier branches.
            branch_feats:     Precomputed features for each branch [N, feature_dim].
            budget_remaining: Remaining compute budget in segments.
            epsilon:          Epsilon-greedy exploration probability.
        """
        if not branches:
            return Action.STOP, -1

        N = len(branches)
        device = branch_feats.device

        # Epsilon-greedy: uniformly pick EXPAND or BRANCH on a random branch.
        if epsilon > 0 and torch.rand(1).item() < epsilon:
            rand_action = Action(torch.randint(1, 3, (1,)).item())
            rand_branch = torch.randint(N, (1,)).item()
            return rand_action, rand_branch

        best_idx = max(range(N), key=lambda i: branches[i].value)
        best_ref = branch_feats[best_idx].unsqueeze(0).expand(N, -1)       # [N, D]
        budget_t = torch.full((N,), budget_remaining, device=device)       # [N]

        with torch.no_grad():
            q_expand, q_branch, q_stop = self.q_net(branch_feats, best_ref, budget_t)

        # Counterfactual STOP baseline: A(s, a) = Q(s, a) – Q(s, STOP)
        # q_stop is [N] but all values come from best_feats so they're identical;
        # use the mean to get a scalar baseline.
        stop_val = q_stop.mean()
        a_expand = q_expand - stop_val
        a_branch = q_branch - stop_val

        best_expand_val, best_expand_idx = a_expand.max(dim=0)
        best_branch_val, best_branch_idx = a_branch.max(dim=0)

        best_adv = max(best_expand_val.item(), best_branch_val.item())
        if best_adv <= 0:
            return Action.STOP, -1

        if best_expand_val.item() >= best_branch_val.item():
            return Action.EXPAND, best_expand_idx.item()
        else:
            return Action.BRANCH, best_branch_idx.item()
