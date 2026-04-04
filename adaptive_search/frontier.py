"""Frontier: a bounded, deduplicated priority queue of Branch objects.

Branches are keyed by their decoded output (Branch.key()), so two branches
that have converged to the same answer are treated as a single entry — the
one with the higher value is kept.
"""

from __future__ import annotations
from typing import Iterator, List, Optional

from .branch import Branch


class Frontier:
    """Bounded, deduplicated set of branches sorted by value (descending).

    Args:
        max_size: Hard cap on the number of branches kept after each prune()
                  call.  Recommended range: 8–32.
    """

    def __init__(self, max_size: int) -> None:
        self.max_size = max_size
        self._branches: List[Branch] = []
        self._key_to_idx: dict[bytes, int] = {}

    # ------------------------------------------------------------------
    # Mutation
    # ------------------------------------------------------------------

    def add(self, branch: Branch) -> bool:
        """Attempt to add a branch.

        If a branch with the same output key already exists and the new one
        has a higher value, it replaces the old entry.  Otherwise the add is
        rejected.

        Returns True if the branch was inserted or replaced, False otherwise.
        """
        k = branch.key()
        if k in self._key_to_idx:
            idx = self._key_to_idx[k]
            if branch.value > self._branches[idx].value:
                self._branches[idx] = branch
                return True
            return False

        self._key_to_idx[k] = len(self._branches)
        self._branches.append(branch)
        return True

    def prune(self) -> None:
        """Trim to max_size, keeping the highest-value branches."""
        if len(self._branches) <= self.max_size:
            return

        self._branches.sort(key=lambda b: b.value, reverse=True)
        removed = self._branches[self.max_size:]
        self._branches = self._branches[: self.max_size]

        # Rebuild index after sort + truncation.
        self._key_to_idx = {b.key(): i for i, b in enumerate(self._branches)}

        # Free GPU memory of evicted carries eagerly.
        for b in removed:
            del b.carry

    # ------------------------------------------------------------------
    # Queries
    # ------------------------------------------------------------------

    def best(self) -> Optional[Branch]:
        """Return the highest-value branch without removing it."""
        if not self._branches:
            return None
        return max(self._branches, key=lambda b: b.value)

    def pop_best(self) -> Optional[Branch]:
        """Remove and return the highest-value branch."""
        if not self._branches:
            return None
        idx = max(range(len(self._branches)), key=lambda i: self._branches[i].value)
        branch = self._branches.pop(idx)
        self._key_to_idx = {b.key(): i for i, b in enumerate(self._branches)}
        return branch

    def __len__(self) -> int:
        return len(self._branches)

    def __iter__(self) -> Iterator[Branch]:
        return iter(self._branches)

    def __bool__(self) -> bool:
        return bool(self._branches)
