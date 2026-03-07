# adaptive_search/config.py
from dataclasses import dataclass
from typing import Callable, Optional

@dataclass
class SearchConfig:
    segment_len_N: int = 32
    max_frontier: int = 16
    branch_m: int = 3

    cost_expand: float = 1.0
    # if BRANCH generates m segments (one per child):
    def cost_branch(self) -> float:
        return float(self.branch_m)

    gamma: float = 0.99
    lambda_cost: float = 0.1

    max_steps_failsafe: int = 10_000  # guardrail, not “budget”

# optional: type hints for required TRM API can go here too
