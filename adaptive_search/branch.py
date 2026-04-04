from __future__ import annotations
from dataclasses import dataclass, field
from typing import Any, Dict, Optional, Tuple, Callable
import torch

CarryCopyFn = Optional[Callable[[Any], Any]]

def default_clone_carry(carry: Any) -> Any:
    """Safe-ish generic carry cloning for common structures."""
    if carry is None:
        return None
    if torch.is_tensor(carry):
        return carry.detach().clone()
    if isinstance(carry, (tuple, list)):
        return type(carry)(default_clone_carry(x) for x in carry)
    if isinstance(carry, dict):
        return {k: default_clone_carry(v) for k, v in carry.items()}
    if hasattr(carry, "clone"):
        try:
            return carry.clone()
        except TypeError:
            pass
    # Fallback: last resort shallow copy (warn-worthy)
    return carry

@dataclass
class Branch:
    batch: Dict[str, torch.Tensor]   # shared input (same for all branches on one problem)
    output: torch.LongTensor          # argmax(lm_head(z_H)), the current decoded answer
    carry: Any                        # TRM carry object (per-branch!)
    value: float = float("-inf")      # Φ / verifier score

    logp: float = 0.0

    steps: int = 0
    cost: float = 0.0

    meta: Dict[str, Any] = field(default_factory=dict)

    def clone(self, carry_copy_fn: CarryCopyFn = None, strict: bool = True) -> "Branch":
        output_copy = self.output.detach().clone()

        if carry_copy_fn is not None:
            carry_copy = carry_copy_fn(self.carry)
        else:
            if strict:
                # Force you to think about carry copying.
                # Flip strict=False if you really want generic cloning.
                raise ValueError("Branch.clone called without carry_copy_fn; this can alias carry.")
            carry_copy = default_clone_carry(self.carry)

        return Branch(
            batch=self.batch,
            output=output_copy,
            carry=carry_copy,
            value=float(self.value),
            logp=float(self.logp),
            steps=int(self.steps),
            cost=float(self.cost),
            meta=dict(self.meta),
        )

    def key(self) -> bytes:
        """Content-based dedup key over the full decoded output grid."""
        return self.output.detach().to("cpu").contiguous().numpy().tobytes()