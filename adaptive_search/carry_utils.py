from typing import Dict
import torch

from models.recursive_reasoning.trm import (
    TinyRecursiveReasoningModel_ACTV1Carry,
    TinyRecursiveReasoningModel_ACTV1InnerCarry,
)

def copy_actv1_carry(c: TinyRecursiveReasoningModel_ACTV1Carry) -> TinyRecursiveReasoningModel_ACTV1Carry:
    # Clone all tensors so branches don't alias memory.
    inner = TinyRecursiveReasoningModel_ACTV1InnerCarry(
        z_H=c.inner_carry.z_H.clone(),
        z_L=c.inner_carry.z_L.clone(),
    )

    current_data: Dict[str, torch.Tensor] = {k: v.clone() for k, v in c.current_data.items()}

    return TinyRecursiveReasoningModel_ACTV1Carry(
        inner_carry=inner,
        steps=c.steps.clone(),
        halted=c.halted.clone(),
        current_data=current_data,
    )
