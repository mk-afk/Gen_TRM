from models.tinyReasoning import TinyReasoningCore
import torch
import torch.nn as nn

class TRM(nn.Module):
    """
    Recursive Token Reasoning Module
    """

    def __init__(self, d_model=128, num_steps=6):
        super().__init__()
        self.num_steps = num_steps
        self.core = TinyReasoningCore(d_model)
        self.norm = nn.LayerNorm(d_model)

    def forward(self, x):
        """
        x: (B, D) input embedding
        returns: final reasoning state h_T
        """
        h = torch.zeros_like(x)

        for _ in range(self.num_steps):
            h = self.core(h, x)
            h = self.norm(h)

        return h
