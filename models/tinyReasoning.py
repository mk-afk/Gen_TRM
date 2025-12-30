import torch
import torch.nn as nn

class TinyReasoningCore(nn.Module):
    """
    f_theta in the paper
    Very small MLP, shared across steps
    """

    def __init__(self, d_model=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(2 * d_model, d_model),
            nn.GELU(),
            nn.Linear(d_model, d_model),
        )

    def forward(self, h, x):
        """
        h: (B, T, D)
        x: (B, T, D)
        """

        return self.net(torch.cat([h, x], dim=-1))
