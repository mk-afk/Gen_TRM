# adaptive_search/delta_net.py

import torch
from torch import nn


class DeltaNet(nn.Module):
    """
    Predicts Δ̂(s): expected marginal value of additional search.

    Input:
        x: FloatTensor [B, D]  (from delta_features)

    Output:
        delta_hat: FloatTensor [B]
    """

    def __init__(
        self,
        input_dim: int = 3,
        hidden_dim: int = 64,
    ):
        super().__init__()

        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

        # Initialize last layer near zero so Δ̂ starts conservative
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: FloatTensor [B, input_dim]

        Returns:
            delta_hat: FloatTensor [B]
        """
        assert x.dim() == 2, "DeltaNet input must be [B, D]"
        return self.net(x).squeeze(-1)
