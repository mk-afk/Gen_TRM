# models/ValueNet.py
import torch
import torch.nn as nn


class ValueNet(nn.Module):
    def __init__(self, d_model):
        super().__init__()

        self.value_head = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.Tanh(),
            nn.Linear(d_model, 1),
        )

    def forward(self, hidden_states):
        """
        hidden_states: (T, D) or (B, T, D)
        returns: (T,) or (B, T)
        """
        if hidden_states.dim() == 3:
            # use last token (pre-action state)
            h = hidden_states[:, -1, :]
        else:
            h = hidden_states[-1]

        return self.value_head(h).squeeze(-1)
