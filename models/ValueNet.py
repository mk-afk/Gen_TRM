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
        if hidden_states.dim() == 3:
            h = hidden_states[:, -1, :]
        elif hidden_states.dim() == 2:
            h = hidden_states[-1]
        else:
            h = hidden_states
        return self.value_head(h).squeeze(-1)
