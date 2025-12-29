# models/EditPolicy.py
import torch
import torch.nn as nn


class EditPolicy(nn.Module):
    def __init__(self, trm_model, vocab_size, num_actions):
        super().__init__()
        self.trm = trm_model
        self.d_model = trm_model.d_model

        self.action_head = nn.Linear(self.d_model, num_actions)
        self.token_head  = nn.Linear(self.d_model, vocab_size)
        self.value_head  = nn.Linear(self.d_model, 1)   # <-- NEW

    def forward(self, tokens):
        """
        tokens:
          (T,) or (B, T)
        returns:
          action_logits, token_logits, value
        """

        is_batched = tokens.dim() == 2
        if not is_batched:
            tokens = tokens.unsqueeze(0)  # (1, T)

        # TRM returns (B, T, D)
        h = self.trm(tokens, return_hidden=True)
        h_last = h[:, -1, :]              # (B, D)

        action_logits = self.action_head(h_last)
        token_logits  = self.token_head(h_last)
        value         = self.value_head(h_last).squeeze(-1)  # (B,)

        if not is_batched:
            action_logits = action_logits.squeeze(0)
            token_logits  = token_logits.squeeze(0)
            value         = value.squeeze(0)

        return action_logits, token_logits, value
