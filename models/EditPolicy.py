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

    def forward(self, tokens):
        if tokens.dim() == 1:
            tokens = tokens.unsqueeze(0)

        out = self.trm(tokens)
        h = out["hidden"]
        h_last = h[:, -1, :]

        action_logits = self.action_head(h_last)
        token_logits  = self.token_head(h_last)

        return {
            "action_logits": action_logits.squeeze(0),
            "token_logits": token_logits.squeeze(0),
            "hidden_states": h_last.squeeze(0),
        }



