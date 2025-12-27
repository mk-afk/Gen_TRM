import torch.nn as nn


class EditPolicy(nn.Module):
    def __init__(self, trm, vocab_size, num_actions):
        super().__init__()
        self.trm = trm
        self.action_head = nn.Linear(trm.d_model, num_actions)
        self.token_head  = nn.Linear(trm.d_model, vocab_size)

    def forward(self, tokens):
        h = self.trm(tokens)          # (T, D)
        h_last = h[-1]                # cursor-focused
        action_logits = self.action_head(h_last)
        token_logits  = self.token_head(h_last)
        return action_logits, token_logits
