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

    def forward(self, tokens, return_hidden=False):
        """
        tokens:
          (T,) or (B, T)

        returns:
          dict with:
            action_logits
            token_logits
            hidden_states (optional)
        """

        is_batched = tokens.dim() == 2
        if not is_batched:
            tokens = tokens.unsqueeze(0)  # (1, T)

        # TRM returns (B, T, D)
        h = self.trm(tokens, return_hidden=True)
        h_last = h[:, -1, :]              # (B, D)

        action_logits = self.action_head(h_last)
        token_logits  = self.token_head(h_last)

        out = {
            "action_logits": action_logits,
            "token_logits": token_logits,
        }

        if return_hidden:
            out["hidden_states"] = h_last

        if not is_batched:
            for k in out:
                out[k] = out[k].squeeze(0)

        return out
