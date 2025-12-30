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
        if tokens.dim() == 1:
            tokens = tokens.unsqueeze(0)

        # LanguageTRM ALWAYS returns a dict
        out = self.trm(tokens)
        h = out["hidden"]                 # (B, T, D)
        h_last = h[:, -1, :]              # (B, D)

        action_logits = self.action_head(h_last)
        token_logits  = self.token_head(h_last)

        result = {
            "action_logits": action_logits,
            "token_logits": token_logits,
        }

        if return_hidden:
            result["hidden_states"] = h_last

        # Unbatch for convenience
        if action_logits.shape[0] == 1:
            for k in result:
                result[k] = result[k].squeeze(0)

        return result


