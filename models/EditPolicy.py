import torch.nn as nn

class EditPolicy(nn.Module):
    def __init__(self, trm_model, vocab_size, num_actions=4):
        super().__init__()
        self.trm = trm_model
        self.action_head = nn.Linear(trm_model.d_model, num_actions)
        self.token_head  = nn.Linear(trm_model.d_model, vocab_size)

    def forward(self, tokens):
        h = self.trm(tokens, return_hidden=True)
        h_last = h[-1]

        action_logits = self.action_head(h_last)
        token_logits  = self.token_head(h_last)
        return action_logits, token_logits
