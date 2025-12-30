# models/LanguageTRM.py
from models.TRM import TRM
import torch.nn as nn


class LanguageTRM(nn.Module):
    def __init__(self, vocab_size, d_model=128, steps=6):
        super().__init__()
        self.vocab_size = vocab_size
        self.d_model = d_model

        self.embed = nn.Embedding(vocab_size, d_model)
        self.trm = TRM(d_model, steps)
        self.lm_head = nn.Linear(d_model, vocab_size)

    def forward(self, tokens):
        """
        tokens: (B, T) or (T,)
        returns dict with:
          - logits: (B, T, V)
          - hidden: (B, T, D)
        """
        if tokens.dim() == 1:
            tokens = tokens.unsqueeze(0)

        x = self.embed(tokens)        # (B, T, D)
        h = self.trm(x)               # (B, T, D)
        logits = self.lm_head(h)      # (B, T, V)

        return {
            "logits": logits,
            "hidden": h,
        }
