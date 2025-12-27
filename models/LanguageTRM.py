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
        tokens: (T,) token window
        """
        x = self.embed(tokens)      # (T, D)
        h = self.trm(x)             # (T, D)
        return self.lm_head(h)      # (T, vocab)
