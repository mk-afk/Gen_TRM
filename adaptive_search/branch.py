from dataclasses import dataclass
import torch

@dataclass
class Branch:
    tokens: torch.Tensor
    score: float
    length: int

    def key(self):
        return tuple(self.tokens.tolist())
