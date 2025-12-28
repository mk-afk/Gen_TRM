import torch
from torch.utils.data import Dataset
from envs.actions import EditAction


class ExpertDataset(Dataset):
    def __init__(self, token_stream, context_length=64):
        """
        token_stream: 1D tensor of token ids (e.g. train_ids)
        """
        self.tokens = token_stream.cpu()
        self.context_length = context_length

    def __len__(self):
        return len(self.tokens) - self.context_length - 1

    def __getitem__(self, idx):
        # Context window
        context = self.tokens[idx : idx + self.context_length]

        # Expert action: ADD next token
        next_token = self.tokens[idx + self.context_length]

        action_target = torch.tensor(
            EditAction.ADD, dtype=torch.long
        )

        return context, action_target, next_token
