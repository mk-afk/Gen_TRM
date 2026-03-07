import torch
from typing import Dict

def make_trm_batch(
    tokens_1d: torch.LongTensor,
    seq_len: int,
    pad_id: int,
    puzzle_identifier: int,
    device: torch.device,
) -> Dict[str, torch.Tensor]:
    """
    tokens_1d: [T] on any device
    returns batch dict with B=1, inputs [1, seq_len], puzzle_identifiers [1]
    """
    tokens_1d = tokens_1d.to(device)
    if tokens_1d.numel() >= seq_len:
        window = tokens_1d[-seq_len:]
    else:
        pad = torch.full((seq_len - tokens_1d.numel(),), pad_id, dtype=tokens_1d.dtype, device=device)
        window = torch.cat([pad, tokens_1d], dim=0)

    batch = {
        "inputs": window.view(1, seq_len).to(torch.int32),
        "puzzle_identifiers": torch.tensor([puzzle_identifier], device=device, dtype=torch.int32),
    }
    return batch
