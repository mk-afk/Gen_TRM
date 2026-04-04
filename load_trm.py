"""
Utility to load a pretrained TRM checkpoint from:
    huggingface.co/arcprize/trm_arc_prize_verification

Usage
-----
    from load_trm import load_trm
    trm, config_dict = load_trm("weights/trm_arc_v1/arc_v1_public/step_<N>/model.pt")
    trm.eval()

Then pass trm.inner to greedy_search / frontier_search / train().
"""

import json
import torch
from models.recursive_reasoning.trm import TinyRecursiveReasoningModel_ACTV1

IDENTIFIERS_PATH = "data/arc1concept-aug-1000/identifiers.json"

def load_identifier_map(path: str = IDENTIFIERS_PATH) -> dict[str, int]:
    """Load puzzle_name → puzzle_id mapping from the built dataset.

    The JSON is a list where index = puzzle_id and value = puzzle name
    (including augmentation suffix for augmented variants).
    Returns only the base names (no augmentation suffix) mapped to their first ID.
    """
    with open(path) as f:
        id_to_name: list[str] = json.load(f)

    name_to_id: dict[str, int] = {}
    for idx, name in enumerate(id_to_name):
        # Base name is the part before the augmentation separator '|||'
        base = name.split("|||")[0]
        if base not in name_to_id:
            name_to_id[base] = idx
    return name_to_id


# Fixed values for the ARC checkpoint (from build_arc_dataset.py + trm.yaml).
_ARC_BASE_CONFIG = dict(
    # Architecture (trm.yaml)
    H_cycles=3,
    L_cycles=4,
    H_layers=0,
    L_layers=2,
    hidden_size=512,
    num_heads=8,
    expansion=4,
    puzzle_emb_ndim=512,
    puzzle_emb_len=16,
    pos_encodings="rope",
    forward_dtype="bfloat16",
    mlp_t=False,
    no_ACT_continue=True,
    halt_max_steps=16,
    halt_exploration_prob=0.1,
    # Data (build_arc_dataset.py)
    seq_len=900,        # 30 * 30
    vocab_size=12,      # PAD + EOS + 10 colors
    batch_size=1,       # 1 for inference
)


def load_trm(
    checkpoint_path: str,
    device: str = "cpu",
) -> tuple[TinyRecursiveReasoningModel_ACTV1, dict]:
    """Load a pretrained TRM from a checkpoint file.

    The checkpoint was saved with torch.compile wrapping a loss head, so keys
    look like '_orig_mod.model.inner.*'.  This function strips those prefixes
    so the weights load cleanly into TinyRecursiveReasoningModel_ACTV1.

    Args:
        checkpoint_path: Path to the .pt checkpoint file.
        device:          Device to load onto ('cpu', 'cuda', etc.)

    Returns:
        (model, config_dict) — model is in eval mode with frozen weights.
    """
    print(f"Loading checkpoint: {checkpoint_path}")
    raw_sd = torch.load(checkpoint_path, map_location=device)

    # Strip torch.compile prefix (_orig_mod.) and loss-head prefix (model.).
    # Resulting keys should match TinyRecursiveReasoningModel_ACTV1.
    strip = "_orig_mod.model."
    sd = {}
    for k, v in raw_sd.items():
        if k.startswith(strip):
            sd[k[len(strip):]] = v
        elif k.startswith("_orig_mod."):
            sd[k[len("_orig_mod."):]] = v
        else:
            sd[k] = v

    # Infer num_puzzle_identifiers from the puzzle embedding weight shape.
    emb_key = "inner.puzzle_emb.weights"
    if emb_key not in sd:
        raise KeyError(
            f"Expected key '{emb_key}' in checkpoint after stripping prefixes. "
            f"Found keys: {list(sd.keys())[:10]} ..."
        )
    num_puzzle_identifiers = sd[emb_key].shape[0]
    print(f"Inferred num_puzzle_identifiers: {num_puzzle_identifiers}")

    config_dict = {**_ARC_BASE_CONFIG, "num_puzzle_identifiers": num_puzzle_identifiers}

    model = TinyRecursiveReasoningModel_ACTV1(config_dict)
    missing, unexpected = model.load_state_dict(sd, strict=False, assign=True)

    if missing:
        print(f"Missing keys ({len(missing)}): {missing[:5]}{'...' if len(missing) > 5 else ''}")
    if unexpected:
        print(f"Unexpected keys ({len(unexpected)}): {unexpected[:5]}{'...' if len(unexpected) > 5 else ''}")

    model = model.to(device).eval()
    for p in model.parameters():
        p.requires_grad_(False)

    print("Checkpoint loaded successfully.")
    return model, config_dict
