"""
Smoke test — run from the project root:
    python test_smoke.py --checkpoint weights/trm_arc_v1/arc_v1_public/step_XXXXX/model.pt
"""

import argparse
import torch
from load_trm import load_trm
from adaptive_search import greedy_search, frontier_search

parser = argparse.ArgumentParser()
parser.add_argument("--checkpoint", required=True, help="Path to .pt checkpoint file")
parser.add_argument("--device", default="cpu")
args = parser.parse_args()

# ── 1. Load model ────────────────────────────────────────────────────────────
trm, cfg = load_trm(args.checkpoint, device=args.device)
print(f"\nConfig: seq_len={cfg['seq_len']}, vocab_size={cfg['vocab_size']}, "
      f"num_puzzle_identifiers={cfg['num_puzzle_identifiers']}\n")

# ── 2. Minimal batch (all-zeros input, puzzle id=1) ──────────────────────────
batch = {
    "inputs": torch.zeros(1, cfg["seq_len"], dtype=torch.int32, device=args.device),
    "puzzle_identifiers": torch.tensor([1], dtype=torch.int32, device=args.device),
}

# ── 3. greedy_search ─────────────────────────────────────────────────────────
print("Testing greedy_search (4 steps)...")
result = greedy_search(trm.inner, batch, num_steps=4)
print(f"  value={result.value:.4f}  output.shape={result.output.shape}  steps={result.steps}")
assert result.output.shape == (1, cfg["seq_len"]), "unexpected output shape"

# ── 4. frontier_search ───────────────────────────────────────────────────────
print("Testing frontier_search (budget=8, frontier=4)...")
result = frontier_search(trm.inner, batch, budget_segments=8, max_frontier=4, branch_m=2)
print(f"  value={result.value:.4f}  output.shape={result.output.shape}  steps={result.steps}")

print("\nAll smoke tests passed.")
