"""
Test greedy_search and frontier_search on a real ARC puzzle.

Usage:
    python test_arc_puzzle.py --checkpoint weights/trm_arc_v1/arc_v1_public/step_518071
    python test_arc_puzzle.py --checkpoint weights/trm_arc_v1/arc_v1_public/step_518071 --puzzle-id 007bbfb7
    python test_arc_puzzle.py --checkpoint weights/trm_arc_v1/arc_v1_public/step_518071 --steps 8
"""

import argparse
import json
import numpy as np
import torch

from load_trm import load_trm, load_identifier_map
from adaptive_search import greedy_search, frontier_search

# ARC encoding constants (from build_arc_dataset.py)
PAD = 0
EOS = 1
COLOR_OFFSET = 2        # color c → token c+2
MAX_GRID = 30
SEQ_LEN = MAX_GRID * MAX_GRID  # 900

# Simple color display map for terminal output
COLOR_CHARS = ".123456789"  # 0 → '.', 1-9 → '1'-'9'


# ---------------------------------------------------------------------------
# Encoding / decoding
# ---------------------------------------------------------------------------

def encode_grid(grid: list[list[int]]) -> np.ndarray:
    """Encode an ARC grid to a flat 900-token sequence (no translation offset)."""
    arr = np.array(grid, dtype=np.int32)
    nrow, ncol = arr.shape
    assert nrow <= MAX_GRID and ncol <= MAX_GRID

    canvas = np.zeros((MAX_GRID, MAX_GRID), dtype=np.int32)
    canvas[:nrow, :ncol] = arr + COLOR_OFFSET  # shift colors

    # EOS markers: row after grid, column after grid
    if nrow < MAX_GRID:
        canvas[nrow, :ncol] = EOS
    if ncol < MAX_GRID:
        canvas[:nrow, ncol] = EOS

    return canvas.flatten()


def decode_output(tokens: np.ndarray, expected_shape: tuple[int, int]) -> np.ndarray:
    """Decode a flat 900-token output to an ARC grid of expected_shape."""
    canvas = tokens.reshape(MAX_GRID, MAX_GRID)
    nrow, ncol = expected_shape
    region = canvas[:nrow, :ncol]
    # Clamp: values should be COLOR_OFFSET..COLOR_OFFSET+9; anything else → 0
    decoded = np.clip(region - COLOR_OFFSET, 0, 9).astype(np.int32)
    return decoded


def grid_to_str(grid: np.ndarray) -> str:
    rows = []
    for row in grid:
        rows.append(" ".join(COLOR_CHARS[min(c, 9)] for c in row))
    return "\n".join(rows)


def accuracy(pred: np.ndarray, gold: np.ndarray) -> float:
    return float((pred == gold).all())


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--puzzle-id", default="007bbfb7",
                        help="ARC puzzle ID")
    parser.add_argument("--split", default="training",
                        choices=["training", "evaluation", "evaluation2", "concept"],
                        help="Which challenge file to load from")
    parser.add_argument("--steps", type=int, default=6,
                        help="Number of greedy_search steps")
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()

    # ── Load model & identifier map ──────────────────────────────────────────
    trm, cfg = load_trm(args.checkpoint, device=args.device)
    id_map = load_identifier_map()

    # ── Load puzzle ──────────────────────────────────────────────────────────
    with open(f"kaggle/combined/arc-agi_{args.split}_challenges.json") as f:
        challenges = json.load(f)
    with open(f"kaggle/combined/arc-agi_{args.split}_solutions.json") as f:
        solutions = json.load(f)

    if args.puzzle_id not in challenges:
        raise ValueError(f"Puzzle '{args.puzzle_id}' not found. "
                         f"Available: {list(challenges.keys())[:5]} ...")

    puzzle = challenges[args.puzzle_id]
    test_input  = puzzle["test"][0]["input"]
    test_output = solutions[args.puzzle_id][0]  # ground truth

    inp_arr  = np.array(test_input,  dtype=np.int32)
    gold_arr = np.array(test_output, dtype=np.int32)

    print(f"\nPuzzle: {args.puzzle_id}")
    print(f"  Input  shape: {inp_arr.shape}")
    print(f"  Output shape: {gold_arr.shape}")

    print("\n── Input ────────────────────────")
    print(grid_to_str(inp_arr))
    print("\n── Expected output ──────────────")
    print(grid_to_str(gold_arr))

    # ── Build batch ──────────────────────────────────────────────────────────
    puzzle_id = id_map.get(args.puzzle_id, 0)
    if puzzle_id == 0:
        print(f"  WARNING: '{args.puzzle_id}' not in identifier map, using blank embedding (id=0)")
    else:
        print(f"  puzzle_id={puzzle_id}")

    tokens = torch.tensor(encode_grid(test_input), dtype=torch.int32, device=args.device)
    batch = {
        "inputs": tokens.unsqueeze(0),                                      # [1, 900]
        "puzzle_identifiers": torch.tensor([puzzle_id], dtype=torch.int32,
                                           device=args.device),
    }

    # ── greedy_search ────────────────────────────────────────────────────────
    print(f"\n── greedy_search ({args.steps} steps) ───────────────────────")
    result = greedy_search(trm.inner, batch, num_steps=args.steps)
    pred_tokens = result.output[0].cpu().numpy()
    pred_arr = decode_output(pred_tokens, gold_arr.shape)

    print(grid_to_str(pred_arr))
    print(f"\n  value={result.value:.4f}  steps={result.steps}  "
          f"exact_match={'YES' if accuracy(pred_arr, gold_arr) else 'NO'}")

    # ── frontier_search ──────────────────────────────────────────────────────
    print(f"\n── frontier_search (budget=12, frontier=4) ──────────────────")
    result_f = frontier_search(
        trm.inner, batch, budget_segments=12, max_frontier=4, branch_m=2
    )
    pred_f = decode_output(result_f.output[0].cpu().numpy(), gold_arr.shape)

    print(grid_to_str(pred_f))
    print(f"\n  value={result_f.value:.4f}  steps={result_f.steps}  "
          f"exact_match={'YES' if accuracy(pred_f, gold_arr) else 'NO'}")


if __name__ == "__main__":
    main()
