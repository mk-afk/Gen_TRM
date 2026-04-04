"""
Run greedy_search across all evaluation puzzles and report accuracy.

Usage:
    python test_batch_eval.py --checkpoint weights/trm_arc_v1/arc_v1_public/step_518071
"""

import argparse
import json
import numpy as np
import torch

from load_trm import load_trm, load_identifier_map
from adaptive_search import greedy_search
from test_arc_puzzle import encode_grid, decode_output

parser = argparse.ArgumentParser()
parser.add_argument("--checkpoint", required=True)
parser.add_argument("--split", default="evaluation",
                    choices=["training", "evaluation", "evaluation2", "concept"])
parser.add_argument("--steps", type=int, default=6)
parser.add_argument("--device", default="cpu")
parser.add_argument("--limit", type=int, default=50, help="Max puzzles to test")
args = parser.parse_args()

trm, cfg = load_trm(args.checkpoint, device=args.device)
id_map = load_identifier_map()

with open(f"kaggle/combined/arc-agi_{args.split}_challenges.json") as f:
    challenges = json.load(f)
with open(f"kaggle/combined/arc-agi_{args.split}_solutions.json") as f:
    solutions = json.load(f)

correct = 0
total = 0
failures = []

for puzzle_id in list(dict.fromkeys(challenges))[:args.limit]:
    puzzle = challenges[puzzle_id]
    test_input  = puzzle["test"][0]["input"]
    test_output = solutions[puzzle_id][0]
    gold = np.array(test_output, dtype=np.int32)

    puzzle_num_id = id_map.get(puzzle_id, 0)
    tokens = torch.tensor(encode_grid(test_input), dtype=torch.int32, device=args.device)
    batch = {
        "inputs": tokens.unsqueeze(0),
        "puzzle_identifiers": torch.tensor([puzzle_num_id], dtype=torch.int32, device=args.device),
    }

    result = greedy_search(trm.inner, batch, num_steps=args.steps)
    pred = decode_output(result.output[0].cpu().numpy(), gold.shape)
    match = bool((pred == gold).all())

    if match:
        correct += 1
    else:
        failures.append(puzzle_id)
    total += 1
    print(f"  {puzzle_id}  {'OK' if match else 'FAIL'}  value={result.value:.4f}")

print(f"\n{args.split} accuracy: {correct}/{total} = {correct/total*100:.1f}%")
if failures:
    print(f"Failed puzzles: {failures[:10]}")
