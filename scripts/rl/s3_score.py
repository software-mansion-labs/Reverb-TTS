"""Scores every group of a groups directory (s2_generate.py) with an RL critic package: critic score, CTC loss,
per-word alignment and CTC gate per take, one scored/<prompt id>.json per group (echo.rl.groups).

    cd /workspace/Echo && PYTHONPATH=. /venv/main/bin/python scripts/rl/s3_score.py --groups runs/rl/round3/groups \
        --critic runs/critic/v2 [--critic-checkpoint runs/final/fp32] [--shard 0/4]

The critic reads takes through --critic-checkpoint, the checkpoint it was fitted with, whatever model made them; its
package goes on record in <groups>/critic.json. A critic holds about 18 GB of GPU memory. --shard i/n scores every
n-th prompt, starting at the i-th. Groups already scored are skipped, so a stopped run resumes, and the pair rules
(s4_pairs.py) can change without rescoring.
"""
import argparse
import hashlib
import json
import time
from pathlib import Path

import numpy as np
import torch

from echo.rl.critic import CHECKPOINT, GroupCritic
from echo.rl.groups import read_prompts, read_seeds, read_take
from echo.rl.prompts import RL_DATA


def main() -> None:
    parser = argparse.ArgumentParser(description="Score every group of takes with an RL critic.")
    parser.add_argument("--groups", type=Path, required=True)
    parser.add_argument("--critic", type=Path, required=True, help="The critic package's folder (critic.py, weights)")
    parser.add_argument("--critic-checkpoint", type=Path, default=CHECKPOINT)
    parser.add_argument("--shard", default="0/1", help="i/n: every n-th prompt, starting at the i-th")
    args = parser.parse_args()

    shard, shards = map(int, args.shard.split("/"))
    if shard == 0:
        weights = hashlib.md5((args.critic / "critic_weights.npz").read_bytes()).hexdigest()
        (args.groups / "critic.json").write_text(json.dumps(dict(package=str(args.critic.resolve()), weights_md5=weights,
                                                                 checkpoint=str(args.critic_checkpoint))))
    seeds = read_seeds(args.groups)
    rows = read_prompts(args.groups)[shard::shards]
    scored = args.groups / "scored"
    scored.mkdir(exist_ok=True)
    todo = [r for r in rows if not (scored / f"{r['id']}.json").exists()]
    torch.set_grad_enabled(False)
    critic = GroupCritic(args.critic, args.critic_checkpoint)
    start, gated = time.time(), []
    for n, row in enumerate(todo, 1):
        keys = [f"{row['id']}_{s}" for s in seeds[row["id"]]]
        prompt = dict(row, reference=torch.from_numpy(np.load(RL_DATA / row["reference"]).astype(np.int64)))
        group = critic.score(prompt, [read_take(args.groups, k) for k in keys], keys)
        (scored / f"{row['id']}.json").write_text(json.dumps(group))
        gated.append(np.mean(group["ctc_gated"]))
        if n % 8 == 0 or n == len(todo):
            print(f"  scored {n}/{len(todo)} groups ({(time.time() - start) / n:.1f} s/group); CTC-gated {np.mean(gated):.1%}",
                  flush=True)
    print(f"done: {scored}", flush=True)


if __name__ == "__main__":
    main()
