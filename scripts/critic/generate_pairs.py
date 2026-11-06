"""Generates A/B take pairs for critic training: for every prompt of a prompts file (scripts/rl/s0_prompt_pool.py), one
take with s8 and one with s<--steps>, both from the runtime sampler sNcb2 + 2 (echo.rl.sampler).

    cd /workspace/Echo && PYTHONPATH=. /venv/main/bin/python scripts/critic/generate_pairs.py --prompts data/rl/prompts.csv --out <dir> [--steps 32]

Prompts are cut into batches of at most --batch-frames target frames and --batch-tokens prefix + target tokens
(echo.rl.sampler.pack). Each take draws its Gumbel noise from its own generator seeded with --seed + the prompt's row
index, as the notebook would for that seed (bf16 rounding still depends on the batch). Target length is the
notebook's rule (estimate_frames); the prompt's language tag is the target text's language.

Output: <out>/codes/<row>_s<steps>.npy (int16 [8, frames]) for both samplers, and <out>/pairs.csv (the prompt
columns + row|frames|seed|codes_s8|codes_s<steps>). Takes already on disk are skipped, so a stopped run resumes.
"""
import argparse
import csv
import time
from pathlib import Path

import numpy as np
import torch
from transformers import Qwen2Tokenizer

from echo.rl.prompts import prompt_text
from echo.rl.sampler import Batch, Echo, estimate_frames, generate, pack

DTYPES = {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate s8 vs sN take pairs for critic training.")
    parser.add_argument("--prompts", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--steps", type=int, default=32, help="Main passes of the stronger sampler.")
    parser.add_argument("--checkpoint", type=Path, default=Path("runs/final/fp32"))
    parser.add_argument("--dtype", choices=DTYPES, default="bf16")
    parser.add_argument("--batch-frames", type=int, default=16384, help="Target frames per batch, padding included.")
    parser.add_argument("--batch-tokens", type=int, default=65536,
                        help="Prefix + target tokens per batch, padding included: caps the KV cache of short targets.")
    parser.add_argument("--remask", type=float, default=0.0, help="REMASK_RATIO of both samplers (notebook: 0.5).")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--limit", type=int, help="Only the first N prompts.")
    args = parser.parse_args()

    torch.set_grad_enabled(False)
    device = "cuda"
    model = Echo(args.checkpoint).to(device, DTYPES[args.dtype]).eval()
    tokenizer = Qwen2Tokenizer.from_pretrained(args.checkpoint)
    with open(args.prompts, encoding="utf-8", newline="") as f:
        rows = list(csv.DictReader(f, delimiter="|", quoting=csv.QUOTE_NONE))[: args.limit]
    prompts = []
    for i, row in enumerate(rows):
        reference = torch.from_numpy(np.load(args.prompts.parent / row["reference"]).astype(np.int64))
        prompts.append(dict(row=i, ids=tokenizer(prompt_text(row), return_tensors="pt").input_ids[0], reference=reference,
                            frames=estimate_frames(row["text"], row["reference_transcript"], reference.size(1)),
                            seed=args.seed + i))
    prompts.sort(key=lambda p: (p["frames"], p["ids"].numel() + p["reference"].size(1)))  # pack()'s order

    codes = args.out / "codes"
    codes.mkdir(parents=True, exist_ok=True)

    def path(p: dict, steps: int) -> Path:
        return codes / f"{p['row']:05d}_s{steps}.npy"

    for steps in (8, args.steps):
        todo = [p for p in prompts if not path(p, steps).exists()]
        start, done, padded = time.time(), 0, 0
        for b in pack(todo, args.batch_frames, args.batch_tokens):
            batch = Batch(model, b, device)
            for p, take in zip(b, generate(batch, [p["seed"] for p in b], steps, args.remask)):
                np.save(path(p, steps), take)
            done += len(b)
            padded += batch.n * batch.F - sum(batch.frames)
            elapsed = time.time() - start
            print(f"s{steps}: {done}/{len(todo)} takes, {done / elapsed:.1f} takes/s, batch {batch.n} x {batch.F} frames, "
                  f"padding {padded / max(sum(p['frames'] for p in todo[:done]), 1):.1%}", flush=True)
        if todo:
            print(f"DONE s{steps}: {len(todo)} takes in {time.time() - start:.0f}s", flush=True)

    fields = list(rows[0]) + ["row", "frames", "seed", "codes_s8", f"codes_s{args.steps}"]
    lines = ["|".join(fields)]
    for p in sorted(prompts, key=lambda p: p["row"]):
        extra = [str(p["row"]), str(p["frames"]), str(p["seed"]), str(path(p, 8).relative_to(args.out)),
                 str(path(p, args.steps).relative_to(args.out))]
        lines.append("|".join([rows[p["row"]][k] for k in rows[0]] + extra))
    (args.out / "pairs.csv").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"DONE: {len(prompts)} pairs in {args.out / 'pairs.csv'}", flush=True)


if __name__ == "__main__":
    main()
