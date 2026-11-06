"""Generates a group of takes per prompt with the deployed sampler (echo.rl.sampler: s8, no remasking, 2-pass polish),
into a groups directory (echo.rl.groups).

    cd /workspace/Echo && PYTHONPATH=. /venv/main/bin/python scripts/rl/s2_generate.py --prompts runs/rl/round3/prompts.csv \
        --out runs/rl/round3/groups --checkpoint <round's start model> --takes-per-prompt 32 --seed-offset 30000000 \
        [--shard 0/4] [--gated runs/rl/round3/groups/scored]

A take's seed is 1000 x its prompt's number + its index + --seed-offset, so give every round its own offset (and a
mining pool another one: the takes of a prompt must not repeat). --gated keeps only the prompts whose scored group
there has a CTC-gated take: a mining pool of more takes where the model breaks. --shard i/n generates every n-th
batch, starting at the i-th; run n shards, one per GPU. The batches don't depend on the shards, nor on where a run
stopped, so neither changes a take. Takes on disk are skipped.
"""
import argparse
import json
from pathlib import Path

from transformers import Qwen2Tokenizer

from echo.rl.groups import write_groups
from echo.rl.prompts import model_inputs, read_rows
from echo.rl.sampler import jobs_for, load_model, pack, take_seeds, write_takes


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate a group of takes per prompt.")
    parser.add_argument("--prompts", type=Path, required=True, help="A prompts file with ids (s1_prompts.py)")
    parser.add_argument("--out", type=Path, required=True, help="The groups directory")
    parser.add_argument("--checkpoint", type=Path, required=True, help="The policy: the round's start model")
    parser.add_argument("--takes-per-prompt", type=int, default=32)
    parser.add_argument("--seed-offset", type=int, required=True)
    parser.add_argument("--gated", type=Path, help="A scored/ directory: only its prompts with a CTC-gated take")
    parser.add_argument("--shard", default="0/1", help="i/n: every n-th batch, starting at the i-th")
    parser.add_argument("--batch-frames", type=int, default=16384, help="Target frames per batch, padding included")
    parser.add_argument("--batch-tokens", type=int, default=65536, help="Prefix + target tokens per batch, padding included")
    args = parser.parse_args()

    rows = read_rows(args.prompts)
    if args.gated is not None:
        rows = [r for r in rows if any(json.loads((args.gated / f"{r['id']}.json").read_text())["ctc_gated"])]
    seeds = {r["id"]: take_seeds(r["id"], args.takes_per_prompt, args.seed_offset) for r in rows}
    shard, shards = map(int, args.shard.split("/"))
    if shard == 0:
        write_groups(args.out, rows, seeds)
    prompts = model_inputs(rows, Qwen2Tokenizer.from_pretrained(args.checkpoint))
    batches = pack(jobs_for(prompts, seeds), args.batch_frames, args.batch_tokens)[shard::shards]
    print(f"{len(rows)} prompts x {args.takes_per_prompt} takes; shard {args.shard}: {len(batches)} batches", flush=True)
    write_takes(load_model(args.checkpoint), batches, args.out / "takes")
    print(f"done: {args.out / 'takes'}", flush=True)


if __name__ == "__main__":
    main()
