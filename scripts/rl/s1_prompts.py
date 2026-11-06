"""Draws a round's prompts from a prompt pool (s0_prompt_pool.py), and keeps what is left of the pool for later
rounds.

    PYTHONPATH=. .venv/bin/python scripts/rl/s1_prompts.py --prompts-pool data/rl/pool.csv --amount 2000 \
        --round runs/rl/round3 [--seed 0]

--amount prompts are drawn at random, never one with a reserved text (echo.rl.prompts.reserved_texts: evaluation
texts and the texts the user ranked), and numbered p0000, p0001, ... Writes <round>/prompts.csv (an id, then the
pool's columns) and <round>/pool_left.csv, the pool without the drawn prompts: the next round's --prompts-pool.
"""
import argparse
import random
from pathlib import Path

from echo.rl.prompts import FIELDS, read_rows, reserved_texts, write_rows


def main() -> None:
    parser = argparse.ArgumentParser(description="Draw a round's prompts from a prompt pool.")
    parser.add_argument("--prompts-pool", type=Path, required=True)
    parser.add_argument("--amount", type=int, required=True)
    parser.add_argument("--round", type=Path, required=True, help="The round's directory")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    if (args.round / "prompts.csv").exists():
        raise SystemExit(f"{args.round / 'prompts.csv'} exists: a round draws its prompts once")
    pool = read_rows(args.prompts_pool)
    reserved = reserved_texts()
    eligible = [i for i, r in enumerate(pool) if r["text"] not in reserved]
    if args.amount > len(eligible):
        raise SystemExit(f"the pool has {len(eligible)} usable prompts, fewer than --amount {args.amount}")
    drawn = random.Random(args.seed).sample(eligible, args.amount)
    rows = [dict(pool[i], id=f"p{n:04d}") for n, i in enumerate(drawn)]
    taken = set(drawn)
    left = [r for i, r in enumerate(pool) if i not in taken]
    write_rows(args.round / "prompts.csv", rows, ["id"] + FIELDS)
    write_rows(args.round / "pool_left.csv", left, FIELDS)
    print(f"{len(rows)} prompts for {len({r['speaker_id'] for r in rows})} speakers in {args.round / 'prompts.csv'}; "
          f"{len(left)} left in {args.round / 'pool_left.csv'} ({len(pool) - len(eligible)} with reserved texts)",
          flush=True)


if __name__ == "__main__":
    main()
