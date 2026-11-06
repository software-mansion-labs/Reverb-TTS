"""Trains one DPO run on a recipe's pairs (s4_pairs.py) from the round's start model (echo.rl.dpo).

    cd /workspace/Echo && PYTHONPATH=. /venv/main/bin/python scripts/rl/s5_train.py --config echo/configs/rl/base.json \
        --pairs runs/rl/round3/pairs/base.jsonl --start <round's start model> --out runs/rl/round3/runs/base_s0 [--seed 0]

The start model must keep its text embedding factorized, as deployed, so the trained model is saved in its layout.
Writes <out>/model (the start model's files with the trained weights), train.log (one line per step), probe.jsonl
(held-out and training pair margins under fixed masks, every probe_every steps) and recipe.json. With tracking on in
the recipe, the run reports to MLflow and its id goes to <out>/mlflow_run_id, where s6_evaluate.py finds it. About
13.5 s per step of 32 pairs on an RTX 5090.
"""
import argparse
from pathlib import Path

from echo.rl.config import RLConfig
from echo.rl.dpo import train
from echo.rl.pairs import read_pairs
from echo.rl.sampler import is_factorized
from echo.rl.tracking import Tracker


def main() -> None:
    parser = argparse.ArgumentParser(description="Train one DPO run.")
    parser.add_argument("--config", type=Path, required=True, help="The DPO recipe (echo/configs/rl/)")
    parser.add_argument("--pairs", type=Path, required=True)
    parser.add_argument("--start", type=Path, required=True, help="The round's start model: policy and reference")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=0, help="Seeds the pair order and the masks")
    parser.add_argument("--max-steps", type=int, help="Stop after this many steps (smoke tests)")
    args = parser.parse_args()

    if (args.out / "model" / "model.safetensors").exists():
        raise SystemExit(f"{args.out / 'model'} exists: the run is done")
    if not is_factorized(args.start):
        raise SystemExit(f"{args.start} keeps its text embedding dense: convert it with scripts/rl/factorize_checkpoint.py")
    config = RLConfig.from_json(args.config)
    pairs = read_pairs(args.pairs)
    tracker = Tracker.start(config, args.out, dict(start=str(args.start), pairs=str(args.pairs), seed=args.seed))
    train(config, pairs, args.start, args.out, args.seed, args.max_steps, tracker)
    if tracker is not None:
        tracker.close()


if __name__ == "__main__":
    main()
