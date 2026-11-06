"""Picks the pairs a DPO recipe trains on from scored groups (echo.rl.pairs), weighted as the recipe asks.

    PYTHONPATH=. .venv/bin/python scripts/rl/s4_pairs.py --config echo/configs/rl/mine.json \
        --groups runs/rl/round3/groups --ctc-only runs/rl/round3/mine --out runs/rl/round3/pairs/mine.jsonl

--groups and --ctc-only are repeatable; a --ctc-only directory (a mining pool) gives CTC pairs only. Writes one pair
per line: groups, prompt, winner, loser, kind, weight.
"""
import argparse
from pathlib import Path

from echo.rl.config import RLConfig
from echo.rl.pairs import load_pairs, weigh, write_pairs


def main() -> None:
    parser = argparse.ArgumentParser(description="Pick a DPO recipe's pairs from scored groups.")
    parser.add_argument("--config", type=Path, required=True, help="The DPO recipe (echo/configs/rl/)")
    parser.add_argument("--groups", type=Path, action="append", required=True)
    parser.add_argument("--ctc-only", type=Path, action="append", default=[])
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    config = RLConfig.from_json(args.config)
    pairs = [p for g in args.groups for p in load_pairs(g, config.pairs)]
    pairs += [p for g in args.ctc_only for p in load_pairs(g, config.pairs, ctc_only=True)]
    weight = weigh(pairs, config.pairs.ctc_share)
    write_pairs(args.out, pairs)
    ctc = sum(p.kind == "ctc" for p in pairs)
    print(f"{len(pairs)} pairs ({ctc} CTC, {len(pairs) - ctc} critic) from {len({(p.groups, p.prompt) for p in pairs})} "
          f"prompts, CTC pair weight {weight:.3f}, in {args.out}", flush=True)


if __name__ == "__main__":
    main()
