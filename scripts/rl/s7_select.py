"""Picks a round's winner among its runs, by the rule fixed before the round, CTC and critic weighed equally
(the user, 2026-11-04; knowledge/training/rl_dpo.md):
    1. a run qualifies only if it doesn't lose to the start model on held-out WER (more paired takes better than
       worse) or on the critic (at least as many confident same-seed test pairs won as lost);
    2. among those, the best rank sum of held-out WER without numbers and the critic's net wins; ties go to the lower
       KL (mean of the last 50 training steps).

    PYTHONPATH=. .venv/bin/python scripts/rl/s7_select.py --round runs/rl/round3

Reads <round>/runs/*/eval/summary.json (s6_evaluate.py against the start model) and train.log; writes
<round>/winner.json and <round>/summary.md. The winner's numbers are inflated by its selection: report it on fresh
seeds (s6_evaluate.py --seed-offset).
"""
import argparse
import json
import re
from pathlib import Path


def kl(run: Path) -> float:
    steps = [line for line in (run / "train.log").read_text().splitlines() if " step " in line]
    values = [float(m.group(1)) for line in steps[-50:] if (m := re.search(r"KL/slot ([0-9.]+)", line))]
    return sum(values) / len(values)


def rank(values: dict[str, float]) -> dict[str, float]:
    """Rank 1 for the lowest value; tied values share their mean rank."""
    order = sorted(values.values())
    return {k: sum(i + 1 for i, x in enumerate(order) if x == v) / order.count(v) for k, v in values.items()}


def main() -> None:
    parser = argparse.ArgumentParser(description="Pick a round's winner among its runs.")
    parser.add_argument("--round", type=Path, required=True)
    args = parser.parse_args()

    runs = {}
    for summary in sorted(args.round.glob("runs/*/eval/summary.json")):
        s, run = json.loads(summary.read_text()), summary.parent.parent
        held, critic = s["vs_baseline"]["held"], s["vs_baseline"]["critic"]
        runs[run.name] = dict(model=str(run / "model"), held=s["held"]["wer_no_numbers"], held_better=held["better"],
                              held_worse=held["worse"], test=s["test"]["wer_no_numbers"], critic_wins=critic["wins"],
                              critic_losses=critic["confident"] - critic["wins"], ctc_change=critic["ctc_change"], kl=kl(run))
    if not runs:
        raise SystemExit(f"no evaluated runs in {args.round / 'runs'}")
    qualified = {k: r for k, r in runs.items() if r["held_better"] >= r["held_worse"] and r["critic_wins"] >= r["critic_losses"]}
    held_rank = rank({k: r["held"] for k, r in qualified.items()})
    critic_rank = rank({k: -(r["critic_wins"] - r["critic_losses"]) for k, r in qualified.items()})
    for k in qualified:
        runs[k]["rank_sum"] = held_rank[k] + critic_rank[k]
    winner = min(qualified, key=lambda k: (runs[k]["rank_sum"], runs[k]["kl"])) if qualified else None

    (args.round / "winner.json").write_text(json.dumps(dict(winner=winner, runs=runs), indent=1))
    lines = ["| run | held-out WER without numbers (paired better / worse) | test WER without numbers | critic won / lost | "
             "CTC change | KL | rank sum |", "|---|---|---|---|---|---|---|"]
    for k, r in runs.items():
        name = f"**{k}**" if k == winner else k
        lines.append(f"| {name} | {r['held']:.2%} ({r['held_better']} / {r['held_worse']}) | {r['test']:.2%} | "
                     f"{r['critic_wins']} / {r['critic_losses']} | {r['ctc_change']:+.4f} | {r['kl']:.3f} | "
                     f"{r.get('rank_sum', 'not qualified')} |")
    (args.round / "summary.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines) + f"\n\nwinner: {winner}", flush=True)


if __name__ == "__main__":
    main()
