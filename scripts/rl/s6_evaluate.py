"""Evaluates one model on the held-out prompts and the test set (echo.rl.evaluate), and against a baseline's
evaluation: paired WER, and the critic and CTC on same-seed test takes.

    cd /workspace/Echo && PYTHONPATH=. /venv/main/bin/python scripts/rl/s6_evaluate.py --run runs/rl/round3/runs/base_s0 \
        --critic runs/critic/v2 --baseline runs/rl/round3/eval/start [--anchor runs/rl/round1/eval/start]
    ... s6_evaluate.py --model <checkpoint> --out runs/rl/round3/eval/start --critic runs/critic/v2

--run evaluates <run>/model into <run>/eval and reports to the run's MLflow run when it has one; --model and --out
evaluate any checkpoint (a round's start model, with no baseline). --anchor is a second baseline for the critic alone,
such as the first RL model, since same-seed pairs within one round can't see a drift the whole round shares.
--seed-offset shifts every seed, for fresh takes (evaluate the baseline at the same offset). --shard i/n only
generates a share of the takes, one shard per GPU; the run without it then reuses them. Writes per set
<eval>/<set>/takes/ and rows.csv (one row per take), and <eval>/summary.json. Takes and rows on disk are reused.
"""
import argparse
import json
import time
from pathlib import Path

import torch
from transformers import Qwen2Tokenizer

from echo.rl.critic import CHECKPOINT, GroupCritic
from echo.rl.evaluate import ROW_FIELDS, SETS, critic_pairs, eval_jobs, paired, transcribe_takes, wer
from echo.rl.prompts import read_rows, write_rows
from echo.rl.sampler import load_model, write_takes
from echo.rl.tracking import Tracker


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate one model, and against a baseline.")
    parser.add_argument("--run", type=Path, help="A training run (s5_train.py): its model, into <run>/eval")
    parser.add_argument("--model", type=Path, help="Any checkpoint, with --out")
    parser.add_argument("--out", type=Path)
    parser.add_argument("--critic", type=Path, required=True, help="The critic package (and its take trim)")
    parser.add_argument("--critic-checkpoint", type=Path, default=CHECKPOINT)
    parser.add_argument("--baseline", type=Path, help="The baseline's evaluation directory")
    parser.add_argument("--anchor", type=Path, help="A second baseline's evaluation directory, for the critic alone")
    parser.add_argument("--seed-offset", type=int, default=0)
    parser.add_argument("--shard", help="i/n: only generate every n-th batch of takes, starting at the i-th, then stop")
    args = parser.parse_args()
    if (args.run is None) == (args.model is None) or (args.model is not None and args.out is None):
        parser.error("give either --run, or --model and --out")
    model, out = (args.run / "model", args.run / "eval") if args.run is not None else (args.model, args.out)

    tokenizer = Qwen2Tokenizer.from_pretrained(model)
    sets = {name: eval_jobs(name, tokenizer, args.seed_offset) for name in SETS}
    torch.set_grad_enabled(False)
    if args.shard is not None:
        shard, shards = map(int, args.shard.split("/"))
        echo = load_model(model)
        for name, (_, batches) in sets.items():
            write_takes(echo, batches[shard::shards], out / name / "takes")
        print(f"done: shard {args.shard} of the takes in {out}", flush=True)
        return
    if not all((out / name / "rows.csv").exists() for name in SETS):
        echo = load_model(model)
        for name, (_, batches) in sets.items():
            print(f"{name}: generating", flush=True)
            write_takes(echo, batches, out / name / "takes")
        del echo
        torch.cuda.empty_cache()

    summary = dict(model=str(model), seed_offset=args.seed_offset)
    comparisons = [(key, path) for key, path in (("vs_baseline", args.baseline), ("vs_anchor", args.anchor)) if path]
    if comparisons:
        critic = GroupCritic(args.critic, args.critic_checkpoint)
        jobs = sets["test"][0]
        for key, path in comparisons:
            print(f"critic {key}: {path}", flush=True)
            summary[key] = dict(evaluation=str(path), critic=critic_pairs(critic, jobs, path / "test" / "takes",
                                                                           out / "test" / "takes"))
        del critic
        torch.cuda.empty_cache()

    for name, (jobs, _) in sets.items():
        if not (out / name / "rows.csv").exists():
            print(f"{name}: transcribing", flush=True)
            write_rows(out / name / "rows.csv", transcribe_takes(jobs, out / name / "takes", args.critic, name == "test"),
                       ROW_FIELDS)
        rows = read_rows(out / name / "rows.csv")
        summary[name] = wer(rows)
        if args.baseline is not None:
            summary["vs_baseline"][name] = paired(read_rows(args.baseline / name / "rows.csv"), rows)
    (out / "summary.json").write_text(json.dumps(summary, indent=1))

    lines = [f"{model}, seed offset {args.seed_offset}"]
    for name in SETS:
        s = summary[name]
        lines.append(f"{name}: WER {s['wer']:.2%}, without numbers {s['wer_no_numbers']:.2%}, takes with an error "
                     f"{s['takes_with_error']:.1%}" + (f", WavLM similarity {s['similarity']:.4f}" if "similarity" in s else ""))
        if args.baseline is not None:
            v = summary["vs_baseline"][name]
            lines.append(f"  vs baseline ({v['baseline_wer']:.2%} / {v['baseline_wer_no_numbers']:.2%}): fewer errors in "
                         f"{v['better']} takes, more in {v['worse']}, p = {v['p']:.3f}")
    for key, _ in comparisons:
        c = summary[key]["critic"]
        lines.append(f"critic {key}: wins {c['wins']} of {c['confident']} confident same-seed pairs; CTC loss change "
                     f"{c['ctc_change']:+.4f} +- {c['ctc_change_se']:.4f}; gated test takes {c['gated_baseline']} -> {c['gated']}")
    print("\n".join(lines), flush=True)

    tracker = Tracker.resume(args.run) if args.run is not None else None
    if tracker is not None:
        metrics = {f"eval/{name}/{k}": v for name in SETS for k, v in summary[name].items()}
        for key, _ in comparisons:
            metrics.update({f"eval/{key}/critic/{k}": v for k, v in summary[key]["critic"].items()})
        if args.baseline is not None:
            metrics.update({f"eval/vs_baseline/{name}/{k}": v for name in SETS
                            for k, v in summary["vs_baseline"][name].items() if not k.startswith("baseline_")})
        tracker.log(metrics, 0)
        tracker.close()
    print(f"done: {out / 'summary.json'} ({time.strftime('%H:%M')})", flush=True)


if __name__ == "__main__":
    main()
