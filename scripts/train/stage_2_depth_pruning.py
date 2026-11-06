"""Stage 2 - the depth loop: prunes the cheapest layer and recovers for --round-epochs, pruning.layers_to_prune times,
then, with --final-recovery, the final recovery.

Each step runs as its own process - scripts/pruning/prune.py, then scripts/train/stage_2_pruning.py under torchrun.
With --fade, prune.py only chooses the layer, and the recovery fades it out over pruning.fade_steps and drops it.
"""

import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import List

import torch

from echo.training.config import EchoTrainingConfig
from echo.utils.config_utils import check

CONFIG = "echo/configs/training/train.json"
PRUNE = "scripts/pruning/prune.py"
TRAIN = "scripts/train/stage_2_pruning.py"
REPORT = "pruning_report.json"   # Written by prune.py next to its checkpoint


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=CONFIG, help="Pruning stage's .json config")
    parser.add_argument("--init", type=Path, required=True, help="Checkpoint the depth loop starts from")
    parser.add_argument("--output-dir", type=Path, required=True, help="Root of every round's dropK/, dropK-train/ and final/")
    parser.add_argument("--gpus", type=int, default=max(torch.cuda.device_count(), 1), help="Processes each training runs on")
    parser.add_argument("--fade", action="store_true", help="Fades every chosen layer out during its recovery, instead of cutting it first")
    parser.add_argument("--round-epochs", type=int, default=1, help="Length of every round's recovery, in epochs")
    parser.add_argument("--final-recovery", action=argparse.BooleanOptionalAction, default=False, help="Runs the final recovery after the last round")
    parser.add_argument("--continues", action="store_true", help="The init carries an earlier loop's damage: its first round also warms up from schedule.warmup_start_ratio")
    args = parser.parse_args()

    config = EchoTrainingConfig.from_json(args.config)
    evaluation = config.evaluation
    # Every pruned checkpoint is evaluated before its recovery, as the reference the later evaluations compare to.
    check(
        evaluation.eval_at_start and (evaluation.eval_on_dev or evaluation.eval_on_test),
        "the depth loop needs evaluation.eval_at_start and eval_on_dev or eval_on_test",
    )

    torchrun = [sys.executable, "-m", "torch.distributed.run", f"--nproc_per_node={args.gpus}", TRAIN, "--config", str(args.config)]
    latest = args.init

    for pruned in range(config.pruning.layers_to_prune):
        pruning = args.output_dir / f"drop{pruned + 1}"
        print(f"\n===== round {pruned + 1} of {config.pruning.layers_to_prune}: pruning {latest}", flush=True)

        prune = [sys.executable, PRUNE, "--config", str(args.config), "--checkpoint", str(latest), "--output", str(pruning), "--drop-layer", "auto"]
        _run(prune + (["--choose-only"] if args.fade else []))

        # A fade starts from the unpruned checkpoint, and removes the chosen layer itself.
        start = latest if args.fade else pruning
        fade = ["--fade-layer", str(json.loads((pruning / REPORT).read_text())["layer"])] if args.fade else []
        # The first round starts from a model with nothing left to repair, so its warmup climbs from zero; the later
        # ones carry the previous round's damage and start from the config's schedule.warmup_start_ratio.
        first = ["--warmup-start-ratio", "0"] if pruned == 0 and not args.continues else []
        _run(torchrun + ["--init", str(start), "--output-dir", f"{pruning}-train", "--epochs", str(args.round_epochs), "--eval-at-end"] + fade + first)
        latest = _newest_checkpoint(Path(f"{pruning}-train"))

    if not args.final_recovery:
        print(f"\n===== done, without the final recovery: {latest}", flush=True)
        return

    print(f"\n===== final recovery of {latest}", flush=True)
    _run(torchrun + ["--init", str(latest), "--output-dir", str(args.output_dir / "final")])


def _run(command: List[str]) -> None:
    """
    Runs one step, and stops the loop if it fails.
    """

    code = subprocess.run(command).returncode
    if code != 0:
        raise SystemExit(f"step failed with exit code {code}: {' '.join(command)}")


def _newest_checkpoint(output_dir: Path) -> Path:
    return max(output_dir.glob("checkpoint-*"), key=lambda path: int(path.name.rsplit("-", 1)[-1]))


if __name__ == "__main__":
    main()
