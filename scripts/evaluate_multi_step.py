"""Runs the whole diffusion process on the test suite, and scores what comes out of it."""

import argparse
import logging
from pathlib import Path

from echo import build_model
from echo.eval import evaluate_multi_step
from echo.eval.constants import MODES
from echo.utils.distributed_utils import setup


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-config", type=Path, required=True, help="Model's .json config")
    parser.add_argument("--dataset-inputs", type=Path, required=True, help="References .csv: codec path, transcript, speaker id, language")
    parser.add_argument("--dataset-targets", type=Path, required=True, help="Targets .csv: speaker id, text, language")
    parser.add_argument("--mode", choices=tuple(MODES), default="s8", help="The schedule the model generates with")
    parser.add_argument("--max-samples", type=int, default=50)
    parser.add_argument("--per-speaker", action="store_true", help="Adds a breakdown by reference voice")
    parser.add_argument("--per-language", action="store_true", help="Adds a breakdown by language")
    args = parser.parse_args()

    # The report goes through the log, so it needs somewhere to land.
    logging.basicConfig(level=logging.INFO, format="%(message)s")

    model = build_model(args.model_config, device=setup())

    evaluate_multi_step(
        model,
        args.dataset_inputs,
        args.dataset_targets,
        mode=args.mode,
        max_samples=args.max_samples,
        per_speaker=args.per_speaker,
        per_language=args.per_language,
    )


if __name__ == "__main__":
    main()
