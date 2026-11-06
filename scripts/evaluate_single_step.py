"""Scores a single denoising step across the trajectory, optionally against a teacher."""

import argparse
import logging
from pathlib import Path

from echo import build_model
from echo.data.dataset import EchoDataset
from echo.data.loader import EchoDataLoader
from echo.data.preprocessor import EchoPreprocessor
from echo.eval import evaluate_single_step
from echo.eval.constants import PREFIX_RATIO
from echo.utils.distributed_utils import setup


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-config", type=Path, required=True, help="Student's .json config")
    parser.add_argument("--teacher-config", type=Path, help="Teacher's .json config - adds its CE and the KL to the report")
    parser.add_argument("--dataset", type=Path, required=True, help="Dataset .csv, as written by scripts/codec_manifest.py")
    parser.add_argument("--data-root", type=Path, help="Root the codec paths are relative to (default: the dataset's directory)")
    parser.add_argument("--max-batches", type=int, default=50)
    parser.add_argument("--per-language", action=argparse.BooleanOptionalAction, default=True,
                        help="A breakdown by language, on unless --no-per-language")
    args = parser.parse_args()

    # The report goes through the log, so it needs somewhere to land.
    logging.basicConfig(level=logging.INFO, format="%(message)s")

    device = setup()
    student = build_model(args.model_config, device=device)
    teacher = build_model(args.teacher_config, device=device) if args.teacher_config else None

    dataset = EchoDataset(args.dataset, data_root=args.data_root)

    # The evaluation draws its own masks, so the loader hands out clean audio,
    # split at a fixed ratio rather than a drawn one.
    loader = EchoDataLoader(
        dataset,
        EchoPreprocessor(
            prefix_min_ratio=PREFIX_RATIO, prefix_max_ratio=PREFIX_RATIO
        ),
        disable_prefix_drop=True,
        disable_masking=True,
    )

    evaluate_single_step(
        student,
        teacher,
        loader,
        max_batches=args.max_batches,
        per_language=args.per_language,
    )


if __name__ == "__main__":
    main()
