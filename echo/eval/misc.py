import logging

from typing import Dict, List, Optional

from echo.eval.constants import (
    BLOCKS,
    BOLD,
    DIM,
    LOSS_COLORS,
    LOSS_THRESHOLDS,
    RESET,
    THRESHOLDS,
)
from echo.eval.types import Breakdown, Row

logger = logging.getLogger(__name__)

# Columns the table carries, and the headers they are printed under. Everything
# past the first one needs a teacher to be measured against.
COLUMNS = {"ce": "CE", "ce_teacher": "CE (T)", "ce_delta": "d CE", "kl": "KL"}


def print_summary(
    results: Dict[str, Row],
    with_teacher: bool,
    languages: Optional[Breakdown] = None,   # Adds a row per language
) -> None:
    """
    Prints the losses as a table, one row per traced sampler state, then their
    weighted pool.

    Beside the losses every row carries the masking profile the point
    produced: one block per codebook, tall where most of it was masked.
    """

    columns = list(COLUMNS) if with_teacher else ["ce"]

    header = (
        f"{'state':>10}  "
        + "  ".join(f"{COLUMNS[column]:>10}" for column in columns)
        + f"    {'MASK c0-c7':<20}{'positions':>12}"
    )
    logger.info(f"\n{BOLD}{header}{RESET}")
    logger.info(f"{DIM}{'-' * len(header)}{RESET}")

    # The traced states, and last their pool weighted by the share of the grid each state decides.
    for index, (label, row) in enumerate(results.items()):
        losses = "  ".join(_cell(column, row[column]) for column in columns)
        name = f"{BOLD}{label:>10}{RESET}" if index == len(results) - 1 else f"{label:>10}"
        logger.info(f"{name}  {losses}    {_profile(row['mask_ratios']):<20}{row['positions']:>12,.0f}")

    if languages:
        _print_languages(languages, columns)

    logger.info("")


def _print_languages(
    languages: Breakdown,
    columns: List[str],   # The same columns the grid above was printed with
) -> None:
    """
    One line per language, pooled like the sampler's row.
    """

    header = f"{'  by language':<12}" + "".join(f"{COLUMNS[c]:>10}" for c in columns)
    logger.info(f"{DIM}{'-' * (len(header) + 13)}{RESET}")
    logger.info(f"{DIM}{header}{'positions':>13}{RESET}")

    for language, row in languages.items():
        losses = "".join(_cell(column, row[column]) for column in columns)
        logger.info(f"  {language:<10}{losses}{row['positions']:>13,.0f}")


def print_metrics(
    results: Dict[str, float],
    mode: str,
    speakers: Optional[Breakdown] = None,    # Adds a row per voice
    languages: Optional[Breakdown] = None,   # Adds a row per language
) -> None:
    """
    Prints what a full diffusion run scored, one row per metric.

    Every value is painted against the range the metric is still good in, so a
    row that needs attention is visible without knowing the thresholds by heart.
    """

    rows = [
        ("word error rate", "wer", "wer", f"{results['wer']:>12.1%}"),
        ("speaker similarity", "similarity", "similarity", f"{results['similarity']:>12.3f}"),
        ("speaking rate", "rate_dev", "rate", f"{results['rate_dev']:>+11.1f}%"),
        ("  deviation", "rate_abs", "rate", f"{results['rate_abs']:>11.1f}%"),
        ("pausing", "pause_dev", "pause", f"{results['pause_dev']:>+10.1f}pp"),
        ("  deviation", "pause_abs", "pause", f"{results['pause_abs']:>10.1f}pp"),
    ]

    header = f"{mode:<20}{'value':>12}"
    logger.info(f"\n{BOLD}{header}{RESET}")
    logger.info(f"{DIM}{'-' * len(header)}{RESET}")

    for label, metric, threshold, value in rows:
        logger.info(f"{label:<20}{_graded(value, results[metric], threshold)}")

    lost = results.get("lost", 0.0)

    if speakers:
        _print_group("by speaker", speakers)
    if languages:
        _print_group("by language", languages)

    logger.info(f"{DIM}{'-' * len(header)}{RESET}")
    logger.info(f"{DIM}{'generations':<20}{results['samples']:>12,.0f}{RESET}")
    logger.info(f"{_lost_color(lost)}{'lost':<20}{lost:>12,.0f}{RESET}\n")


def _print_group(title: str, groups: Breakdown) -> None:
    """
    One line per group, holding the same metrics as the table above it.
    """

    header = f"{'  ' + title:<20}{'WER':>7}{'SIM':>7}{'RATE':>8}{'PAUSE':>9}{'n':>5}"
    logger.info(f"{DIM}{'-' * len(header)}{RESET}")
    logger.info(f"{DIM}{header}{RESET}")

    for speaker, row in groups.items():
        cells = (
            _graded(f"{row['wer']:>7.1%}", row["wer"], "wer")
            + _graded(f"{row['similarity']:>7.3f}", row["similarity"], "similarity")
            + _graded(f"{row['rate_dev']:>+8.1f}", row["rate_dev"], "rate")
            + _graded(f"{row['pause_dev']:>+9.1f}", row["pause_dev"], "pause")
        )
        logger.info(f"  {speaker:<18}{cells}{row['samples']:>5,.0f}")


def _lost_color(lost: float) -> str:
    """
    Dims a clean run, and marks one that lost takes along the way.
    """

    return DIM if lost == 0.0 else LOSS_COLORS[1]


def _graded(text: str, value: float, threshold: str) -> str:
    """
    Paints a metric by the thresholds it is expected to stay within.
    """

    good, acceptable = THRESHOLDS[threshold]

    if threshold == "similarity":
        rank = 0 if value >= good else 1 if value >= acceptable else 2
    else:
        value = abs(value)
        rank = 0 if value <= good else 1 if value <= acceptable else 2

    return f"{LOSS_COLORS[rank]}{text}{RESET}"


def _cell(column: str, value: float) -> str:
    """
    Renders one loss, painted against the range it is expected to stay within.

    A cross entropy on its own has no such range - only the comparisons against
    the teacher do - so it is left uncolored.
    """

    # The gap over the teacher is the one column worth reading with its sign.
    text = f"{value:>+10.4f}" if column == "ce_delta" else f"{value:>10.4f}"

    if column not in LOSS_THRESHOLDS:
        return text

    good, acceptable = LOSS_THRESHOLDS[column]
    rank = 0 if value <= good else 1 if value <= acceptable else 2

    return f"{LOSS_COLORS[rank]}{text}{RESET}"


def _profile(ratios: List[float]) -> str:
    """
    Draws the per-codebook masking ratios as blocks, plus the two ends of them.
    """

    blocks = "".join(
        BLOCKS[min(int(ratio * len(BLOCKS)), len(BLOCKS) - 1)] for ratio in ratios
    )

    return f"{blocks}  {ratios[0]:.2f}-{ratios[-1]:.2f}"
