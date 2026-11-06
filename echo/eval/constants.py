from typing import Any, Dict

from echo.eval.types import MultiStepMode, Trajectory

# ---------- Trajectory sweep ----------

# The states the deployed two-stage sampler (s8: 8 main-path passes over codebooks 0-1, remask 0.5, then one polish
# pass over codebooks 2-7) feeds the model on its main path: the masking rate of every codebook at each pass, traced
# from real generation (84 test targets x 2 seeds), and the share of codebook 0-1 tokens that pass decides, which
# weights the main-path row. See knowledge/inference/two_stage_sampling.md.
TRAJECTORY: Trajectory = {
    "s8_f0": ((1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00), 0.002),
    "s8_f1": ((0.98, 0.99, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00), 0.002),
    "s8_f2": ((0.96, 0.99, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00), 0.003),
    "s8_f3": ((0.94, 0.98, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00), 0.005),
    "s8_f4": ((0.91, 0.98, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00), 0.008),
    "s8_f5": ((0.87, 0.96, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00), 0.017),
    "s8_f6": ((0.80, 0.94, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00), 0.183),
    "s8_f7": ((0.68, 0.88, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00), 0.780),
}

# The polish pass's state: codebooks 0-1 complete, the rest fully masked.
POLISH_STATE = (0.00, 0.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00)

# Masks drawn per sample at every grid point. One draw is a noisy view of the
# model's loss at that progress, so a few of them steady the estimate.
REPEATS = 3

# Fixed prefix ratio for single-step evals, so that they are reproducible.
PREFIX_RATIO = 0.25

# Seed the whole sweep is derived from.
# Makes the evaluation results reproductible and allow to directly compare two evaluation runs
# for two different models.
EVAL_SEED = 1234


# ---------- Reporting ----------

# ANSI escapes the summary table is drawn with.
BOLD = "\033[1m"
DIM = "\033[2m"
RESET = "\033[0m"

# Colors the values are painted with, from good through acceptable to poor.
LOSS_COLORS = ("\033[32m", "\033[33m", "\033[31m")

# Where a single step loss stops being good, and where it stops being
# acceptable. Lower is better in both, and the gap over the teacher is read with
# its sign, so a student below its teacher is always green.
#
# Only the two comparisons are graded. A cross entropy on its own is absolute
# nats, and what counts as a good one depends on how much of the target was
# masked - a fully masked target costs over 6 nats however healthy the model is.
LOSS_THRESHOLDS = {
    "ce_delta": (0.03, 0.1),
    "kl": (0.03, 0.15),
}

# Blocks the per-codebook masking ratios are drawn with, from none to all.
BLOCKS = "\u2581\u2582\u2583\u2584\u2585\u2586\u2587\u2588"


# ---------- The full diffusion process ----------

# Generation settings every evaluated mode stands for: the step count of the two-stage sampler's main path. The rest
# of the sampler comes from the model's default generation config.
MODES: Dict[MultiStepMode, Dict[str, Any]] = {
    "s6": {"num_step": 6},
    "s8": {"num_step": 8},
    "s10": {"num_step": 10},
    "s12": {"num_step": 12},
}


# ---------- Metrics ----------

# Sample rate the metric models are trained at. Every waveform is resampled to
# it before it is measured.
METRIC_SAMPLE_RATE = 16_000

# Transcriber the word error rate is measured with. It is a CTranslate2 build,
# run through faster-whisper rather than through transformers.
ASR_MODEL = "Systran/faster-whisper-large-v3"

# Device the metric models run on, if available.
METRIC_DEVICE = "cuda"

# Precision the transcriber runs at.
ASR_COMPUTE_TYPE = {"cuda": "float16", "cpu": "int8"}

# Speaker verification model the similarity is measured with. It returns an
# x-vector per clip, and the metric is the cosine between two of them.
SPEAKER_MODEL = "microsoft/wavlm-base-plus-sv"

# Share of the median frame energy a frame has to clear to count as spoken.
VOICED_RATIO = 0.15

# Where a metric stops being good, and where it stops being acceptable. Lower is
# better everywhere except the speaker similarity, which is flagged the other
# way round.
THRESHOLDS = {
    "wer": (0.05, 0.15),
    "similarity": (0.88, 0.75),
    "rate": (5.0, 15.0),
    "pause": (5.0, 15.0),
}
