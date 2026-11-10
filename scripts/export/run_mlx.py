"""Generates speech codes with an Echo .pte from scripts/export/export_mlx.py or export_xnnpack.py: the runtime notebooks'
sNcb2 + 2 sampler (N main passes over codebooks 0-1, then 2 polish passes) around the exported passes.

Writes the [8, frames] Higgs codec codes as .npy; the codec decodes them, as in scripts/runtime/fixed_reference.ipynb.
Run it with the ExecuTorch venv: ../executorch/.venv/bin/python scripts/export/run_mlx.py ...
"""

import argparse
import math
import unicodedata
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import torch
from executorch.extension.pybindings import portable_lib
from executorch.runtime import Runtime
from transformers import Qwen2Tokenizer

MASK = 1024  # the id of a masked slot, in every codebook

T_SHIFT = 0.1               # warps the commit schedule: few commits early, most in the last passes
CODEBOOK_PENALTY = 5.0      # per codebook index, so codebook 0 tends to be committed before 1
POSITION_TEMPERATURE = 5.0  # scales confidence and penalty against the Gumbel noise
REMASK_RATIO = 0.5          # earlier commits re-masked per new commit, on every main pass but the last
POLISH_BIAS = 1.0           # attention bias towards the reference frames, in the polish passes only
POLISH_GUIDANCE = 0.5       # the second polish pass's weight against the first, for codebooks 4-7


def estimate_frames(text: str, reference_text: str, reference_frames: int) -> int:
    """
    The target's length: the text's spoken weight at the reference's rate (OmniVoice's rule, for alphabetic scripts).
    """

    def weight(string: str) -> float:
        kinds = (unicodedata.category(char)[0] for char in string)
        return sum({"Z": 0.2, "M": 0.0, "P": 0.5, "S": 0.5, "N": 3.5}.get(kind, 1.0) for kind in kinds)

    frames = weight(text) / (weight(reference_text) / reference_frames)
    if frames < 50:  # short texts get relatively more time
        frames = 50 * (frames / 50) ** (1 / 3)
    return max(1, int(frames))


class EchoMLX:
    def __init__(self, pte: Path, tokenizer: Path, threads: Optional[int] = None) -> None:
        if threads:
            # Before loading: XNNPACK keeps the thread pool it was loaded with.
            portable_lib._unsafe_reset_threadpool(threads)
        program = Runtime.get().load_program(pte)
        self.prefill_method, self.step_method, self.polish_method = (
            program.load_method(name) for name in ("prefill", "step", "polish")
        )
        self.tokenizer = Qwen2Tokenizer.from_pretrained(tokenizer)

    def prefill(self, text: str, language: str, reference_text: str, reference: torch.Tensor) -> Tuple[torch.Tensor, ...]:
        """
        Runs the prefix once; returns its keys and values, as the program lays them out.
        """

        prompt = (
            f"<|denoise|><|lang_start|>{language}<|lang_end|><|instruct_start|>None<|instruct_end|>"
            f"<|text_start|>{reference_text} {text}<|text_end|>"
        )
        ids = self.tokenizer(prompt, return_tensors="pt").input_ids[0]
        return tuple(self.prefill_method.execute([ids, reference]))

    def generate(self, cache: Tuple[torch.Tensor, ...], frames: int, reference_frames: int, steps: int) -> torch.Tensor:
        """
        Fills a fully masked [8, frames] target: 'steps' main passes over codebooks 0-1, then 2 polish passes.
        """

        tokens = torch.full((8, frames), MASK, dtype=torch.long)
        codebook = torch.arange(8)[:, None]
        t = torch.linspace(0, 1, steps + 1)
        t = (T_SHIFT * t / (1 - (1 - T_SHIFT) * t)).tolist()
        confidence = torch.zeros(8, frames)
        prediction = torch.zeros(8, frames, dtype=torch.long)

        for step in range(steps):
            confidence[:2], prediction[:2] = self.step_method.execute([tokens, *cache])[:2]
            masked = (tokens == MASK) & (codebook < 2)
            revealed = (tokens != MASK) & (codebook < 2)
            last = step == steps - 1
            # This pass's share of the 2 * frames slots; the last pass fills whatever remasking left open.
            k = int(masked.sum()) if last else math.ceil(2 * frames * (t[step + 1] - t[step]))
            gumbel = -torch.log(-torch.log(torch.rand_like(confidence) + 1e-10) + 1e-10)
            score = (confidence - CODEBOOK_PENALTY * codebook) / POSITION_TEMPERATURE + gumbel
            chosen = score.masked_fill(~masked, -math.inf).flatten().topk(k).indices
            tokens.view(-1)[chosen] = prediction.view(-1)[chosen]
            if not last:  # remasking
                count = min(round(REMASK_RATIO * k), int(revealed.sum()))
                withdrawn = confidence.masked_fill(~revealed, math.inf).flatten().topk(count, largest=False).indices
                tokens.view(-1)[withdrawn] = MASK

        prefix = cache[-1].shape[-2]  # the values end in [prefix length, head dim] in both layouts
        bias = torch.zeros(1, 1, 1, prefix, dtype=cache[0].dtype)
        bias[..., prefix - reference_frames :] = POLISH_BIAS
        guidance = torch.tensor([POLISH_GUIDANCE])
        tokens[2:] = self.polish_method.execute([tokens, bias, guidance, *cache])[0]
        return tokens


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--pte", type=Path, required=True, help="Program written by export_mlx.py")
    parser.add_argument("--tokenizer", type=Path, help="Directory of the text tokenizer (default: the .pte's)")
    parser.add_argument("--reference", type=Path, required=True, help="The reference clip's codec codes, [8, frames] .npy")
    parser.add_argument("--reference-text", required=True, help="Transcript of the reference clip")
    parser.add_argument("--text", required=True, help="Text to speak")
    parser.add_argument("--language", required=True, help="Two-letter code, e.g. en")
    parser.add_argument("--frames", type=int, help="Target length at 25 frames per second (default: estimated from the text)")
    parser.add_argument("--steps", type=int, default=8, help="Main passes")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--threads", type=int, help="CPU threads, for XNNPACK programs (6 is fastest on an M4 Pro)")
    parser.add_argument("--output", type=Path, required=True, help=".npy the [8, frames] codes are written to")
    args = parser.parse_args()

    echo = EchoMLX(args.pte, args.tokenizer or args.pte.parent, args.threads)
    reference = torch.from_numpy(np.load(args.reference)).long()
    frames = args.frames or estimate_frames(args.text, args.reference_text, reference.shape[1])

    cache = echo.prefill(args.text, args.language, args.reference_text, reference)
    torch.manual_seed(args.seed)
    tokens = echo.generate(cache, frames, reference.shape[1], args.steps)
    np.save(args.output, tokens.numpy())
    print(f"{frames} frames ({frames / 25:.2f} s) -> {args.output}")


if __name__ == "__main__":
    main()
