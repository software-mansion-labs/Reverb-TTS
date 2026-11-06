"""Speed of the Echo .pte exports against the original OmniVoice, at the four standard lengths: 5 s or 10 s of
reference, 5 s or 10 s of target.

- Echo (--pte, with the ExecuTorch venv): from the text to the target's codes - prompt tokens, prefill, 8 main passes,
  the polish passes and the sampler between them. XNNPACK programs want --threads 6.
- OmniVoice (--omnivoice, with Echo's venv): its generate() as shipped, 16 steps with guidance 2.0 - in fp16 on MPS, or
  in fp32 with --device cpu. The table times its token generation, the same span as Echo's; the codec decode after it
  is common to both.

Each run adds its rows to --results and prints the table of every row there.
"""

import argparse
import csv
import io
import json
import statistics
import sys
import time
import zipfile
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
FRAME_RATE = 25

# The references, cut to 5 s and 10 s, and target texts whose estimated lengths are about 5 s and 10 s. The target
# lengths are then set to exactly 5 s and 10 s, for both models.
REFERENCES = {5: "english/codecs/en_2_kim.npy", 10: "english/codecs/en_1_diana.npy"}
TEXTS = {
    5: "We're sorry, that is not a valid selection. Please try again.",
    10: "Keep in mind that the tax treatment of an asset can differ significantly from how it is presented in the "
        "financial statements.",
}
SETUPS = [(5, 5), (5, 10), (10, 5), (10, 10)]  # (reference, target) seconds

WARMUP = 2


def load_setups(data: Path) -> List[dict]:
    transcripts = {row["name"]: row["transcript"] for row in csv.DictReader(open(data / "test_in_processed.csv"), delimiter="|")}
    setups = []
    for reference, target in SETUPS:
        name = REFERENCES[reference]
        language, member = name.split("/", 1)
        with zipfile.ZipFile(data / language / "codecs.zip") as archive:
            codes = np.load(io.BytesIO(archive.read(member)))[:, : reference * FRAME_RATE]
        setups.append({
            "name": f"{reference} s + {target} s",
            "reference": torch.from_numpy(codes).long(),
            "reference_text": transcripts[name],
            "text": TEXTS[target],
            "frames": target * FRAME_RATE,
        })

    return setups


def timed(run: Callable[[], Tuple[float, ...]], repeats: int) -> List[Tuple[float, ...]]:
    for _ in range(WARMUP):
        run()
    return [run() for _ in range(repeats)]


def bench_echo(pte: Path, setups: List[dict], repeats: int, threads: Optional[int]) -> Dict[str, dict]:
    sys.path.insert(0, str(Path(__file__).parent))
    from run_mlx import EchoMLX

    echo = EchoMLX(pte, pte.parent, threads)
    rows = {}
    for setup in setups:
        def run() -> Tuple[float, float]:
            start = time.perf_counter()
            cache = echo.prefill(setup["text"], "en", setup["reference_text"], setup["reference"])
            prefilled = time.perf_counter()
            echo.generate(cache, setup["frames"], setup["reference"].shape[1], steps=8)
            return time.perf_counter() - start, prefilled - start

        times = timed(run, repeats)
        rows[setup["name"]] = {
            "seconds": statistics.median(t[0] for t in times),
            "prefill": statistics.median(t[1] for t in times),
        }
        print(f"{pte.stem} {setup['name']}: {1000 * rows[setup['name']]['seconds']:.1f} ms", flush=True)

    return rows


def bench_omnivoice(setups: List[dict], repeats: int, device: str) -> Dict[str, dict]:
    from omnivoice.models.omnivoice import OmniVoice, OmniVoiceGenerationConfig, VoiceClonePrompt

    dtype = torch.float16 if device == "mps" else torch.float32
    model = OmniVoice.from_pretrained("k2-fsa/OmniVoice", device_map=device, dtype=dtype)
    sync = torch.mps.synchronize if device == "mps" else lambda: None
    config = OmniVoiceGenerationConfig(num_step=16, guidance_scale=2.0)

    # Times the token generation inside generate(), so it can be read apart from the codec decode.
    spans = []
    generate_tokens = model._generate_iterative

    def timed_tokens(*args, **kwargs) -> list:
        sync()
        start = time.perf_counter()
        tokens = generate_tokens(*args, **kwargs)
        sync()
        spans.append(time.perf_counter() - start)
        return tokens

    model._generate_iterative = timed_tokens

    rows = {}
    for setup in setups:
        prompt = VoiceClonePrompt(ref_audio_tokens=setup["reference"], ref_text=setup["reference_text"], ref_rms=None)

        def run() -> Tuple[float, float]:
            start = time.perf_counter()
            model.generate(text=setup["text"], language="en", voice_clone_prompt=prompt,
                           duration=setup["frames"] / FRAME_RATE, generation_config=config)
            return spans[-1], time.perf_counter() - start

        times = timed(run, repeats)
        rows[setup["name"]] = {
            "seconds": statistics.median(t[0] for t in times),
            "generate": statistics.median(t[1] for t in times),
        }
        print(f"omnivoice {device} {setup['name']}: {1000 * rows[setup['name']]['seconds']:.1f} ms", flush=True)

    return rows


def print_table(results: Dict[str, Dict[str, dict]]) -> None:
    models = list(results)
    print("\nms per generation (x realtime) - median\n")
    print(" | ".join(["setup".ljust(12)] + [model.rjust(20) for model in models]))
    for reference, target in SETUPS:
        name = f"{reference} s + {target} s"
        cells = [
            f"{1000 * results[model][name]['seconds']:.0f} ({target / results[model][name]['seconds']:.1f}x)".rjust(20)
            if name in results[model] else "-".rjust(20)
            for model in models
        ]
        print(" | ".join([name.ljust(12)] + cells))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--pte", type=Path, nargs="*", default=[], help="Echo programs written by export_mlx.py")
    parser.add_argument("--threads", type=int, help="CPU threads for the Echo programs (XNNPACK: 6)")
    parser.add_argument("--omnivoice", action="store_true", help="Benchmarks k2-fsa/OmniVoice instead")
    parser.add_argument("--device", default="mps", choices=("mps", "cpu"), help="Where OmniVoice runs")
    parser.add_argument("--repeats", type=int, default=10, help="Timed generations per setup, after 2 warm-up ones")
    parser.add_argument("--data", type=Path, default=ROOT / "data/test", help="The test set, with its codecs.zip files")
    parser.add_argument("--results", type=Path, default=ROOT / "runs/final/mlx/benchmark.json")
    args = parser.parse_args()

    setups = load_setups(args.data)
    results = json.loads(args.results.read_text()) if args.results.exists() else {}
    if args.omnivoice:
        results[f"omnivoice_{args.device}"] = bench_omnivoice(setups, args.repeats, args.device)
    for pte in args.pte:
        results[pte.stem] = bench_echo(pte, setups, args.repeats, args.threads)

    args.results.parent.mkdir(parents=True, exist_ok=True)
    args.results.write_text(json.dumps(results, indent=2))
    print_table(results)


if __name__ == "__main__":
    main()
