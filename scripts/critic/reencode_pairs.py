"""Round-trips the takes of a generate_pairs.py output through the Higgs codec, decode then encode, into a sibling
directory. Re-encoded codes are what codes from real audio look like, so a critic trained on them can't tell the
samplers apart by token statistics that the audio doesn't carry.

    cd /workspace/Echo && /venv/main/bin/python scripts/critic/reencode_pairs.py /workspace/pairs/s8_s32 [--workers 4]

Writes <dir>_reencoded/codes/<same names>.npy (int16 [8, frames]) and a copy of pairs.csv; takes already on disk
are skipped. Each take goes through the codec alone, as the dataset's codecs were encoded: in a batch the codec's
convolutions round differently, and 3-5% of the tokens change. Batch-1 calls leave the GPU mostly idle, so
--workers processes share it, each with its own slice of the takes. Run it on CUDA, since CPU and MPS encode 3-10%
different tokens too.
"""
import argparse
import multiprocessing
import shutil
import time
from pathlib import Path

import numpy as np
import torch
from transformers import HiggsAudioV2TokenizerModel

CODEC = "eustlb/higgs-audio-v2-tokenizer"


def reencode(worker: int, paths: list[Path], out: Path, total: int) -> None:
    torch.set_grad_enabled(False)
    codec = HiggsAudioV2TokenizerModel.from_pretrained(CODEC).cuda().eval()
    start = time.time()
    for done, path in enumerate(paths, 1):
        codes = torch.from_numpy(np.load(path).astype(np.int64))[None].cuda()
        again = codec.encode(codec.decode(codes).audio_values).audio_codes[0]
        assert again.shape == codes.shape[1:], (path, again.shape)
        np.save(out / "codes" / path.name, again.short().cpu().numpy())
        if worker == 0 and (done % 500 == 0 or done == len(paths)):  # worker 0 speaks for all, which keep its pace
            rate = done * total / len(paths) / (time.time() - start)
            print(f"about {done * total // len(paths)}/{total} takes, {rate:.0f} takes/s", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description="Decode and re-encode generated takes.")
    parser.add_argument("pairs", type=Path, help="A generate_pairs.py output directory.")
    parser.add_argument("--workers", type=int, default=4, help="Processes sharing the GPU.")
    args = parser.parse_args()

    out = args.pairs.with_name(args.pairs.name + "_reencoded")
    (out / "codes").mkdir(parents=True, exist_ok=True)
    shutil.copy(args.pairs / "pairs.csv", out / "pairs.csv")
    paths = [p for p in sorted((args.pairs / "codes").glob("*.npy")) if not (out / "codes" / p.name).exists()]
    context = multiprocessing.get_context("spawn")  # CUDA can't be used in forked children
    workers = [context.Process(target=reencode, args=(i, paths[i:: args.workers], out, len(paths)))
               for i in range(args.workers)]
    for w in workers:
        w.start()
    for w in workers:
        w.join()
    failed = [w.exitcode for w in workers if w.exitcode]
    print(f"DONE: {len(paths)} takes re-encoded into {out}" if not failed else f"FAILED: worker exit codes {failed}",
          flush=True)


if __name__ == "__main__":
    main()
