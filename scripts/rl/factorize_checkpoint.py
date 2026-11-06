"""Converts an RL checkpoint saved with a dense text embedding (every RL model before 2026-11-05) to the deployed
layout of its original pre-RL checkpoint, so echo.build_model, the export scripts and RL training can load it.

    PYTHONPATH=. .venv/bin/python scripts/rl/factorize_checkpoint.py runs/rl/test_run <out> --original runs/final/fp32

The [V, H] embedding is refitted as a rank-r table and r -> H projection by SVD, the best rank-r fit, r being the
original's rank. RL's change to the embedding isn't of rank r, so the refit keeps only part of it; the script prints
how much. The codebook offsets come from the original, and the audio heads are uncentred by its means (the same
softmax). Every other file is copied.
"""
import argparse
import shutil
from pathlib import Path

import torch
from safetensors.torch import load_file, save_file

from echo.rl.sampler import CODEBOOKS, CODES


def main() -> None:
    parser = argparse.ArgumentParser(description="Convert a dense-embedding RL checkpoint to the deployed layout.")
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("out", type=Path)
    parser.add_argument("--original", type=Path, required=True, help="The pre-RL checkpoint the RL chain started from")
    args = parser.parse_args()

    weights = load_file(args.checkpoint / "model.safetensors")
    original = load_file(args.original / "model.safetensors")
    if "llm.embed_tokens.weight" not in weights:
        raise SystemExit(f"{args.checkpoint} keeps its text embedding factorized already")
    dense = weights.pop("llm.embed_tokens.weight").double()
    table, project = original["llm.embed_tokens.table.weight"].double(), original["llm.embed_tokens.project.weight"].double()
    rank = table.size(1)
    u, s, vh = torch.linalg.svd(dense, full_matrices=False)
    weights["llm.embed_tokens.table.weight"] = (u[:, :rank] * s[:rank]).float().contiguous()
    weights["llm.embed_tokens.project.weight"] = vh[:rank].T.float().contiguous()
    weights["codebook_layer_offsets"] = original["codebook_layer_offsets"]
    heads = weights["audio_heads.weight"].view(CODEBOOKS, CODES, -1)
    means = original["audio_heads.weight"].view(CODEBOOKS, CODES, -1).mean(1, keepdim=True)
    weights["audio_heads.weight"] = (heads + means).view(CODEBOOKS * CODES, -1).contiguous()

    args.out.mkdir(parents=True, exist_ok=True)
    save_file(weights, args.out / "model.safetensors")
    for f in args.checkpoint.iterdir():
        if f.is_file() and f.name != "model.safetensors":
            shutil.copy(f, args.out / f.name)
    refit = weights["llm.embed_tokens.table.weight"].double() @ weights["llm.embed_tokens.project.weight"].double().T
    change = dense - table @ project.T
    print(f"{args.out}: text embedding refitted to rank {rank}; it keeps {1 - float((dense - refit).norm() / change.norm()):.1%} "
          f"of RL's change to the embedding (a change of {float(change.norm() / dense.norm()):.2%} of its size)", flush=True)


if __name__ == "__main__":
    main()
