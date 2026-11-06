"""
The runtime notebook's sampler (scripts/runtime/voice_cloning.ipynb), batched: main passes over codebooks 0-1, then
two polish passes over codebooks 2-7. RL generates, trains and evaluates on its s8 form without remasking.

    - one prefill per batch; every pass runs over the target frames only, against the cached prefix;
    - padded keys are masked out and padded slots are never committed;
    - each take draws its Gumbel noise from its own generator, seeded with the take's seed. Its bf16 rounding still
      depends on the batch: regenerated in other batches, 78 of 96 takes changed (2026-11-05), so a take is
      reproducible only from the same batches.
"""

import json
import math
import time
import unicodedata

from pathlib import Path
from typing import Dict, List

import numpy as np
import torch

from safetensors import safe_open
from safetensors.torch import load_file
from transformers import DynamicCache, Qwen3Config, Qwen3Model

from echo.model.model import FactorizedEmbedding

MASK, CODEBOOKS, CODES = 1024, 8, 1025

# The runtime notebook's sampler settings.
T_SHIFT, CODEBOOK_PENALTY, POSITION_TEMPERATURE = 0.1, 5.0, 5.0
POLISH_BIAS, POLISH_GUIDANCE = 1.0, 0.5


class Echo(torch.nn.Module):
    """
    A checkpoint's weights, loaded as the runtime notebook loads them, except that a factorized text embedding stays
    factorized, so RL trains the form that is deployed. checkpoint_weights() saves a trained model back.
    """

    def __init__(self, checkpoint: Path) -> None:
        super().__init__()
        config = json.loads((checkpoint / "config.json").read_text())
        weights = load_file(checkpoint / "model.safetensors")
        self.llm = Qwen3Model(Qwen3Config.from_dict({**config["llm_config"], "attn_implementation": "sdpa"}))
        hidden = self.llm.config.hidden_size
        if "llm.embed_tokens.table.weight" in weights:
            table = weights["llm.embed_tokens.table.weight"]
            self.llm.embed_tokens = FactorizedEmbedding(table.size(0), hidden, table.size(1))
        self.audio_embeddings = torch.nn.Embedding(CODEBOOKS * CODES, hidden)
        self.audio_heads = torch.nn.Linear(hidden, CODEBOOKS * CODES, bias=False)
        heads = weights["audio_heads.weight"].view(CODEBOOKS, CODES, hidden)  # centred: same softmax, bf16-safe logits
        weights["audio_heads.weight"] = (heads - heads.mean(1, keepdim=True)).view(CODEBOOKS * CODES, hidden)
        weights.pop("codebook_layer_offsets", None)
        self.load_state_dict(weights)


def is_factorized(checkpoint: Path) -> bool:
    """
    Whether a checkpoint keeps its text embedding factorized, as deployed; RL models saved before 2026-11-05 don't
    (scripts/rl/factorize_checkpoint.py converts them).
    """

    with safe_open(checkpoint / "model.safetensors", "pt") as f:
        return "llm.embed_tokens.table.weight" in f.keys()


def checkpoint_weights(model: Echo, start: Path) -> Dict[str, torch.Tensor]:
    """
    A model's weights in the layout of the checkpoint it started from: its audio heads uncentred by the start's means
    (the same softmax), and the start's codebook offsets.
    """

    original = load_file(start / "model.safetensors")
    weights = {k: v.detach().cpu().contiguous() for k, v in model.state_dict().items()}
    heads, means = weights["audio_heads.weight"], original["audio_heads.weight"].view(CODEBOOKS, CODES, -1).mean(1, keepdim=True)
    weights["audio_heads.weight"] = (heads.view(CODEBOOKS, CODES, -1) + means.to(heads.dtype)).view_as(heads).contiguous()
    if "codebook_layer_offsets" in original:
        weights["codebook_layer_offsets"] = original["codebook_layer_offsets"]
    return weights


def estimate_frames(text: str, reference_text: str, reference_frames: int) -> int:
    """
    The notebook's target length: the text's spoken weight at the reference's rate.
    """

    def weight(string: str) -> float:
        kinds = (unicodedata.category(char)[0] for char in string)
        return sum({"Z": 0.2, "M": 0.0, "P": 0.5, "S": 0.5, "N": 3.5}.get(kind, 1.0) for kind in kinds)

    frames = weight(text) / (weight(reference_text) / reference_frames)
    if frames < 50:
        frames = 50 * (frames / 50) ** (1 / 3)
    return max(1, int(frames))


class Batch:
    """
    Prompts padded together: prefix [B, P] on the right, target [B, F] on the right.
    """

    def __init__(self, model: Echo, prompts: List[dict], device: str) -> None:
        self.model, self.device, self.n = model, device, len(prompts)
        dtype = model.audio_heads.weight.dtype
        embed = model.llm.embed_tokens
        offsets = CODES * torch.arange(CODEBOOKS, device=device)[:, None]
        prefixes = [torch.cat([embed(p["ids"].to(device)), model.audio_embeddings(p["reference"].to(device) + offsets).sum(0)])
                    for p in prompts]
        self.lengths = torch.tensor([x.size(0) for x in prefixes], device=device)
        self.reference_frames = torch.tensor([p["reference"].size(1) for p in prompts], device=device)
        self.frames = [p["frames"] for p in prompts]
        self.P, self.F = int(self.lengths.max()), max(self.frames)
        frames = torch.tensor(self.frames, device=device)
        prefix = torch.nn.utils.rnn.pad_sequence(prefixes, batch_first=True)

        keys = torch.arange(self.P, device=device)
        valid = keys[None] < self.lengths[:, None]                                                   # [B, P]
        mask = torch.where(valid[:, None, None], 0.0, -math.inf).expand(-1, 1, self.P, -1).clone()
        mask[..., 0] = 0.0  # padded queries need one key, or their softmax is NaN
        self.cache = DynamicCache()
        model.llm(inputs_embeds=prefix, attention_mask=mask.to(dtype), past_key_values=self.cache,
                  position_ids=keys[None].expand(self.n, -1))

        slots = torch.arange(self.F, device=device)
        self.target = slots[None] < frames[:, None]                                                  # [B, F]
        keys = torch.cat([valid, self.target], 1)                                                    # [B, P + F]
        plain = torch.where(keys[:, None, None], 0.0, -math.inf).expand(-1, 1, self.F, -1).clone()
        reference = (torch.arange(self.P, device=device)[None] >= (self.lengths - self.reference_frames)[:, None]) & valid
        polish = plain.clone()
        polish[..., : self.P] += POLISH_BIAS * reference[:, None, None].to(polish.dtype)
        self.masks = {0.0: plain.to(dtype), POLISH_BIAS: polish.to(dtype)}
        self.positions = self.lengths[:, None] + slots[None]

    def log_probs(self, tokens: torch.Tensor, codebooks: range, bias: float = 0.0) -> torch.Tensor:
        """
        One pass over the targets: [B, 8, F] tokens -> [B, len(codebooks), F, 1024] fp32 log-probabilities.
        """

        model = self.model
        offsets = CODES * torch.arange(CODEBOOKS, device=self.device)[None, :, None]
        inputs = model.audio_embeddings(tokens + offsets).sum(1)                                     # [B, F, hidden]
        hidden = model.llm(inputs_embeds=inputs, attention_mask=self.masks[bias], position_ids=self.positions,
                           past_key_values=self.cache).last_hidden_state
        self.cache.crop(-self.F)  # drop the pass's target keys and values; keep the prefix's
        weight = model.audio_heads.weight[codebooks.start * CODES: codebooks.stop * CODES]
        logits = (hidden @ weight.T).float().view(self.n, self.F, len(codebooks), CODES).transpose(1, 2)
        return logits.log_softmax(-1)[..., :MASK]


def rank(score: torch.Tensor, descending: bool) -> torch.Tensor:
    """
    Each entry's rank within its row.
    """

    order = score.argsort(-1, descending=descending)
    return torch.empty_like(order).scatter_(-1, order, torch.arange(score.size(-1), device=score.device).expand_as(order))


def generate(batch: Batch, seeds: List[int], steps: int = 8, remask: float = 0.0) -> List[np.ndarray]:
    """
    The notebook's generate() for a whole batch: `steps` main passes over codebooks 0-1, then 2 polish passes.
    Returns one int16 [8, frames] array per take.
    """

    B, F, device = batch.n, batch.F, batch.device
    tokens = torch.full((B, CODEBOOKS, F), MASK, device=device)
    open_ = batch.target[:, None].expand(-1, 2, -1)                                                  # [B, 2, F]
    penalty = CODEBOOK_PENALTY * torch.arange(2, device=device)[None, :, None]
    generators = [torch.Generator(device).manual_seed(s) for s in seeds]
    t = torch.linspace(0, 1, steps + 1)
    t = (T_SHIFT * t / (1 - (1 - T_SHIFT) * t)).tolist()
    for step in range(steps):
        confidence, prediction = batch.log_probs(tokens, range(0, 2)).max(-1)                        # [B, 2, F]
        main = tokens[:, :2]
        masked, revealed = (main == MASK) & open_, (main != MASK) & open_
        last = step == steps - 1
        if last:
            k = masked.flatten(1).sum(1)
        else:
            k = torch.tensor([math.ceil(2 * f * (t[step + 1] - t[step])) for f in batch.frames], device=device)
        noise = torch.stack([torch.nn.functional.pad(torch.rand((CODEBOOKS, f), generator=g, device=device)[:2], (0, F - f))
                             for g, f in zip(generators, batch.frames)])
        gumbel = -torch.log(-torch.log(noise + 1e-10) + 1e-10)
        score = ((confidence - penalty) / POSITION_TEMPERATURE + gumbel).masked_fill(~masked, -math.inf).flatten(1)
        chosen = ((rank(score, True) < k[:, None]) & score.isfinite()).view_as(masked)
        main = torch.where(chosen, prediction, main)
        if not last and remask:
            count = torch.minimum((remask * k.double()).round().long(), revealed.flatten(1).sum(1))  # round() as Python's
            calm = confidence.masked_fill(~revealed, math.inf).flatten(1)
            main = main.masked_fill(((rank(calm, False) < count[:, None]).view_as(revealed) & revealed), MASK)
        tokens[:, :2] = main
    first = batch.log_probs(tokens, range(2, 8), POLISH_BIAS)
    tokens[:, 2:4] = first[:, :2].argmax(-1)
    second = batch.log_probs(tokens, range(4, 8), POLISH_BIAS)
    tokens[:, 4:] = (second + POLISH_GUIDANCE * (second - first[:, 2:])).argmax(-1)
    return [tokens[i, :, :f].short().cpu().numpy() for i, f in enumerate(batch.frames)]


def pack(jobs: List[dict], batch_frames: int = 16384, batch_tokens: int = 65536) -> List[List[dict]]:
    """
    Takes to generate ({ids, reference, frames, ...}) cut into batches that pad little: sorted by target, then prefix
    length, at most `batch_frames` target frames and `batch_tokens` prefix + target tokens per batch, padding included.
    """

    jobs = sorted(jobs, key=lambda j: (j["frames"], j["ids"].numel() + j["reference"].size(1)))
    batches, current, prefix = [], [], 0
    for j in jobs:  # sorted by frames, so a batch's longest target is its last
        length = j["ids"].numel() + j["reference"].size(1)
        n, longest = len(current) + 1, max(prefix, length)
        if current and (n * j["frames"] > batch_frames or n * (longest + j["frames"]) > batch_tokens):
            batches.append(current)
            current, longest = [], length
        current.append(j)
        prefix = longest
    return batches + ([current] if current else [])


def write_takes(model: Echo, batches: List[List[dict]], out: Path) -> None:
    """
    Generates every batch of takes ({key, seed, ...}) whose takes aren't all in `out` yet, as <out>/<key>.npy. A
    batch is regenerated whole, so a take doesn't depend on where a run stopped.
    """

    out.mkdir(parents=True, exist_ok=True)
    todo = [b for b in batches if not all((out / f"{j['key']}.npy").exists() for j in b)]
    total, done, start = sum(len(b) for b in todo), 0, time.time()
    with torch.no_grad():
        for b in todo:
            for j, codes in zip(b, generate(Batch(model, b, "cuda"), [j["seed"] for j in b])):
                np.save(out / f"{j['key']}.npy", codes)
            done += len(b)
            print(f"  generated {done}/{total} takes, {done / (time.time() - start):.1f} takes/s", flush=True)


def load_model(checkpoint: Path) -> Echo:
    """
    A checkpoint ready to generate: bf16 on CUDA, as RL always generates.
    """

    return Echo(checkpoint).to("cuda", torch.bfloat16).eval()


def take_seeds(prompt_id: str, takes: int, offset: int) -> List[int]:
    """
    The seeds of a prompt's takes: 1000 x the prompt's number + the take's index + `offset`.
    """

    return [1000 * int(prompt_id[1:]) + k + offset for k in range(takes)]


def jobs_for(prompts: List[Dict], seeds: Dict[str, List[int]]) -> List[dict]:
    """
    One job per (prompt, seed): the prompt's model inputs, its seed and its take key <prompt id>_<seed>.
    """

    return [dict(p, key=f"{p['id']}_{s}", seed=s) for p in prompts for s in seeds[p["id"]]]
