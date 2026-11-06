"""Quantizes the weights of the Higgs Audio v2 codec's encode path (audio -> codes) and measures what it costs against
the reference encode: fp32 on CUDA, with cuDNN's TF32 off.

    PYTHONPATH=. /venv/main/bin/python scripts/export/quantize_encoder.py --config '{"name": "int8", "bits": 8}' ...

Each --config is a JSON object over DEFAULTS, run in turn on the 57 test references and the dev clips. The encode path is
HuBERT (semantic_model), the DAC encoder (acoustic_encoder), encoder_semantic, fc and the residual quantizer, 163M
parameters; the decoder's modules are dropped. Quantization is simulated: each quantized weight is rounded to its grid
and put back as a float, so a run computes in `dtype` with the values an int8 or int4 checkpoint would hold.

Reported per split, against the reference: token agreement per codebook; the first codebook that differs in each frame;
how near a decision boundary the first flips sat (the median rank of their margin among all frames' margins in that
codebook: a low rank means near a boundary); the SNR of each sub-module's output and of the pre-quantizer embedding (fc),
mean and min over clips; and the bytes of the encode path's weights, scales included. Every run is appended to --results.

--save writes each run's weights as a checkpoint (int8, or int4 packed two per byte, with fp16 scales, and
quantization.json), and checks that it encodes the test set as the run did; load() rebuilds the encode path from one.
"""

import argparse
import copy
import csv
import hashlib
import json
import math
import random
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torchaudio
from safetensors.torch import load_file, save_file
from transformers import AutoFeatureExtractor, HiggsAudioV2TokenizerModel

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from echo.model.quantization import _pack, _unpack  # noqa: E402
from scripts.__style__ import print_info, print_section, print_test_title  # noqa: E402
from scripts.normalization.s3_ctc_rescoring import decoded  # noqa: E402
from scripts.normalization.s8_codecs import MODEL, RESAMPLE  # noqa: E402

CODEBOOKS = 8
DECODE_ONLY = ("acoustic_decoder", "decoder_semantic", "fc1", "fc2")
NETWORK = ("semantic_model", "acoustic_encoder", "encoder_semantic", "fc")
SIGNALS = ("hubert", "semantic", "acoustic", "embedding")
DTYPES = {"fp32": torch.float32, "fp16": torch.float16, "bf16": torch.bfloat16}
DEFAULTS: dict[str, Any] = {
    "name": "reference",
    "init": None,  # a checkpoint written by --save or by qat_encoder.py, scored as it is: no quantization keys apply
    "device": "cuda" if torch.cuda.is_available() else "cpu",
    "tf32": False,  # cuDNN convolutions in TF32, torch's default on CUDA; off, the test set's stored codes come out
    "dtype": "fp32",  # the network's weights and activations
    "rvq": "fp32",  # the residual quantizer's weights and distances
    "storage": "fp32",  # every float the quantization leaves is rounded to this
    "snake": None,  # "rewrite" forces the fp16-safe Snake, which is otherwise used only when dtype isn't fp32
    "bits": None,  # 8 or 4: the width of the quantized weights
    "modules": list(NETWORK),  # prefixes of the modules whose Linear and Conv1d weights are quantized
    "keep": [],  # prefixes left out of those
    "overrides": [],  # [prefix, {key: value}] pairs: other settings for the weights under a prefix; later ones win
    "granularity": "channel",  # channel | group: group_size inputs x every kernel tap | tap: group_size inputs of one tap
    "group_size": None,
    "symmetric": True,
    "method": "rtn",  # rtn: round to nearest | gptq: rounding that compensates each error, from calibration clips
    "damp": 0.01,  # GPTQ's dampening, a share of the mean diagonal of x x^T
    "watch": False,  # records the largest |output| of every module, and the modules that overflow
}


class Snake(torch.nn.Module):
    """
    Snake1d, x + sin(a x)^2 / a, as x + sign(a) (sin(a x) / sqrt|a|)^2, as in export_codec.py: 1/a overflows fp16 for the
    encoder's 5 alphas below 1.5e-5 (the smallest is 2.8e-6), while 1/sqrt|a| stays below 600.
    """

    def __init__(self, alpha: torch.Tensor) -> None:
        super().__init__()
        self.register_buffer("alpha", alpha)
        self.register_buffer("rsqrt", (alpha.abs() + 1e-9).rsqrt())
        self.register_buffer("sign", alpha.sign())

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        r = torch.sin(self.alpha * x) * self.rsqrt
        return x + self.sign * (r * r)


def under(name: str, prefixes: list[str]) -> bool:
    return any(name == p or name.startswith(p + ".") for p in prefixes)


def load_codec() -> HiggsAudioV2TokenizerModel:
    """The codec's encode path in fp32 on the CPU, with the positional conv's weight norm folded into its weight."""
    model = HiggsAudioV2TokenizerModel.from_pretrained(MODEL).eval()
    for name in DECODE_ONLY:
        delattr(model, name)
    torch.nn.utils.parametrize.remove_parametrizations(model.semantic_model.encoder.pos_conv_embed.conv, "weight")
    return model


def encode_tensors(model: HiggsAudioV2TokenizerModel) -> dict[str, torch.Tensor]:
    """Every weight encode reads: the parameters and the codebooks, not the quantizer's training buffers."""
    tensors = dict(model.named_parameters())
    tensors.update({n: b for n, b in model.named_buffers() if n.endswith("codebook.embed")})
    return tensors


def manifest_rows(manifest: Path) -> list[dict[str, str]]:
    with open(manifest, newline="") as f:
        return list(csv.DictReader(f, delimiter="|", quoting=csv.QUOTE_NONE))


def load_clips(manifest: Path, rows: list[dict[str, str]], extractor: Any) -> list[dict[str, Any]]:
    """Clips of a manifest of codec paths, with their stored codes and their audio prepared as s8_codecs.py does."""
    paths = [manifest.parent / row["name"] for row in rows]
    clips = []
    for path, row, result in zip(paths, rows, decoded([(p.parent.parent, p.stem) for p in paths])):
        if isinstance(result, Exception):
            raise result
        y, sr, _ = result
        audio = torch.as_tensor(y, dtype=torch.float32)
        if sr != extractor.sampling_rate:
            audio = torchaudio.functional.resample(audio, sr, extractor.sampling_rate, **RESAMPLE)
        inputs = extractor(raw_audio=audio.numpy(), sampling_rate=extractor.sampling_rate, return_tensors="pt")
        clips.append({"name": row["name"], "stored": np.load(path), "input": inputs["input_values"]})
    return clips


def layout(weight: torch.Tensor, config: dict[str, Any]) -> tuple[torch.Tensor, int]:
    """
    The weight as [out, columns], in the order its groups run, and how many columns a group spans. A group never spans
    more than group_size inputs: where the inputs don't divide by it (HuBERT's positional conv has 48 per conv group),
    it shrinks to the gcd.
    """
    w = weight.float()
    w = w[:, :, None] if w.dim() == 2 else w
    out, inputs, taps = w.shape
    if config["granularity"] == "channel":
        return w.reshape(out, -1), inputs * taps
    size = math.gcd(inputs, config["group_size"])
    if config["granularity"] == "tap":
        return w.transpose(1, 2).reshape(out, -1), size
    return w.reshape(out, -1), size * taps


def unlayout(columns: torch.Tensor, weight: torch.Tensor, config: dict[str, Any]) -> torch.Tensor:
    if config["granularity"] == "tap" and weight.dim() == 3:
        columns = columns.reshape(weight.shape[0], weight.shape[2], weight.shape[1]).transpose(1, 2)
    return columns.reshape(weight.shape).to(weight.dtype)


def grid_params(groups: torch.Tensor, config: dict[str, Any]) -> tuple[torch.Tensor, torch.Tensor, int, int]:
    """
    The fp16 scale and the zero point of each group (the last dim), and the range of the integers. An asymmetric range
    always spans 0, so a group of one weight (the waveform convs have one input) still lands on its grid.
    """
    bits = config["bits"]
    if config["symmetric"]:
        top = 2 ** (bits - 1) - 1
        scale = (groups.abs().amax(-1, keepdim=True) / top).half().float().clamp(min=2**-24)
        return scale, torch.zeros_like(scale), -top, top
    top = 2**bits - 1
    low, high = groups.amin(-1, keepdim=True).clamp(max=0), groups.amax(-1, keepdim=True).clamp(min=0)
    scale = ((high - low) / top).half().float().clamp(min=2**-24)
    return scale, (-low / scale).round().clamp(0, top), 0, top


def to_ints(w: torch.Tensor, scale: torch.Tensor, zero: torch.Tensor, low: int, high: int) -> torch.Tensor:
    return ((w / scale).round() + zero).clamp(low, high)


def grid(weight: torch.Tensor, config: dict[str, Any]) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """
    The weight rounded to the nearest point of its grid, and what a checkpoint holds for it: the integers [out, columns]
    in the layout's column order, and each group's scale and zero point [out, groups].
    """
    columns, width = layout(weight, config)
    groups = columns.reshape(columns.shape[0], -1, width)
    scale, zero, low, high = grid_params(groups, config)
    ints = to_ints(groups, scale, zero, low, high)
    stored = {"ints": ints.reshape(columns.shape), "scale": scale[..., 0], "zero": zero[..., 0]}
    return unlayout(((ints - zero) * scale).reshape(columns.shape), weight, config), stored


def unfold(padded: torch.Tensor, conv: torch.nn.Conv1d, start: int, stop: int) -> torch.Tensor:
    """
    The columns a conv's weight multiplies for output positions [start, stop) of its padded input: [in x taps,
    batch x positions].
    """
    k, d, s = conv.kernel_size[0], conv.dilation[0], conv.stride[0]
    x = padded[..., start * s:(stop - 1) * s + d * (k - 1) + 1].unfold(2, d * (k - 1) + 1, s)[..., ::d]
    return x.permute(1, 3, 0, 2).reshape(x.shape[1] * k, -1)


@torch.inference_mode()
def hessians(model: HiggsAudioV2TokenizerModel, layers: dict[str, torch.nn.Module], clips: list[dict[str, Any]],
             device: str) -> dict[str, torch.Tensor]:
    """
    The mean x x^T of each layer's input over the calibration clips; a conv's input unfolded to its columns. In fp64:
    neighbouring taps at 24 kHz are so alike that fp32 sums leave eigenvalues below zero, deeper than GPTQ's dampening.
    """
    sums: dict[str, torch.Tensor] = {}
    counts: dict[str, int] = {}

    def hook(name: str) -> Any:
        def accumulate(module: torch.nn.Module, args: tuple, output: torch.Tensor) -> None:
            x = args[0]
            if isinstance(module, torch.nn.Conv1d):
                padded, positions = torch.nn.functional.pad(x, (module.padding[0],) * 2), output.shape[-1]
                chunks = (unfold(padded, module, a, min(a + 16384, positions)) for a in range(0, positions, 16384))
            else:
                chunks = iter([x.reshape(-1, x.shape[-1]).T])
            for chunk in chunks:
                chunk = chunk.double()
                if name not in sums:
                    sums[name], counts[name] = chunk.new_zeros(chunk.shape[0], chunk.shape[0]), 0
                sums[name].addmm_(chunk, chunk.T)
                counts[name] += chunk.shape[1]
        return accumulate

    handles = [module.register_forward_hook(hook(name)) for name, module in layers.items()]
    for clip in clips:
        model.encode(clip["input"].to(device))
    for handle in handles:
        handle.remove()
    # Parked on the CPU: the DAC's take 2.4 GB, and GPTQ needs only one at a time.
    return {name: sums[name].div_(counts[name]).cpu() for name in sums}


def gptq(weight: torch.Tensor, hessian: torch.Tensor, config: dict[str, Any]) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """
    GPTQ: the columns are rounded one by one, and each one's error is spread over the columns not yet rounded, as the
    calibration inputs' x x^T says it is best compensated. A group's scale is taken when its first column comes up.
    Returns what grid() does; `hessian` is dampened in place.
    """
    columns, width = layout(weight, config)
    W, H = columns.double(), hessian
    if config["granularity"] == "tap" and weight.dim() == 3:
        order = torch.arange(H.shape[0], device=H.device).reshape(weight.shape[1], weight.shape[2]).T.flatten()
        H = H[order][:, order]
    dead = H.diagonal() == 0
    H[dead, dead] = 1
    W[:, dead] = 0
    H.diagonal().add_(config["damp"] * H.diagonal().mean())
    U = torch.linalg.cholesky(torch.cholesky_inverse(torch.linalg.cholesky(H)), upper=True)
    Q, ints = torch.zeros_like(W), torch.zeros_like(W)
    scales, zeros = W.new_zeros(W.shape[0], W.shape[1] // width), W.new_zeros(W.shape[0], W.shape[1] // width)
    params = grid_params(W[:, None, :], config) if config["granularity"] == "channel" else None
    if params:
        scales[:, 0], zeros[:, 0] = params[0].reshape(-1), params[1].reshape(-1)
    block = 128 if params else width * max(1, 128 // width)
    for b0 in range(0, W.shape[1], block):
        b1 = min(b0 + block, W.shape[1])
        Wb, errors = W[:, b0:b1].clone(), torch.zeros_like(W[:, b0:b1])
        for i in range(b1 - b0):
            if config["granularity"] != "channel" and i % width == 0:
                params = grid_params(Wb[:, None, i:i + width], config)
                scales[:, (b0 + i) // width], zeros[:, (b0 + i) // width] = params[0].reshape(-1), params[1].reshape(-1)
            scale, zero, low, high = params
            n = to_ints(Wb[:, i], scale.reshape(-1), zero.reshape(-1), low, high)
            q = (n - zero.reshape(-1)) * scale.reshape(-1)
            errors[:, i] = (Wb[:, i] - q) / U[b0 + i, b0 + i]
            Wb[:, i:] -= errors[:, i:i + 1] * U[b0 + i, b0 + i:b1][None]
            Q[:, b0 + i], ints[:, b0 + i] = q, n
        W[:, b1:] -= errors @ U[b0:b1, b1:]
    return unlayout(Q, weight, config), {"ints": ints, "scale": scales, "zero": zeros}


def quantized_weights(model: HiggsAudioV2TokenizerModel, config: dict[str, Any]) -> dict[str, tuple[torch.nn.Module, dict]]:
    """Each quantized weight's module and its own settings: the config, with the overrides that cover it applied."""
    layers = {}
    for name, module in model.named_modules():
        if not isinstance(module, (torch.nn.Linear, torch.nn.Conv1d)):
            continue
        if not under(name, config["modules"]) or under(name, config["keep"]):
            continue
        settings = dict(config)
        for prefix, values in config["overrides"]:
            if under(name, [prefix]):
                settings.update(values)
        if settings["bits"]:
            layers[f"{name}.weight"] = (module, settings)
    return layers


def uses_gptq(config: dict[str, Any]) -> bool:
    return config["method"] == "gptq" or any(values.get("method") == "gptq" for _, values in config["overrides"])


def size(model: HiggsAudioV2TokenizerModel, config: dict[str, Any]) -> int:
    """Bytes of the encode path's weights: quantized ones at `bits` plus an fp16 scale (and a zero point) per group."""
    if config["init"]:
        return sum(t.numel() * t.element_size() for t in load_file(Path(config["init"]) / "encoder.safetensors").values())
    quantized = quantized_weights(model, config)
    total = 0.0
    for name, tensor in encode_tensors(model).items():
        if name in quantized:
            s = quantized[name][1]
            scales = grid(tensor, s)[1]["scale"].numel()
            total += tensor.numel() * s["bits"] / 8 + scales * (2 + (0 if s["symmetric"] else s["bits"] / 8))
        else:
            compute = config["rvq"] if name.startswith("quantizer.") else config["dtype"]
            total += tensor.numel() * min(DTYPES[config["storage"]].itemsize, DTYPES[compute].itemsize)
    return int(total)


def _cast(module: torch.nn.Module, args: tuple) -> tuple:
    dtype = next(module.parameters()).dtype
    return tuple(a.to(dtype) if torch.is_tensor(a) and a.is_floating_point() else a for a in args)


def build(base: HiggsAudioV2TokenizerModel, config: dict[str, Any],
          calibration: list[dict[str, Any]]) -> tuple[HiggsAudioV2TokenizerModel, dict[str, dict]]:
    """The encode path as the config computes it, and what a checkpoint of its quantized weights holds."""
    model, stored = copy.deepcopy(load(Path(config["init"])) if config["init"] else base).to(config["device"]), {}
    # A checkpoint's weights are on their grid and at their storage width already.
    quantized = {} if config["init"] else quantized_weights(model, config)
    # One sub-module at a time, in the order the data flows, so encoder_semantic and fc are calibrated on the rounded
    # outputs before them. GPTQ needs a conv's columns to match its input's: the grouped positional conv is rounded.
    for part in NETWORK:
        layers = {n: ms for n, ms in quantized.items() if under(n, [part])}
        exact = {n: m for n, (m, s) in layers.items() if s["method"] == "gptq" and getattr(m, "groups", 1) == 1}
        start = time.time()
        stats = hessians(model, exact, calibration, config["device"]) if exact else {}
        with torch.no_grad():
            for name, (module, s) in layers.items():
                if name in exact:
                    values, held = gptq(module.weight, stats.pop(name).to(config["device"], copy=True), s)
                else:
                    values, held = grid(module.weight, s)
                module.weight.copy_(values)
                stored[name] = {"ints": held["ints"].short().cpu(), "scale": held["scale"].cpu(),
                                "zero": held["zero"].cpu(), "settings": s}
        if exact:
            print(f"    GPTQ on {part}: {len(exact)} layers, {time.time() - start:.0f} s", flush=True)
    with torch.no_grad():
        storage = DTYPES[config["storage"]]
        for name, tensor in encode_tensors(model).items():
            if name not in quantized and not config["init"]:
                tensor.copy_(tensor.to(storage))
    if config["snake"] == "rewrite" or (config["snake"] is None and config["dtype"] != "fp32"):
        for module in list(model.acoustic_encoder.modules()):
            for name, child in module.named_children():
                if type(child).__name__ == "Snake1d":
                    setattr(module, name, Snake(child.alpha.detach()))
    for name in NETWORK:
        getattr(model, name).to(DTYPES[config["dtype"]]).register_forward_pre_hook(_cast)
    model.quantizer.to(DTYPES[config["rvq"]])
    for quantizer in model.quantizer.quantizers:
        quantizer.project_in.register_forward_pre_hook(_cast)
        quantizer.project_out.register_forward_pre_hook(_cast)
    return model, stored


def save(base: HiggsAudioV2TokenizerModel, config: dict[str, Any], stored: dict[str, dict], directory: Path) -> None:
    """
    The encode path's weights as a checkpoint: each quantized weight as int8 (int4 packed two per byte, the first in the
    low nibble) in its layout's column order, with an fp16 scale (and a uint8 zero point) per group; every other float
    at `storage`. load() reads it back.
    """
    directory.mkdir(parents=True, exist_ok=True)
    state, weights = {}, {}
    for name, tensor in encode_tensors(base).items():
        if name not in stored:
            state[name] = tensor.detach().to(DTYPES[config["storage"]]).contiguous()
            continue
        held, s = stored[name], stored[name]["settings"]
        if s["bits"] == 4:
            state[name] = _pack(held["ints"].flatten().float() - (0 if s["symmetric"] else 8))
        else:
            state[name] = held["ints"].to(torch.int8 if s["symmetric"] else torch.uint8)
        state[f"{name}_scale"] = held["scale"].half()
        if not s["symmetric"]:
            state[f"{name}_zero"] = held["zero"].to(torch.uint8)
        weights[name] = {k: s[k] for k in ("bits", "granularity", "symmetric")} | {"shape": list(tensor.shape)}
    save_file(state, directory / "encoder.safetensors", metadata={"format": "pt"})
    (directory / "quantization.json").write_text(json.dumps({"config": config, "weights": weights}, indent=1))


def load(directory: Path) -> HiggsAudioV2TokenizerModel:
    """An encode path written by save(), its weights dequantized to fp32."""
    meta = json.loads((directory / "quantization.json").read_text())
    state = load_file(directory / "encoder.safetensors")
    for name, w in meta["weights"].items():
        scale = state.pop(f"{name}_scale").float()[..., None]
        zero = state.pop(f"{name}_zero").float()[..., None] if not w["symmetric"] else torch.zeros_like(scale)
        ints = _unpack(state[name]) + (0 if w["symmetric"] else 8) if w["bits"] == 4 else state[name].float()
        values = (ints.reshape(w["shape"][0], scale.shape[1], -1) - zero) * scale
        state[name] = unlayout(values.reshape(w["shape"][0], -1), torch.empty(w["shape"]), w)
    model = load_codec()
    missing, unexpected = model.load_state_dict({k: v.float() for k, v in state.items()}, strict=False)
    if unexpected or any(not k.endswith(("inited", "cluster_size", "embed_avg")) for k in missing):
        raise ValueError(f"{directory}: missing {missing}, unexpected {unexpected}")
    return model


@torch.inference_mode()
def run(model: HiggsAudioV2TokenizerModel, clips: list[dict[str, Any]], config: dict[str, Any]) -> tuple[list[dict], dict]:
    """Each clip's codes and sub-module outputs; with `watch`, also every module's peak |output| and the overflows."""
    captured: dict[str, torch.Tensor] = {}
    peaks: dict[str, torch.Tensor] = {}
    overflows: dict[str, torch.Tensor] = {}
    hooks = [
        model.encoder_semantic.register_forward_hook(lambda m, a, o: captured.update(hubert=a[0], semantic=o)),
        model.acoustic_encoder.register_forward_hook(lambda m, a, o: captured.update(acoustic=o)),
        model.fc.register_forward_hook(lambda m, a, o: captured.update(embedding=o)),
    ]

    def watch(name: str) -> Any:
        def hook(module: torch.nn.Module, args: tuple, output: Any) -> None:
            if torch.is_tensor(output) and output.is_floating_point():
                finite = torch.isfinite(output)
                peak, bad = output.abs().where(finite, 0).amax().float(), (~finite).any()
                peaks[name] = torch.maximum(peaks[name], peak) if name in peaks else peak
                overflows[name] = overflows[name] | bad if name in overflows else bad
        return hook

    if config["watch"]:
        hooks += [m.register_forward_hook(watch(n)) for n, m in model.named_modules() if not list(m.children())]
    outputs, start = [], time.time()
    for k, clip in enumerate(clips, 1):
        codes = model.encode(clip["input"].to(config["device"])).audio_codes[0]
        outputs.append({"codes": codes.short().cpu().numpy(), **{s: captured[s][0].float().cpu() for s in SIGNALS}})
        if k % 100 == 0 or k == len(clips):
            print(f"    {k}/{len(clips)} clips, {time.time() - start:.0f} s", flush=True)
    for hook in hooks:
        hook.remove()
    top = sorted(((n, p.item()) for n, p in peaks.items()), key=lambda item: -item[1])[:5]
    return outputs, {"peaks": [[n, round(p, 1)] for n, p in top], "overflows": sorted(n for n, b in overflows.items() if b)}


@torch.inference_mode()
def margins(quantizer: torch.nn.Module, embedding: torch.Tensor, codes: np.ndarray) -> np.ndarray:
    """
    [CODEBOOKS, T]: how far each frame's residual sits from the boundary between the entry the reference chose and the
    nearest other one, along the reference's own residuals; a flip needs a perturbation at least this large.
    """
    residual, out = embedding.double(), []
    for k, q in enumerate(quantizer.quantizers):
        h = residual @ q.project_in.weight.double().T + q.project_in.bias.double()
        e = q.codebook.embed.double()
        d = (h * h).sum(1, keepdim=True) - 2 * h @ e.T + (e * e).sum(1)
        chosen = torch.as_tensor(codes[k], dtype=torch.long)[:, None]
        own = d.gather(1, chosen)[:, 0]
        other = d.scatter(1, chosen, float("inf")).argmin(1)
        gap = (e[other] - e[chosen[:, 0]]).norm(dim=1)
        out.append(((d.gather(1, other[:, None])[:, 0] - own) / (2 * gap)).float())
        residual = residual - (e[chosen[:, 0]] @ q.project_out.weight.double().T + q.project_out.bias.double())
    return torch.stack(out).numpy()


def snr(reference: torch.Tensor, signal: torch.Tensor) -> float:
    noise = (signal.double() - reference.double()).pow(2).sum()
    return float(10 * torch.log10(reference.double().pow(2).sum() / noise)) if noise > 0 else float("inf")


def compare(reference: list[dict], outputs: list[dict], ranks: list[np.ndarray]) -> dict[str, Any]:
    agree, frames, first, flips, flips0, nonfinite = np.zeros(CODEBOOKS), 0, np.zeros(CODEBOOKS + 1), [], [], 0
    snrs: dict[str, list[float]] = {s: [] for s in SIGNALS}
    for ref, out, rank in zip(reference, outputs, ranks):
        if out["codes"].shape != ref["codes"].shape:
            raise ValueError(f"{out['codes'].shape} codes against the reference's {ref['codes'].shape}")
        differ = out["codes"] != ref["codes"]
        agree += (~differ).sum(1)
        frames += differ.shape[1]
        at = np.where(differ.any(0), differ.argmax(0), CODEBOOKS)
        first += np.bincount(at, minlength=CODEBOOKS + 1)
        t = np.where(at < CODEBOOKS)[0]
        flips += list(rank[at[t], t])
        flips0 += list(rank[0, t[at[t] == 0]])
        for s in SIGNALS:
            nonfinite += int((~torch.isfinite(out[s])).sum())
            snrs[s].append(snr(ref[s], out[s]))
    return {
        "frames": int(frames),
        "agreement": [round(float(a / frames), 5) for a in agree],
        "first": [round(float(f / frames), 5) for f in first],
        "margin_rank": round(float(np.median(flips)), 4) if flips else None,
        "margin_rank_cb0": round(float(np.median(flips0)), 4) if flips0 else None,
        "snr": {s: {"mean": round(float(np.mean(v)), 2), "min": round(float(np.min(v)), 2)} for s, v in snrs.items()},
        "nonfinite": nonfinite,
    }


def report(name: str, split: str, result: dict[str, Any]) -> None:
    a, f, s = result["agreement"], result["first"], result["snr"]
    rank = "-" if result["margin_rank"] is None else f"{100 * result['margin_rank']:.1f}%"
    print(f"  {split:4s} {name:28s} {result['size'] / 1e6:6.1f} MB | cb0 {100 * a[0]:6.2f} cb1 {100 * a[1]:6.2f} "
          f"cb2-7 {100 * np.mean(a[2:]):6.2f} | same {100 * f[-1]:5.1f}, first 0: {100 * f[0]:4.2f} 1: {100 * f[1]:4.2f} "
          f"2-7: {100 * sum(f[2:-1]):4.1f} | rank {rank} | SNR emb {s['embedding']['mean']:.1f} "
          f"(min {s['embedding']['min']:.1f}) hubert {s['hubert']['mean']:.1f} sem {s['semantic']['mean']:.1f} "
          f"ac {s['acoustic']['mean']:.1f}" + (f" | {result['nonfinite']} non-finite" if result["nonfinite"] else "")
          + (f" | stored cb0 {100 * result['stored'][0]:.2f} cb1 {100 * result['stored'][1]:.2f}" if "stored" in result else ""),
          flush=True)


def stored_agreement(clips: list[dict[str, Any]], outputs: list[dict]) -> list[float]:
    """Agreement per codebook with the codes stored next to the audio, made on another machine."""
    agree = sum((c["stored"] == o["codes"]).sum(1) for c, o in zip(clips, outputs))
    return [round(float(a), 5) for a in agree / sum(o["codes"].shape[1] for o in outputs)]


def write_codes(directory: Path, manifest: Path, clips: list[dict], outputs: list[dict]) -> None:
    """The test codes laid out like data/test, with its manifest, for scripts/evaluate_multi_step.py."""
    for clip, out in zip(clips, outputs):
        (directory / clip["name"]).parent.mkdir(parents=True, exist_ok=True)
        np.save(directory / clip["name"], out["codes"])
    (directory / manifest.name).write_text(manifest.read_text())


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--config", action="append", default=[], help="a JSON object over DEFAULTS; repeat it for more runs")
    parser.add_argument("--test", type=Path, default=Path("data/test/test_in_processed.csv"))
    parser.add_argument("--dev", type=Path, default=Path("data/dev/dev_processed.csv"))
    parser.add_argument("--calibration", type=int, default=64, help="dev clips held out for calibration, never scored")
    parser.add_argument("--dev-limit", type=int, help="scores only this many of the other dev clips")
    parser.add_argument("--results", type=Path, default=Path("runs/encoder_quantization/results.jsonl"))
    parser.add_argument("--codes", type=Path, help="writes each run's test codes to <codes>/<name>/, laid out like data/test")
    parser.add_argument("--save", type=Path, help="writes each run's weights to <save>/<name>/, and checks they encode the same")
    parser.add_argument("--memory", type=float, default=6.0, help="GB of GPU memory this process may take (shared box)")
    args = parser.parse_args()

    if torch.cuda.is_available():
        torch.cuda.set_per_process_memory_fraction(args.memory * 2**30 / torch.cuda.get_device_properties(0).total_memory)
    torch.set_num_threads(16)
    script = hashlib.sha1(Path(__file__).read_bytes()).hexdigest()[:10]
    configs = [{**DEFAULTS, **json.loads(c)} for c in args.config]

    print_test_title("Higgs encoder quantization")
    extractor, base = AutoFeatureExtractor.from_pretrained(MODEL), load_codec()
    dev = manifest_rows(args.dev)
    random.Random(0).shuffle(dev)
    held_out, dev = dev[:args.calibration], dev[args.calibration:][:args.dev_limit]
    splits = {"test": load_clips(args.test, manifest_rows(args.test), extractor), "dev": load_clips(args.dev, dev, extractor)}
    calibration = load_clips(args.dev, held_out, extractor) if any(uses_gptq(c) for c in configs) else []
    # At most 10 s each (whole frames): the DAC's 24 kHz activations, unfolded for x x^T, stay small.
    for clip in calibration:
        clip["input"] = clip["input"][..., :10 * extractor.sampling_rate]
    print_info("clips", ", ".join(f"{s} {len(c)}" for s, c in splits.items()) + f", calibration {len(held_out)} dev")
    print_info("encode path", f"{sum(t.numel() for t in encode_tensors(base).values()) / 1e6:.2f}M weights")

    print_section(f"reference: fp32 on {DEFAULTS['device']}, against the stored codes")
    torch.backends.cudnn.allow_tf32 = DEFAULTS["tf32"]
    model, _ = build(base, DEFAULTS, [])
    reference, ranks = {}, {}
    for split, clips in splits.items():
        reference[split], _ = run(model, clips, DEFAULTS)
        raw = [margins(base.quantizer, r["embedding"], r["codes"]) for r in reference[split]]
        pooled = [np.sort(np.concatenate([m[k] for m in raw])) for k in range(CODEBOOKS)]
        ranks[split] = [np.stack([np.searchsorted(pooled[k], m[k]) / len(pooled[k]) for k in range(CODEBOOKS)]) for m in raw]
        same = [i for i, (c, r) in enumerate(zip(clips, reference[split])) if c["stored"].shape == r["codes"].shape]
        exact = sum(bool((clips[i]["stored"] == reference[split][i]["codes"]).all()) for i in same)
        stored = [{**reference[split][i], "codes": clips[i]["stored"].astype(np.int16)} for i in same]
        result = compare(stored, [reference[split][i] for i in same], [ranks[split][i] for i in same])
        report("stored codes", split, {**result, "size": size(base, DEFAULTS)})
        print_info(f"{split} clips", f"{exact}/{len(clips)} reproduced exactly, {len(clips) - len(same)} of another length")
    del model

    args.results.parent.mkdir(parents=True, exist_ok=True)
    for config in configs:
        print_section(f"{config['name']}: " + json.dumps({k: v for k, v in config.items() if DEFAULTS[k] != v and k != "name"}))
        torch.backends.cudnn.allow_tf32 = config["tf32"]
        (model, stored), nbytes = build(base, config, calibration), size(base, config)
        for split, clips in splits.items():
            start = time.time()
            outputs, watched = run(model, clips, config)
            result = {**compare(reference[split], outputs, ranks[split]), "size": nbytes, **watched,
                      "stored": stored_agreement(clips, outputs)}
            report(config["name"], split, result)
            if config["watch"]:
                print_info("peaks", result["peaks"])
                print_info("overflows", result["overflows"] or "none")
            if args.codes and split == "test":
                write_codes(args.codes / config["name"], args.test, clips, outputs)
            with open(args.results, "a") as f:
                f.write(json.dumps({"config": config, "split": split, "clips": len(clips), "script": script,
                                    "seconds": round(time.time() - start, 1), **result}) + "\n")
            if args.save and split == "test":
                save(base, config, stored, args.save / config["name"])
                # Its floats are what the checkpoint holds already: rounding them to `storage` again would move them.
                plain = {**config, "bits": None, "overrides": [], "storage": "fp32"}
                loaded, _ = build(load(args.save / config["name"]), plain, [])
                again, _ = run(loaded, clips, config)
                exact = sum(bool((a["codes"] == o["codes"]).all()) for a, o in zip(again, outputs))
                print_info("saved", f"{args.save / config['name']}: {exact}/{len(clips)} test clips encode the same")
                del loaded
        # transformers caches _get_conv1d_layers with lru_cache on the method, which keeps every encoding model alive.
        del model
        HiggsAudioV2TokenizerModel._get_conv1d_layers.cache_clear()
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
