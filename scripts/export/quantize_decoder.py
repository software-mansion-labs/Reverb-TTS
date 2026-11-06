"""Quantizes the weights of the Higgs Audio v2 codec's decode path (codes -> 24 kHz waveform) and measures what it costs
against the fp32 decode (TF32 off); `qat` then trains a saved checkpoint's scales to reproduce the fp32 decode.

    PYTHONPATH=. /venv/main/bin/python scripts/export/quantize_decoder.py score --config '{"name": "int8", "bits": 8}' ...
    PYTHONPATH=. /venv/main/bin/python scripts/export/quantize_decoder.py qat --init <checkpoint> --output <checkpoint>

The decode path is the codebooks with their project_out, fc2 and the DAC decoder: 21.5M weights. Configs, grids, GPTQ
and the checkpoint format are quantize_encoder.py's; a transposed conv is quantized as the conv it equals (its weight
transposed, its taps reversed, over its input stuffed with zeros). Scored per split on the stored codes (the 57 test
references, and the dev clips but the 64 held out for calibration): the waveform's SNR, the log-mel distance (dB) and
the error in 4-8 kHz relative to the signal, mean and worst over clips; each decoder block's output SNR; and the bytes
of the weights, scales included. The QAT keeps a checkpoint's integers and learns a factor per scale, as
qat_encoder.py's --learn scales does, on crops of the train codes, the languages drawn evenly.
"""

import argparse
import copy
import hashlib
import io
import json
import math
import random
import time
import zipfile
from pathlib import Path
from typing import Any, Iterator

import numpy as np
import torch
import torch.nn.functional as F
import torchaudio
from safetensors.torch import load_file, save_file

import quantize_encoder as q
from echo.model.quantization import _pack, _unpack
from qat_encoder import OnGrid

ENCODE_ONLY = ("semantic_model", "acoustic_encoder", "encoder_semantic", "fc", "fc1", "decoder_semantic")
PARTS = ("quantizer", "fc2", "acoustic_decoder")
RATE = 24000
DEFAULTS: dict[str, Any] = {**q.DEFAULTS, "modules": list(PARTS)}
LAYERS = (torch.nn.Linear, torch.nn.Conv1d, torch.nn.ConvTranspose1d)


def load_codec() -> q.HiggsAudioV2TokenizerModel:
    """The codec's decode path in fp32 on the CPU."""
    model = q.HiggsAudioV2TokenizerModel.from_pretrained(q.MODEL).eval()
    for name in ENCODE_ONLY:
        delattr(model, name)
    return model


def decode_tensors(model: q.HiggsAudioV2TokenizerModel) -> dict[str, torch.Tensor]:
    """Every weight decode reads: no project_in, and of the quantizer's buffers only the codebooks."""
    tensors = {n: p for n, p in model.named_parameters() if ".project_in." not in n}
    tensors.update({n: b for n, b in model.named_buffers() if n.endswith("codebook.embed")})
    return tensors


def as_conv(module: torch.nn.Module, weight: torch.Tensor) -> torch.Tensor:
    """A weight as the [out, in, taps] cross-correlation it computes; for a transposed conv this is its own inverse."""
    return weight.transpose(0, 1).flip(-1) if isinstance(module, torch.nn.ConvTranspose1d) else weight


def quantized_weights(model: q.HiggsAudioV2TokenizerModel, config: dict[str, Any]) -> dict[str, tuple[torch.nn.Module, dict]]:
    """Each quantized weight's module and its own settings, as quantize_encoder.quantized_weights, transposed convs too."""
    layers = {}
    for name, module in model.named_modules():
        if not isinstance(module, LAYERS) or ".project_in" in name:
            continue
        if not q.under(name, config["modules"]) or q.under(name, config["keep"]):
            continue
        settings = dict(config)
        for prefix, values in config["overrides"]:
            if q.under(name, [prefix]):
                settings.update(values)
        if settings["bits"]:
            layers[f"{name}.weight"] = (module, settings)
    return layers


def size(model: q.HiggsAudioV2TokenizerModel, config: dict[str, Any]) -> int:
    """Bytes of the decode path's weights: quantized ones at `bits` plus an fp16 scale (and a zero point) per group."""
    if config["init"]:
        return sum(t.numel() * t.element_size() for t in load_file(Path(config["init"]) / "decoder.safetensors").values())
    quantized = quantized_weights(model, config)
    total = 0.0
    for name, tensor in decode_tensors(model).items():
        if name in quantized:
            module, s = quantized[name]
            scales = q.grid(as_conv(module, tensor), s)[1]["scale"].numel()
            total += tensor.numel() * s["bits"] / 8 + scales * (2 + (0 if s["symmetric"] else s["bits"] / 8))
        else:
            total += tensor.numel() * min(q.DTYPES[config["storage"]].itemsize, q.DTYPES[config["dtype"]].itemsize)
    return int(total)


def columns(module: torch.nn.Module, x: torch.Tensor) -> Iterator[torch.Tensor]:
    """The columns a layer's weight, in its conv form, multiplies: [in x taps, positions], a chunk at a time."""
    if isinstance(module, torch.nn.Linear):
        yield x.reshape(-1, x.shape[-1]).T
        return
    k, d = module.kernel_size[0], module.dilation[0]
    if isinstance(module, torch.nn.ConvTranspose1d):
        s, p = module.stride[0], module.padding[0]
        stuffed = x.new_zeros(*x.shape[:2], (x.shape[2] - 1) * s + 1)
        stuffed[..., ::s] = x
        padded, stride = F.pad(stuffed, (k - 1 - p, k - 1 - p + module.output_padding[0])), 1
    else:
        padded, stride = F.pad(x, (module.padding[0],) * 2), module.stride[0]
    positions = (padded.shape[-1] - d * (k - 1) - 1) // stride + 1
    for a in range(0, positions, 16384):
        b = min(a + 16384, positions)
        chunk = padded[..., a * stride:(b - 1) * stride + d * (k - 1) + 1].unfold(2, d * (k - 1) + 1, stride)[..., ::d]
        yield chunk.permute(1, 3, 0, 2).reshape(x.shape[1] * k, -1)


@torch.inference_mode()
def hessians(model: q.HiggsAudioV2TokenizerModel, layers: dict[str, torch.nn.Module], clips: list[dict[str, Any]],
             device: str) -> dict[str, torch.Tensor]:
    """The mean x x^T of each layer's columns over the calibration clips, in fp64, parked on the CPU."""
    sums: dict[str, torch.Tensor] = {}
    counts: dict[str, int] = {}

    def hook(name: str) -> Any:
        def accumulate(module: torch.nn.Module, args: tuple, output: torch.Tensor) -> None:
            for chunk in columns(module, args[0]):
                chunk = chunk.double()
                if name not in sums:
                    sums[name], counts[name] = chunk.new_zeros(chunk.shape[0], chunk.shape[0]), 0
                sums[name].addmm_(chunk, chunk.T)
                counts[name] += chunk.shape[1]
        return accumulate

    handles = [module.register_forward_hook(hook(name)) for name, module in layers.items()]
    for clip in clips:
        model.decode(clip["codes"].to(device))
    for handle in handles:
        handle.remove()
    return {name: sums[name].div_(counts[name]).cpu() for name in sums}


def build(base: q.HiggsAudioV2TokenizerModel, config: dict[str, Any],
          calibration: list[dict[str, Any]]) -> tuple[q.HiggsAudioV2TokenizerModel, dict[str, dict]]:
    """The decode path as the config computes it, and what a checkpoint of its quantized weights holds."""
    model, stored = copy.deepcopy(load(Path(config["init"])) if config["init"] else base).to(config["device"]), {}
    quantized = {} if config["init"] else quantized_weights(model, config)
    for part in PARTS:
        layers = {n: ms for n, ms in quantized.items() if q.under(n, [part])}
        exact = {n: m for n, (m, s) in layers.items() if s["method"] == "gptq"}
        stats = hessians(model, exact, calibration, config["device"]) if exact else {}
        with torch.no_grad():
            for name, (module, s) in layers.items():
                weight = as_conv(module, module.weight)
                if name in exact:
                    values, held = q.gptq(weight, stats.pop(name).to(config["device"], copy=True), s)
                else:
                    values, held = q.grid(weight, s)
                module.weight.copy_(as_conv(module, values))
                stored[name] = {"ints": held["ints"].short().cpu(), "scale": held["scale"].cpu(),
                                "zero": held["zero"].cpu(), "settings": s, "shape": list(weight.shape)}
    with torch.no_grad():
        storage = q.DTYPES[config["storage"]]
        for name, tensor in decode_tensors(model).items():
            if name not in quantized and not config["init"]:
                tensor.copy_(tensor.to(storage))
    if config["snake"] == "rewrite" or (config["snake"] is None and config["dtype"] != "fp32"):
        for module in list(model.acoustic_decoder.modules()):
            for name, child in module.named_children():
                if type(child).__name__ == "Snake1d":
                    setattr(module, name, q.Snake(child.alpha.detach()))
    for name in PARTS:
        getattr(model, name).to(q.DTYPES[config["dtype"]])
    for module in [model.fc2, model.acoustic_decoder] + [layer.project_out for layer in model.quantizer.quantizers]:
        module.register_forward_pre_hook(q._cast)
    return model, stored


def save(base: q.HiggsAudioV2TokenizerModel, config: dict[str, Any], stored: dict[str, dict], directory: Path) -> None:
    """quantize_encoder.save's format for the decode path; each quantized weight's integers in its conv form."""
    directory.mkdir(parents=True, exist_ok=True)
    state, weights = {}, {}
    for name, tensor in decode_tensors(base).items():
        if name not in stored:
            state[name] = tensor.detach().to(q.DTYPES[config["storage"]]).contiguous()
            continue
        held, s = stored[name], stored[name]["settings"]
        if s["bits"] == 4:
            state[name] = _pack(held["ints"].flatten().float() - (0 if s["symmetric"] else 8))
        else:
            state[name] = held["ints"].to(torch.int8 if s["symmetric"] else torch.uint8)
        state[f"{name}_scale"] = held["scale"].half()
        if not s["symmetric"]:
            state[f"{name}_zero"] = held["zero"].to(torch.uint8)
        weights[name] = {k: s[k] for k in ("bits", "granularity", "symmetric")} | {"shape": held["shape"]}
    save_file(state, directory / "decoder.safetensors", metadata={"format": "pt"})
    (directory / "quantization.json").write_text(json.dumps({"config": config, "weights": weights}, indent=1))


def load(directory: Path) -> q.HiggsAudioV2TokenizerModel:
    """A decode path written by save(), its weights dequantized to fp32."""
    meta = json.loads((directory / "quantization.json").read_text())
    state = load_file(directory / "decoder.safetensors")
    model = load_codec()
    for name, w in meta["weights"].items():
        scale = state.pop(f"{name}_scale").float()[..., None]
        zero = state.pop(f"{name}_zero").float()[..., None] if not w["symmetric"] else torch.zeros_like(scale)
        ints = _unpack(state[name]) + (0 if w["symmetric"] else 8) if w["bits"] == 4 else state[name].float()
        values = (ints.reshape(w["shape"][0], scale.shape[1], -1) - zero) * scale
        conv = q.unlayout(values.reshape(w["shape"][0], -1), torch.empty(w["shape"]), w)
        state[name] = as_conv(model.get_submodule(name[: -len(".weight")]), conv)
    missing, unexpected = model.load_state_dict({k: v.float() for k, v in state.items()}, strict=False)
    allowed = (".project_in.weight", ".project_in.bias", "inited", "cluster_size", "embed_avg")
    if unexpected or any(not k.endswith(allowed) for k in missing):
        raise ValueError(f"{directory}: missing {missing}, unexpected {unexpected}")
    return model


def load_clips(manifest: Path, rows: list[dict[str, str]]) -> list[dict[str, Any]]:
    return [{"name": r["name"], "codes": torch.as_tensor(np.load(manifest.parent / r["name"]).astype(np.int64))[None]}
            for r in rows]


class Metrics:
    """Waveform SNR, log-mel L1 (dB, the mel power floored at 1e-5) and the 4-8 kHz error relative to the signal (dB)."""

    def __init__(self, device: str) -> None:
        self.mel = torchaudio.transforms.MelSpectrogram(RATE, n_fft=1024, hop_length=256, n_mels=80).to(device)
        self.window = torch.hann_window(1024, device=device)
        self.band = slice(round(4000 * 1024 / RATE), round(8000 * 1024 / RATE))

    def __call__(self, reference: torch.Tensor, signal: torch.Tensor) -> dict[str, float]:
        log_mel = [10 * torch.log10(self.mel(x).clamp(min=1e-5)) for x in (reference, signal)]
        spectra = [torch.stft(x, 1024, 256, window=self.window, return_complex=True)[self.band] for x in (reference, signal)]
        band = (spectra[0] - spectra[1]).abs().pow(2).sum() / spectra[0].abs().pow(2).sum()
        return {"snr": q.snr(reference, signal), "mel": float((log_mel[0] - log_mel[1]).abs().mean()),
                "band": float(10 * torch.log10(band))}


@torch.inference_mode()
def score(reference: torch.nn.Module, model: torch.nn.Module, clips: list[dict[str, Any]], config: dict[str, Any],
          metrics: Metrics) -> dict[str, Any]:
    """Every clip decoded by the reference (TF32 off) and by the model; per-clip metrics, pooled as mean and worst."""
    blocks = len(model.acoustic_decoder.block)
    captured: dict[str, dict[int, torch.Tensor]] = {"reference": {}, "model": {}}
    hooks = [m.acoustic_decoder.block[i].register_forward_hook(
        lambda mod, a, o, side=side, i=i: captured[side].__setitem__(i, o.float()))
        for side, m in (("reference", reference), ("model", model)) for i in range(blocks)]
    rows, start = [], time.time()
    for k, clip in enumerate(clips, 1):
        codes = clip["codes"].to(config["device"])
        torch.backends.cudnn.allow_tf32 = False
        r = reference.decode(codes).audio_values[0, 0].float()
        torch.backends.cudnn.allow_tf32 = config["tf32"]
        x = model.decode(codes).audio_values[0, 0].float()
        rows.append({**metrics(r, x), **{f"block{i}": q.snr(captured["reference"][i], captured["model"][i])
                                        for i in range(blocks)}})
        if k % 100 == 0 or k == len(clips):
            print(f"    {k}/{len(clips)} clips, {time.time() - start:.0f} s", flush=True)
    for hook in hooks:
        hook.remove()
    mean = {key: round(float(np.mean([r[key] for r in rows])), 3) for key in rows[0]}
    return {**mean, "snr_min": round(min(r["snr"] for r in rows), 2), "mel_max": round(max(r["mel"] for r in rows), 3)}


def report(name: str, split: str, result: dict[str, Any]) -> None:
    blocks = " ".join(f"{result[k]:.1f}" for k in sorted(k for k in result if k.startswith("block")))
    print(f"  {split:4s} {name:24s} {result['size'] / 1e6:5.1f} MB | SNR {result['snr']:5.1f} (min {result['snr_min']:5.1f}) dB"
          f" | log-mel {result['mel']:.3f} (max {result['mel_max']:.3f}) dB | 4-8 kHz {result['band']:6.1f} dB"
          f" | blocks {blocks}", flush=True)


class Transposed(torch.nn.Module):
    """A transposed conv's weight parametrized through the conv it equals."""

    def __init__(self, inner: torch.nn.Module) -> None:
        super().__init__()
        self.inner = inner

    def forward(self, weight: torch.Tensor) -> torch.Tensor:
        return self.inner(weight.transpose(0, 1).flip(-1)).transpose(0, 1).flip(-1)


class Codes(torch.utils.data.Dataset):
    """`count` crops of `frames` frames of the train codes, the languages drawn evenly; crop i is the same every run."""

    def __init__(self, root: Path, frames: int, count: int, seed: int) -> None:
        self.frames, self.count, self.seed, self.zips = frames, count, seed, {}
        self.members: dict[str, list[tuple[str, str]]] = {}
        for path in sorted(root.glob("*/**/codecs.zip")):
            with zipfile.ZipFile(path) as z:
                names = [n for n in z.namelist() if n.endswith(".npy")]
            self.members.setdefault(path.relative_to(root).parts[0], []).extend((str(path), n) for n in names)
        self.languages = sorted(self.members)

    def __len__(self) -> int:
        return self.count

    def __getitem__(self, i: int) -> torch.Tensor:
        rng = random.Random(self.seed * 1_000_003 + i)
        while True:
            path, name = rng.choice(self.members[rng.choice(self.languages)])
            # Each worker opens the zips itself: a handle shared across processes reads garbage.
            z = self.zips.get(path) or self.zips.setdefault(path, zipfile.ZipFile(path))
            codes = np.load(io.BytesIO(z.read(name))).astype(np.int64)
            if codes.shape[1] >= self.frames:
                start = rng.randrange(codes.shape[1] - self.frames + 1)
                return torch.as_tensor(codes[:, start:start + self.frames])


def main_score(args: argparse.Namespace) -> None:
    if torch.cuda.is_available():
        torch.cuda.set_per_process_memory_fraction(args.memory * 2**30 / torch.cuda.get_device_properties(0).total_memory)
    configs = [{**DEFAULTS, **json.loads(c)} for c in args.config]
    script = hashlib.sha1(Path(__file__).read_bytes()).hexdigest()[:10]
    dev = q.manifest_rows(args.dev)
    random.Random(0).shuffle(dev)
    held_out, dev = dev[:args.calibration], dev[args.calibration:]
    splits = {"test": load_clips(args.test, q.manifest_rows(args.test)), "dev": load_clips(args.dev, dev)}
    calibration = load_clips(args.dev, held_out)
    base, device = load_codec(), DEFAULTS["device"]
    torch.backends.cudnn.allow_tf32 = False
    reference, metrics = build(base, DEFAULTS, [])[0], Metrics(device)
    print(f"decode path: {sum(t.numel() for t in decode_tensors(base).values()) / 1e6:.2f}M weights; "
          f"test {len(splits['test'])}, dev {len(splits['dev'])}, calibration {len(calibration)} clips", flush=True)
    args.results.parent.mkdir(parents=True, exist_ok=True)
    for config in configs:
        print(f"--- {config['name']}: " + json.dumps({k: v for k, v in config.items() if DEFAULTS[k] != v and k != "name"}),
              flush=True)
        torch.backends.cudnn.allow_tf32 = config["tf32"]
        (model, stored), nbytes = build(base, config, calibration), size(base, config)
        for split, clips in splits.items():
            result = {**score(reference, model, clips, config, metrics), "size": nbytes}
            report(config["name"], split, result)
            with open(args.results, "a") as f:
                f.write(json.dumps({"config": config, "split": split, "clips": len(clips), "script": script, **result}) + "\n")
        if args.save:
            save(base, config, stored, args.save / config["name"])
            print(f"  saved {args.save / config['name']}", flush=True)
        del model
        q.HiggsAudioV2TokenizerModel._get_conv1d_layers.cache_clear()
        torch.cuda.empty_cache()


def main_qat(args: argparse.Namespace) -> None:
    device = "cuda"
    torch.backends.cudnn.allow_tf32 = False
    torch.manual_seed(args.seed)
    meta = json.loads((args.init / "quantization.json").read_text())
    state = load_file(args.init / "decoder.safetensors")
    teacher = load_codec().to(device).eval().requires_grad_(False)
    student = load(args.init).to(device).eval().requires_grad_(False)
    for name, settings in meta["weights"].items():
        module = student.get_submodule(name[: -len(".weight")])
        scale = state[f"{name}_scale"].to(device)
        zero = state.get(f"{name}_zero", torch.zeros_like(scale)).to(device)
        on_grid = OnGrid(scale, zero, settings, as_conv(module, module.weight.detach()))
        transposed = isinstance(module, torch.nn.ConvTranspose1d)
        torch.nn.utils.parametrize.register_parametrization(module, "weight", Transposed(on_grid) if transposed else on_grid)
    thetas = [m.theta for m in student.modules() if isinstance(m, OnGrid)]
    for theta in thetas:
        theta.requires_grad_(True)

    train = Codes(args.train, args.frames, args.steps * args.batch, args.seed)
    held_out = Codes(args.train, args.frames, 16 * args.batch, args.seed + 1)
    held_out = [torch.stack([held_out[i * args.batch + j] for j in range(args.batch)]) for i in range(16)]
    loader = torch.utils.data.DataLoader(train, batch_size=args.batch, num_workers=args.workers, prefetch_factor=4)
    print(f"{len(thetas)} scales learn a factor; {', '.join(f'{k} {len(v)}' for k, v in train.members.items())} clips; "
          f"{args.steps} steps of {args.batch} x {args.frames} frames", flush=True)

    @torch.no_grad()
    def validate(step: int) -> None:
        noise = energy = mel = 0.0
        metrics = Metrics(device)
        for codes in held_out:
            for r, x in zip(teacher.decode(codes.to(device)).audio_values, student.decode(codes.to(device)).audio_values):
                noise, energy = noise + float(((r - x) ** 2).sum()), energy + float((r ** 2).sum())
                mel += metrics(r[0], x[0])["mel"] / (16 * args.batch)
        result = {"step": step, "snr": round(10 * math.log10(energy / noise), 2), "mel": round(mel, 4)}
        print(json.dumps(result), flush=True)
        with open(args.output.parent / f"{args.output.name}.jsonl", "a") as f:
            f.write(json.dumps(result) + "\n")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    (args.output.parent / f"{args.output.name}.jsonl").unlink(missing_ok=True)
    validate(0)
    optimizer = torch.optim.AdamW(thetas, lr=args.lr, weight_decay=0.0)
    schedule = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda step: min(1.0, (step + 1) / 100) * (
        0.1 + 0.45 * (1 + math.cos(math.pi * min(1.0, step / max(args.steps, 1))))))
    start = time.time()
    for step, codes in enumerate(loader, 1):
        codes = codes.to(device, non_blocking=True)
        with torch.no_grad():
            t = teacher.decode(codes).audio_values
        s = student.decode(codes).audio_values
        error = ((s - t) ** 2).sum() / (t ** 2).sum()
        optimizer.zero_grad(set_to_none=True)
        torch.log10(error).backward()
        torch.nn.utils.clip_grad_norm_(thetas, 1.0)
        optimizer.step()
        schedule.step()
        if step % 50 == 0:
            print(f"step {step}/{args.steps}: SNR {-10 * math.log10(float(error.detach())):.2f} dB, "
                  f"lr {schedule.get_last_lr()[0]:.2e}, {(time.time() - start) / step:.2f} s/step", flush=True)
        if step % args.eval_every == 0 or step == args.steps:
            validate(step)

    stored = {}
    for name, settings in meta["weights"].items():
        module = student.get_submodule(name[: -len(".weight")])
        on_grid = next(m for m in module.parametrizations.weight.modules() if isinstance(m, OnGrid))
        stored[name] = {"ints": on_grid.fixed.reshape(on_grid.fixed.shape[0], -1).short().cpu(),
                        "scale": on_grid.effective_scale().detach()[..., 0].cpu(), "zero": on_grid.zero[..., 0].cpu(),
                        "settings": settings, "shape": list(as_conv(module, module.weight).shape)}
        torch.nn.utils.parametrize.remove_parametrizations(module, "weight", leave_parametrized=True)
    config = {**meta["config"], "qat": {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items() if k != "func"}}
    save(student, config, stored, args.output)
    print(f"wrote {args.output}", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    commands = parser.add_subparsers(required=True)
    scoring = commands.add_parser("score", help="quantizes and scores configurations")
    scoring.set_defaults(func=main_score)
    scoring.add_argument("--config", action="append", default=[], help="a JSON object over DEFAULTS; repeat it for more runs")
    scoring.add_argument("--test", type=Path, default=Path("data/test/test_in_processed.csv"))
    scoring.add_argument("--dev", type=Path, default=Path("data/dev/dev_processed.csv"))
    scoring.add_argument("--calibration", type=int, default=64, help="dev clips held out for calibration, never scored")
    scoring.add_argument("--results", type=Path, default=Path("runs/decoder_quantization/results.jsonl"))
    scoring.add_argument("--save", type=Path, help="writes each run's weights to <save>/<name>/")
    scoring.add_argument("--memory", type=float, default=24.0, help="GB of GPU memory this process may take")
    training = commands.add_parser("qat", help="trains a saved checkpoint's scales against the fp32 decode")
    training.set_defaults(func=main_qat)
    training.add_argument("--init", type=Path, required=True, help="checkpoint written by score --save")
    training.add_argument("--output", type=Path, required=True)
    training.add_argument("--train", type=Path, default=Path("data/train"), help="root of the train codes' zips")
    training.add_argument("--steps", type=int, default=3000)
    training.add_argument("--batch", type=int, default=8)
    training.add_argument("--frames", type=int, default=100, help="crop length in frames (25 per second)")
    training.add_argument("--lr", type=float, default=5e-4, help="peak, on each scale's factor; 100 steps of warmup, cosine")
    training.add_argument("--eval-every", type=int, default=500)
    training.add_argument("--workers", type=int, default=8)
    training.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
