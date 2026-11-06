"""Exports the decoder of Echo's audio codec (Higgs Audio v2) to ExecuTorch for the CoreML backend (CPU and GPU):
forward(codes [1, 8, T]) -> 24 kHz waveform [1, 1, 960 T], for T from 2 to 1024 frames. Computed in fp16, with fp16
weights, or with --int8 int8 ones: symmetric, one scale per 8 input channels of an output channel (Core ML's per-block
quantization, which needs iOS 18).

Run it with the ExecuTorch venv: ../executorch/.venv/bin/python scripts/export/export_codec.py ...
"""

import argparse
import json
import time
from pathlib import Path

import coremltools as ct
import torch
from executorch.backends.apple.coreml.compiler import CoreMLBackend
from executorch.backends.apple.coreml.partition import CoreMLPartitioner
from executorch.exir import to_edge_transform_and_lower
from huggingface_hub import snapshot_download
from safetensors.torch import load_file
from torch.export import Dim, export
from transformers import DacConfig
from transformers.models.dac.modeling_dac import DacDecoder

CODEC = "eustlb/higgs-audio-v2-tokenizer"
CODEBOOKS = 8
MAX_FRAMES = 1024


class Snake(torch.nn.Module):
    """
    Snake1d, x + sin(a x)^2 / a, as x + sign(a) (sin(a x) / sqrt|a|)^2: the smallest |a| is 4.1e-7, so 1/a overflows
    fp16, while 1/sqrt|a| stays below 1560.
    """

    def __init__(self, alpha: torch.Tensor) -> None:
        super().__init__()
        self.register_buffer("alpha", alpha)
        self.register_buffer("rsqrt", (alpha.abs() + 1e-9).rsqrt())
        self.register_buffer("sign", alpha.sign())

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        r = torch.sin(self.alpha * x) * self.rsqrt
        return x + self.sign * (r * r)


def linear(weights: dict, name: str) -> torch.nn.Linear:
    weight = weights[f"{name}.weight"]
    layer = torch.nn.Linear(weight.shape[1], weight.shape[0])
    layer.load_state_dict({"weight": weight, "bias": weights[f"{name}.bias"]})
    return layer


class Decoder(torch.nn.Module):
    """
    The codec's decode path: the codebooks' entries, projected and summed, then fc2 and the DAC decoder.
    """

    def __init__(self) -> None:
        super().__init__()
        path = Path(snapshot_download(CODEC, allow_patterns=["config.json", "model.safetensors"]))
        config = json.loads((path / "config.json").read_text())
        weights = load_file(path / "model.safetensors")

        quantizers = [f"quantizer.quantizers.{i}" for i in range(CODEBOOKS)]
        self.codebooks = torch.nn.ModuleList(
            torch.nn.Embedding.from_pretrained(weights[f"{q}.codebook.embed"]) for q in quantizers
        )
        self.projections = torch.nn.ModuleList(linear(weights, f"{q}.project_out") for q in quantizers)
        self.fc2 = linear(weights, "fc2")

        self.decoder = DacDecoder(DacConfig.from_dict(config["acoustic_model_config"]))
        prefix = "acoustic_decoder."
        self.decoder.load_state_dict({k[len(prefix):]: v for k, v in weights.items() if k.startswith(prefix)})
        # The codec's changes to DAC: output padding on the upsampling convolutions, and no tanh at the end.
        for module in self.decoder.modules():
            if isinstance(module, torch.nn.ConvTranspose1d):
                module.output_padding = (module.stride[0] % 2,)
        self.decoder.tanh = torch.nn.Identity()
        for module in list(self.decoder.modules()):
            for name, child in module.named_children():
                if type(child).__name__ == "Snake1d":
                    setattr(module, name, Snake(child.alpha.detach()))

    def forward(self, codes: torch.Tensor) -> torch.Tensor:
        x = sum(project(codebook(c)) for c, codebook, project in zip(codes[0], self.codebooks, self.projections))
        return self.decoder(self.fc2(x).t()[None])


def lower(decoder: Decoder, int8: bool) -> bytes:
    with torch.no_grad():
        program = export(
            decoder,
            (torch.zeros(1, CODEBOOKS, 250, dtype=torch.long),),
            dynamic_shapes=({2: Dim("frames", min=2, max=MAX_FRAMES)},),
            strict=False,
        )

    specs = CoreMLBackend.generate_compile_specs(
        # The Neural Engine takes no dynamic shapes, and with ALL Core ML would still compile for it on the first load.
        compute_unit=ct.ComputeUnit.CPU_AND_GPU,
        minimum_deployment_target=ct.target.iOS18 if int8 else ct.target.iOS17,
        compute_precision=ct.precision.FLOAT16,
        model_type=CoreMLBackend.MODEL_TYPE.COMPILED_MODEL,
        op_linear_quantizer_config=(
            {"mode": "linear_symmetric", "dtype": "int8", "granularity": "per_block", "block_size": 8} if int8 else None
        ),
    )
    edge = to_edge_transform_and_lower(program, partitioner=[CoreMLPartitioner(compile_specs=specs)])
    return edge.to_executorch().buffer


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--output", type=Path, required=True, help=".pte file to write, e.g. runs/final/coreml/codec_int8.pte")
    parser.add_argument("--int8", action="store_true", help="int8 weights instead of fp16")
    args = parser.parse_args()

    start = time.time()
    program = lower(Decoder().eval(), args.int8)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_bytes(program)
    print(f"Lowered in {time.time() - start:.0f} s: {args.output} ({len(program) / 1e6:.1f} MB)", flush=True)


if __name__ == "__main__":
    main()
