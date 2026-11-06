"""Exports an Echo checkpoint to ExecuTorch for the MLX backend (Apple silicon GPU): fp16 from the fp32 checkpoint,
int8 or int4 weights from the quantized ones (W8A16, W4A16), computed in fp16.

The program has three methods, sharing one copy of the weights:
- prefill(text [L], reference [8, R]) -> keys, values [layers, 1, kv heads, L + R, head dim]: the prefix, run once.
- step(tokens [8, T], keys, values) -> confidence, prediction [2, T]: one main pass, codebooks 0-1 only.
- polish(tokens [8, T], bias [1, 1, 1, L + R], keys, values) -> [6, T]: both polish passes, codebooks 2-7.

Run it with the ExecuTorch venv: ../executorch/.venv/bin/python scripts/export/export_mlx.py ...
"""

import argparse
import json
import shutil
import time
from pathlib import Path
from typing import List, Optional, Tuple

import torch
import torch.nn.functional as F
from safetensors.torch import load_file
from torch.export import Dim, export

import executorch.backends.mlx.custom_ops  # noqa: F401 - registers mlx::rope
import torchao  # noqa: F401 - registers torchao::dequantize_affine

from executorch.backends.mlx import MLXPartitioner
from executorch.backends.mlx.passes import get_default_passes
from executorch.exir import EdgeCompileConfig, ExecutorchBackendConfig, to_edge_transform_and_lower
from executorch.exir.passes import MemoryPlanningPass

DTYPE = torch.float16

# Upper bounds of the dynamic lengths, in positions: the prefix (prompt tokens + reference frames) and the target.
MAX_PREFIX = 1024
MAX_FRAMES = 1024

# The polish passes guide codebooks 4-7 against the first pass's view of them, as the runtime notebooks do.
POLISH_GUIDANCE = 0.5


def frozen(t: torch.Tensor) -> torch.nn.Parameter:
    """
    A constant held as a parameter: ExecuTorch drops the parameters a method doesn't use, but keeps every buffer.
    """

    return torch.nn.Parameter(t, requires_grad=False)


class Linear(torch.nn.Module):
    """
    A projection with fp16 weights, stored transposed so that it lowers to a single matmul.
    """

    def __init__(self, weight: torch.Tensor) -> None:
        super().__init__()
        self.weight = frozen(weight.t().contiguous().to(DTYPE))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x @ self.weight


class QuantizedLinear(torch.nn.Module):
    """
    A projection with the checkpoint's symmetric int weights, as is: the MLX backend fuses the dequantization into
    its quantized matmul. One scale per output channel (int8) or per group of inputs (int4).
    """

    def __init__(self, values: torch.Tensor, scale: torch.Tensor, bits: int) -> None:
        super().__init__()
        self.values = frozen(values)
        # In fp16, as a quantized matmul runs in the dtype of its scales.
        self.scale = frozen(scale.to(DTYPE))
        self.zero = frozen(torch.zeros_like(scale, dtype=torch.int8))
        self.block = [1, values.shape[1] // scale.shape[1]]
        self.bits = bits

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        limit = 2 ** (self.bits - 1)
        weight = torch.ops.torchao.dequantize_affine(
            self.values, self.block, self.scale, self.zero, torch.int8, -limit, limit - 1, output_dtype=DTYPE
        )
        return F.linear(x, weight)


def load_projections(weights: dict, quantization: Optional[dict]) -> dict:
    """
    The decoder projections, popped from 'weights': float weights, or int8 values and their scales.
    """

    projections = {}
    for key in [key for key in weights if key.startswith("llm.layers.") and key.endswith("_proj.weight")]:
        weight = weights.pop(key)
        if quantization is None:
            projections[key] = (weight.float(),)
            continue
        scale = weights.pop(key + "_scale").float()
        if quantization["bits"] == 4:
            # Two values per byte, the first one in the low nibble, offset by 8.
            weight = torch.stack([weight & 0xF, weight >> 4], dim=-1).flatten(-2).to(torch.int8) - 8
        projections[key] = (weight, scale)

    return projections


def make_linear(parts: List[tuple], bits: Optional[int]) -> torch.nn.Module:
    """
    One projection from the stacked output channels of 'parts' - e.g. q, k and v as a single matmul.
    """

    if bits is None:
        return Linear(torch.cat([part[0] for part in parts]))
    return QuantizedLinear(torch.cat([part[0] for part in parts]), torch.cat([part[1] for part in parts]), bits)


class Layer(torch.nn.Module):
    def __init__(self, prefix: str, projections: dict, weights: dict, bits: Optional[int], config: dict) -> None:
        super().__init__()
        self.heads, self.kv_heads, self.head_dim = config["heads"], config["kv_heads"], config["head_dim"]
        self.eps, self.theta = config["eps"], config["theta"]

        def linear(*names: str) -> torch.nn.Module:
            return make_linear([projections[f"{prefix}{name}.weight"] for name in names], bits)

        # q, k and v in one matmul: ~10% faster per pass in fp16.
        self.qkv = linear("self_attn.q_proj", "self_attn.k_proj", "self_attn.v_proj")
        self.o = linear("self_attn.o_proj")
        self.gate = linear("mlp.gate_proj")
        self.up = linear("mlp.up_proj")
        self.down = linear("mlp.down_proj")
        self.input_norm = frozen(weights[prefix + "input_layernorm.weight"].to(DTYPE))
        self.post_norm = frozen(weights[prefix + "post_attention_layernorm.weight"].to(DTYPE))
        self.q_norm = frozen(weights[prefix + "self_attn.q_norm.weight"].to(DTYPE))
        self.k_norm = frozen(weights[prefix + "self_attn.k_norm.weight"].to(DTYPE))

    def forward(
        self,
        x: torch.Tensor,
        position: int,
        past: Optional[Tuple[torch.Tensor, torch.Tensor]],
        mask: Optional[torch.Tensor],
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        [n, hidden] -> [n, hidden], and the n positions' keys and values, [1, kv heads, n, head dim] each.
        """

        n, d = x.shape[0], self.head_dim
        h = F.rms_norm(x, (x.shape[1],), self.input_norm, self.eps)
        q, k, v = self.qkv(h).split([self.heads * d, self.kv_heads * d, self.kv_heads * d], dim=-1)
        q = F.rms_norm(q.view(1, n, self.heads, d), (d,), self.q_norm, self.eps).transpose(1, 2)
        k = F.rms_norm(k.view(1, n, self.kv_heads, d), (d,), self.k_norm, self.eps).transpose(1, 2)
        v = v.view(1, n, self.kv_heads, d).transpose(1, 2)
        q = torch.ops.mlx.rope(q, d, position, False, self.theta, 1.0, None)
        k = torch.ops.mlx.rope(k, d, position, False, self.theta, 1.0, None)
        keys, values = (k, v) if past is None else (torch.cat([past[0], k], dim=2), torch.cat([past[1], v], dim=2))
        a = F.scaled_dot_product_attention(q, keys, values, attn_mask=mask, enable_gqa=True)
        x = x + self.o(a.transpose(1, 2).reshape(n, self.heads * d))
        h = F.rms_norm(x, (x.shape[1],), self.post_norm, self.eps)
        return x + self.down(F.silu(self.gate(h)) * self.up(h)), k, v


class Echo(torch.nn.Module):
    """
    The checkpoint's weights: the decoder in fp16 (or int weights), the audio heads in fp32 - the logits reach
    hundreds, where fp16 rounds to a quarter, and the sampler reads them at the ties.
    """

    def __init__(self, checkpoint: Path) -> None:
        super().__init__()
        config = json.loads(Path(checkpoint, "config.json").read_text())
        quantization = config.get("quantization")
        if quantization and (quantization.get("activations") or quantization.get("rotation")):
            raise NotImplementedError("MLX has no int8 activations nor a Hadamard op: export the weight-only checkpoints")
        llm = config["llm_config"]
        geometry = {
            "heads": llm["num_attention_heads"],
            "kv_heads": llm["num_key_value_heads"],
            "head_dim": llm["head_dim"],
            "eps": llm["rms_norm_eps"],
            "theta": float(llm["rope_parameters"]["rope_theta"]),
        }
        self.books, self.codes, self.mask_id = config["num_audio_codebook"], config["audio_vocab_size"], config["audio_mask_id"]
        # The mask is the last code, so dropping the last column drops it from every prediction.
        assert self.mask_id == self.codes - 1

        weights = load_file(Path(checkpoint, "model.safetensors"))
        projections = load_projections(weights, quantization)
        # Embedding tables stored in int8, one scale per row: dequantized, as the program keeps them in fp16.
        for key in [key for key in weights if key.endswith("_scale")]:
            weights[key[: -len("_scale")]] = weights[key[: -len("_scale")]].float() * weights.pop(key).float()
        weights = {key: value.float() if value.is_floating_point() else value for key, value in weights.items()}
        bits = quantization["bits"] if quantization else None
        self.layers = torch.nn.ModuleList(
            Layer(f"llm.layers.{i}.", projections, weights, bits, geometry) for i in range(llm["num_hidden_layers"])
        )
        self.eps = geometry["eps"]
        self.norm = frozen(weights["llm.norm.weight"].to(DTYPE))
        # The text embedding stays factorized, as the checkpoint stores it: a [vocab, rank] table and a projection.
        self.table = frozen(weights["llm.embed_tokens.table.weight"].to(DTYPE))
        self.project = frozen(weights["llm.embed_tokens.project.weight"].t().contiguous().to(DTYPE))
        self.audio = frozen(weights["audio_embeddings.weight"].to(DTYPE))
        self.offsets = frozen(self.codes * torch.arange(self.books)[:, None])

        # Centering each codebook's rows leaves its softmax as it is, and keeps the logits small.
        heads = weights["audio_heads.weight"].view(self.books, self.codes, -1)
        heads = (heads - heads.mean(1, keepdim=True)).flatten(0, 1).t()
        self.main_heads = frozen(heads[:, : 2 * self.codes].contiguous())
        self.polish_heads = frozen(heads[:, 2 * self.codes :].contiguous())

    def embed_audio(self, codes: torch.Tensor) -> torch.Tensor:
        """
        [8, n] codes -> [n, hidden]: each frame is the sum of its 8 codebook embeddings.
        """

        return F.embedding(codes + self.offsets, self.audio).sum(0)

    def run(
        self, x: torch.Tensor, position: int, past: Optional[Tuple[torch.Tensor, torch.Tensor]], mask: Optional[torch.Tensor]
    ) -> Tuple[torch.Tensor, List[torch.Tensor], List[torch.Tensor]]:
        keys, values = [], []
        for i, layer in enumerate(self.layers):
            x, k, v = layer(x, position, None if past is None else (past[0][i], past[1][i]), mask)
            keys.append(k)
            values.append(v)

        return F.rms_norm(x, (x.shape[1],), self.norm, self.eps), keys, values

    def logits(self, h: torch.Tensor, heads: torch.Tensor) -> torch.Tensor:
        """
        [n, hidden] -> [n, codebooks, codes], in fp32.
        """

        return (h.float() @ heads).view(h.shape[0], -1, self.codes)


# RoPE only sees relative positions, so the whole sequence is shifted by the prefix length: the MLX rope op takes only
# a symbolic offset, and the prefix length is one that prefill and the passes both know. The prefix sits at
# [P, 2P), the target from 2P on.


class Prefill(torch.nn.Module):
    def __init__(self, echo: Echo) -> None:
        super().__init__()
        self.echo = echo

    def forward(self, text: torch.Tensor, reference: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        x = torch.cat([F.embedding(text, self.echo.table) @ self.echo.project, self.echo.embed_audio(reference)])
        _, keys, values = self.echo.run(x, x.shape[0], None, None)
        return torch.stack(keys), torch.stack(values)


class Step(torch.nn.Module):
    """
    One main pass: the log-probability of the best code of codebooks 0-1 in every slot, and that code. The softmax
    counts the mask, but the mask is never a prediction.
    """

    def __init__(self, echo: Echo) -> None:
        super().__init__()
        self.echo = echo

    def forward(self, tokens: torch.Tensor, keys: torch.Tensor, values: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        h, _, _ = self.echo.run(self.echo.embed_audio(tokens), 2 * keys.shape[3], (keys, values), None)
        logits = self.echo.logits(h, self.echo.main_heads)
        normalizer = torch.logsumexp(logits, dim=-1)
        logits = logits[..., : self.echo.mask_id]
        return (logits.amax(-1) - normalizer).t(), logits.argmax(-1).t()


class Polish(torch.nn.Module):
    """
    Both polish passes, with codebooks 0-1 final and 'bias' added to the attention scores towards the prefix: codebooks
    2-3 from the first, then 4-7 from the second, guided against the first. The log-softmax normalizers are constant
    per slot, so the guided argmax runs on the logits.
    """

    def __init__(self, echo: Echo) -> None:
        super().__init__()
        self.echo = echo

    def forward(self, tokens: torch.Tensor, bias: torch.Tensor, keys: torch.Tensor, values: torch.Tensor) -> torch.Tensor:
        mask = torch.cat([bias, bias.new_zeros(1, 1, 1, tokens.shape[1])], dim=-1)
        position, mid = 2 * keys.shape[3], self.echo.mask_id
        h, _, _ = self.echo.run(self.echo.embed_audio(tokens), position, (keys, values), mask)
        first = self.echo.logits(h, self.echo.polish_heads)[..., :mid]
        middle = first[:, :2].argmax(-1)
        tokens = torch.cat([tokens[:2], middle.t(), tokens[4:]])
        h, _, _ = self.echo.run(self.echo.embed_audio(tokens), position, (keys, values), mask)
        second = self.echo.logits(h, self.echo.polish_heads[:, 2 * self.echo.codes :])[..., :mid]
        fine = ((1 + POLISH_GUIDANCE) * second - POLISH_GUIDANCE * first[:, 2:]).argmax(-1)
        return torch.cat([middle, fine], dim=1).t()


def export_methods(echo: Echo) -> dict:
    layers, kv_heads, head_dim = len(echo.layers), echo.layers[0].kv_heads, echo.layers[0].head_dim
    prefix, frames = Dim("prefix", min=4, max=MAX_PREFIX), Dim("frames", min=4, max=MAX_FRAMES)
    text, reference = Dim("text", min=2, max=MAX_PREFIX), Dim("reference", min=2, max=MAX_PREFIX)
    cache = tuple(torch.zeros(layers, 1, kv_heads, 300, head_dim, dtype=DTYPE) for _ in range(2))
    tokens = torch.zeros(echo.books, 120, dtype=torch.long)

    with torch.no_grad():
        return {
            "prefill": export(
                Prefill(echo),
                (torch.zeros(60, dtype=torch.long), torch.zeros(echo.books, 240, dtype=torch.long)),
                dynamic_shapes=({0: text}, {1: reference}),
                strict=False,
            ),
            "step": export(
                Step(echo), (tokens, *cache), dynamic_shapes=({1: frames}, {3: prefix}, {3: prefix}), strict=False
            ),
            "polish": export(
                Polish(echo),
                (tokens, torch.zeros(1, 1, 1, 300, dtype=DTYPE), *cache),
                dynamic_shapes=({1: frames}, {3: prefix}, {3: prefix}, {3: prefix}),
                strict=False,
            ),
        }


def lower(methods: dict) -> bytes:
    edge = to_edge_transform_and_lower(
        methods,
        transform_passes=get_default_passes(),
        partitioner=[MLXPartitioner()],
        compile_config=EdgeCompileConfig(_check_ir_validity=False, _skip_dim_order=True),
    )
    for name in methods:
        stray = [
            node for node in edge.exported_program(name).graph.nodes
            if node.op == "call_function" and "executorch_call_delegate" not in str(node.target)
            and "getitem" not in str(node.target)
        ]
        if stray:
            raise RuntimeError(f"{name}: ops left outside the MLX delegate: {sorted({str(n.target) for n in stray})}")

    # Inputs are not copied into planned memory: the K/V cache goes back in on every pass.
    config = ExecutorchBackendConfig(
        extract_delegate_segments=True, memory_planning_pass=MemoryPlanningPass(alloc_graph_input=False)
    )
    return edge.to_executorch(config).buffer


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--checkpoint", type=Path, required=True, help="Checkpoint directory, e.g. runs/final/w4a16")
    parser.add_argument("--output", type=Path, required=True, help=".pte file to write; the tokenizer goes next to it")
    args = parser.parse_args()

    start = time.time()
    echo = Echo(args.checkpoint).eval()
    methods = export_methods(echo)
    print(f"Exported in {time.time() - start:.0f} s", flush=True)

    start = time.time()
    program = lower(methods)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_bytes(program)
    for name in ("tokenizer.json", "tokenizer_config.json"):
        shutil.copy(args.checkpoint / name, args.output.parent / name)
    print(f"Lowered in {time.time() - start:.0f} s: {args.output} ({len(program) / 1e6:.1f} MB)", flush=True)


if __name__ == "__main__":
    main()
