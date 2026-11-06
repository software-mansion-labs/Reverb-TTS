"""Exports the W8A8 Echo checkpoint to ExecuTorch for the XNNPACK backend (CPU): the checkpoint's int8 weights, and the
inputs of every decoder linear quantized to int8 on the fly, one asymmetric scale per token. Everything else is fp32.

The program has the methods of scripts/export/export_mlx.py's, with the cache as one key and one value tensor per layer:
- prefill(text [L], reference [8, R]) -> keys [kv heads, head dim, P], values [kv heads, P, head dim], for every layer.
- step(tokens [8, T], *cache) -> confidence, prediction [2, T]: one main pass, codebooks 0-1 only.
- polish(tokens [8, T], bias [1, 1, 1, P], *cache) -> [6, T]: both polish passes, codebooks 2-7.

XNNPACK takes no slices of tensors with dynamic shapes, so the graph is built to need none: q and k come as each head's
two halves from separate projections, which RoPE combines elementwise.

Run it with the ExecuTorch venv: ../executorch/.venv/bin/python scripts/export/export_xnnpack.py ...
Run the program with 6 threads on an M4 Pro (scripts/export/run_mlx.py --threads 6): more cost more than they give.
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

import torchao  # noqa: F401 - registers torchao's quantization ops

from executorch.backends.xnnpack.partition.xnnpack_partitioner import XnnpackPartitioner
from executorch.exir import EdgeCompileConfig, ExecutorchBackendConfig, to_edge_transform_and_lower
from executorch.exir.passes import MemoryPlanningPass

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


class QuantizedLinear(torch.nn.Module):
    """
    A projection with the checkpoint's int8 weights, one scale per output channel, and int8 inputs, quantized per token
    on the fly. XNNPACK runs it as a single int8 matmul.
    """

    def __init__(self, values: torch.Tensor, scale: torch.Tensor) -> None:
        super().__init__()
        self.values = frozen(values.contiguous())
        self.scale = frozen(scale.reshape(-1, 1).float())
        self.zero = frozen(torch.zeros_like(self.scale, dtype=torch.int8))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        block = [1, x.shape[-1]]
        scale, zero = torch.ops.torchao.choose_qparams_affine(x, "ASYMMETRIC", block, torch.int8, -128, 127, None, None, torch.int8)
        x = torch.ops.torchao.quantize_affine(x, block, scale, zero, torch.int8, -128, 127)
        x = torch.ops.torchao.dequantize_affine(x, block, scale, zero, torch.int8, -128, 127)
        weight = torch.ops.torchao.dequantize_affine(self.values, block, self.scale, self.zero, torch.int8, -128, 127)
        return F.linear(x, weight)


def rms(x: torch.Tensor, weight: torch.Tensor, eps: float) -> torch.Tensor:
    """
    RMSNorm of [n, d] rows. XNNPACK only averages over the two inner dims of a 4-D tensor.
    """

    x4 = x.view(1, -1, 1, x.shape[-1])
    return (x4 * torch.rsqrt((x4 * x4).mean(dim=(2, 3), keepdim=True) + eps) * weight).view(-1, x.shape[-1])


def norm_rope(a: torch.Tensor, b: torch.Tensor, wa: torch.Tensor, wb: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor,
              eps: float) -> torch.Tensor:
    """
    Per-head RMSNorm and RoPE, given each head's first (a) and second (b) halves, [n, heads * half] each, and cos and
    sin as [1, n * heads, 1, half]. Returns [1, n * heads, 1, head dim].

    The norm averages over a half (a^2 + b^2): twice the full mean, which the weights' factor sqrt(2) makes up for.
    """

    a, b = a.view(1, -1, 1, cos.shape[-1]), b.view(1, -1, 1, cos.shape[-1])
    r = torch.rsqrt((a * a + b * b).mean(dim=(2, 3), keepdim=True) + 2 * eps)
    a, b = a * wa, b * wb
    return torch.cat([a * cos - b * sin, b * cos + a * sin], dim=-1) * r


def halves(t: torch.Tensor, heads: int) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    The rows of a per-head projection, [heads * head dim, ...], split into every head's first and second halves.
    """

    t = t.view(heads, 2, -1, *t.shape[1:])
    return t[:, 0].flatten(0, 1), t[:, 1].flatten(0, 1)


class Layer(torch.nn.Module):
    def __init__(self, prefix: str, weights: dict, config: dict) -> None:
        super().__init__()
        self.heads, self.kv_heads, self.head_dim, self.eps = config["heads"], config["kv_heads"], config["head_dim"], config["eps"]

        def linear(name: str) -> Tuple[torch.Tensor, torch.Tensor]:
            return weights[f"{prefix}{name}.weight"], weights[f"{prefix}{name}.weight_scale"]

        (q, q_scale), (k, k_scale) = linear("self_attn.q_proj"), linear("self_attn.k_proj")
        (q1, q2), (q1_scale, q2_scale) = halves(q, self.heads), halves(q_scale, self.heads)
        (k1, k2), (k1_scale, k2_scale) = halves(k, self.kv_heads), halves(k_scale, self.kv_heads)
        self.q1, self.q2 = QuantizedLinear(q1, q1_scale), QuantizedLinear(q2, q2_scale)
        self.k1, self.k2 = QuantizedLinear(k1, k1_scale), QuantizedLinear(k2, k2_scale)
        self.v = QuantizedLinear(*linear("self_attn.v_proj"))
        self.o = QuantizedLinear(*linear("self_attn.o_proj"))
        self.gate = QuantizedLinear(*linear("mlp.gate_proj"))
        self.up = QuantizedLinear(*linear("mlp.up_proj"))
        self.down = QuantizedLinear(*linear("mlp.down_proj"))
        self.input_norm = frozen(weights[prefix + "input_layernorm.weight"].float())
        self.post_norm = frozen(weights[prefix + "post_attention_layernorm.weight"].float())
        # The attention's 1 / sqrt(head dim) goes into the q norm.
        q_norm = weights[prefix + "self_attn.q_norm.weight"].float() * 2**0.5 / self.head_dim**0.5
        k_norm = weights[prefix + "self_attn.k_norm.weight"].float() * 2**0.5
        self.q_norm1, self.q_norm2 = (frozen(w) for w in q_norm.chunk(2))
        self.k_norm1, self.k_norm2 = (frozen(w) for w in k_norm.chunk(2))

    def forward(
        self,
        x: torch.Tensor,
        rope: Tuple[Tuple[torch.Tensor, torch.Tensor], Tuple[torch.Tensor, torch.Tensor]],
        past: Optional[Tuple[torch.Tensor, torch.Tensor]],
        mask: Optional[torch.Tensor],
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        [n, hidden] -> [n, hidden], and the n positions' keys [kv heads, head dim, n] and values [kv heads, n, head dim].
        """

        d, groups = self.head_dim, self.kv_heads
        h = rms(x, self.input_norm, self.eps)
        q = norm_rope(self.q1(h), self.q2(h), self.q_norm1, self.q_norm2, *rope[0], self.eps)
        k = norm_rope(self.k1(h), self.k2(h), self.k_norm1, self.k_norm2, *rope[1], self.eps)
        # Each kv head serves 2 q heads: their rows are stacked, so one batched matmul per kv head needs no repeat.
        permute = torch.ops.aten.permute_copy
        q = permute(q.view(-1, groups, self.heads // groups, d), [1, 0, 2, 3]).view(groups, -1, d)
        k = permute(k.view(-1, groups, d), [1, 2, 0])
        v = permute(self.v(h).view(-1, groups, d), [1, 0, 2])
        keys, values = (k, v) if past is None else (torch.cat([past[0], k], dim=2), torch.cat([past[1], v], dim=1))
        scores = torch.bmm(q, keys)
        if mask is not None:
            scores = scores + mask
        a = torch.bmm(torch.softmax(scores, dim=-1), values)
        x = x + self.o(permute(a.view(groups, -1, self.heads // groups, d), [1, 0, 2, 3]).view(-1, self.heads * d))
        h = rms(x, self.post_norm, self.eps)
        gate = self.gate(h)
        return x + self.down(gate * torch.sigmoid(gate) * self.up(h)), k, v


class Echo(torch.nn.Module):
    """
    The checkpoint's weights: the decoder linears in int8, everything else in fp32.
    """

    def __init__(self, checkpoint: Path) -> None:
        super().__init__()
        config = json.loads(Path(checkpoint, "config.json").read_text())
        quantization = config.get("quantization") or {}
        if quantization.get("bits") != 8 or quantization.get("activations") != "per_token" or quantization.get("rotation"):
            raise NotImplementedError("this export is for the W8A8 checkpoint: int8 weights, int8 inputs, no rotation")
        llm = config["llm_config"]
        geometry = {
            "heads": llm["num_attention_heads"],
            "kv_heads": llm["num_key_value_heads"],
            "head_dim": llm["head_dim"],
            "eps": llm["rms_norm_eps"],
        }
        self.books, self.codes, self.mask_id = config["num_audio_codebook"], config["audio_vocab_size"], config["audio_mask_id"]
        # The mask is the last code, so dropping the last column drops it from every prediction.
        assert self.mask_id == self.codes - 1

        weights = load_file(Path(checkpoint, "model.safetensors"))
        weights = {key: value.float() if value.dtype in (torch.bfloat16, torch.float16) else value for key, value in weights.items()}
        self.layers = torch.nn.ModuleList(
            Layer(f"llm.layers.{i}.", weights, geometry) for i in range(llm["num_hidden_layers"])
        )
        self.eps, self.heads, self.kv_heads = geometry["eps"], geometry["heads"], geometry["kv_heads"]
        self.norm = frozen(weights["llm.norm.weight"])
        # The text embedding stays factorized, as the checkpoint stores it: a [vocab, rank] table and a projection.
        self.table = frozen(weights["llm.embed_tokens.table.weight"])
        self.project = frozen(weights["llm.embed_tokens.project.weight"])
        self.audio = frozen(weights["audio_embeddings.weight"])
        self.offsets = frozen(self.codes * torch.arange(self.books)[:, None])
        half = geometry["head_dim"] // 2
        self.inv_freq = frozen(1.0 / float(llm["rope_parameters"]["rope_theta"]) ** (torch.arange(half) / half))
        self.q_ones = frozen(torch.ones(1, 1, self.heads, half))
        self.kv_ones = frozen(torch.ones(1, 1, self.kv_heads, half))

        # Centering each codebook's rows leaves its softmax as it is, and keeps the logits small. The heads are split by
        # codebook, so that every pass multiplies by whole parameters: XNNPACK takes no slice of one.
        heads = weights["audio_heads.weight"].view(self.books, self.codes, -1)
        heads = (heads - heads.mean(1, keepdim=True)).flatten(0, 1)
        self.main_heads = frozen(heads[: 2 * self.codes].contiguous())
        self.middle_heads = frozen(heads[2 * self.codes : 4 * self.codes].contiguous())
        self.fine_heads = frozen(heads[4 * self.codes :].contiguous())

    def embed_audio(self, codes: torch.Tensor) -> torch.Tensor:
        """
        [8, n] codes -> [n, hidden]: each frame is the sum of its 8 codebook embeddings.
        """

        rows = codes + self.offsets
        x = F.embedding(rows[0], self.audio)
        for book in range(1, self.books):
            x = x + F.embedding(rows[book], self.audio)
        return x

    def rope(self, start: int, n: int) -> Tuple[Tuple[torch.Tensor, torch.Tensor], Tuple[torch.Tensor, torch.Tensor]]:
        """
        cos and sin of positions [start, start + n), repeated over the q heads and over the kv heads.
        """

        angles = (torch.arange(n, dtype=torch.float32) + start)[:, None] * self.inv_freq[None, :]
        cos, sin = angles.cos().view(1, -1, 1, angles.shape[1]), angles.sin().view(1, -1, 1, angles.shape[1])

        def tile(t: torch.Tensor, ones: torch.Tensor) -> torch.Tensor:
            return (t * ones).view(1, -1, 1, t.shape[-1])

        return (tile(cos, self.q_ones), tile(sin, self.q_ones)), (tile(cos, self.kv_ones), tile(sin, self.kv_ones))

    def run(
        self, x: torch.Tensor, start: int, past: Optional[Tuple[torch.Tensor, ...]], mask: Optional[torch.Tensor]
    ) -> Tuple[torch.Tensor, List[torch.Tensor]]:
        rope = self.rope(start, x.shape[0])
        cache = []
        for i, layer in enumerate(self.layers):
            x, k, v = layer(x, rope, None if past is None else (past[2 * i], past[2 * i + 1]), mask)
            cache += [k, v]

        return rms(x, self.norm, self.eps), cache

    def logits(self, h: torch.Tensor, heads: torch.Tensor) -> torch.Tensor:
        """
        [n, hidden] -> [n, codebooks, codes].
        """

        return F.linear(h, heads).view(h.shape[0], -1, self.codes)


class Prefill(torch.nn.Module):
    def __init__(self, echo: Echo) -> None:
        super().__init__()
        self.echo = echo

    def forward(self, text: torch.Tensor, reference: torch.Tensor) -> Tuple[torch.Tensor, ...]:
        x = torch.cat([F.linear(F.embedding(text, self.echo.table), self.echo.project), self.echo.embed_audio(reference)])
        _, cache = self.echo.run(x, 0, None, None)
        return tuple(cache)


class Step(torch.nn.Module):
    """
    One main pass: the log-probability of the best code of codebooks 0-1 in every slot, and that code. The softmax
    counts the mask, but the mask is never a prediction.
    """

    def __init__(self, echo: Echo) -> None:
        super().__init__()
        self.echo = echo

    def forward(self, tokens: torch.Tensor, *past: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        h, _ = self.echo.run(self.echo.embed_audio(tokens), past[1].shape[1], past, None)
        log_probs = torch.log_softmax(self.echo.logits(h, self.echo.main_heads), dim=-1)[..., : self.echo.mask_id]
        return log_probs.amax(-1).t(), log_probs.argmax(-1).t()


class Polish(torch.nn.Module):
    """
    Both polish passes, with codebooks 0-1 final and 'bias' added to the attention scores towards the prefix: codebooks
    2-3 from the first, then 4-7 from the second, guided against the first. The log-softmax normalizers are constant
    per slot, so the guided argmax runs on the logits.
    """

    def __init__(self, echo: Echo) -> None:
        super().__init__()
        self.echo = echo

    def forward(self, tokens: torch.Tensor, bias: torch.Tensor, *past: torch.Tensor) -> torch.Tensor:
        bias = bias.view(1, 1, -1)
        mask = torch.cat([bias, bias.new_zeros(1, 1, tokens.shape[1])], dim=-1)
        start, mid = past[1].shape[1], self.echo.mask_id
        h, _ = self.echo.run(self.echo.embed_audio(tokens), start, past, mask)
        middle = self.echo.logits(h, self.echo.middle_heads)[..., :mid].argmax(-1)
        first = self.echo.logits(h, self.echo.fine_heads)[..., :mid]
        tokens = torch.cat([tokens[:2], middle.t(), tokens[4:]])
        h, _ = self.echo.run(self.echo.embed_audio(tokens), start, past, mask)
        second = self.echo.logits(h, self.echo.fine_heads)[..., :mid]
        fine = ((1 + POLISH_GUIDANCE) * second - POLISH_GUIDANCE * first).argmax(-1)
        return torch.cat([middle, fine], dim=1).t()


def export_methods(echo: Echo) -> dict:
    layers, kv_heads, head_dim = len(echo.layers), echo.kv_heads, echo.layers[0].head_dim
    prefix, frames = Dim("prefix", min=4, max=MAX_PREFIX), Dim("frames", min=4, max=MAX_FRAMES)
    text, reference = Dim("text", min=2, max=MAX_PREFIX), Dim("reference", min=2, max=MAX_PREFIX)
    cache = []
    for _ in range(layers):
        cache += [torch.zeros(kv_heads, head_dim, 300), torch.zeros(kv_heads, 300, head_dim)]
    cache_shapes = tuple({2: prefix} if i % 2 == 0 else {1: prefix} for i in range(2 * layers))
    tokens = torch.zeros(echo.books, 120, dtype=torch.long)

    with torch.no_grad():
        return {
            "prefill": export(
                Prefill(echo),
                (torch.zeros(60, dtype=torch.long), torch.zeros(echo.books, 240, dtype=torch.long)),
                dynamic_shapes=({0: text}, {1: reference}),
                strict=False,
            ),
            "step": export(Step(echo), (tokens, *cache), dynamic_shapes=({1: frames}, cache_shapes), strict=False),
            "polish": export(
                Polish(echo),
                (tokens, torch.zeros(1, 1, 1, 300), *cache),
                dynamic_shapes=({1: frames}, {3: prefix}, cache_shapes),
                strict=False,
            ),
        }


def lower(methods: dict) -> bytes:
    edge = to_edge_transform_and_lower(
        methods, partitioner=[XnnpackPartitioner()], compile_config=EdgeCompileConfig(_check_ir_validity=False)
    )
    # Inputs are not copied into planned memory: the cache goes back in on every pass.
    config = ExecutorchBackendConfig(
        extract_delegate_segments=True, memory_planning_pass=MemoryPlanningPass(alloc_graph_input=False)
    )
    return edge.to_executorch(config).buffer


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--checkpoint", type=Path, required=True, help="The W8A8 checkpoint directory, e.g. runs/final/w8a8")
    parser.add_argument("--output", type=Path, required=True, help=".pte file to write; the tokenizer goes next to it")
    args = parser.parse_args()

    start = time.time()
    methods = export_methods(Echo(args.checkpoint).eval())
    print(f"Exported in {time.time() - start:.0f} s; lowering takes about 10 minutes", flush=True)

    start = time.time()
    program = lower(methods)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_bytes(program)
    for name in ("tokenizer.json", "tokenizer_config.json"):
        shutil.copy(args.checkpoint / name, args.output.parent / name)
    print(f"Lowered in {time.time() - start:.0f} s: {args.output} ({len(program) / 1e6:.1f} MB)", flush=True)


if __name__ == "__main__":
    main()
