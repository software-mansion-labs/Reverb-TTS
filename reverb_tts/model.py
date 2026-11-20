"""
The network and its checkpoints: a Qwen3 backbone over text tokens and audio-codec tokens, loaded from a folder holding
config.json and model.safetensors - full precision, or a quantized variant whose weights are expanded on load.
"""

import json
from pathlib import Path
from typing import Dict, Optional

import torch
from safetensors.torch import load_file
from transformers import Qwen3Config, Qwen3Model

CODEBOOKS = 8         # codec codebooks per frame, from coarse (0) to fine (7)
CODES = 1025          # entries per codebook: 1024 codes and the mask
MASK = 1024           # the id of a masked slot, in every codebook
QUANTIZED = "llm.layers."   # the decoder projections, the only tensors a quantized checkpoint holds as int4 or int8 codes
SCALE = "_scale"      # suffix of the tensor holding a quantized tensor's scales


class FactorizedEmbedding(torch.nn.Module):
    """
    The text embedding as a rank-r product: a [vocab, r] table, then an r -> hidden projection.
    """

    def __init__(self, vocab: int, hidden: int, rank: int) -> None:
        super().__init__()
        self.table = torch.nn.Embedding(vocab, rank)
        self.project = torch.nn.Linear(rank, hidden, bias=False)

    def forward(self, ids: torch.Tensor) -> torch.Tensor:
        return self.project(self.table(ids))


class Network(torch.nn.Module):
    """
    A checkpoint's weights: the Qwen3 backbone with its text embedding, and the audio embeddings and heads. A frame's
    input is the sum of its 8 codebook embeddings; codebook c owns embedding rows and head rows c * 1025 + code.
    """

    def __init__(self, llm_config: Dict, rank: Optional[int] = None) -> None:
        super().__init__()
        self.llm = Qwen3Model(Qwen3Config.from_dict({**llm_config, "attn_implementation": "sdpa"}))
        hidden = self.llm.config.hidden_size
        if rank is not None:
            self.llm.embed_tokens = FactorizedEmbedding(self.llm.config.vocab_size, hidden, rank)
        self.audio_embeddings = torch.nn.Embedding(CODEBOOKS * CODES, hidden)
        self.audio_heads = torch.nn.Linear(hidden, CODEBOOKS * CODES, bias=False)

    @classmethod
    def load(cls, folder: Path) -> "Network":
        config = json.loads((folder / "config.json").read_text(encoding="utf-8"))
        weights = load_file(folder / "model.safetensors")
        quantization = config.get("quantization")
        if quantization:
            weights = dequantize(weights, quantization)
        weights = {key: value.float() if value.is_floating_point() else value for key, value in weights.items()}
        weights.pop("codebook_layer_offsets", None)

        table = weights.get("llm.embed_tokens.table.weight")
        network = cls(config["llm_config"], None if table is None else table.size(1))
        # Centring each codebook's head rows leaves its softmax unchanged, and keeps the logits small enough for bf16.
        heads = weights["audio_heads.weight"].view(CODEBOOKS, CODES, -1)
        weights["audio_heads.weight"] = (heads - heads.mean(1, keepdim=True)).flatten(0, 1)
        network.load_state_dict(weights)
        return network.eval()


def dequantize(weights: Dict[str, torch.Tensor], quantization: Dict) -> Dict[str, torch.Tensor]:
    """
    A quantized checkpoint's tensors as floats: each code times its scale. Decoder projections have one scale per
    output channel or per group of inputs; int8 embedding tables have one per row.
    """

    if quantization.get("rotation"):
        raise NotImplementedError("checkpoints with rotated projections are not supported")
    if quantization.get("activations"):
        raise NotImplementedError("checkpoints with quantized activations, such as int8a8, are not supported")
    bits, group = quantization["bits"], quantization.get("group_size")
    for key in [key for key in weights if key.endswith(SCALE)]:
        name, scale = key[: -len(SCALE)], weights.pop(key).float()
        codes = weights[name]
        projection = name.startswith(QUANTIZED)
        if projection and bits == 4:
            # Two codes per byte, the first in the low nibble, each stored as code + 8.
            codes = torch.stack([codes & 0xF, codes >> 4], dim=-1).flatten(-2).to(torch.int16) - 8
        codes = codes.float()
        if projection and group:
            weights[name] = (codes.unflatten(-1, (-1, group)) * scale[..., None]).flatten(-2)
        else:
            weights[name] = codes * scale
    return weights

