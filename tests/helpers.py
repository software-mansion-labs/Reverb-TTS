import json
from pathlib import Path
from types import SimpleNamespace
from typing import Dict, Optional

import torch
from safetensors.torch import save_file

from placeholder_tts.model import QUANTIZED, SCALE, Network

LLM_CONFIG = {
    "vocab_size": 64, "hidden_size": 32, "intermediate_size": 64, "num_hidden_layers": 2, "num_attention_heads": 4,
    "num_key_value_heads": 2, "head_dim": 8, "max_position_embeddings": 1024, "rms_norm_eps": 1e-6,
    "rope_theta": 10000.0, "tie_word_embeddings": False,
}
TABLES = ("llm.embed_tokens.table.weight", "audio_embeddings.weight")


def quantize_state(state: Dict[str, torch.Tensor], bits: int, group: Optional[int]) -> Dict[str, torch.Tensor]:
    """
    The release format: decoder projections as int8 (per output channel) or packed int4 (per group of inputs) codes,
    embedding tables as int8 with one fp16 scale per row, everything else in bf16.
    """

    out = {}
    for key, value in state.items():
        if key.startswith(QUANTIZED) and key.endswith("proj.weight"):
            limit = 2 ** (bits - 1) - 1
            grouped = value.unflatten(-1, (-1, group)) if group else value
            scale = grouped.abs().amax(-1, keepdim=True) / limit
            codes = (grouped / scale).round().clamp(-limit, limit)
            if group:
                codes, scale = codes.flatten(-2), scale.squeeze(-1)
            if bits == 4:
                nibbles = (codes.to(torch.int16) + 8).to(torch.uint8).view(*codes.shape[:-1], -1, 2)
                codes = nibbles[..., 0] | (nibbles[..., 1] << 4)
            else:
                codes = codes.to(torch.int8)
            out[key], out[key + SCALE] = codes, scale.half()
        elif key in TABLES:
            scale = value.abs().amax(-1, keepdim=True) / 127
            out[key], out[key + SCALE] = (value / scale).round().to(torch.int8), scale.half()
        else:
            out[key] = value.bfloat16() if value.is_floating_point() else value
    return out


def write_checkpoint(folder: Path, network: Network, quantization: Optional[Dict] = None) -> Path:
    folder.mkdir(parents=True, exist_ok=True)
    state = {key: value.detach().clone().contiguous() for key, value in network.state_dict().items()}
    config: Dict = {"llm_config": LLM_CONFIG}
    if quantization:
        state = quantize_state(state, quantization["bits"], quantization.get("group_size"))
        config["quantization"] = quantization
    save_file(state, folder / "model.safetensors")
    (folder / "config.json").write_text(json.dumps(config))
    return folder


class StubTokenizer:
    """
    Token ids from characters, as many as the prompt has.
    """

    def __call__(self, prompt: str, return_tensors: str) -> SimpleNamespace:
        return SimpleNamespace(input_ids=torch.tensor([[ord(c) % LLM_CONFIG["vocab_size"] for c in prompt[:200]]]))


class StubCodec:
    """
    The codec's interface: 960 samples per frame at 24 kHz; decodes every frame to a tone.
    """

    config = SimpleNamespace(sample_rate=24000, hop_length=960)

    def encode(self, wav: torch.Tensor) -> SimpleNamespace:
        frames = wav.size(-1) // 960
        return SimpleNamespace(audio_codes=(torch.arange(8 * frames).view(1, 8, frames) * 7) % 1024)

    def decode(self, codes: torch.Tensor) -> SimpleNamespace:
        t = torch.arange(codes.size(-1) * 960) / 24000
        return SimpleNamespace(audio_values=(0.3 * torch.sin(2 * torch.pi * 220 * t))[None, None])
