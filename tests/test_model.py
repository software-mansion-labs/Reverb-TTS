import json
from pathlib import Path

import pytest
import torch

from reverb_tts.model import CODEBOOKS, CODES, Network
from helpers import write_checkpoint


def test_full_precision_checkpoint_loads_with_centred_heads(network: Network, tmp_path: Path) -> None:
    loaded = Network.load(write_checkpoint(tmp_path, network))
    heads = loaded.audio_heads.weight.view(CODEBOOKS, CODES, -1)
    assert heads.mean(1).abs().max() < 1e-5
    hidden = torch.randn(3, 32)
    original = (hidden @ network.audio_heads.weight.T).view(3, CODEBOOKS, CODES).log_softmax(-1)
    centred = (hidden @ loaded.audio_heads.weight.T).view(3, CODEBOOKS, CODES).log_softmax(-1)
    assert torch.allclose(original, centred, atol=1e-4)


@pytest.mark.parametrize("bits, group", [(8, None), (4, 8)])
def test_quantized_checkpoints_are_expanded_on_load(network: Network, tmp_path: Path, bits: int, group: int) -> None:
    quantization = {"bits": bits, "group_size": group, "activations": None, "rotation": None}
    loaded = Network.load(write_checkpoint(tmp_path, network, quantization))
    for name in ("llm.layers.0.self_attn.q_proj.weight", "llm.layers.1.mlp.down_proj.weight"):
        original, expanded = network.state_dict()[name], loaded.state_dict()[name]
        step = original.abs().amax() / (2 ** (bits - 1) - 1)
        assert (original - expanded).abs().max() <= step / 2 + 1e-3
    table = network.state_dict()["llm.embed_tokens.table.weight"]
    assert (table - loaded.state_dict()["llm.embed_tokens.table.weight"]).abs().max() <= table.abs().amax() / 254 + 1e-3


def test_int8a8_checkpoint_rounds_projection_inputs(network: Network, tmp_path: Path) -> None:
    quantization = {"bits": 8, "group_size": None, "activations": "per_token", "rotation": None}
    loaded = Network.load(write_checkpoint(tmp_path, network, quantization))
    hooked = [m for m in loaded.llm.layers.modules() if isinstance(m, torch.nn.Linear) and m._forward_pre_hooks]
    assert len(hooked) == 7 * 2  # q, k, v, o, gate, up, down in each of 2 layers


def test_rotated_checkpoints_are_rejected(network: Network, tmp_path: Path) -> None:
    folder = write_checkpoint(tmp_path, network, {"bits": 8, "group_size": None, "activations": None, "rotation": None})
    config = json.loads((folder / "config.json").read_text())
    config["quantization"]["rotation"] = "hadamard_512"
    (folder / "config.json").write_text(json.dumps(config))
    with pytest.raises(NotImplementedError, match="rotated"):
        Network.load(folder)
