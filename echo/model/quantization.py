import copy
import torch

from pathlib import Path
from safetensors.torch import load_file, save_file
from transformers.utils import SAFE_WEIGHTS_NAME
from typing import Optional, Tuple, Union

from echo.model.model import EchoModel
from omnivoice.models.omnivoice import OmniVoiceConfig

# Only the decoder projections are quantized. The embeddings, the audio heads and the norms are a small share
# of the model, and stay in full precision.
QUANTIZED = "llm.layers."

# Suffix of the tensor holding a quantized weight's scales in a checkpoint.
SCALE = "_scale"

# Inputs whose width is not a power of two (o_proj's 1536) are rotated in blocks of this size.
ROTATION_BLOCK = 512


def _quantize(x: torch.Tensor, bits: int = 8, group: Optional[int] = None) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Symmetric quantization along the last dim, one scale per row - or per group of 'group' entries of it.
    Returns the rounded values, still as floats, and their scales; with a group, both keep it as an extra dim.
    """

    if group is not None:
        x = x.unflatten(-1, (-1, group))
    limit = 2 ** (bits - 1) - 1
    scale = x.abs().amax(dim=-1, keepdim=True).clamp(min=1e-12) / limit

    return (x / scale).round().clamp(-limit, limit), scale


def fake_quant(x: torch.Tensor, bits: int = 8, group: Optional[int] = None) -> torch.Tensor:
    """
    Rounds x to its grid. The gradient passes through as if it was never rounded (straight-through estimator).
    """

    full = x.float()
    values, scale = _quantize(full, bits, group)
    rounded = (values * scale).flatten(-2) if group is not None else values * scale

    return (full + (rounded - full).detach()).to(x.dtype)


def _pack(values: torch.Tensor) -> torch.Tensor:
    """
    int4 values in [-7, 7], as floats -> uint8 holding two of them, the first one in the low nibble.
    """

    nibbles = (values.to(torch.int16) + 8).to(torch.uint8).view(*values.shape[:-1], -1, 2)

    return nibbles[..., 0] | (nibbles[..., 1] << 4)


def _unpack(packed: torch.Tensor) -> torch.Tensor:
    """
    The inverse of _pack: uint8 -> int4 values, as floats.
    """

    nibbles = torch.stack([packed & 0xF, packed >> 4], dim=-1).flatten(-2)

    return nibbles.float() - 8


def hadamard(n: int) -> torch.Tensor:
    """
    The orthogonal rotation an input of width n is quantized in: a normalized Hadamard matrix, block-diagonal
    when n is not a power of two.
    """

    size = n if n & (n - 1) == 0 else ROTATION_BLOCK
    block = torch.ones(1, 1)
    while block.shape[0] < size:
        block = torch.cat([torch.cat([block, block], 1), torch.cat([block, -block], 1)], 0)

    return torch.block_diag(*[block / size**0.5] * (n // size))


class QuantizedLinear(torch.nn.Linear):
    """
    A decoder projection computing with quantized weights - 'bits' wide, one scale per output channel or per group
    of 'group_size' inputs - and optionally int8 inputs, one scale per token.

    With a rotation, the weight is held as W @ H and the input is turned to x @ H before the rounding - the product
    stays the same, while outliers that move between channels from token to token are spread over all of them.
    """

    activations: bool = False
    bits: int = 8
    group_size: Optional[int] = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.rotation is not None:
            x = x @ self.rotation.to(x.dtype)
        if self.activations:
            x = fake_quant(x)

        return torch.nn.functional.linear(x, fake_quant(self.weight, self.bits, self.group_size), self.bias)


def quantize(model: EchoModel, activations: bool, bits: int = 8, group_size: Optional[int] = None) -> int:
    """
    Turns the decoder projections into QuantizedLinear in place - e.g. W8A8 with 'activations', W4A16 with
    bits=4, a group size and no 'activations'.

    The parameters stay as they are, so the checkpoint keys, the optimizer and the gradient exchange see no difference.
    """

    count = 0
    for name, module in model.named_modules():
        if name.startswith(QUANTIZED) and isinstance(module, torch.nn.Linear):
            module.__class__ = QuantizedLinear
            module.activations = activations
            module.bits, module.group_size = bits, group_size
            # A buffer, so it follows the model across devices - never saved, as the checkpoint names it instead.
            module.register_buffer("rotation", None, persistent=False)
            count += 1

    return count


@torch.no_grad()
def rotate(model: EchoModel) -> None:
    """
    Rotates the inputs of the quantized linears, and their weights with them - an exact change of basis.
    """

    for module in model.modules():
        if isinstance(module, QuantizedLinear):
            module.register_buffer("rotation", hadamard(module.in_features).to(module.weight.device), persistent=False)
            module.weight.copy_(module.weight @ module.rotation)


def save_quantized(model: EchoModel, path: Union[str, Path]) -> None:
    """
    Writes a quantized model: int8 weights (int4 ones packed two per byte) and fp32 scales for its QuantizedLinear,
    bf16 for every other float.
    """

    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)

    quantized = {
        f"{name}.weight": module
        for name, module in model.named_modules()
        if isinstance(module, QuantizedLinear)
    }
    if not quantized:
        raise ValueError("the model has no quantized linears - call quantize() first")
    first = next(iter(quantized.values()))
    if first.bits not in (4, 8):
        raise NotImplementedError("only int8 and int4 weights are written so far")

    state = {}
    for key, tensor in model.state_dict().items():
        # The codec is never part of a checkpoint.
        if key.startswith("audio_tokenizer."):
            continue
        if key in quantized:
            values, scale = _quantize(tensor.float(), first.bits, first.group_size)
            if first.group_size is not None:
                values, scale = values.flatten(-2), scale.squeeze(-1)
            values = _pack(values) if first.bits == 4 else values.to(torch.int8)
            state[key], state[key + SCALE] = values.cpu(), scale.cpu()
        else:
            state[key] = (tensor.to(torch.bfloat16) if tensor.is_floating_point() else tensor).cpu()

    config = copy.deepcopy(model.config)
    config.quantization = {
        "bits": first.bits,
        "symmetric": True,
        "weights": "per_channel" if first.group_size is None else "per_group",
        "group_size": first.group_size,
        "packing": "uint8_low_first" if first.bits == 4 else None,
        "activations": "per_token" if first.activations else None,
        "rotation": f"hadamard_{ROTATION_BLOCK}" if first.rotation is not None else None,
        "modules": f"{QUANTIZED}*",
    }
    config.save_pretrained(path)
    save_file(state, path / SAFE_WEIGHTS_NAME, metadata={"format": "pt"})


def load_quantized(config: OmniVoiceConfig, path: Union[str, Path]) -> EchoModel:
    """
    Builds a model from a checkpoint written by save_quantized: the weights dequantized, the projections quantized again.
    """

    quantization = config.quantization
    bits, group = quantization["bits"], quantization.get("group_size")
    model = EchoModel(config)

    state = load_file(Path(path) / SAFE_WEIGHTS_NAME)
    for key in [key for key in state if key.endswith(SCALE)]:
        scale = state.pop(key)
        weight = key[: -len(SCALE)]
        # Embedding tables stored in int8, one scale per row (embeddings: int8_per_row).
        if not weight.startswith(QUANTIZED):
            state[weight] = state[weight].float() * scale.float()
            continue
        values = _unpack(state[weight]) if bits == 4 else state[weight].float()
        if group is not None:
            values = (values.unflatten(-1, (-1, group)) * scale[..., None]).flatten(-2)
        else:
            values = values * scale
        state[weight] = values
    model.load_state_dict(state, strict=False)

    # The weights are plain floats now, and a save writes them as such.
    del model.config.quantization
    quantize(model, quantization["activations"] is not None, bits, group)

    # The weights were stored rotated, so only the inputs are left to turn.
    if quantization.get("rotation") is not None:
        for module in model.modules():
            if isinstance(module, QuantizedLinear):
                module.register_buffer("rotation", hadamard(module.in_features), persistent=False)

    return model
