"""Writes a checkpoint with its decoder linears quantized: int8 or int4 weights (W8A16, W4A16), and their inputs too (W8A8) with --activations."""

import argparse
from pathlib import Path

from echo import EchoConfig, build_model
from echo.model.quantization import quantize, save_quantized
from echo.training.checkpoint import _copy_tokenizer
from omnivoice.models.omnivoice import OmniVoiceConfig, _resolve_model_path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--init", type=Path, required=True, help="Checkpoint to quantize, e.g. the last one of a QAT run")
    parser.add_argument("--output", type=Path, required=True, help="Directory the quantized checkpoint is written to")
    parser.add_argument("--activations", action="store_true", help="Quantizes the inputs of the linears as well (W8A8)")
    parser.add_argument("--bits", type=int, default=8, choices=(4, 8), help="Width of the quantized weights")
    parser.add_argument("--group-size", type=int, help="Inputs sharing one weight scale - one scale per output channel if unset")
    args = parser.parse_args()

    # The checkpoint keeps its own settings, so it generates as it did before.
    init = _resolve_model_path(str(args.init))
    config = EchoConfig.from_omnivoice_config(OmniVoiceConfig.from_pretrained(init))
    config.init_from = init

    model = build_model(config, train=True)
    print(f"Quantized {quantize(model, args.activations, args.bits, args.group_size)} linears")

    save_quantized(model, args.output)
    _copy_tokenizer(Path(init), args.output)


if __name__ == "__main__":
    main()
