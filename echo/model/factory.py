import json
import logging
import os
import torch

from pathlib import Path
from typing import Any, Dict, Optional, Set, Union
from safetensors import safe_open
from transformers import AutoFeatureExtractor, AutoTokenizer, HiggsAudioV2TokenizerModel

from echo.model.config import EchoConfig
from echo.model.model import EchoModel
from echo.model.quantization import load_quantized
from omnivoice.models.omnivoice import RuleDurationEstimator, _resolve_model_path

logger = logging.getLogger(__name__)

# The codec the model was trained with, when a checkpoint carries none of its own.
DEFAULT_AUDIO_TOKENIZER = "eustlb/higgs-audio-v2-tokenizer"

# The attention every model here is built with, teacher and student alike.
#
# Prefix blocking speaks in dense 4-D masks, and that is the SDPA path: flex
# attention wants a BlockMask, which the forward only builds for packed
# sequences, so it would be handed a tensor it cannot read. Left unset the
# backend follows whatever the installed transformers happens to default to -
# which is why every training script in the reference repo pins it too.
ATTENTION = "sdpa"


def _checkpoint_keys(path: str) -> Set[str]:
    index = Path(path) / "model.safetensors.index.json"
    if index.is_file():
        weight_map: Dict[str, Any] = json.loads(index.read_text())["weight_map"]
        return set(weight_map)
    with safe_open(Path(path) / "model.safetensors", framework="pt") as f:
        return set(f.keys())


def build_model(
    config: Union[str, Path, EchoConfig],
    *,
    train: bool = False,   # Loads the model to be trained: no tokenizers attached, and it comes back in training mode
    device: Optional[Union[str, torch.device]] = None,
    dtype: Optional[torch.dtype] = None,
    strict_weights: bool = False,
) -> EchoModel:
    """
    Builds the model a config describes, loading ``init_from`` weights if set.

    Inference needs the text tokenizer and the audio codec, and they are loaded
    from the checkpoint's own directory - which a checkpoint written during
    training does not have. Such a checkpoint is therefore loadable with
    'train' alone.
    """

    echo_config = (
        config if isinstance(config, EchoConfig) else EchoConfig.from_json(config)
    )
    omnivoice_config = echo_config.to_omnivoice_config()
    omnivoice_config.llm_config._attn_implementation = ATTENTION

    if echo_config.init_from is None:
        model = EchoModel(echo_config)

        if dtype is not None:
            model = model.to(dtype)
    else:
        path = _resolve_model_path(echo_config.init_from)
        if getattr(omnivoice_config, "quantization", None) is not None:
            model = load_quantized(omnivoice_config, path)
            if dtype is not None:
                model = model.to(dtype)
            if not train:
                attach_tokenizers(model, path)
        else:
            kwargs: Dict[str, Any] = {
                "config": omnivoice_config,
                "train": train,
                "attn_implementation": ATTENTION,
            }
            if dtype is not None:
                kwargs["dtype"] = dtype
            model = EchoModel.from_pretrained(path, **kwargs)

        # The audio tokenizer is attached from its own subdirectory, so its
        # weights are never part of the checkpoint itself.
        expected = {
            key
            for key in model.state_dict()
            if not key.startswith("audio_tokenizer.")
        }
        
        missing = sorted(expected - _checkpoint_keys(path))
        if missing:
            message = f"{len(missing)} parameter(s) not in {path}: {missing[:5]}"
            if strict_weights:
                raise ValueError(message)
            logger.warning("Randomly initialized %s", message)

    if device is not None:
        model = model.to(device)

    return model.train() if train else model.eval()


# Inference helper
def attach_tokenizers(
    model: EchoModel,
    path: Union[str, Path],   # A model directory holding the tokenizer files, such as the one the run started from
) -> EchoModel:
    """
    Gives a model loaded for training what it needs to generate in the multi step evaluation phase.

    A checkpoint written mid-run carries weights alone, so a student that has to
    be evaluated by generating takes its tokenizers from elsewhere - usually the
    pretrained model the stage started from.
    """

    resolved = _resolve_model_path(str(path))
    codec = os.path.join(resolved, "audio_tokenizer")

    if not os.path.isdir(codec):
        codec = _resolve_model_path(DEFAULT_AUDIO_TOKENIZER)

    model.text_tokenizer = AutoTokenizer.from_pretrained(resolved)

    # The codec stays in fp32, and off MPS, exactly as the inference path loads it.
    device = "cpu" if str(model.device).startswith("mps") else model.device
    model.audio_tokenizer = HiggsAudioV2TokenizerModel.from_pretrained(
        codec, device_map=device
    ).to(model.device, torch.float32)

    model.feature_extractor = AutoFeatureExtractor.from_pretrained(codec)
    model.sampling_rate = model.feature_extractor.sampling_rate
    model.duration_estimator = RuleDurationEstimator()

    return model
