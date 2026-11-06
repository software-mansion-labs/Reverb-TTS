import csv
import logging
import numpy as np
import random
import torch

from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

from echo.eval.constants import EVAL_SEED, MODES, REPEATS
from echo.eval.metrics import (
    prosody,
    speaker_similarity,
    speaking_rate,
    transcribe,
    word_errors,
)
from echo.eval.misc import print_metrics
from echo.eval.types import Breakdown, MultiStepMode
from echo.model.model import EchoModel
from echo.utils.config_utils import check_choice
from echo.utils.seed_utils import seed_everything, seed_of
from omnivoice.models.omnivoice import VoiceClonePrompt

logger = logging.getLogger(__name__)


def build_test_suite(
    test_manifest_inputs: Union[str, Path],     # References: codec path, transcript, speaker id, language
    test_manifest_targets: Union[str, Path],    # Targets: speaker id, text, language
    data_root: Optional[Union[str, Path]] = None,   # Root the codec paths are relative to, defaults to the inputs' directory
) -> List[Dict[str, Any]]:
    """
    Pairs every reference voice with the texts it is asked to say.
    """

    def _read_manifest(path: Union[str, Path]) -> List[Dict[str, str]]:
        with open(path, newline="") as f:
            return list(csv.DictReader(f, delimiter="|", quoting=csv.QUOTE_NONE))

    root = Path(data_root) if data_root else Path(test_manifest_inputs).parent
    references: Dict[str, List[VoiceClonePrompt]] = {}

    for row in _read_manifest(test_manifest_inputs):
        codec = np.load(root / row["name"]).astype(np.int64)
        references.setdefault(row["speaker_id"], []).append(
            VoiceClonePrompt(
                ref_audio_tokens=torch.from_numpy(codec),
                ref_text=row["transcript"],
                # The references keep their own loudness, so the generated audio
                # is normalized on its own peak instead.
                ref_rms=None,
            )
        )

    suite = []

    for row in _read_manifest(test_manifest_targets):
        if row["speaker_id"] not in references:
            raise ValueError(f"no reference audio for speaker {row['speaker_id']!r}")

        suite.extend(
            {
                "text": row["text"],
                "language": row["language"],
                "voice_clone_prompt": prompt,
                # Not an argument of generate() - it is dropped before the call,
                # and only used to group the results by voice.
                "speaker_id": row["speaker_id"],
            }
            for prompt in references[row["speaker_id"]]
        )

    return suite


def evaluate_multi_step(
    model: EchoModel,
    test_manifest_inputs: Union[str, Path],     # References: codec path, transcript, speaker id, language
    test_manifest_targets: Union[str, Path],    # Targets: speaker id, text, language
    mode: MultiStepMode = "s8",                # The schedule the model generates with
    max_samples: int = 50,
    repeats: int = REPEATS,                     # Generations per sample, each from its own RNG state
    seed: int = EVAL_SEED,
    per_speaker: bool = False,                  # Adds a breakdown by reference voice to the report
    per_language: bool = False,                 # Adds a breakdown by language to the report
) -> Dict[str, Any]:
    """
    Runs the whole diffusion process and measures what comes out of it.

    Every reference-target pair is generated 'repeats' times, each from its own RNG state.
    """

    check_choice("mode", mode, MultiStepMode)

    # A model loaded for training carries no tokenizers, and this evaluation
    # decodes the references and generates - both need them.
    if model.audio_tokenizer is None or model.text_tokenizer is None:
        raise RuntimeError(
            "the model has no tokenizers - load it without 'train', or call "
            "echo.attach_tokenizers(model, path) first"
        )

    suite = build_test_suite(test_manifest_inputs, test_manifest_targets)[:max_samples]
    generation_config = model.echo_config.default_generation_config(**MODES[mode])

    references: Dict[int, np.ndarray] = {}
    generated: List[Tuple[np.ndarray, Dict[str, Any]]] = []
    lost = 0

    was_training = model.training
    model.eval()

    # The generation draws from the global RNG, so the caller's stream is put
    # back afterwards - otherwise the batches following an evaluation replay it.
    python_state, torch_state = random.getstate(), torch.get_rng_state()

    try:
        for index, sample in enumerate(suite):
            prompt = sample["voice_clone_prompt"]
            if id(prompt) not in references:
                references[id(prompt)] = _decode(model, prompt.ref_audio_tokens)

            arguments = {
                key: value for key, value in sample.items() if key != "speaker_id"
            }

            for repeat in range(repeats):
                seed_everything(seed_of(seed, index, repeat))

                # A cover for generations that accidentaly end in a near-silence.
                try:
                    audio = model.generate(
                        **arguments, generation_config=generation_config
                    )[0]
                except Exception as error:
                    logger.warning("generation %d/%d failed: %s", index, repeat, error)
                    lost += 1
                    continue

                if audio.size == 0:
                    logger.warning("generation %d/%d came back empty", index, repeat)
                    lost += 1
                    continue

                generated.append((audio, sample))
    finally:
        random.setstate(python_state)
        torch.set_rng_state(torch_state)
        model.train(was_training)

    if not generated:
        raise RuntimeError(f"every one of the {lost} generations failed")

    takes = _measure(model, generated, references)

    results = _aggregate(takes)
    results["lost"] = float(lost)

    speakers = _grouped(takes, "speaker") if per_speaker else None
    languages = _grouped(takes, "language") if per_language else None

    print_metrics(results, mode, speakers, languages)

    if per_speaker:
        results["speakers"] = speakers
    if per_language:
        results["languages"] = languages

    return results


def _grouped(takes: List[Dict[str, Any]], key: str) -> Breakdown:
    """
    The same numbers as the run's, taken over each value of one field.
    """

    return {
        value: _aggregate([take for take in takes if take[key] == value])
        for value in sorted({take[key] for take in takes})
    }


def _measure(
    model: EchoModel,
    generated: List[Tuple[np.ndarray, Dict[str, Any]]],
    references: Dict[int, np.ndarray],
) -> List[Dict[str, Any]]:
    """
    Scores every generated take on its own, so the run can be pooled as a whole
    and broken down by voice from the same numbers.
    """

    rate = model.sampling_rate
    spoken = transcribe(
        [audio for audio, _ in generated],
        [sample["language"] for _, sample in generated],
        rate,
    )

    takes = []

    for (audio, sample), hypothesis in zip(generated, spoken):
        reference = references[id(sample["voice_clone_prompt"])]

        errors, words = word_errors(sample["text"], hypothesis)
        delivery, reference_delivery = prosody(audio, rate), prosody(reference, rate)
        reference_text = sample["voice_clone_prompt"].ref_text

        takes.append({
            "speaker": sample["speaker_id"],
            "language": sample["language"],
            "errors": errors,
            "words": words,
            "similarity": speaker_similarity(audio, reference, rate),
            # Both deviations are signed: a positive rate means the model speaks
            # faster than the reference, a positive pause that it pauses more.
            "rate_dev": 100.0
            * (
                speaking_rate(sample["text"], delivery)
                / max(speaking_rate(reference_text, reference_delivery), 1e-6)
                - 1.0
            ),
            "pause_dev": 100.0
            * (reference_delivery["speech_frac"] - delivery["speech_frac"]),
        })

    return takes


def _aggregate(takes: List[Dict[str, Any]]) -> Dict[str, float]:
    """
    Pools the takes into the numbers that are reported.

    The word error rate is pooled over words rather than over takes, so a long
    target weighs more than a short one; the rest are plain means.
    """

    rate_deviations = [take["rate_dev"] for take in takes]
    pause_deviations = [take["pause_dev"] for take in takes]

    return {
        "wer": sum(t["errors"] for t in takes) / max(sum(t["words"] for t in takes), 1),
        "similarity": float(np.mean([take["similarity"] for take in takes])),
        "rate_dev": float(np.mean(rate_deviations)),
        "rate_abs": float(np.mean(np.abs(rate_deviations))),
        "pause_dev": float(np.mean(pause_deviations)),
        "pause_abs": float(np.mean(np.abs(pause_deviations))),
        "samples": float(len(takes)),
    }


def _decode(model: EchoModel, tokens: torch.Tensor) -> np.ndarray:
    """
    The waveform a reference's codec stands for, as the model hears it.
    """

    with torch.inference_mode():
        return (
            model.audio_tokenizer.decode(
                tokens.to(model.audio_tokenizer.device).unsqueeze(0)
            )
            .audio_values[0]
            .cpu()
            .numpy()
        )
