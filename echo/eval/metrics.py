import logging
import numpy as np
import unicodedata
import warnings

from contextlib import contextmanager
from functools import lru_cache
from typing import Any, Dict, Iterator, List, Tuple

from echo.eval.constants import (
    ASR_COMPUTE_TYPE,
    ASR_MODEL,
    METRIC_DEVICE,
    METRIC_SAMPLE_RATE,
    SPEAKER_MODEL,
    VOICED_RATIO,
)


# ---------- Transcription ----------


def transcribe(
    audios: List[np.ndarray],   # Waveforms, one per sample
    languages: List[str],       # Language of every waveform, as a two letter code
    sample_rate: int,
) -> List[str]:
    """
    Transcribes every waveform with Whisper, in the language it is spoken in.
    """

    model = _asr()
    spoken = []

    with _quiet():
        for audio, language in zip(audios, languages):
            # The segments are a generator, and transcribing only happens while
            # it is consumed.
            segments, _ = model.transcribe(
                _resampled(audio, sample_rate), language=language, task="transcribe"
            )
            spoken.append("".join(segment.text for segment in segments).strip())

    return spoken


def word_errors(reference: str, hypothesis: str) -> Tuple[int, int]:
    """
    Substitutions, deletions and insertions between the two texts, and the
    number of words they are measured against.

    The counts are returned raw rather than as a ratio, so a whole run can be
    pooled over its words instead of over its samples.
    """

    truth, spoken = _normalize(reference), _normalize(hypothesis)

    # Levenshtein distance over words, kept to two rows.
    previous = list(range(len(spoken) + 1))

    for i, word in enumerate(truth, start=1):
        current = [i]
        for j, other in enumerate(spoken, start=1):
            current.append(
                previous[j - 1]
                if word == other
                else 1 + min(previous[j - 1], previous[j], current[j - 1])
            )
        previous = current

    return previous[-1], len(truth)


# ---------- Speaker identity ----------


def speaker_similarity(
    generated: np.ndarray,
    reference: np.ndarray,
    sample_rate: int,
) -> float:
    """
    Cosine similarity of the two x-vectors, in [-1, 1].

    The same speaker lands around 0.95, two different ones around 0.45.
    """

    import torch

    extractor, model = _speaker()
    embeddings = []

    with _quiet(), torch.no_grad():
        for audio in (generated, reference):
            inputs = extractor(
                _resampled(audio, sample_rate),
                sampling_rate=METRIC_SAMPLE_RATE,
                return_tensors="pt",
            ).to(model.device)
            embeddings.append(
                torch.nn.functional.normalize(model(**inputs).embeddings, dim=-1)[0]
            )

    return float(embeddings[0] @ embeddings[1])


# ---------- Delivery ----------


def prosody(audio: np.ndarray, sample_rate: int) -> Dict[str, float]:
    """
    How much of the clip is spoken, and how long it is.

    A frame counts as voiced when its energy clears a fraction of the clip's own
    median, so the split follows the recording instead of a fixed threshold.
    """

    import librosa

    audio = _resampled(audio, sample_rate)
    energy = librosa.feature.rms(y=audio, frame_length=1024, hop_length=256)[0]

    spoken = energy[energy > 0]
    threshold = VOICED_RATIO * np.median(spoken) if spoken.size else 0.0
    voiced = energy > threshold

    return {
        "speech_frac": float(voiced.mean()) if voiced.size else 0.0,
        "duration_s": float(len(audio) / METRIC_SAMPLE_RATE),
    }


def speaking_rate(text: str, prosody: Dict[str, float]) -> float:
    """
    Words per second of spoken audio, pauses left out of the count.
    """

    spoken_seconds = prosody["speech_frac"] * prosody["duration_s"]

    return len(_normalize(text)) / max(spoken_seconds, 1e-6)


# ---------- Loading, quietly ----------


@lru_cache(maxsize=1)
def _asr() -> Any:
    """
    The transcriber, loaded once and kept.
    """

    with _quiet():
        from faster_whisper import WhisperModel

        device = _device()

        return WhisperModel(
            ASR_MODEL, device=device, compute_type=ASR_COMPUTE_TYPE[device]
        )


@lru_cache(maxsize=1)
def _speaker() -> Tuple[Any, Any]:
    """
    The x-vector model and its feature extractor, loaded once and kept.
    """

    with _quiet():
        from transformers import AutoFeatureExtractor, WavLMForXVector

        return (
            AutoFeatureExtractor.from_pretrained(SPEAKER_MODEL),
            WavLMForXVector.from_pretrained(SPEAKER_MODEL).to(_device()).eval(),
        )


def _device() -> str:
    """
    Where the metric models run: the accelerator when there is a usable one.
    """

    import torch

    return METRIC_DEVICE if torch.cuda.is_available() else "cpu"


@contextmanager
def _quiet() -> Iterator[None]:
    """
    Keeps model loading and inference out of the console.

    The metric models are loaded during an evaluation whose output is a table,
    and their warnings, progress bars and config dumps would break it apart.
    """

    from transformers.utils import logging as hf_logging

    verbosity = hf_logging.get_verbosity()
    hf_logging.set_verbosity_error()
    hf_logging.disable_progress_bar()
    logging.disable(logging.INFO)

    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            yield
    finally:
        logging.disable(logging.NOTSET)
        hf_logging.enable_progress_bar()
        hf_logging.set_verbosity(verbosity)


# ---------- Helpers ----------


def _resampled(audio: np.ndarray, sample_rate: int) -> np.ndarray:
    """
    The waveform as the metric models want it: mono, float32, 16 kHz.
    """

    import librosa

    audio = np.asarray(audio, dtype=np.float32)
    if audio.ndim > 1:
        audio = audio.mean(axis=0)

    if sample_rate != METRIC_SAMPLE_RATE:
        audio = librosa.resample(
            audio, orig_sr=sample_rate, target_sr=METRIC_SAMPLE_RATE
        )

    return audio


def _normalize(text: str) -> List[str]:
    """
    The words a text is scored as: lowercased, and stripped of punctuation.

    Punctuation is dropped by unicode category rather than by a list, so the
    languages of the test set are treated alike.
    """

    stripped = "".join(
        " " if unicodedata.category(char).startswith("P") else char
        for char in text.lower()
    )

    return stripped.split()
