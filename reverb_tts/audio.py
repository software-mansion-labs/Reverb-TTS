"""
Reference audio in, generated audio out: loading and resampling a recording, measuring how long its speech lasts, and
trimming a take's silent edges.
"""

from pathlib import Path
from typing import Optional, Union

import numpy as np
import soundfile as sf
import torch
import torchaudio

# librosa's kaiser_best, which the training references were resampled with before encoding.
RESAMPLE = {"lowpass_filter_width": 64, "rolloff": 0.9475937167399596, "resampling_method": "sinc_interp_kaiser",
            "beta": 14.769656459379392}

AudioInput = Union[str, Path, np.ndarray, torch.Tensor]

SILENCE_DB = -50.0  # a 10 ms window quieter than this (dBFS) is silence


def load_audio(audio: AudioInput, sample_rate: Optional[int], target_rate: int) -> torch.Tensor:
    """
    A recording as mono float32 samples at `target_rate`: a file path, or samples ([samples] or [channels, samples])
    with their `sample_rate`.
    """

    if isinstance(audio, (str, Path)):
        samples, sample_rate = sf.read(str(audio), dtype="float32", always_2d=True)  # [samples, channels]
        wav = torch.from_numpy(samples.mean(1))
    else:
        if sample_rate is None:
            raise ValueError("sample_rate is required when audio is given as samples")
        wav = torch.as_tensor(audio, dtype=torch.float32)
        if wav.dim() == 2:
            wav = wav.mean(0)
        if wav.dim() != 1:
            raise ValueError(f"expected samples shaped [samples] or [channels, samples], got {tuple(wav.shape)}")
    if sample_rate != target_rate:
        wav = torchaudio.functional.resample(wav, sample_rate, target_rate, **RESAMPLE)
    return wav


def trim_silence(audio: np.ndarray, sample_rate: int, keep: float = 0.1) -> np.ndarray:
    """
    Cuts leading and trailing silence (10 ms windows under SILENCE_DB), keeping `keep` seconds of it on each side.
    """

    window = sample_rate // 100
    level = _levels(audio, window)
    loud = np.flatnonzero(level >= SILENCE_DB)
    if len(loud) == 0:
        return audio
    count = len(level)
    sounds = np.split(loud, np.flatnonzero(np.diff(loud) > 20) + 1)  # separated by over 200 ms of silence
    # A sound of at most 120 ms, alone at either edge, is a codec burst rather than speech.
    if len(sounds) > 1 and len(sounds[0]) <= 12:
        sounds = sounds[1:]
    if len(sounds) > 1 and len(sounds[-1]) <= 12:
        sounds = sounds[:-1]
    margin = round(keep * 100)
    return audio[max(sounds[0][0] - margin, 0) * window: min(sounds[-1][-1] + 1 + margin, count) * window]


def speech_length(audio: np.ndarray, sample_rate: int, max_pause: float = 0.2, lead: float = 0.1,
                  trail: float = 0.2) -> float:
    """
    Seconds of a reference recording that set the speaker's pace: the recording without its silent edges (keeping
    `lead` and `trail` seconds) and with every pause inside the speech counted as at most `max_pause` seconds.
    """

    window = sample_rate // 100
    count = len(audio) // window
    if count == 0:
        return len(audio) / sample_rate
    x = np.asarray(audio[: count * window], dtype=np.float64)
    level = _levels(x * 0.1 / max(float(np.sqrt(np.mean(x ** 2))), 1e-8), window)
    loud = np.flatnonzero(level >= SILENCE_DB)
    if len(loud) == 0:
        return len(audio) / sample_rate
    first, last = int(loud[0]), int(loud[-1])
    gaps = np.diff(loud) - 1  # silent windows between consecutive loud ones
    kept = len(loud) + int(np.minimum(gaps, round(max_pause * 100)).sum())
    edges = min(first, round(lead * 100)) + min(count - 1 - last, round(trail * 100))

    return (kept + edges) / 100


def _levels(audio: np.ndarray, window: int) -> np.ndarray:
    """The level of every whole `window` of samples, in dBFS."""

    count = len(audio) // window

    return 20 * np.log10(np.sqrt((audio[: count * window].reshape(count, window) ** 2).mean(1)) + 1e-10)
