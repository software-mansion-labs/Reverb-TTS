import numpy as np
import pytest
import torch

from placeholder_tts.audio import load_audio, trim_silence

RATE = 24000


def test_trim_silence_keeps_a_margin_and_drops_edge_bursts() -> None:
    tone = 0.3 * np.sin(2 * np.pi * 220 * np.arange(RATE) / RATE).astype(np.float32)
    silence, burst = np.zeros(RATE, np.float32), np.full(RATE // 20, 0.3, np.float32)
    audio = np.concatenate([burst, silence, tone, silence])
    trimmed = trim_silence(audio, RATE)
    assert len(trimmed) == pytest.approx(len(tone) + 2 * RATE // 10, abs=RATE // 50)


def test_trim_silence_leaves_silent_and_tiny_audio_alone() -> None:
    assert len(trim_silence(np.zeros(RATE, np.float32), RATE)) == RATE
    assert len(trim_silence(np.zeros(10, np.float32), RATE)) == 10


def test_load_audio_mixes_down_and_resamples() -> None:
    stereo = np.random.default_rng(0).standard_normal((2, 48000)).astype(np.float32) * 0.1
    wav = load_audio(stereo, 48000, RATE)
    assert wav.shape == (RATE,) and wav.dtype == torch.float32


def test_load_audio_needs_the_rate_of_raw_samples() -> None:
    with pytest.raises(ValueError, match="sample_rate"):
        load_audio(np.zeros(100, np.float32), None, RATE)
