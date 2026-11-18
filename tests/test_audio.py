import numpy as np
import pytest
import torch

from reverb_tts.audio import LEVEL_DB, PEAK, load_audio, raise_quiet, speech_length, trim_silence

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


def tone(seconds: float, amplitude: float = 0.3) -> np.ndarray:
    return amplitude * np.sin(2 * np.pi * 220 * np.arange(round(seconds * RATE)) / RATE).astype(np.float32)


def quiet(seconds: float) -> np.ndarray:
    return np.zeros(round(seconds * RATE), np.float32)


def test_speech_length_keeps_short_edges_and_caps_pauses() -> None:
    # 1 s of edge silence on each side, a 1.5 s and a 0.3 s pause between 1 s sounds.
    audio = np.concatenate([quiet(1), tone(1), quiet(1.5), tone(1), quiet(0.3), tone(1), quiet(1)])
    # 3 s of sound, the long pause counted as 0.5 s, the sentence-length one whole, and 0.1 s + 0.2 s of the edges.
    assert speech_length(audio, RATE) == pytest.approx(3 + 0.5 + 0.3 + 0.1 + 0.2, abs=0.02)
    assert speech_length(audio, RATE, max_pause=0.2) == pytest.approx(3 + 0.2 + 0.2 + 0.1 + 0.2, abs=0.02)


def test_speech_length_of_clean_speech_is_its_length() -> None:
    assert speech_length(tone(2), RATE) == pytest.approx(2, abs=0.01)


def test_speech_length_levels_a_quiet_recording_first() -> None:
    # The whole recording sits under -50 dBFS, yet its sounds are told from its silence.
    audio = np.concatenate([tone(1, 1e-4), quiet(2), tone(1, 1e-4)])
    assert speech_length(audio, RATE) == pytest.approx(2.5, abs=0.02)


def test_speech_length_counts_a_silent_or_tiny_recording_whole() -> None:
    assert speech_length(quiet(2), RATE) == 2
    assert speech_length(np.zeros(10, np.float32), RATE) == 10 / RATE


def test_raise_quiet_levels_only_a_quiet_recording() -> None:
    tone = torch.sin(2 * torch.pi * 220 * torch.arange(RATE) / RATE) * np.sqrt(2)  # RMS 1
    def db(x: torch.Tensor) -> float:
        return 20 * float(torch.log10(x.square().mean().sqrt()))
    assert db(raise_quiet(tone * 10 ** (-60 / 20))) == pytest.approx(LEVEL_DB, abs=0.01)
    typical = tone * 10 ** (-30 / 20)
    assert torch.equal(raise_quiet(typical), typical)
    assert torch.equal(raise_quiet(torch.zeros(RATE)), torch.zeros(RATE))


def test_raise_quiet_keeps_a_loud_click_under_the_peak() -> None:
    # A -60 dBFS tone with one click at 0.5: raising it to LEVEL_DB would take the click to ~50, so the peak limits it.
    tone = torch.sin(2 * torch.pi * 220 * torch.arange(RATE) / RATE) * np.sqrt(2) * 10 ** (-60 / 20)
    tone[RATE // 2] = 0.5
    raised = raise_quiet(tone)
    assert float(raised.abs().max()) == pytest.approx(PEAK, abs=1e-6)
    assert float(raised.square().mean().sqrt()) > float(tone.square().mean().sqrt())  # still raised, as far as the peak allows
