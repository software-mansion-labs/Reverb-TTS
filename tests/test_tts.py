import json
from pathlib import Path
from typing import List

import numpy as np
import pytest
import torch

import reverb_tts.tts as tts_module
from reverb_tts import TTS, SamplerConfig, Voice
from reverb_tts.sampler import Prompt
from reverb_tts.tts import _resolve

RATE = 24000


@pytest.fixture
def voice(tts: TTS) -> Voice:
    return tts.voice(np.zeros(2 * RATE, np.float32), "I have 2 cats.", "en", sample_rate=RATE)


def test_a_voice_holds_codes_and_a_prepared_transcript(voice: Voice) -> None:
    assert voice.codes.shape == (8, 50) and voice.codes.dtype == torch.int64
    assert voice.text == "I have two cats." and voice.duration == 2.0


def test_a_voice_survives_a_round_trip_to_disk(voice: Voice, tmp_path: Path) -> None:
    voice.save(tmp_path / "voice.npz")
    loaded = Voice.load(tmp_path / "voice.npz")
    assert torch.equal(loaded.codes, voice.codes) and (loaded.text, loaded.language) == (voice.text, voice.language)


def write_preset(path: Path, codes: List[int], language: str = "en") -> Path:
    path.write_text(json.dumps({"name": path.stem, "language": language, "transcript": "Hello there.", "codes": codes}))
    return path


def test_a_preset_file_loads_as_a_voice(tmp_path: Path) -> None:
    loaded = Voice.load(write_preset(tmp_path / "anna.json", list(range(8 * 5))))
    assert loaded.codes.shape == (8, 5) and loaded.codes[1, 0] == 5  # row-major: one codebook per row
    assert (loaded.text, loaded.language) == ("Hello there.", "en")


@pytest.mark.parametrize("codes, language", [(list(range(9)), "en"), ([1024] * 8, "en"), ([0] * 8, "xx")])
def test_a_malformed_preset_file_is_rejected(tmp_path: Path, codes: List[int], language: str) -> None:
    with pytest.raises(ValueError):
        Voice.load(write_preset(tmp_path / "bad.json", codes, language))


def test_preset_voices_come_from_a_local_folder(tts: TTS, tmp_path: Path) -> None:
    write_preset(tmp_path / "anna.json", [3] * 16)
    write_preset(tmp_path / "piotr.json", [4] * 16, "pl")
    tts.presets = tmp_path
    assert tts.preset_voices() == ["anna", "piotr"]
    assert tts.preset_voice("piotr").language == "pl"
    with pytest.raises(ValueError, match="anna, piotr"):
        tts.preset_voice("nobody")


def test_generate_returns_one_take_or_a_list(tts: TTS, voice: Voice) -> None:
    single = tts.generate("Hello there.", voice, "en", seed=0)
    assert isinstance(single, np.ndarray) and single.dtype == np.float32 and len(single) > 0
    batch = tts.generate(["Hello there.", "Hallo, wie geht's?"], voice, ["en", "de"], seed=0)
    assert isinstance(batch, list) and len(batch) == 2


def test_duration_sets_the_length(tts: TTS, voice: Voice) -> None:
    audio = tts.generate("Hello there.", voice, "en", seed=0, duration=2.0, trim=False)
    assert len(audio) == 50 * 960 - int(0.2 * RATE)


def test_sampler_overrides_reach_the_sampler(tts: TTS, voice: Voice, monkeypatch: pytest.MonkeyPatch) -> None:
    seen: List[SamplerConfig] = []
    original = tts_module.sample

    def spy(batch, seeds, config):
        seen.append(config)
        return original(batch, seeds, config)

    monkeypatch.setattr(tts_module, "sample", spy)
    tts.generate("Hello there.", voice, "en", seed=0, steps=3, remasking=False)
    assert seen[-1].steps == 3 and not seen[-1].remasking and seen[-1].polish_guidance == 1.0
    with pytest.raises(TypeError):
        tts.generate("Hello there.", voice, "en", seed=0, not_a_setting=1)


def test_a_long_text_is_spoken_in_pieces(tts: TTS, voice: Voice, monkeypatch: pytest.MonkeyPatch) -> None:
    sampled: List[Prompt] = []
    original = tts.__class__._sample

    def spy(self, prompts, config):
        sampled.extend(prompts)
        return original(self, prompts, config)

    monkeypatch.setattr(TTS, "_sample", spy)
    tts.generate(" ".join(["This sentence keeps the text going for a while."] * 40), voice, "en", seed=7)
    assert len(sampled) > 1 and [p.seed for p in sampled] == list(range(7, 7 + len(sampled)))


def test_mismatched_lists_are_rejected(tts: TTS, voice: Voice) -> None:
    with pytest.raises(ValueError, match="voices"):
        tts.generate(["One.", "Two."], [voice], "en")


def test_resolve_finds_a_release_layout(tmp_path: Path) -> None:
    (tmp_path / "model" / "int8").mkdir(parents=True)
    (tmp_path / "model" / "int8" / "config.json").write_text(json.dumps({}))
    (tmp_path / "model" / "tokenizer.json").write_text("{}")
    assert _resolve(tmp_path, "int8", None, None) == (tmp_path / "model" / "int8", tmp_path / "model")
    with pytest.raises(FileNotFoundError):
        _resolve(tmp_path, "int4", None, None)
    with pytest.raises(ValueError, match="variant"):
        _resolve(tmp_path, "int2", None, None)


def test_resolve_finds_the_safetensors_layout(tmp_path: Path) -> None:
    weights = tmp_path / "model" / "safetensors" / "int4"
    weights.mkdir(parents=True)
    (weights / "config.json").write_text(json.dumps({}))
    (tmp_path / "model" / "tokenizer.json").write_text("{}")
    assert _resolve(tmp_path, "int4", None, None) == (weights, tmp_path / "model")
