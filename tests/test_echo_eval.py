import numpy as np
import pytest
import torch

from dataclasses import dataclass
from typing import Any, Dict, List, Optional

from echo.data.constants import AUDIO_MASK_ID, IGNORE_INDEX
from echo.eval.single_step import evaluate_single_step
from echo.data.inputs import build_model_inputs

CODEBOOKS = 8
VOCAB = 16


def _sample(prompt: int, text: int, frames: int, prefix_frames: int) -> Dict[str, Any]:
    return {
        "prompt_tokens": torch.arange(prompt),
        "text_tokens": torch.arange(text),
        "audio_tokens": torch.randint(0, VOCAB, (CODEBOOKS, frames)),
        "prefix_length": prompt + text + prefix_frames,
        "target_length": frames - prefix_frames,
    }


def _batch(samples: List[Dict[str, Any]], languages: Optional[List[str]] = None) -> Dict[str, List[Any]]:
    batch = {key: [sample[key] for sample in samples] for key in samples[0]}
    batch["language"] = languages or ["english"] * len(samples)
    lengths = [
        sample["prefix_length"] + sample["target_length"] for sample in samples
    ]
    longest = max(lengths)
    positions = torch.arange(longest).unsqueeze(0)
    batch["valid"] = positions < torch.tensor(lengths).unsqueeze(1)

    return batch


# ---------- Model inputs ----------


def test_model_inputs_carry_the_masked_positions_only() -> None:
    samples = [_sample(4, 6, 30, 5), _sample(3, 5, 32, 8)]
    batch = _batch(samples)
    masks = [torch.rand(sample["audio_tokens"].shape) < 0.5 for sample in samples]

    inputs = build_model_inputs(batch, masks)
    length = samples[0]["prefix_length"] + samples[0]["target_length"]

    assert inputs["input_ids"].shape == (2, CODEBOOKS, length)
    assert inputs["labels"].shape == inputs["input_ids"].shape
    assert inputs["attention_mask"].shape == (2, 1, length, length)

    for index, (sample, mask) in enumerate(zip(samples, masks)):
        prefix = sample["prompt_tokens"].numel() + sample["text_tokens"].numel()
        audio_in = inputs["input_ids"][index, :, prefix:]
        audio_labels = inputs["labels"][index, :, prefix:]

        assert not inputs["audio_mask"][index, :prefix].any()
        assert inputs["audio_mask"][index, prefix:].all()
        assert (audio_in[mask] == AUDIO_MASK_ID).all()
        assert torch.equal(audio_in[~mask], sample["audio_tokens"][~mask])
        assert torch.equal(audio_labels[mask], sample["audio_tokens"][mask])
        assert (audio_labels[~mask] == IGNORE_INDEX).all()
        assert (inputs["labels"][index, :, :prefix] == IGNORE_INDEX).all()


def test_model_inputs_accept_a_dropped_prefix() -> None:
    sample = _sample(0, 0, 40, 0)
    inputs = build_model_inputs(_batch([sample]))

    assert inputs["input_ids"].shape == (1, CODEBOOKS, 40)
    assert inputs["audio_mask"].all()
    assert "labels" not in inputs


# ---------- The sweep ----------


@dataclass
class _StubOutput:
    loss: torch.Tensor
    logits: torch.Tensor


class _StubModel:
    """A model whose logits depend only on the inputs, so runs stay comparable."""

    normalized_audio_codebook_weights = [1.0 / CODEBOOKS] * CODEBOOKS
    training = False

    def __init__(self, bias: float = 0.0) -> None:
        self.bias = bias

    def parameters(self):
        yield torch.zeros(1)

    def eval(self) -> None:
        pass

    def train(self, mode: bool = True) -> None:
        pass

    def __call__(
        self, input_ids: torch.Tensor, labels: Optional[torch.Tensor] = None, **_: Any
    ) -> _StubOutput:
        logits = (input_ids.unsqueeze(-1) % VOCAB).float().expand(
            *input_ids.shape, VOCAB
        ) + self.bias
        return _StubOutput(loss=logits.mean(), logits=logits)


def _loader() -> List[Dict[str, List[Any]]]:
    torch.manual_seed(3)
    return [_batch([_sample(4, 6, 30, 5), _sample(3, 5, 32, 8)]) for _ in range(2)]


def test_sweep_reports_every_grid_point_and_repeats_exactly() -> None:
    loader = _loader()
    trajectory = {"s_f0": ((1.0,) * CODEBOOKS, 0.5), "s_f1": ((0.5,) * CODEBOOKS, 0.5)}

    first = evaluate_single_step(_StubModel(), _StubModel(0.5), loader, trajectory=trajectory)
    second = evaluate_single_step(_StubModel(), _StubModel(0.5), loader, trajectory=trajectory)

    assert list(first) == ["s_f0", "s_f1", "s_polish", "s_main", "s"]
    assert first == second
    assert all(row["positions"] > 0 for row in first.values())
    assert first["s_f0"]["positions"] > first["s_f1"]["positions"]


def test_sweep_scores_traced_states_and_pools_them_by_weight() -> None:
    loader = _loader()
    trajectory = {"s_f0": ((1.0,) * CODEBOOKS, 0.25), "s_f1": ((0.5,) * CODEBOOKS, 0.75)}

    results = evaluate_single_step(_StubModel(), _StubModel(0.5), loader, trajectory=trajectory)

    # A half masked state scores about half the positions of a fully masked one.
    assert results["s_f1"]["positions"] == pytest.approx(results["s_f0"]["positions"] / 2, rel=0.15)
    for key in ("ce", "ce_teacher", "ce_delta", "kl"):
        assert results["s_main"][key] == pytest.approx(0.25 * results["s_f0"][key] + 0.75 * results["s_f1"][key])
        # Main path and polish split as training splits the loss.
        assert results["s"][key] == pytest.approx(0.75 * results["s_main"][key] + 0.25 * results["s_polish"][key])

    # Main-path states are scored on codebooks 0-1, the polish state on 2-7.
    frames = sum(b["target_length"][i] for b in loader for i in range(2)) * 3
    assert results["s_f0"]["positions"] == 2 * frames
    assert results["s_polish"]["positions"] == 6 * frames


def test_sweep_leaves_the_caller_rng_untouched() -> None:
    loader = _loader()

    torch.manual_seed(11)
    expected = torch.rand(4)

    torch.manual_seed(11)
    evaluate_single_step(_StubModel(), None, loader, repeats=2)

    assert torch.equal(torch.rand(4), expected)


# ---------- The full diffusion process ----------


def test_word_errors_counts_edits_against_the_truth() -> None:
    from echo.eval.metrics import word_errors

    assert word_errors("The cat sat.", "the cat sat") == (0, 3)
    assert word_errors("the cat sat", "the dog sat") == (1, 3)
    assert word_errors("the cat sat", "the cat") == (1, 3)
    assert word_errors("the cat sat", "the big cat sat") == (1, 3)


def test_prosody_separates_speech_from_silence() -> None:
    from echo.eval.metrics import prosody, speaking_rate

    rate = 16_000
    speech = np.random.default_rng(0).normal(0, 0.2, rate).astype(np.float32)
    clip = np.concatenate([speech, np.zeros(rate, dtype=np.float32)])

    measured = prosody(clip, rate)

    assert 1.9 < measured["duration_s"] < 2.1
    assert 0.4 < measured["speech_frac"] < 0.6
    # Six words over roughly one second of speech.
    assert 4.0 < speaking_rate("one two three four five six", measured) < 8.0


def test_lost_generations_are_counted_and_the_run_survives(monkeypatch) -> None:
    from echo.eval import multi_step

    rng = np.random.default_rng(0)
    takes = [
        np.zeros(0, dtype=np.float32),              # Trimmed away to nothing
        rng.normal(0, 0.1, 8_000).astype(np.float32),
        ValueError("zero-size array to reduction operation maximum"),
        rng.normal(0, 0.1, 8_000).astype(np.float32),
    ]

    class _Model:
        training = False
        sampling_rate = 16_000
        echo_config = type(
            "_Config", (), {"default_generation_config": lambda self, **kw: kw}
        )()
        # The stubs decode through a patched '_decode', but the guard still
        # checks a real model would be able to.
        audio_tokenizer = object()
        text_tokenizer = object()

        def eval(self) -> None:
            pass

        def train(self, mode: bool = True) -> None:
            pass

        def generate(self, **_: Any) -> List[np.ndarray]:
            take = takes.pop(0)
            if isinstance(take, Exception):
                raise take
            return [take]

    prompt = type("_Prompt", (), {"ref_audio_tokens": None, "ref_text": "a reference"})()
    suite = [
        {
            "text": "one two three",
            "language": "en",
            "voice_clone_prompt": prompt,
            "speaker_id": f"speaker-{index}",
        }
        for index in range(2)
    ]

    monkeypatch.setattr(multi_step, "build_test_suite", lambda *a, **k: suite)
    monkeypatch.setattr(
        multi_step, "_decode", lambda model, tokens: np.zeros(16_000, dtype=np.float32)
    )
    monkeypatch.setattr(multi_step, "transcribe", lambda audios, languages, rate: ["one two three"] * len(audios))
    monkeypatch.setattr(multi_step, "speaker_similarity", lambda *a: 0.9)

    results = multi_step.evaluate_multi_step(
        _Model(), "inputs.csv", "targets.csv", max_samples=2, repeats=2
    )

    assert results["lost"] == 2.0
    assert results["samples"] == 2.0
    assert results["wer"] == 0.0


def test_per_speaker_breaks_the_run_down_by_voice(monkeypatch) -> None:
    from echo.eval import multi_step

    rng = np.random.default_rng(1)

    class _Model:
        training = False
        sampling_rate = 16_000
        echo_config = type(
            "_Config", (), {"default_generation_config": lambda self, **kw: kw}
        )()
        # The stubs decode through a patched '_decode', but the guard still
        # checks a real model would be able to.
        audio_tokenizer = object()
        text_tokenizer = object()

        def eval(self) -> None:
            pass

        def train(self, mode: bool = True) -> None:
            pass

        def generate(self, **kwargs: Any) -> List[np.ndarray]:
            assert "speaker_id" not in kwargs   # It is not an argument of generate
            return [rng.normal(0, 0.1, 8_000).astype(np.float32)]

    prompt = type("_Prompt", (), {"ref_audio_tokens": None, "ref_text": "a reference"})()
    suite = [
        {
            "text": "one two three",
            "language": "en",
            "voice_clone_prompt": prompt,
            "speaker_id": speaker,
        }
        for speaker in ("anna", "bob")
    ]

    monkeypatch.setattr(multi_step, "build_test_suite", lambda *a, **k: suite)
    monkeypatch.setattr(
        multi_step, "_decode", lambda model, tokens: np.zeros(16_000, dtype=np.float32)
    )
    monkeypatch.setattr(
        multi_step, "transcribe", lambda audios, languages, rate: ["one two three"] * len(audios)
    )
    monkeypatch.setattr(multi_step, "speaker_similarity", lambda *a: 0.9)

    results = multi_step.evaluate_multi_step(
        _Model(), "inputs.csv", "targets.csv", max_samples=2, repeats=2, per_speaker=True
    )

    assert sorted(results["speakers"]) == ["anna", "bob"]
    assert all(row["samples"] == 2.0 for row in results["speakers"].values())
    assert results["samples"] == 4.0

    plain = multi_step.evaluate_multi_step(
        _Model(), "inputs.csv", "targets.csv", max_samples=2, repeats=2
    )
    assert "speakers" not in plain


def test_weighted_ce_matches_the_models_own_loss() -> None:
    from echo.utils.loss_utils import weighted_ce

    torch.manual_seed(5)
    logits = torch.randn(2, CODEBOOKS, 12, VOCAB)
    labels = torch.randint(0, VOCAB, (2, CODEBOOKS, 12))
    labels[:, :, :4] = IGNORE_INDEX
    weights = torch.full((CODEBOOKS,), 1.0 / CODEBOOKS)

    # The reduction the base model applies to its own logits.
    per_token = torch.nn.functional.cross_entropy(
        logits.permute(0, 3, 1, 2), labels, reduction="none", ignore_index=IGNORE_INDEX
    )
    scored = (labels != IGNORE_INDEX).float()
    expected = (
        (per_token * scored).sum(dim=(0, 2)) / scored.sum(dim=(0, 2)).clamp(min=1.0) * weights
    ).sum()

    assert weighted_ce(logits, labels, weights) == pytest.approx(float(expected))


def test_single_step_breaks_down_by_language() -> None:
    loader = [
        _batch(
            [_sample(4, 6, 30, 5), _sample(3, 5, 32, 8)],
            languages=["english", "polish"],
        )
    ]

    trajectory = {"s_f0": ((1.0,) * CODEBOOKS, 0.25), "s_f1": ((0.5,) * CODEBOOKS, 0.75)}
    results = evaluate_single_step(
        _StubModel(), _StubModel(0.5), loader, trajectory=trajectory, per_language=True
    )

    languages = results["languages"]
    assert sorted(languages) == ["english", "polish"]

    # Every scored position belongs to exactly one language, and the languages
    # are pooled over the whole grid.
    assert sum(row["positions"] for row in languages.values()) == results["s"]["positions"]

    plain = evaluate_single_step(_StubModel(), None, loader, trajectory=trajectory)
    assert "languages" not in plain


def test_multi_step_breaks_down_by_language(monkeypatch) -> None:
    from echo.eval import multi_step

    rng = np.random.default_rng(2)

    class _Model:
        training = False
        sampling_rate = 16_000
        echo_config = type(
            "_Config", (), {"default_generation_config": lambda self, **kw: kw}
        )()
        # The stubs decode through a patched '_decode', but the guard still
        # checks a real model would be able to.
        audio_tokenizer = object()
        text_tokenizer = object()

        def eval(self) -> None:
            pass

        def train(self, mode: bool = True) -> None:
            pass

        def generate(self, **_: Any) -> List[np.ndarray]:
            return [rng.normal(0, 0.1, 8_000).astype(np.float32)]

    prompt = type("_Prompt", (), {"ref_audio_tokens": None, "ref_text": "a reference"})()
    suite = [
        {
            "text": "one two three",
            "language": language,
            "voice_clone_prompt": prompt,
            "speaker_id": speaker,
        }
        for speaker, language in (("anna", "en"), ("bob", "pl"), ("cleo", "en"))
    ]

    monkeypatch.setattr(multi_step, "build_test_suite", lambda *a, **k: suite)
    monkeypatch.setattr(
        multi_step, "_decode", lambda model, tokens: np.zeros(16_000, dtype=np.float32)
    )
    monkeypatch.setattr(
        multi_step, "transcribe", lambda audios, languages, rate: ["one two three"] * len(audios)
    )
    monkeypatch.setattr(multi_step, "speaker_similarity", lambda *a: 0.9)

    results = multi_step.evaluate_multi_step(
        _Model(), "inputs.csv", "targets.csv", max_samples=3, repeats=2,
        per_speaker=True, per_language=True,
    )

    assert sorted(results["languages"]) == ["en", "pl"]
    assert results["languages"]["en"]["samples"] == 4.0   # Two speakers, twice each
    assert results["languages"]["pl"]["samples"] == 2.0
    assert sorted(results["speakers"]) == ["anna", "bob", "cleo"]


def test_multi_step_says_what_is_missing_without_tokenizers() -> None:
    from echo.model.config import EchoConfig
    from echo.eval import evaluate_multi_step
    from echo.model.model import EchoModel
    from tests.test_echo import _base_dict

    model = EchoModel(EchoConfig(base=_base_dict()))

    with pytest.raises(RuntimeError, match="attach_tokenizers"):
        evaluate_multi_step(model, "inputs.csv", "targets.csv", max_samples=1)
