"""Tests for the echo data loader, on a synthetic manifest so they need no
dataset on disk."""

import random
from pathlib import Path
from typing import List

import numpy as np
import pytest
import torch

from echo.data.constants import AUDIO_MASK_ID, IGNORE_INDEX
from echo.data.inputs import build_model_inputs
from echo.data import (
    EchoBatchSampler,
    EchoDataLoader,
    EchoDataset,
    EchoPreprocessor,
)
from echo.masking.audio_masking import build_main_stage_audio_mask, build_polish_stage_audio_mask

CODEBOOKS = 8
PREFIX_TOKENS = 40
MAX_BATCH_SIZE = 16
MAX_BATCH_TOKENS = 2048
MAX_PAD_RATIO = 0.3
BATCHING = {
    "max_batch_size": MAX_BATCH_SIZE,
    "max_batch_tokens": MAX_BATCH_TOKENS,
    "max_pad_ratio": MAX_PAD_RATIO,
}


def _manifest(tmp_path: Path, frame_counts: List[int]) -> Path:
    """A manifest plus matching codecs, laid out like data/{train,dev}."""
    codecs = tmp_path / "lang" / "codecs"
    codecs.mkdir(parents=True)

    lines = ["name|transcript|speaker_id|language|tokens"]
    for i, frames in enumerate(frame_counts):
        codec = np.random.randint(0, 1024, (CODEBOOKS, frames), dtype=np.int16)
        np.save(codecs / f"utt{i}.npy", codec)
        lang = "en" if i % 2 else "pl"
        lines.append(
            f"lang/codecs/utt{i}.npy|transcript {i}|spk{i % 7}|{lang}|{frames + PREFIX_TOKENS}"
        )

    manifest = tmp_path / "manifest.csv"
    manifest.write_text("\n".join(lines) + "\n")
    return manifest


def _dataset(tmp_path: Path, count: int = 400, seed: int = 0) -> EchoDataset:
    rng = random.Random(seed)
    frames = [rng.randint(50, 900) for _ in range(count)]
    return EchoDataset(_manifest(tmp_path, frames))


def test_manifest_columns_round_trip(tmp_path: Path) -> None:
    dataset = _dataset(tmp_path, count=5)
    sample = dataset[3]

    assert len(dataset) == 5
    assert sample["text"] == "transcript 3"
    assert sample["speaker_id"] == "spk3"
    assert sample["language"] == "en"
    assert sample["audio_tokens"].dtype == torch.int64
    assert sample["audio_tokens"].shape[0] == CODEBOOKS
    assert sample["audio_tokens"].shape[1] == sample["tokens"] - PREFIX_TOKENS

    on_disk = np.load(tmp_path / "lang" / "codecs" / "utt3.npy")
    assert torch.equal(sample["audio_tokens"], torch.from_numpy(on_disk.astype(np.int64)))


def test_batches_respect_every_limit(tmp_path: Path) -> None:
    dataset = _dataset(tmp_path)

    for batch in EchoBatchSampler(dataset, **BATCHING).build_batches(epoch=0):
        lengths = dataset.tokens[batch]
        padded = len(batch) * int(lengths.max())

        assert len(batch) <= MAX_BATCH_SIZE
        assert padded <= MAX_BATCH_TOKENS
        assert 1 - lengths.sum() / padded <= MAX_PAD_RATIO + 1e-9


def test_every_sample_appears_exactly_once(tmp_path: Path) -> None:
    dataset = _dataset(tmp_path)
    batches = EchoBatchSampler(dataset, **BATCHING).build_batches(epoch=0)

    placed = [index for batch in batches for index in batch]
    assert sorted(placed) == list(range(len(dataset)))


def test_long_samples_are_batched_too(tmp_path: Path) -> None:
    """The longest samples must not be starved by the padding limit."""
    dataset = _dataset(tmp_path)
    longest = int(np.argmax(dataset.tokens))
    sampler = EchoBatchSampler(dataset, **BATCHING)

    seen = [b for b in sampler.build_batches(epoch=0) if longest in b]
    assert len(seen) == 1


def test_batch_membership_changes_between_epochs(tmp_path: Path) -> None:
    sampler = EchoBatchSampler(_dataset(tmp_path), **BATCHING)

    partners = []
    for epoch in (0, 1):
        for batch in sampler.build_batches(epoch):
            if 0 in batch:
                partners.append(set(batch) - {0})

    assert partners[0] != partners[1]


def test_batches_are_deterministic_per_epoch(tmp_path: Path) -> None:
    sampler = EchoBatchSampler(_dataset(tmp_path), **BATCHING)

    assert sampler.build_batches(epoch=2) == sampler.build_batches(epoch=2)
    assert sampler.build_batches(epoch=2) != sampler.build_batches(epoch=3)


def test_sorted_packing_places_every_sample_within_the_limits(tmp_path: Path) -> None:
    dataset = _dataset(tmp_path)
    sorted_batches = EchoBatchSampler(dataset, packing="sorted", packing_chunk=100, **BATCHING).build_batches(epoch=0)
    greedy_batches = EchoBatchSampler(dataset, **BATCHING).build_batches(epoch=0)

    def padding(batches: List[List[int]]) -> float:
        return 1 - sum(int(dataset.tokens[b].sum()) for b in batches) / sum(len(b) * int(dataset.tokens[b].max()) for b in batches)

    assert sorted(i for batch in sorted_batches for i in batch) == list(range(len(dataset)))
    for batch in sorted_batches:
        assert len(batch) <= MAX_BATCH_SIZE
        assert len(batch) * int(dataset.tokens[batch].max()) <= MAX_BATCH_TOKENS
    assert padding(sorted_batches) < padding(greedy_batches) / 2


def test_sorted_packing_changes_between_epochs(tmp_path: Path) -> None:
    sampler = EchoBatchSampler(_dataset(tmp_path), packing="sorted", packing_chunk=100, **BATCHING)

    assert sampler.build_batches(epoch=2) == sampler.build_batches(epoch=2)
    assert sampler.build_batches(epoch=2) != sampler.build_batches(epoch=3)


def test_oversized_sample_is_rejected(tmp_path: Path) -> None:
    dataset = EchoDataset(_manifest(tmp_path, [10, MAX_BATCH_TOKENS]))

    with pytest.raises(ValueError, match="never fits"):
        EchoBatchSampler(dataset, max_batch_tokens=MAX_BATCH_TOKENS)


def test_loader_yields_the_sampled_batches(tmp_path: Path) -> None:
    dataset = _dataset(tmp_path, count=60)
    loader = EchoDataLoader(dataset, num_workers=2, **BATCHING)

    batches = list(loader)
    expected = [len(batch) for batch in loader.batch_sampler.batches]

    assert [len(batch["text"]) for batch in batches] == expected
    assert sum(expected) == len(dataset)

    first = batches[0]
    assert set(first) == {"audio_tokens", "text", "speaker_id", "language", "tokens", "valid"}
    assert all(item.shape[0] == CODEBOOKS for item in first["audio_tokens"])

    loader.set_epoch(1)
    assert [len(b) for b in loader.batch_sampler.batches] != expected or len(batches) == 1


def test_sampler_length_matches_batches(tmp_path: Path) -> None:
    dataset = _dataset(tmp_path, count=100)
    sampler = EchoBatchSampler(dataset, **BATCHING)

    assert len(sampler) == len(list(iter(sampler)))


def _preprocessor() -> EchoPreprocessor:
    return EchoPreprocessor()


def test_preprocessor_tokenizes_prompt_and_text(tmp_path: Path) -> None:
    dataset = _dataset(tmp_path, count=4)
    pre = _preprocessor()

    processed = pre([dataset[0], dataset[1]])
    assert len(processed) == 2

    sample, tok = processed[0], pre.tokenizer
    assert tok.decode(sample["prompt_tokens"]) == (
        "<|lang_start|>pl<|lang_end|><|instruct_start|>None<|instruct_end|>"
    )
    assert tok.decode(sample["text_tokens"]) == "<|text_start|>transcript 0<|text_end|>"
    assert sample["prompt_tokens"].dtype == torch.int64


def test_preprocessor_reports_spans_matching_the_manifest(tmp_path: Path) -> None:
    dataset = _dataset(tmp_path, count=4)
    pre = _preprocessor()

    raw = dataset[2]
    sample = pre.process(raw)

    frames = raw["audio_tokens"].size(-1)
    prompt_and_text = len(sample["prompt_tokens"]) + len(sample["text_tokens"])

    assert sample["length"] == prompt_and_text + frames
    assert sample["prefix_length"] + sample["target_length"] == sample["length"]
    assert sample["audio_tokens"].shape == raw["audio_tokens"].shape


def test_loader_applies_the_preprocessor(tmp_path: Path) -> None:
    dataset = _dataset(tmp_path, count=40)
    loader = EchoDataLoader(dataset, preprocessor=_preprocessor(), num_workers=0, **BATCHING)

    batch = next(iter(loader))
    assert set(batch) == {
        "prompt_tokens", "text_tokens", "audio_tokens", "mask", "polish", "valid",
        "prefix_length", "target_length", "length", "speaker_id", "language",
    }
    assert all(t.dim() == 1 for t in batch["text_tokens"])
    assert all(a.shape[0] == CODEBOOKS for a in batch["audio_tokens"])


def test_collate_pads_every_sequence_to_the_batch_maximum(tmp_path: Path) -> None:
    dataset = _dataset(tmp_path, count=40)
    pre = _preprocessor()
    loader = EchoDataLoader(dataset, preprocessor=pre, num_workers=0, **BATCHING)

    batch = next(iter(loader))
    totals = {
        len(prompt) + len(text) + audio.size(-1)
        for prompt, text, audio in zip(
            batch["prompt_tokens"], batch["text_tokens"], batch["audio_tokens"]
        )
    }

    assert len(totals) == 1
    assert totals.pop() == max(batch["length"])


def test_collate_pads_with_the_mask_token(tmp_path: Path) -> None:
    dataset = _dataset(tmp_path, count=40)
    pre = _preprocessor()
    loader = EchoDataLoader(dataset, preprocessor=pre, num_workers=0, **BATCHING)

    batch = next(iter(loader))
    shortest = int(np.argmin(batch["length"]))
    audio = batch["audio_tokens"][shortest]
    padded = max(batch["length"]) - batch["length"][shortest]

    assert padded > 0
    assert (audio[:, audio.size(-1) - padded :] == AUDIO_MASK_ID).all()
    assert (audio[:, : audio.size(-1) - padded] != AUDIO_MASK_ID).any()


def test_collate_pads_raw_samples_to_the_longest_codec(tmp_path: Path) -> None:
    dataset = _dataset(tmp_path, count=40)
    loader = EchoDataLoader(dataset, num_workers=0, **BATCHING)

    batch = next(iter(loader))
    frames = {audio.size(-1) for audio in batch["audio_tokens"]}

    assert len(frames) == 1


def test_collate_marks_real_tokens_in_the_valid_mask(tmp_path: Path) -> None:
    dataset = _dataset(tmp_path, count=40)
    loader = EchoDataLoader(dataset, preprocessor=_preprocessor(), num_workers=0, **BATCHING)

    batch = next(iter(loader))
    valid = batch["valid"]

    assert valid.shape == (len(batch["length"]), max(batch["length"]))
    assert valid.dtype == torch.bool
    assert valid.sum(dim=1).tolist() == batch["length"]
    assert all(row[:length].all() and not row[length:].any()
               for row, length in zip(valid, batch["length"]))


def _audio(frames: int) -> torch.Tensor:
    return torch.zeros(CODEBOOKS, frames, dtype=torch.long)


def test_split_covers_the_whole_utterance() -> None:
    pre = EchoPreprocessor()

    for frames in (30, 120, 750):
        prefix, target = pre.split_audio(_audio(frames))
        assert prefix + target == frames
        assert prefix >= 0 and target >= 0


def test_split_honours_the_ratio_bounds() -> None:
    frames = 400
    pre = EchoPreprocessor(prefix_min_dur=0.0, prefix_min_ratio=0.2, prefix_max_ratio=0.4)

    prefixes = [pre.split_audio(_audio(frames))[0] for _ in range(200)]
    assert min(prefixes) >= int(0.2 * frames)
    assert max(prefixes) <= int(0.4 * frames)


def test_split_uses_a_fixed_ratio_when_bounds_are_equal() -> None:
    frames = 400
    pre = EchoPreprocessor(prefix_min_dur=0.0, prefix_min_ratio=0.25, prefix_max_ratio=0.25)

    assert {pre.split_audio(_audio(frames))[0] for _ in range(20)} == {int(0.25 * frames)}


def test_split_keeps_the_minimum_prefix_duration() -> None:
    pre = EchoPreprocessor(prefix_min_dur=1.0, prefix_min_ratio=0.0, prefix_max_ratio=0.1)
    assert pre.prefix_min_frames == 25

    # A tenth of 100 frames is below the one-second floor, so the floor wins.
    prefixes = [pre.split_audio(_audio(100))[0] for _ in range(50)]
    assert min(prefixes) == 25


def test_split_draws_above_the_minimum_instead_of_clamping() -> None:
    random.seed(0)
    frames = 210   # 8.4 s: the clamp would pin a third of the prefixes at the 37-frame floor
    drawing, clamping = EchoPreprocessor(prefix_min_dur=1.5), EchoPreprocessor(prefix_min_dur=1.5, prefix_clamp=True)
    drawn = [drawing.split_audio(_audio(frames))[0] for _ in range(2000)]
    clamped = [clamping.split_audio(_audio(frames))[0] for _ in range(2000)]

    assert min(drawn) >= 37 and max(drawn) <= frames // 2
    assert sum(p == 37 for p in drawn) / len(drawn) < 0.02
    assert sum(p == 37 for p in clamped) / len(clamped) == pytest.approx(0.35, abs=0.05)


def test_split_never_outgrows_a_very_short_utterance() -> None:
    pre = EchoPreprocessor(prefix_min_dur=2.0)
    prefix, target = pre.split_audio(_audio(10))

    assert prefix == 10
    assert target == 0


def test_split_differs_between_samples_of_one_batch(tmp_path: Path) -> None:
    dataset = _dataset(tmp_path, count=40)
    pre = EchoPreprocessor(prefix_min_dur=0.0)

    processed = pre([dataset[i] for i in range(40)])
    ratios = {
        round((s["prefix_length"] - len(s["prompt_tokens"]) - len(s["text_tokens"]))
              / (s["length"] - len(s["prompt_tokens"]) - len(s["text_tokens"])), 3)
        for s in processed
    }

    assert len(ratios) > 20


def test_split_feeds_the_reported_spans(tmp_path: Path) -> None:
    dataset = _dataset(tmp_path, count=8)
    pre = EchoPreprocessor()

    raw = dataset[4]
    sample = pre.process(raw)
    prompt_and_text = len(sample["prompt_tokens"]) + len(sample["text_tokens"])

    assert sample["prefix_length"] > prompt_and_text
    assert sample["length"] == prompt_and_text + raw["audio_tokens"].size(-1)
    assert sample["target_length"] == sample["length"] - sample["prefix_length"]


def test_main_stage_mask_masks_the_polished_codebooks_fully() -> None:
    torch.manual_seed(0)
    audio = _audio(2000)

    for _ in range(50):
        mask = build_main_stage_audio_mask(audio, prefix_frames=100)
        rates = mask[:, 100:].float().mean(dim=1)

        assert not mask[:, :100].any()
        assert (rates[2:] == 1.0).all()
        # Codebook 1 trails codebook 0, and neither goes past the last pass of the longest sampler.
        assert 0.45 < float(rates[0]) and float(rates[0]) <= float(rates[1]) + 0.05


def test_main_stage_mask_replays_the_decisions() -> None:
    torch.manual_seed(0)
    audio = _audio(2000)

    # Most draws sit at the last pass of a sampler, where codebook 0 is 0.60-0.74 masked.
    c0 = [float(build_main_stage_audio_mask(audio, 0).float().mean(dim=1)[0]) for _ in range(1000)]
    assert sum(0.55 < c < 0.79 for c in c0) / len(c0) > 0.7

    # An all-early share keeps codebook 0 nearly fully masked; an all-tail one reaches past every sampler.
    early = [float(build_main_stage_audio_mask(audio, 0, (1, 0, 0, 0, 0)).float().mean(dim=1)[0]) for _ in range(200)]
    tail = [float(build_main_stage_audio_mask(audio, 0, (0, 0, 0, 0, 1)).float().mean(dim=1)[0]) for _ in range(200)]
    assert min(early) > 0.9
    assert min(tail) < 0.5 and max(tail) < 0.85

    # Drawn uniformly, the passes before the last two - codebook 0 above 0.85 - take most of the draws.
    uniform = [float(build_main_stage_audio_mask(audio, 0, decision_power=0.0).float().mean(dim=1)[0]) for _ in range(1000)]
    assert sum(c > 0.85 for c in uniform) / len(uniform) == pytest.approx(0.65, abs=0.1)


def test_polish_stage_mask_reveals_codebooks_0_1_and_masks_the_rest() -> None:
    audio = _audio(300)
    mask = build_polish_stage_audio_mask(audio, prefix_frames=100)

    assert not mask[:, :100].any()
    assert not mask[:2].any()
    assert mask[2:, 100:].all()


def test_preprocessor_masks_only_the_target(tmp_path: Path) -> None:
    dataset = _dataset(tmp_path, count=8)
    pre = EchoPreprocessor(prefix_min_dur=1.0, prefix_min_ratio=0.3, prefix_max_ratio=0.3)

    raw = dataset[5]
    sample = pre.process(raw)
    audio = sample["audio_tokens"]
    prefix_frames = sample["prefix_length"] - len(sample["prompt_tokens"]) - len(sample["text_tokens"])

    mask = sample["mask"]

    assert torch.equal(audio, raw["audio_tokens"])   # The tokens are left as they are
    assert mask.shape == raw["audio_tokens"].shape
    assert not mask[:, :prefix_frames].any()         # The prefix is never masked
    assert mask[:, prefix_frames:].any()



def test_preprocessor_draws_the_polish_share(tmp_path: Path) -> None:
    dataset = _dataset(tmp_path, count=50)
    pre = EchoPreprocessor(polish_share=0.25)
    random.seed(0)

    samples = [pre.process(dataset[i % 50]) for i in range(800)]
    polish = [sample for sample in samples if sample["polish"]]

    assert len(polish) / len(samples) == pytest.approx(0.25, abs=0.05)
    assert all(not sample["mask"][:2].any() for sample in polish)


def test_dropped_prefix_count_is_exact(tmp_path: Path) -> None:
    dataset = _dataset(tmp_path, count=20)
    pre = EchoPreprocessor(drop_prefix_ratio=0.1)
    raw = [dataset[i] for i in range(20)]

    for _ in range(10):
        dropped = [s for s in pre(raw) if s["prefix_length"] == 0]
        assert len(dropped) == 2


def test_dropped_sample_is_all_target(tmp_path: Path) -> None:
    dataset = _dataset(tmp_path, count=8)
    pre = EchoPreprocessor(drop_prefix_ratio=1.0)

    raw = dataset[3]
    sample = pre([raw])[0]
    frames = raw["audio_tokens"].size(-1)

    assert len(sample["prompt_tokens"]) == 0
    assert len(sample["text_tokens"]) == 0
    assert sample["prefix_length"] == 0
    assert sample["target_length"] == frames
    assert sample["length"] == frames
    assert sample["audio_tokens"].shape == raw["audio_tokens"].shape
    assert sample["mask"].any()


def test_no_prefix_is_dropped_by_default(tmp_path: Path) -> None:
    dataset = _dataset(tmp_path, count=20)
    pre = EchoPreprocessor()

    for sample in pre([dataset[i] for i in range(20)]):
        assert len(sample["prompt_tokens"]) > 0
        assert len(sample["text_tokens"]) > 0
        assert sample["prefix_length"] > 0


def test_disable_flags_skip_prefix_drop_and_masking(tmp_path: Path) -> None:
    dataset = _dataset(tmp_path, count=20)
    pre = EchoPreprocessor(drop_prefix_ratio=0.5)
    raw = [dataset[i] for i in range(20)]

    samples = pre(raw, disable_prefix_drop=True, disable_masking=True)

    for sample, source in zip(samples, raw):
        assert sample["prefix_length"] > 0
        assert torch.equal(sample["audio_tokens"], source["audio_tokens"])


def test_fixed_prefix_ratio_makes_the_split_deterministic(tmp_path: Path) -> None:
    dataset = _dataset(tmp_path, count=8)
    pre = EchoPreprocessor(prefix_min_ratio=0.25, prefix_max_ratio=0.25)
    raw = [dataset[i] for i in range(8)]

    first = [s["prefix_length"] for s in pre(raw)]
    second = [s["prefix_length"] for s in pre(raw)]

    assert first == second

    # A quarter of the frames, unless the one second floor asks for more.
    for sample, split in zip(raw, pre(raw)):
        frames = sample["audio_tokens"].size(-1)
        prefix_frames = (
            split["prefix_length"]
            - split["prompt_tokens"].numel()
            - split["text_tokens"].numel()
        )
        assert prefix_frames == max(int(0.25 * frames), 25)


def test_the_mask_travels_beside_the_true_tokens(tmp_path: Path) -> None:
    """Masking late keeps the labels: the preprocessor marks, collate pads, the
    model inputs apply it."""

    dataset = _dataset(tmp_path, count=6)
    loader = EchoDataLoader(dataset, EchoPreprocessor(), num_workers=0, **BATCHING)
    batch = next(iter(loader))

    inputs = build_model_inputs(batch, batch["mask"])

    for index, mask in enumerate(batch["mask"]):
        prefix = batch["prompt_tokens"][index].numel() + batch["text_tokens"][index].numel()
        audio = inputs["input_ids"][index, :, prefix:]
        labels = inputs["labels"][index, :, prefix:]

        assert mask.shape == batch["audio_tokens"][index].shape   # Padded alike
        assert (audio[mask] == AUDIO_MASK_ID).all()
        assert torch.equal(labels[mask], batch["audio_tokens"][index][mask])
        assert (labels[~mask] == IGNORE_INDEX).all()

        # The padded tail is never marked, so it is never scored.
        assert not mask[:, batch["length"][index] - prefix:].any()


def test_main_samples_are_scored_on_codebooks_0_1_alone(tmp_path: Path) -> None:
    dataset = _dataset(tmp_path, count=12)
    loader = EchoDataLoader(dataset, EchoPreprocessor(polish_share=0.5), num_workers=0, **BATCHING)
    batch = next(iter(loader))

    inputs = build_model_inputs(batch, batch["mask"], batch["polish"])

    for index, (mask, polish) in enumerate(zip(batch["mask"], batch["polish"])):
        prefix = batch["prompt_tokens"][index].numel() + batch["text_tokens"][index].numel()
        labels = inputs["labels"][index, :, prefix:]
        scored = labels != IGNORE_INDEX

        # Both kinds mask codebooks 2-7; only the polish samples are scored there.
        assert torch.equal(scored[2:], mask[2:] if polish else torch.zeros_like(mask[2:]))
        assert torch.equal(scored[:2], mask[:2])
