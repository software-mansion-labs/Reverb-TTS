"""Tests for the RL pipeline's pure parts: pair rules, take masks, evaluation counts, batching and recipes."""

import random

import pytest
import torch

from echo.rl.config import PairsConfig, RLConfig
from echo.rl.dpo import SHARES, masks
from echo.rl.evaluate import only_numbers, sign_p
from echo.rl.pairs import Pair, group_pairs, weigh
from echo.rl.sampler import pack


def _group(loss: list, score: list | None = None, words: list | None = None, gated: list | None = None) -> dict:
    n = len(loss)
    return dict(ctc_loss=loss, score=score or [0.0] * n, words=words or [[0.9, 0.9]] * n, ctc_gated=gated or [False] * n)


def test_ctc_pairs_take_the_largest_gaps_within_the_caps() -> None:
    # median 0.875: a unit of 0.4375; every pair but the two lowest takes clears it
    pairs = group_pairs(_group([0.2, 0.25, 1.5, 2.0]), PairsConfig())

    assert [p for p in pairs if p[2] == "ctc"] == [(0, 3, "ctc"), (1, 3, "ctc"), (0, 2, "ctc"), (1, 2, "ctc")]


def test_a_dropped_word_admits_half_a_unit_and_takes_are_reused_twice() -> None:
    words = [[0.9, 0.9]] * 3 + [[0.9, 0.3]]   # take 3 drops its second word
    pairs = group_pairs(_group([0.2, 0.2, 0.2, 0.45], words=words), PairsConfig())
    plain = group_pairs(_group([0.2, 0.2, 0.2, 0.45]), PairsConfig())

    assert pairs == [(2, 3, "ctc"), (1, 3, "ctc")]
    assert plain == []


def test_critic_pairs_need_acceptable_takes_ctc_leaves_undecided() -> None:
    pairs = group_pairs(_group([0.2, 0.25, 1.5, 2.0], score=[0.0, 5.0, 20.0, -20.0]), PairsConfig())

    assert [p for p in pairs if p[2] == "critic"] == [(1, 0, "critic")]


def test_the_ctc_filter_needs_a_gated_take_and_a_worse_word() -> None:
    words = [[0.9, 0.9], [0.9, 0.9], [0.9, 0.9], [0.9, 0.3]]
    loss = [0.2, 0.25, 1.5, 2.0]
    config = PairsConfig(ctc_filter=True)

    assert group_pairs(_group(loss, words=words), config) == []
    assert group_pairs(_group(loss, words=words, gated=[False, False, False, True]), config) == [
        (0, 3, "ctc"), (1, 3, "ctc")]


def test_ctc_pairs_are_weighted_to_their_share_within_1_to_2() -> None:
    def pairs(ctc: int, critic: int) -> list:
        return [Pair("g", "p0000", "a", "b", "ctc")] * ctc + [Pair("g", "p0000", "a", "b", "critic")] * critic

    assert weigh(pairs(10, 30), 0.4) == 2.0
    assert weigh(pairs(30, 30), 0.4) == 1.0
    assert weigh(pairs(20, 45), 0.4) == pytest.approx(1.5)
    assert weigh(pairs(20, 45), 0.0) == 1.0


def test_masks_follow_the_sampler_shares_from_a_random_pass() -> None:
    rng, copy = random.Random(0), random.Random(0)
    copy.randrange(2**31)
    offset = copy.randrange(8)
    ms = masks(20_000, 8, rng)

    assert len(ms) == 8
    for i, m in enumerate(ms):
        assert m.shape == (2, 20_000) and m[0].any()
        assert torch.allclose(m.float().mean(1), SHARES[:, (i + offset) % 8], atol=0.015)


def test_only_number_errors_are_left_out() -> None:
    assert only_numbers("I have three cats.", "I have 3 cats")
    assert not only_numbers("I have three cats.", "I have cats")
    assert not only_numbers("I have three cats and a dog.", "I have 3 cats")
    assert sign_p(5, 5) == 1.0
    assert sign_p(10, 0) == pytest.approx(2 / 1024)


def test_pack_keeps_every_batch_within_its_budgets() -> None:
    jobs = [dict(ids=torch.zeros(20 + i % 7), reference=torch.zeros(8, 100 + i % 5), frames=50 + 13 * i) for i in range(40)]
    batches = pack(jobs, batch_frames=1000, batch_tokens=4000)

    assert sum(len(b) for b in batches) == 40
    for b in batches:
        assert len(b) * max(j["frames"] for j in b) <= 1000 or len(b) == 1
        assert len(b) * (max(j["ids"].numel() + j["reference"].size(1) for j in b) + max(j["frames"] for j in b)) <= 4000 or len(b) == 1


@pytest.mark.parametrize("recipe", ["base", "mine"])
def test_recipes_load_strictly(recipe: str, tmp_path) -> None:
    config = RLConfig.from_json(f"echo/configs/rl/{recipe}.json")
    (tmp_path / "bad.json").write_text('{"pairs": {"ctc_pair": 4}}')

    assert config.name == recipe and config.train.winner_anchor == 2.0
    with pytest.raises(ValueError):
        RLConfig.from_json(tmp_path / "bad.json")
