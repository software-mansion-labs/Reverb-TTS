"""Tests for echo.pruning, on the tiny randomly initialized model of tests/test_echo.py."""

from typing import List

import pytest
import torch

from echo import ArchitectureConfig, CFGConfig, EchoConfig, EchoModel, build_model
from echo.data.constants import IGNORE_INDEX
from echo.masking.attention_masking import build_dense_bidirectional_mask
from echo.masking.span_masking import gather_target
from echo.data.inputs import build_unconditional_inputs
from echo.pruning.calibration import CalibrationInput, attention_grams, branch_kl, ffn_grams
from echo.pruning.search import layer_costs, next_layer
from echo.model.model import FactorizedEmbedding
from echo.pruning.surgery import LayerFade, drop_kv_groups, drop_layers, factorize_embedding, faded_layer, prune_vocab, shrink_ffn, without_layers
from tests.test_echo import AUDIO_VOCAB_SIZE, PREFIX_LEN, _batch, _config


def _inputs() -> List[CalibrationInput]:
    input_ids, audio_mask, prefix_lengths, target_lengths = _batch()
    masked = torch.zeros_like(input_ids, dtype=torch.bool)
    masked[:, :, PREFIX_LEN::2] = True
    labels = torch.where(masked, input_ids, IGNORE_INDEX)
    ids = input_ids.masked_fill(masked, AUDIO_VOCAB_SIZE - 1)
    valid = torch.ones(audio_mask.shape, dtype=torch.bool)

    cond = {
        "input_ids": ids,
        "audio_mask": audio_mask,
        "attention_mask": build_dense_bidirectional_mask(valid),
        "prefix_lengths": prefix_lengths,
        "target_lengths": target_lengths,
        "labels": labels,
    }
    uncond = build_unconditional_inputs(ids, prefix_lengths, target_lengths)
    span_labels, _ = gather_target(labels, prefix_lengths, target_lengths)

    return [CalibrationInput(cond, uncond, span_labels, valid, uncond["audio_mask"])]


def _logits(model: EchoModel, item: CalibrationInput) -> torch.Tensor:
    with torch.no_grad():
        return model(**{k: v for k, v in item.cond.items() if k != "labels"}).logits


def test_shrinking_to_the_full_size_changes_nothing() -> None:
    model = EchoModel(_config()).eval()
    inputs = _inputs()
    before = _logits(model, inputs[0])

    shrink_ffn(model, 64, ffn_grams(model, inputs))

    torch.testing.assert_close(_logits(model, inputs[0]), before, atol=1e-3, rtol=1e-3)


def test_a_pruned_model_saves_reloads_and_describes_itself(tmp_path) -> None:
    model = EchoModel(_config()).eval()
    inputs = _inputs()
    shrink_ffn(model, 32, ffn_grams(model, inputs))
    drop_layers(model, [1])
    drop_layers(model, [1])      # Originally layer 2
    expected = _logits(model, inputs[0])

    model.save_pretrained(tmp_path)
    reloaded = build_model(EchoConfig(init_from=str(tmp_path)), train=True).eval()

    assert len(reloaded.llm.layers) == 2
    assert reloaded.llm.layers[0].mlp.gate_proj.weight.shape == (32, 32)
    assert reloaded.echo_config.architecture.kept_layers == [0, 3]
    assert reloaded.echo_config.architecture.intermediate_size == 32
    assert torch.equal(_logits(reloaded, inputs[0]), expected)

    # The architecture describes the checkpoint - it can not reshape it.
    with pytest.raises(ValueError, match="does not match"):
        build_model(EchoConfig(init_from=str(tmp_path), architecture=ArchitectureConfig(intermediate_size=64)), train=True)


def test_keeping_every_kv_group_changes_nothing() -> None:
    model = EchoModel(_config()).eval()
    inputs = _inputs()
    before = _logits(model, inputs[0])

    drop_kv_groups(model, 2, attention_grams(model, inputs))

    # Not exact: the refit's ridge shrinks o_proj a little even with nothing dropped.
    torch.testing.assert_close(_logits(model, inputs[0]), before, atol=5e-3, rtol=5e-3)


def test_a_kv_pruned_model_saves_reloads_and_describes_itself(tmp_path) -> None:
    model = EchoModel(_config()).eval()
    inputs = _inputs()
    drop_kv_groups(model, 1, attention_grams(model, inputs))   # Of 2 KV heads, each shared by 2 query heads
    expected = _logits(model, inputs[0])

    model.save_pretrained(tmp_path)
    reloaded = build_model(EchoConfig(init_from=str(tmp_path)), train=True).eval()

    architecture = reloaded.echo_config.architecture
    assert (reloaded.llm.config.num_attention_heads, reloaded.llm.config.num_key_value_heads) == (2, 1)
    assert (architecture.num_attention_heads, architecture.num_key_value_heads) == (2, 1)
    assert len(architecture.kept_kv_heads) == 4 and all(len(heads) == 1 for heads in architecture.kept_kv_heads)
    assert torch.equal(_logits(reloaded, inputs[0]), expected)

    # The provenance follows the layers it describes.
    kept = architecture.kept_kv_heads
    drop_layers(reloaded, [1])
    assert reloaded.echo_config.architecture.kept_kv_heads == [kept[0]] + kept[2:]


def test_a_vocab_pruned_model_gives_the_same_logits_on_renumbered_ids_and_reloads(tmp_path) -> None:
    model = EchoModel(_config()).eval()
    item = _inputs()[0]
    before = _logits(model, item)

    rows = list(range(AUDIO_VOCAB_SIZE)) + [40, 90]   # Every text id the batch uses, and two more
    prune_vocab(model, rows)
    assert model.llm.embed_tokens.num_embeddings == len(rows) and model.llm.config.vocab_size == len(rows)
    # The batch's text ids all lie below len(rows), where the renumbering is the identity.
    torch.testing.assert_close(_logits(model, item), before, atol=0, rtol=0)

    model.save_pretrained(tmp_path)
    reloaded = build_model(EchoConfig(init_from=str(tmp_path)), train=True).eval()
    assert reloaded.echo_config.architecture.vocab_size == len(rows)
    assert torch.equal(reloaded.llm.embed_tokens.weight[-1], model.llm.embed_tokens.weight[-1])
    assert torch.equal(_logits(reloaded, item), before)


def test_a_full_rank_factorization_changes_nothing() -> None:
    model = EchoModel(_config()).eval()
    item = _inputs()[0]
    before = _logits(model, item)

    factorize_embedding(model, 32)

    assert isinstance(model.llm.embed_tokens, FactorizedEmbedding)
    torch.testing.assert_close(_logits(model, item), before, atol=1e-4, rtol=1e-4)


def test_a_factorized_model_saves_reloads_and_describes_itself(tmp_path) -> None:
    model = EchoModel(_config()).eval()
    item = _inputs()[0]
    factorize_embedding(model, 8)
    expected = _logits(model, item)

    model.save_pretrained(tmp_path)
    reloaded = build_model(EchoConfig(init_from=str(tmp_path)), train=True, strict_weights=True).eval()

    assert reloaded.echo_config.architecture.embedding_rank == 8
    assert reloaded.llm.embed_tokens.table.weight.shape == (128, 8)
    assert torch.equal(_logits(reloaded, item), expected)
    with pytest.raises(ValueError, match="does not match"):
        build_model(EchoConfig(init_from=str(tmp_path), architecture=ArchitectureConfig(embedding_rank=16)), train=True)


def test_a_fully_faded_layer_is_a_removed_one() -> None:
    model = EchoModel(_config()).eval()
    item = _inputs()[0]
    whole = _logits(model, item)

    with without_layers(model, [1]):
        removed = _logits(model, item)
    with faded_layer(model, 1, 0.0):
        torch.testing.assert_close(_logits(model, item), removed)
    with faded_layer(model, 1, 1.0):
        torch.testing.assert_close(_logits(model, item), whole)

    # The scaling goes with the context.
    torch.testing.assert_close(_logits(model, item), whole)


def test_a_layer_fades_on_schedule_holds_over_budget_and_drops_cleanly() -> None:
    model = EchoModel(_config()).eval()
    item = _inputs()[0]
    fade = LayerFade(model, 1, steps=4, power=2.0, budget=0.1)

    assert not any(parameter.requires_grad for parameter in model.llm.layers[1].parameters())
    assert all(parameter.requires_grad for parameter in model.llm.layers[2].parameters())

    fade.update(1.0)
    assert fade.alpha == pytest.approx(1 - 0.25 ** 0.5)
    fade.update(10.0)   # Far over the budget: held
    assert fade.progress == 1

    fade.progress, fade.level = fade.steps - 1, fade.start   # Back within the budget
    fade.update(1.0)
    assert fade.done and fade.alpha == 0.0

    # Without a budget the fade never holds.
    unbounded = LayerFade(model, 2, steps=4, power=2.0, budget=None)
    unbounded.update(1.0)
    unbounded.update(10.0)
    assert unbounded.progress == 2
    for handle in unbounded.handles:
        handle.remove()

    faded = _logits(model, item)
    drop_layers(model, [1])
    torch.testing.assert_close(_logits(model, item), faded)


def test_unconditional_inputs_match_the_models_own_branch() -> None:
    model = EchoModel(_config(cfg=CFGConfig(mode="teacher"))).eval()
    input_ids, audio_mask, prefix_lengths, target_lengths = _batch()

    with torch.no_grad():
        branch = model(input_ids, audio_mask, prefix_lengths=prefix_lengths, target_lengths=target_lengths).uncond_logits
        direct = model(**build_unconditional_inputs(input_ids, prefix_lengths, target_lengths), unconditional=False).logits

    assert torch.equal(branch, direct)


def _two_codebook_weights(monkeypatch) -> None:
    # The tiny model has 2 codebooks, the two-stage loss weights 8.
    monkeypatch.setattr("echo.pruning.calibration.codebook_weights", lambda: [0.5, 0.5])


def test_both_branches_are_scored(monkeypatch) -> None:
    _two_codebook_weights(monkeypatch)
    torch.manual_seed(0)
    model, teacher = EchoModel(_config()).eval(), EchoModel(_config()).eval()

    kl = branch_kl(model, teacher, _inputs())
    assert set(kl) == {"cond", "uncond"}
    assert kl["cond"] > 0.0 and kl["uncond"] != pytest.approx(kl["cond"])

    # Against itself a model has lost nothing.
    assert branch_kl(teacher, teacher, _inputs())["cond"] == pytest.approx(0.0, abs=1e-6)


def test_kl_weights_only_the_scored_codebooks(monkeypatch) -> None:
    torch.manual_seed(0)
    model, teacher = EchoModel(_config()).eval(), EchoModel(_config()).eval()
    inputs = _inputs()
    inputs[0].labels[:, 1] = IGNORE_INDEX   # A single state, scoring codebook 0 alone

    monkeypatch.setattr("echo.pruning.calibration.codebook_weights", lambda: [0.5, 0.5])
    kl = branch_kl(model, teacher, inputs)
    monkeypatch.setattr("echo.pruning.calibration.codebook_weights", lambda: [1.0, 0.0])

    assert kl["cond"] == pytest.approx(branch_kl(model, teacher, inputs)["cond"])


def test_layer_search_skips_protected_layers_and_takes_the_cheapest(monkeypatch) -> None:
    _two_codebook_weights(monkeypatch)
    model = EchoModel(_config()).eval()
    assert set(layer_costs(model, EchoModel(_config()).eval(), _inputs())) == {1, 2}

    # Ranked by the conditional ΔKL alone - the unconditional one is only reported.
    costs = {
        1: {"cond": 0.03, "uncond": 0.0},
        2: {"cond": 0.02, "uncond": 0.2},
    }
    choice = next_layer(costs, {"cond": 0.09, "uncond": 0.0})
    assert choice.layer == 2
    assert choice.total["cond"] == pytest.approx(0.11)
