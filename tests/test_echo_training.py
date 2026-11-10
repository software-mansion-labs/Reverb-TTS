import random
from types import SimpleNamespace

import numpy as np
import pytest
import torch
import torch.nn.functional as F

from echo.data.constants import IGNORE_INDEX
from echo.data.loader import EchoDataLoader
from echo.data.sampler import EchoBatchSampler
from echo.data.preprocessor import EchoPreprocessor
from echo.training.builder import build_optimizer, build_schedule
from echo.training.config import EchoTrainingConfig, PathLossConfig
from echo.utils.distributed_utils import is_main, reduce_mean, world
from echo.training.checkpoint import load_checkpoint, save_checkpoint
from echo.training.meter import Meter
from echo.training.step import distillation_step, plain_step
from echo.model.model import EchoModel
from tests.test_echo import _config as _model_config
from echo.utils.seed_utils import seed_everything, seed_of, seed_worker
from echo.utils.loss_utils import (
    weighted_ce,
    distillation_loss,
    reducible_tokens,
    weighted_kl,
    weighted_kl_logp,
)

CONFIG = "echo/configs/training/train.json"


def _config(**overrides) -> EchoTrainingConfig:
    config = EchoTrainingConfig.from_json(CONFIG)
    for section, values in overrides.items():
        for key, value in values.items():
            setattr(getattr(config, section), key, value)

    return config


# ---------- The config itself ----------


def test_example_config_loads_strictly() -> None:
    config = EchoTrainingConfig.from_json(CONFIG, strict=True)
    config.validate()

    assert config.masking.polish_share == 0.25
    assert config.distilling is True


def test_dtype_is_a_torch_dtype() -> None:
    assert _config(optim={"mixed_precision": "bf16"}).dtype is torch.bfloat16
    assert _config(optim={"mixed_precision": "fp32"}).dtype is torch.float32


def test_added_fields_are_validated() -> None:
    with pytest.raises(ValueError, match="temperature"):
        _config(loss={"temperature": 0.0}).validate()
    with pytest.raises(ValueError, match="min_lr_ratio"):
        _config(schedule={"min_lr_ratio": 1.5}).validate()
    with pytest.raises(ValueError, match="layers_to_prune"):
        _config(pruning={"layers_to_prune": -1}).validate()


@pytest.mark.parametrize(
    "split",
    [
        {"prefix_min_dur": -0.5},
        {"prefix_min_ratio": -0.1},
        {"prefix_max_ratio": 1.5},
        {"prefix_min_ratio": 0.6, "prefix_max_ratio": 0.4},
    ],
)
def test_invalid_split_settings_are_rejected(split: dict) -> None:
    config = _config()
    for key, value in split.items():
        setattr(config.preprocessing.prefix_split, key, value)

    with pytest.raises(ValueError, match="prefix_split"):
        config.validate()


def test_ce_selection_is_validated() -> None:
    with pytest.raises(ValueError, match="ce_selection"):
        _config(loss={"polish": PathLossConfig(ce=1.0, ce_selection=0.0)}).validate()
    with pytest.raises(ValueError, match="models.teacher"):
        _config(loss={"polish": PathLossConfig(ce=1.0, kl=0.0, ce_selection=0.5), "main": PathLossConfig(ce=1.0, kl=0.0)},
                models={"teacher": None}).validate()
    with pytest.raises(ValueError, match="CE term"):
        _config(loss={"polish": PathLossConfig(ce=0.0, kl=1.0, ce_selection=0.5)}).validate()
    _config(loss={"polish": PathLossConfig(ce=1.0, kl=1.0, ce_selection=0.5)}).validate()

    config = _config(loss={"main": PathLossConfig(ce=1.0, kl=0.0), "polish": PathLossConfig(ce=1.0, kl=0.0, ce_selection=0.5)})
    config.validate()
    assert config.selecting and not config.distilling


def test_packing_reaches_the_sampler() -> None:
    config = _config(data={"packing": "sorted", "packing_chunk": 64})

    assert config.sampler_kwargs["packing"] == "sorted"
    assert config.sampler_kwargs["packing_chunk"] == 64
    with pytest.raises(ValueError, match="data.packing"):
        _config(data={"packing": "tight"}).validate()


def test_invalid_drop_prefix_ratio_is_rejected() -> None:
    with pytest.raises(ValueError, match="drop_prefix_ratio"):
        _config(preprocessing={"drop_prefix_ratio": 1.5}).validate()


def test_masking_fields_are_validated() -> None:
    with pytest.raises(ValueError, match="polish_share"):
        _config(masking={"polish_share": 1.5}).validate()
    with pytest.raises(ValueError, match="polish_weights"):
        _config(masking={"polish_weights": (8.0, 8.0)}).validate()
    with pytest.raises(ValueError, match="stage_shares"):
        _config(masking={"stage_shares": (0.5, 0.5, 0.5, 0.0, 0.0)}).validate()
    with pytest.raises(ValueError, match="tail_masked"):
        _config(masking={"tail_masked": 1.0}).validate()

    with pytest.raises(ValueError, match="samplers"):
        _config(masking={"samplers": {"s8": 1.0, "s12": 1.0}}).validate()
    with pytest.raises(ValueError, match="samplers"):
        _config(masking={"samplers": {"s8": 0.0}}).validate()
    with pytest.raises(ValueError, match="min_pass_share"):
        _config(masking={"min_pass_share": 0.2}).validate()

    _config(masking={"stage_shares": (0.04, 0.15, 0.20, 0.41, 0.20)}).validate()
    _config(masking={"samplers": {"s8": 1.0, "s6_rmc025": 0.5}, "min_pass_share": 0.02}).validate()


# ---------- What the properties hand to the constructors ----------


def test_dataset_kwargs_match_the_dataset(tmp_path) -> None:
    from echo.data.dataset import EchoDataset

    manifest = tmp_path / "manifest.csv"
    manifest.write_text("name|transcript|speaker_id|language|tokens\n")

    dataset = EchoDataset(manifest, **_config().dataset_kwargs)
    sampler = EchoBatchSampler(dataset, **_config().sampler_kwargs)

    assert dataset.data_root == manifest.parent   # 'None' falls back to the manifest's directory
    assert sampler.max_batch_tokens == _config().data.max_batch_tokens


def test_preprocessor_kwargs_reach_the_mask_builders() -> None:
    config = _config(masking={"polish_share": 0.4, "stage_shares": (0, 0, 0, 0, 1), "tail_masked": 0.3})
    pre = EchoPreprocessor(**config.preprocessor_kwargs)

    assert pre.polish_share == 0.4
    assert pre.masking_kwargs == {"stage_shares": (0, 0, 0, 0, 1), "tail_masked": 0.3, "samplers": None,
                                  "min_pass_share": 0.01}

    audio = torch.randint(0, 1000, (8, 400))
    assert not pre.draw_mask(audio, 50, polish=True)[:2].any()
    assert pre.draw_mask(audio, 50, polish=False)[2:, 50:].all()


def test_codebook_weights_split_the_loss_between_the_stages() -> None:
    weights = _config(masking={"polish_share": 0.25, "polish_weights": (8.0, 8.0, 4.0, 4.0, 2.0, 2.0)}).codebook_weights

    assert weights[:2] == [0.375, 0.375]
    assert sum(weights[2:]) == pytest.approx(0.25)
    assert weights[2] == pytest.approx(4 * weights[6])


# ---------- Optimizer and schedule ----------


def _model() -> torch.nn.Module:
    return torch.nn.Sequential(torch.nn.Linear(4, 4), torch.nn.LayerNorm(4))


def _rates(config: EchoTrainingConfig, total: int) -> list:
    optimizer = build_optimizer(_model(), config)
    schedule = build_schedule(optimizer, config, total)

    seen = []
    for _ in range(total):
        seen.append(optimizer.param_groups[0]["lr"])
        optimizer.step()
        schedule.step()

    return seen


def test_only_matrices_are_decayed() -> None:
    config = _config(optim={"weight_decay": 0.01})
    optimizer = build_optimizer(_model(), config)

    decayed, plain = optimizer.param_groups
    assert [p.ndim for p in decayed["params"]] == [2]        # The linear's weight
    assert [p.ndim for p in plain["params"]] == [1, 1, 1]    # Its bias, and the norm
    assert (decayed["weight_decay"], plain["weight_decay"]) == (0.01, 0.0)


def test_frozen_parameters_are_left_out() -> None:
    model = _model()
    model[0].weight.requires_grad_(False)

    groups = build_optimizer(model, _config()).param_groups

    assert groups[0]["params"] == []
    assert len(groups[1]["params"]) == 3


def test_schedule_warms_up_then_decays_to_the_floor() -> None:
    config = _config(schedule={"steps": 1000, "min_lr_ratio": 0.1, "warmup_ratio": 0.03})
    peak = config.optim.learning_rate
    warmup = config.warmup_steps(1000)

    rates = _rates(config, 1000)

    assert rates[0] == pytest.approx(peak / warmup)
    assert rates[warmup - 1] == pytest.approx(peak)
    assert rates[-1] == pytest.approx(peak * 0.1, rel=1e-3)
    assert all(a >= b for a, b in zip(rates[warmup:], rates[warmup + 1:]))


def test_schedule_restores_its_place(tmp_path) -> None:
    config = _config(schedule={"steps": 100})

    optimizer = build_optimizer(_model(), config)
    schedule = build_schedule(optimizer, config, 100)
    for _ in range(40):
        optimizer.step()
        schedule.step()

    resumed_optimizer = build_optimizer(_model(), config)
    resumed = build_schedule(resumed_optimizer, config, 100)
    resumed.load_state_dict(schedule.state_dict())

    assert resumed.get_last_lr() == schedule.get_last_lr()

    # Loading does not write the rate back onto the optimizer - torch applies it
    # on the next step - so a resume has to do that itself, or spend one step at
    # the rate the schedule started from.
    assert resumed_optimizer.param_groups[0]["lr"] != optimizer.param_groups[0]["lr"]

    for group, rate in zip(resumed_optimizer.param_groups, resumed.get_last_lr()):
        group["lr"] = rate

    assert resumed_optimizer.param_groups[0]["lr"] == optimizer.param_groups[0]["lr"]


def test_unknown_scheduler_is_rejected() -> None:
    with pytest.raises(ValueError, match="cosine"):
        _config(schedule={"lr_scheduler_type": "linear"}).validate()


# ---------- Losses ----------

CODEBOOKS = 8
VOCAB = 16
MASK_ID = VOCAB - 1


def _logits(seed: int) -> torch.Tensor:
    torch.manual_seed(seed)
    return torch.randn(2, CODEBOOKS, 6, VOCAB)


def _labels() -> torch.Tensor:
    labels = torch.randint(0, VOCAB, (2, CODEBOOKS, 6))
    labels[:, :, :2] = IGNORE_INDEX

    return labels


def _weights() -> torch.Tensor:
    return torch.full((CODEBOOKS,), 1.0 / CODEBOOKS)


def test_logp_kl_agrees_with_the_logits_one() -> None:
    student, teacher = _logits(0), _logits(1)
    labels, weights = _labels(), _weights()

    against_logits = weighted_kl(student, teacher, labels, weights)
    against_log_probs = weighted_kl_logp(
        student, torch.log_softmax(teacher, dim=-1), labels, weights
    )

    assert float(against_log_probs) == pytest.approx(float(against_logits), rel=1e-5)


def test_logp_kl_survives_a_forbidden_token() -> None:
    student, teacher = _logits(2), _logits(3)
    labels, weights = _labels(), _weights()

    # What a CFG mixture hands over: the mask token is ruled out entirely.
    target = torch.log_softmax(teacher, dim=-1)
    target[..., MASK_ID] = -float("inf")
    target = torch.log_softmax(target, dim=-1)

    divergence = weighted_kl_logp(student, target, labels, weights)

    assert torch.isfinite(divergence)
    assert float(divergence) > 0.0


def test_reverse_logp_kl_rules_out_what_the_target_forbids() -> None:
    student, teacher = _logits(4).requires_grad_(True), _logits(5)
    labels, weights = _labels(), _weights()

    target = torch.log_softmax(teacher, dim=-1)
    target[..., MASK_ID] = -float("inf")
    target = torch.log_softmax(target, dim=-1)

    divergence = weighted_kl_logp(student, target, labels, weights, reverse=True)
    divergence.backward()

    assert torch.isfinite(divergence) and float(divergence) > 0.0
    assert torch.isfinite(student.grad).all()
    # A student whose allowed tokens match the target costs nothing, whatever it puts on the forbidden one.
    assert float(weighted_kl_logp(teacher, target, labels, weights, reverse=True)) == pytest.approx(0.0, abs=1e-6)


def test_temperature_rescales_the_divergence() -> None:
    student, teacher = _logits(4), _logits(5)
    labels, weights = _labels(), _weights()

    plain = weighted_kl(student, teacher, labels, weights)
    softened = weighted_kl(student, teacher, labels, weights, temperature=2.0)

    assert float(plain) != pytest.approx(float(softened))
    assert torch.isfinite(softened)


def test_distillation_loss_weighs_and_reports_both_terms() -> None:
    student, teacher = _logits(6), _logits(7)
    labels, weights = _labels(), _weights()

    terms = distillation_loss(
        student, labels, weights, teacher_logits=teacher, ce_weight=0.5, kl_weight=2.0
    )

    assert float(terms["loss"]) == pytest.approx(
        0.5 * float(terms["ce"]) + 2.0 * float(terms["kl"])
    )
    assert float(terms["ce"]) > 0.0 and float(terms["kl"]) > 0.0


def test_a_term_with_no_weight_is_not_computed() -> None:
    student, labels, weights = _logits(8), _labels(), _weights()

    # Without a teacher the cross entropy alone has to carry the step.
    terms = distillation_loss(student, labels, weights, ce_weight=1.0, kl_weight=0.0)

    assert float(terms["kl"]) == 0.0
    assert float(terms["loss"]) == pytest.approx(float(terms["ce"]))


def test_distillation_loss_needs_exactly_one_teacher() -> None:
    student, labels, weights = _logits(9), _labels(), _weights()

    with pytest.raises(ValueError, match="exactly one"):
        distillation_loss(student, labels, weights, kl_weight=1.0)

    with pytest.raises(ValueError, match="exactly one"):
        distillation_loss(
            student,
            labels,
            weights,
            teacher_logits=student,
            teacher_log_probs=torch.log_softmax(student, dim=-1),
            kl_weight=1.0,
        )


def test_reducible_tokens_keep_the_largest_excess_of_every_codebook() -> None:
    student, reference, labels = _logits(0), _logits(1), _labels()
    selected = reducible_tokens(student, reference, labels, keep=0.5)

    excess = F.cross_entropy(student.permute(0, 3, 1, 2), labels, reduction="none", ignore_index=IGNORE_INDEX) - F.cross_entropy(
        reference.permute(0, 3, 1, 2), labels, reduction="none", ignore_index=IGNORE_INDEX
    )
    for codebook in range(CODEBOOKS):
        scored = labels[:, codebook] != IGNORE_INDEX
        kept = selected[:, codebook] != IGNORE_INDEX
        assert int(kept.sum()) == int(scored.sum()) // 2
        assert excess[:, codebook][kept].min() >= excess[:, codebook][scored & ~kept].max()
    assert torch.equal(selected[selected != IGNORE_INDEX], labels[selected != IGNORE_INDEX])
    assert reducible_tokens(student, reference, labels, keep=1.0) is labels

    # Codebooks below 'first_codebook' keep every token.
    partial = reducible_tokens(student, reference, labels, keep=0.5, first_codebook=1)
    assert torch.equal(partial[:, 0], labels[:, 0])
    assert torch.equal(partial[:, 1:], selected[:, 1:])


def test_the_teacher_runs_outside_the_autocast() -> None:
    seen = {}

    def teacher(**inputs: torch.Tensor) -> SimpleNamespace:
        seen["autocast"] = torch.is_autocast_enabled("cpu")
        seen["precision"] = torch.get_float32_matmul_precision()
        return SimpleNamespace(logits=_logits(1))

    def student(**inputs: torch.Tensor) -> SimpleNamespace:
        return SimpleNamespace(logits=_logits(0))

    inputs = {"input_ids": torch.zeros(2, CODEBOOKS, 6, dtype=torch.long), "labels": _labels()}
    with torch.autocast("cpu", dtype=torch.bfloat16):
        distillation_step(student, teacher, inputs, _config(), _weights())

    assert seen == {"autocast": False, "precision": "high"}
    assert torch.get_float32_matmul_precision() == "highest"


def test_distillation_weighs_the_main_and_polish_codebooks_apart() -> None:
    student, teacher, labels, weights = _logits(0), _logits(1), _labels(), _weights()
    inputs = {"input_ids": torch.zeros(2, CODEBOOKS, 6, dtype=torch.long), "labels": labels}

    def run(main: PathLossConfig, polish: PathLossConfig) -> dict:
        return distillation_step(
            lambda **_: SimpleNamespace(logits=student), lambda **_: SimpleNamespace(logits=teacher), inputs,
            _config(loss={"main": main, "polish": polish}), weights
        )

    # CE on the main path, KL on the polish codebooks: each term sees only its own codebooks.
    terms = run(PathLossConfig(ce=1.0, kl=0.0), PathLossConfig(ce=0.0, kl=1.0))
    main, polish = weights.clone(), weights.clone()
    main[2:], polish[:2] = 0.0, 0.0
    assert torch.allclose(terms["ce"], weighted_ce(student, labels, main))
    assert torch.allclose(terms["kl"], weighted_kl(student, teacher, labels, polish))
    assert torch.allclose(terms["loss"], terms["ce"] + terms["kl"])

    # Selection narrows the polish CE, and leaves the KL on every token.
    selected = run(PathLossConfig(ce=1.0, kl=1.0), PathLossConfig(ce=1.0, kl=1.0, ce_selection=0.5))
    kept = reducible_tokens(student, teacher, labels, 0.5, first_codebook=2)
    assert torch.allclose(selected["ce"], weighted_ce(student, kept, weights))
    assert torch.allclose(selected["kl"], weighted_kl(student, teacher, labels, weights))
    assert torch.equal(inputs["labels"], labels)   # the batch's own labels stay untouched

    # Each path selects on its own share: the main codebooks keep half of their tokens too.
    both = run(PathLossConfig(ce=1.0, kl=0.0, ce_selection=0.5), PathLossConfig(ce=1.0, kl=0.0, ce_selection=0.5))
    assert torch.allclose(both["ce"], weighted_ce(student, reducible_tokens(student, teacher, labels, 0.5), weights))


def test_the_loss_carries_gradients() -> None:
    student = _logits(10).requires_grad_(True)
    teacher, labels, weights = _logits(11), _labels(), _weights()

    distillation_loss(
        student, labels, weights, teacher_logits=teacher, ce_weight=1.0, kl_weight=1.0
    )["loss"].backward()

    assert student.grad is not None
    assert float(student.grad.abs().sum()) > 0.0


# ---------- Seeding ----------


def _draws() -> tuple:
    return random.random(), float(np.random.rand()), float(torch.rand(1))


def test_seeding_replays_every_stream() -> None:
    seed_everything(7)
    first = _draws()

    seed_everything(7)
    assert _draws() == first

    seed_everything(8)
    assert _draws() != first


def test_seed_of_depends_on_the_whole_place() -> None:
    assert seed_of(1234, 2, 3) == seed_of(1234, 2, 3)
    assert seed_of(1234, 2, 3) != seed_of(1234, 3, 2)   # Order carries meaning
    assert seed_of(1234, 2, 3) != seed_of(1234, 2, 4)

    assert 0 <= seed_of(2**40, 999, 5) < 2**31          # Fits what torch accepts


def test_workers_do_not_share_a_stream() -> None:
    torch.manual_seed(11)
    seed_worker(0)
    first = _draws()

    torch.manual_seed(11)
    seed_worker(1)
    second = _draws()

    assert first != second

    # The same worker of the same run draws the same noise again.
    torch.manual_seed(11)
    seed_worker(0)
    assert _draws() == first


# ---------- The meter ----------


def test_meter_averages_over_micro_batches_and_resets() -> None:
    meter = Meter()

    # Two optimizer steps' worth, at a gradient accumulation of two.
    for loss in (1.0, 2.0, 3.0, 4.0):
        meter.add({"loss": loss, "kl": loss / 2})

    reported = meter.report(step=2)

    assert reported["loss"] == pytest.approx(2.5)   # Over the four micro-batches
    assert reported["kl"] == pytest.approx(1.25)
    assert reported["steps_per_second"] > 0.0

    meter.add({"loss": 10.0})
    assert meter.report(step=3)["loss"] == pytest.approx(10.0)


def test_a_resumed_meter_counts_only_its_own_steps() -> None:
    meter = Meter(start_step=40_000)
    meter.started -= 10.0   # Ten seconds into the resumed run

    assert meter.report(step=40_050)["steps_per_second"] == pytest.approx(5.0, rel=0.01)


def test_meter_reports_nothing_it_was_not_given() -> None:
    meter = Meter()
    reported = meter.report(step=0)

    assert "loss" not in reported
    assert reported["steps_per_second"] == pytest.approx(0.0)


# ---------- MLflow tracking ----------


def test_eval_metrics_keep_the_ce_gap_at_four_states() -> None:
    from echo.training.tracking import LOGGED_STATES, eval_metrics

    row = {"ce": 1.0, "ce_teacher": 0.9, "ce_delta": 0.1, "kl": 0.05, "positions": 10.0}
    results = {state: row for state in ("s8_f0", "s8_f1", "s8_f3", "s8_f7", "s8_polish", "s8_main", "s8")}
    results["languages"] = {"en": row}

    assert eval_metrics("dev", results, with_teacher=True) == {f"dev/ce_delta/{state}": 0.1 for state in LOGGED_STATES}
    assert eval_metrics("test", results, with_teacher=False) == {f"test/ce/{state}": 1.0 for state in LOGGED_STATES}


# ---------- Worker seeding through the loader ----------


def test_workers_draw_their_own_splits(tmp_path) -> None:
    from tests.test_echo_data import BATCHING, _dataset

    dataset = _dataset(tmp_path, count=24)
    loader = EchoDataLoader(
        dataset,
        EchoPreprocessor(prefix_min_ratio=0.0, prefix_max_ratio=0.5),
        num_workers=2,
        persistent_workers=False,
        prefetch_factor=2,
        **BATCHING,
    )

    splits = [
        length - prompt.numel() - text.numel()
        for batch in loader
        for length, prompt, text in zip(
            batch["prefix_length"], batch["prompt_tokens"], batch["text_tokens"]
        )
    ]

    # Two workers replaying one stream would make the halves line up.
    assert len(set(splits)) > 1


# ---------- Checkpoints ----------


def _run(steps: int, config: EchoTrainingConfig, total: int = 100) -> tuple:
    """A toy run, far enough along that its state is distinguishable."""

    model = EchoModel(_model_config())
    optimizer = build_optimizer(model, config)
    schedule = build_schedule(optimizer, config, total)

    for _ in range(steps):
        model.audio_heads.weight.grad = torch.ones_like(model.audio_heads.weight)
        optimizer.step()
        schedule.step()

    return model, optimizer, schedule


def test_checkpoint_holds_the_weights_and_the_rest_of_the_run(tmp_path) -> None:
    config = _config(schedule={"steps": 100})
    model, optimizer, schedule = _run(40, config)

    path = save_checkpoint(tmp_path, model, optimizer, schedule, step=40, epoch=2)

    assert path.name == "checkpoint-40"
    assert {"config.json", "model.safetensors", "state.pt"} <= {p.name for p in path.iterdir()}


def test_resume_continues_where_it_stopped(tmp_path) -> None:
    config = _config(schedule={"steps": 100})
    model, optimizer, schedule = _run(40, config)

    torch.manual_seed(5)
    save_checkpoint(tmp_path, model, optimizer, schedule, step=40, epoch=2)
    expected_draw = torch.rand(3)
    rate = optimizer.param_groups[0]["lr"]

    fresh_model = EchoModel(_model_config())
    resumed_optimizer = build_optimizer(fresh_model, config)
    resumed_schedule = build_schedule(resumed_optimizer, config, 100)

    where = load_checkpoint(
        tmp_path / "checkpoint-40", fresh_model, resumed_optimizer, resumed_schedule
    )

    assert where == {"step": 40, "epoch": 2}
    assert resumed_optimizer.param_groups[0]["lr"] == rate   # Not one step behind
    assert torch.equal(torch.rand(3), expected_draw)         # The stream carries on

    # And the optimizer's own memory came back.
    moments = list(resumed_optimizer.state.values())[0]
    assert float(moments["exp_avg"].abs().sum()) > 0.0

    # As did the weights it was moving.
    for name, weight in model.state_dict().items():
        assert torch.equal(fresh_model.state_dict()[name], weight), name


def test_rotation_keeps_the_last_few(tmp_path) -> None:
    config = _config(schedule={"steps": 100})
    model, optimizer, schedule = _run(1, config)

    for step in (10, 20, 30, 40):
        save_checkpoint(tmp_path, model, optimizer, schedule, step=step, keep_last_n=2)

    kept = sorted(p.name for p in tmp_path.glob("checkpoint-*"))

    assert kept == ["checkpoint-30", "checkpoint-40"]


def test_tokenizer_travels_with_the_checkpoint(tmp_path) -> None:
    config = _config(schedule={"steps": 100})
    model, optimizer, schedule = _run(1, config)

    source = tmp_path / "base"
    source.mkdir()
    (source / "tokenizer.json").write_text("{}")
    (source / "tokenizer_config.json").write_text("{}")

    path = save_checkpoint(
        tmp_path / "run", model, optimizer, schedule, step=1, tokenizer_from=source
    )

    assert (path / "tokenizer.json").is_file()
    assert (path / "tokenizer_config.json").is_file()
    assert not (path / "audio_tokenizer").exists()   # 768 MB that never needs copying


def test_tokenizer_travels_from_a_hub_model(tmp_path, monkeypatch) -> None:
    from echo.training import checkpoint

    snapshot = tmp_path / "snapshot"
    snapshot.mkdir()
    (snapshot / "tokenizer.json").write_text("{}")
    monkeypatch.setattr(checkpoint, "_resolve_model_path", lambda name: str(snapshot))

    model, optimizer, schedule = _run(1, _config(schedule={"steps": 100}))
    path = save_checkpoint(
        tmp_path / "run", model, optimizer, schedule, step=1, tokenizer_from="k2-fsa/OmniVoice"
    )

    assert (path / "tokenizer.json").is_file()


def test_evaluation_pins_the_split_the_training_draws() -> None:
    config = _config(evaluation={"prefix_ratio": 0.3})

    training = EchoPreprocessor(**config.preprocessor_kwargs)
    evaluating = EchoPreprocessor(**config.eval_preprocessor_kwargs)

    assert (training.prefix_min_ratio, training.prefix_max_ratio) == (0.0, 0.5)
    assert (evaluating.prefix_min_ratio, evaluating.prefix_max_ratio) == (0.3, 0.3)

    # Everything else is shared, so an evaluation sees the training's masking.
    assert evaluating.masking_kwargs == training.masking_kwargs
    assert evaluating.polish_share == training.polish_share


def test_an_impossible_eval_split_is_rejected() -> None:
    with pytest.raises(ValueError, match="prefix_ratio"):
        _config(evaluation={"prefix_ratio": 1.0}).validate()


# ---------- Several ranks ----------


def test_every_rank_gets_its_own_share(tmp_path) -> None:
    from tests.test_echo_data import BATCHING, _dataset

    dataset = _dataset(tmp_path, count=60)
    whole = EchoBatchSampler(dataset, **BATCHING)
    shares = [EchoBatchSampler(dataset, rank, 3, **BATCHING) for rank in range(3)]

    lengths = {len(share) for share in shares}
    assert len(lengths) == 1                     # Or the ranks would wait on each other
    assert lengths.pop() == len(whole) // 3

    seen = [tuple(batch) for share in shares for batch in share.batches]
    assert len(seen) == len(set(seen))           # No sample is trained on twice per step
    assert set(seen) <= {tuple(batch) for batch in whole.batches}


def test_the_shares_follow_the_epoch(tmp_path) -> None:
    from tests.test_echo_data import BATCHING, _dataset

    dataset = _dataset(tmp_path, count=60)
    shares = [EchoBatchSampler(dataset, rank, 2, **BATCHING) for rank in range(2)]
    first = [tuple(batch) for batch in shares[0].batches]

    for epoch in (1, 2):
        for share in shares:
            share.set_epoch(epoch)

        # The packing is drawn afresh, so an epoch holds a different number of
        # batches - but the same number on every rank, which is what keeps them
        # in step.
        assert len({len(share.batches) for share in shares}) == 1

    assert [tuple(batch) for batch in shares[0].batches] != first


def test_a_world_of_one_changes_nothing(tmp_path) -> None:
    from tests.test_echo_data import BATCHING, _dataset

    sampler = EchoBatchSampler(_dataset(tmp_path, count=60), **BATCHING)

    assert sampler.batches == sampler.build_batches(0)
    assert world() == (0, 1)
    assert is_main() is True
    assert reduce_mean(2.5) == 2.5


# ---------- Recomputing instead of holding ----------


def test_checkpointing_follows_the_config(tmp_path) -> None:
    from echo.model.model import EchoModel
    from echo.training.trainer import Trainer
    from tests.test_echo_data import BATCHING, _dataset

    dataset = _dataset(tmp_path, count=8)

    def _built(enabled: bool) -> EchoModel:
        config = _config(schedule={"steps": 10}, optim={"gradient_checkpointing": enabled})
        config.checkpoint.output_dir = str(tmp_path)

        student = EchoModel(_model_config())
        Trainer(
            config=config,
            student=student,
            train_loader=EchoDataLoader(dataset, EchoPreprocessor(), num_workers=0, **BATCHING),
            loss_fn=plain_step,
        )

        return student

    assert _built(True).llm.is_gradient_checkpointing is True
    assert _built(False).llm.is_gradient_checkpointing is False


# ---------- CFG distillation ----------


def test_guidance_rises_as_its_shape_prescribes() -> None:
    from echo.training.config import GuidanceConfig

    constant = GuidanceConfig(scale=2.0, anneal_shape="constant")
    linear = GuidanceConfig(scale=2.0, anneal_shape="linear", anneal_ratio=0.25)
    power = GuidanceConfig(scale=2.0, anneal_shape="power", anneal_ratio=0.25, anneal_power=0.5)

    assert constant.at(0, 1000) == 2.0
    for ramp in (linear, power):
        assert ramp.at(0, 1000) == 0.0
        assert ramp.at(250, 1000) == 2.0
        assert ramp.at(900, 1000) == 2.0
    assert linear.at(125, 1000) == pytest.approx(1.0)
    assert power.at(125, 1000) == pytest.approx(2.0 * 0.5 ** 0.5)


def test_warmup_climbs_from_its_start_after_a_short_ramp() -> None:
    config = _config(schedule={"warmup_ratio": 0.45, "warmup_start_ratio": 0.5, "min_lr_ratio": 0.5})
    peak = config.optim.learning_rate

    rates = _rates(config, 1000)

    assert rates[0] == pytest.approx(0.05 * peak)      # The ramp, 10 steps up to half the peak
    assert rates[9] == pytest.approx(0.5 * peak)
    assert rates[229] == pytest.approx(0.75 * peak, rel=0.01)
    assert rates[449] == pytest.approx(peak)
    assert rates[-1] == pytest.approx(0.5 * peak, rel=0.01)

    # Starting from zero is the warmup as it always was.
    plain = _config(schedule={"warmup_ratio": 0.45})
    assert _rates(plain, 1000)[0] == pytest.approx(peak / 450)


def test_held_schedule_decays_over_the_rest_of_the_run() -> None:
    plain = _config(schedule={"warmup_ratio": 0.1, "hold_ratio": 0.0, "min_lr_ratio": 0.1})
    held = _config(schedule={"warmup_ratio": 0.1, "hold_ratio": 0.25, "min_lr_ratio": 0.1})
    peak = plain.optim.learning_rate

    rates = _rates(held, 100)

    # No hold is the schedule as it always was.
    assert _rates(plain, 100)[20] < peak
    assert rates[20] == pytest.approx(peak)
    assert rates[26] < peak
    assert rates[-1] == pytest.approx(0.1 * peak, rel=0.05)


def test_cfg_step_distils_the_guided_mixture_of_a_copy(monkeypatch) -> None:
    import copy

    from echo.training import step
    from tests.test_echo import AUDIO_VOCAB_SIZE, PREFIX_LEN, _batch

    # The tiny model's own mask token, in place of the real codec's.
    monkeypatch.setattr(step, "AUDIO_MASK_ID", AUDIO_VOCAB_SIZE - 1)

    student = EchoModel(_model_config()).eval()
    teacher = copy.deepcopy(student)
    input_ids, audio_mask, prefix_lengths, target_lengths = _batch()
    labels = input_ids.clone()
    labels[:, :, :PREFIX_LEN] = IGNORE_INDEX
    inputs = {
        "input_ids": input_ids, "audio_mask": audio_mask, "labels": labels,
        "prefix_lengths": prefix_lengths, "target_lengths": target_lengths,
    }
    config = _config(loss={"reverse_kl": True})
    weights = torch.tensor(student.normalized_audio_codebook_weights)

    kl = {}
    for scale in (0.0, 2.0):
        teacher.echo_config.cfg.guidance_scale = scale
        with torch.no_grad():
            terms = step.cfg_step(student, teacher, inputs, config, weights)
        assert float(terms["guidance"]) == scale
        kl[scale] = float(terms["kl"])

    # Unguided, the copy's mixture is the student's own conditional pass.
    assert kl[0.0] == pytest.approx(0.0, abs=1e-5)
    assert kl[2.0] > 1e-3
