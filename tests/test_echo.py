"""Tests for the echo config / model / factory, on a tiny randomly initialized
model so they need neither a checkpoint nor a GPU."""

from typing import Any, Tuple

import json

import pytest
import torch
import torch.nn.functional as F
from transformers.models.auto.configuration_auto import CONFIG_MAPPING

from echo import (
    freeze,
    ArchitectureConfig,
    CFGConfig,
    EchoConfig,
    EchoMeta,
    EchoModel,
    PrefixBlockingConfig,
    build_model,
)
from echo.pruning.surgery import drop_layers, without_layers
from echo.masking.attention_masking import (
    apply_prefix_blocking_4d,
    build_dense_bidirectional_mask,
)
from omnivoice.models.omnivoice import GenerationTask, OmniVoice

BATCH = 2
SEQ_LEN = 12
PREFIX_LEN = 5
NUM_CODEBOOKS = 2
AUDIO_VOCAB_SIZE = 17


def _base_dict(num_hidden_layers: int = 4) -> dict:
    llm_config = CONFIG_MAPPING["qwen3"](
        vocab_size=128,
        hidden_size=32,
        num_hidden_layers=num_hidden_layers,
        num_attention_heads=4,
        num_key_value_heads=2,
        intermediate_size=64,
        max_position_embeddings=128,
        attn_implementation="sdpa",
    )
    from omnivoice.models.omnivoice import OmniVoiceConfig

    return OmniVoiceConfig(
        audio_vocab_size=AUDIO_VOCAB_SIZE,
        audio_mask_id=AUDIO_VOCAB_SIZE - 1,
        num_audio_codebook=NUM_CODEBOOKS,
        audio_codebook_weights=[2, 1],
        llm_config=llm_config,
    ).to_dict()


def _config(**overrides: Any) -> EchoConfig:
    return EchoConfig(base=_base_dict(), **overrides)


def _batch() -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    torch.manual_seed(0)
    input_ids = torch.randint(0, AUDIO_VOCAB_SIZE - 1, (BATCH, NUM_CODEBOOKS, SEQ_LEN))
    audio_mask = torch.zeros(BATCH, SEQ_LEN, dtype=torch.bool)
    audio_mask[:, PREFIX_LEN:] = True
    prefix_lengths = torch.full((BATCH,), PREFIX_LEN, dtype=torch.long)
    target_lengths = torch.full((BATCH,), SEQ_LEN - PREFIX_LEN, dtype=torch.long)
    return input_ids, audio_mask, prefix_lengths, target_lengths


def test_config_roundtrip(tmp_path) -> None:
    config = _config(
        meta=EchoMeta(name="round-trip"),
        cfg=CFGConfig(mode="teacher", guidance_scale=3.0),
    )
    path = tmp_path / "echo.json"
    config.to_json(path)

    loaded = EchoConfig.from_json(path)
    assert loaded.to_dict(include_base=False) == config.to_dict(include_base=False)
    assert loaded.base["llm_config"]["model_type"] == "qwen3"

    base = loaded.resolve_base_config()
    assert base.audio_vocab_size == AUDIO_VOCAB_SIZE
    assert base.llm_config.num_hidden_layers == 4


def test_unknown_key_is_dropped_unless_strict() -> None:
    data = {"meta": {"name": "x"}, "cfg": {"mode": "default", "guidence_scale": 2.0}}

    assert EchoConfig.from_dict(data).cfg.guidance_scale == 2.0
    with pytest.raises(ValueError, match="guidence_scale"):
        EchoConfig.from_dict(data, strict=True)


def test_config_drives_architecture_and_generation_defaults() -> None:
    config = _config(cfg=CFGConfig(mode="teacher", guidance_scale=2.0))
    model = EchoModel(config)

    # The architecture describes the base it was built from, and does not reshape it.
    assert len(model.llm.layers) == 4
    assert model.echo_config.architecture.num_hidden_layers == 4
    assert model.echo_config.architecture.intermediate_size == 64
    with pytest.raises(ValueError, match="does not match"):
        EchoModel(_config(architecture=ArchitectureConfig(num_hidden_layers=2)))

    gen_config = config.default_generation_config()
    assert gen_config.guidance_scale == 2.0
    assert gen_config.prefix_blocked is False

    # A plain model still guides at inference - only a distilled student does not.
    plain = _config(cfg=CFGConfig(mode="default", guidance_scale=2.0))
    assert plain.default_generation_config().guidance_scale == 2.0

    student = _config(cfg=CFGConfig(mode="student", guidance_scale=2.0))
    assert student.default_generation_config().guidance_scale == 0.0


def test_width_override_is_rejected() -> None:
    with pytest.raises(NotImplementedError):
        _config(architecture=ArchitectureConfig(hidden_size=64)).validate()


def test_prefix_blocking_cuts_only_prefix_to_target() -> None:
    target_mask = torch.zeros(1, SEQ_LEN, dtype=torch.bool)
    target_mask[:, PREFIX_LEN:] = True
    mask = torch.ones(1, 1, SEQ_LEN, SEQ_LEN, dtype=torch.bool)

    blocked = apply_prefix_blocking_4d(mask, target_mask)

    assert not blocked[0, 0, :PREFIX_LEN, PREFIX_LEN:].any()
    assert blocked[0, 0, :PREFIX_LEN, :PREFIX_LEN].all()
    assert blocked[0, 0, PREFIX_LEN:, :].all()


def test_forward_delegates_and_prefix_blocking_changes_output() -> None:
    config = _config()
    model = EchoModel(config).eval()
    input_ids, audio_mask, prefix_lengths, target_lengths = _batch()
    attention_mask = build_dense_bidirectional_mask(
        torch.ones(BATCH, SEQ_LEN, dtype=torch.bool)
    )

    with torch.no_grad():
        echo_out = model(input_ids, audio_mask, attention_mask=attention_mask)
        base_out = OmniVoice.forward(
            model, input_ids, audio_mask, attention_mask=attention_mask
        )
        blocked_out = model(
            input_ids,
            audio_mask,
            attention_mask=attention_mask,
            prefix_lengths=prefix_lengths,
            target_lengths=target_lengths,
            prefix_blocked=True,
        )

    assert torch.equal(echo_out.logits, base_out.logits)
    assert not torch.allclose(
        blocked_out.logits[:, :, :PREFIX_LEN], echo_out.logits[:, :, :PREFIX_LEN]
    )


def test_prefix_blocking_without_spans_raises() -> None:
    model = EchoModel(_config()).eval()
    input_ids, audio_mask, _, _ = _batch()

    with pytest.raises(ValueError, match="prefix_lengths"):
        model(input_ids, audio_mask, prefix_blocked=True)


def test_cfg_teacher_returns_both_branches() -> None:
    config = _config(cfg=CFGConfig(mode="teacher", guidance_scale=2.0))
    model = EchoModel(config).eval()
    input_ids, audio_mask, prefix_lengths, target_lengths = _batch()
    spans = {"prefix_lengths": prefix_lengths, "target_lengths": target_lengths}

    with torch.no_grad():
        guided = model(input_ids, audio_mask, **spans)
        plain = model(input_ids, audio_mask, unconditional=False, **spans)

    target_logits = guided.logits[:, :, PREFIX_LEN:]
    assert guided.cond_logits.shape == target_logits.shape
    assert guided.uncond_logits.shape == target_logits.shape
    assert guided.target_valid.all()

    # The conditional branch is the target slice of the full pass, and dropping
    # the prefix has to change what the model predicts there.
    assert torch.equal(guided.cond_logits, target_logits)
    assert not torch.allclose(guided.cond_logits, guided.uncond_logits)

    assert plain.cond_logits is None
    assert plain.uncond_logits is None
    assert plain.target_valid is None


def test_audio_heads_are_centered_without_changing_the_distribution() -> None:
    model = EchoModel(_config()).eval()
    heads = model.audio_heads

    with torch.no_grad():
        # A large row that every token of a codebook shares, as training leaves on the fine codebooks.
        shared = 5 * torch.randn(NUM_CODEBOOKS, 1, heads.in_features)
        heads.weight.view(NUM_CODEBOOKS, -1, heads.in_features).add_(shared)
        hidden = torch.randn(BATCH, SEQ_LEN, heads.in_features)
        logits = heads(hidden).view(BATCH, SEQ_LEN, NUM_CODEBOOKS, -1)
        plain = F.linear(hidden, heads.weight).view(BATCH, SEQ_LEN, NUM_CODEBOOKS, -1)

    assert torch.allclose(logits.mean(dim=-1), torch.zeros(()), atol=1e-4)
    assert torch.allclose(logits.log_softmax(dim=-1), plain.log_softmax(dim=-1), atol=1e-4)
    assert "audio_heads.weight" in model.state_dict()


def test_a_plain_model_runs_one_pass() -> None:
    model = EchoModel(_config(cfg=CFGConfig(mode="default"))).eval()
    input_ids, audio_mask, prefix_lengths, target_lengths = _batch()

    with torch.no_grad():
        out = model(input_ids, audio_mask)

    assert out.logits.shape[:3] == (BATCH, NUM_CODEBOOKS, SEQ_LEN)
    assert out.uncond_logits is None


def test_guided_mixture_matches_base_decoder() -> None:
    from echo.utils.cfg_utils import guided_log_probs
    from omnivoice.models.omnivoice import OmniVoiceGenerationConfig

    model = EchoModel(_config()).eval()
    torch.manual_seed(1)
    cond = torch.randn(1, NUM_CODEBOOKS, 4, AUDIO_VOCAB_SIZE)
    uncond = torch.randn(1, NUM_CODEBOOKS, 4, AUDIO_VOCAB_SIZE)
    gen_config = OmniVoiceGenerationConfig(guidance_scale=2.0, class_temperature=0.0)

    tokens, _ = model._predict_tokens_with_scoring(cond, uncond, gen_config)
    mixed = guided_log_probs(cond, uncond, 2.0)
    mixed[..., model.config.audio_mask_id] = -float("inf")

    assert torch.equal(mixed.argmax(dim=-1), tokens)


def test_factory_builds_and_roundtrips_through_save_pretrained(tmp_path) -> None:
    config = _config(
        meta=EchoMeta(name="factory"),
        prefix_blocking=PrefixBlockingConfig(enabled=True),
        cfg=CFGConfig(mode="teacher", guidance_scale=1.5),
    )
    model = build_model(config, dtype=torch.float32)
    assert model.prefix_blocked is True
    assert model.unconditional is True
    assert model.guidance_scale == 1.5

    model.save_pretrained(tmp_path)
    reloaded = EchoModel.from_pretrained(tmp_path, train=True)

    assert reloaded.echo_config.cfg.mode == "teacher"
    assert reloaded.prefix_blocked is True
    assert reloaded.echo_config.meta.name == "factory"


def test_generate_makes_forward_a_plain_pass(monkeypatch) -> None:
    """The base decoder blocks and guides in its own batch, and calls forward
    without spans - so both must be off for the length of the call."""

    config = _config(
        prefix_blocking=PrefixBlockingConfig(enabled=True),
        cfg=CFGConfig(mode="teacher", guidance_scale=2.0),
    )
    model = EchoModel(config)
    seen = {}

    def _record(self, text, *args, **kwargs):
        seen["blocked"] = self.prefix_blocked
        seen["unconditional"] = self.unconditional
        raise RuntimeError("boom")

    monkeypatch.setattr(OmniVoice, "generate", _record)

    with pytest.raises(RuntimeError, match="boom"):
        model.generate("hello")

    assert seen == {"blocked": False, "unconditional": False}

    # And restored, even though the call raised.
    assert model.prefix_blocked is True
    assert model.unconditional is True


def test_two_stage_sampler_leaves_the_polished_codebooks_to_the_polish_pass(monkeypatch) -> None:
    """With the main path on codebook 0 alone, every step's input keeps codebook 1 fully masked; only the
    polish pass, with codebook 0 complete, fills it."""

    model = EchoModel(_config()).eval()
    mask_id, target = model.config.audio_mask_id, 12

    def _inputs(self, text, target_len, *_):
        ids = torch.full((1, NUM_CODEBOOKS, PREFIX_LEN + target_len), mask_id)
        ids[..., :PREFIX_LEN] = 1
        return {"input_ids": ids, "audio_mask": torch.arange(PREFIX_LEN + target_len).unsqueeze(0) >= PREFIX_LEN}

    seen, forward = [], model.forward

    def _record(*args, **kwargs):
        seen.append(kwargs["input_ids"][0, :, -target:].clone())
        return forward(*args, **kwargs)

    monkeypatch.setattr(EchoModel, "_prepare_inference_inputs", _inputs)
    monkeypatch.setattr(model, "forward", _record)

    config = model.echo_config.default_generation_config(
        num_step=4, guidance_scale=0.0, main_codebooks=1, remask_codebooks=1, polish_codebooks_from=1
    )
    task = GenerationTask(1, ["x"], [target], [None], [None], [None], [None], [None])
    with torch.inference_mode():
        tokens = model._generate_iterative(task, config)[0]

    assert len(seen) == 5
    assert all((step[1] == mask_id).all() for step in seen)
    assert not (seen[-1][0] == mask_id).any()
    assert not (tokens == mask_id).any()


# ---------- Loading for training, and layer surgery ----------


def test_training_load_needs_no_tokenizer_files(tmp_path) -> None:
    """A checkpoint written mid-run holds weights alone, and must still load."""

    EchoModel(_config()).save_pretrained(tmp_path)
    assert sorted(p.name for p in tmp_path.iterdir()) == ["config.json", "model.safetensors"]

    resumed = _config(init_from=str(tmp_path))

    with pytest.raises(ValueError, match="tokenizer"):
        build_model(resumed)

    model = build_model(resumed, train=True)
    assert model.training is True
    assert all(p.requires_grad for p in model.llm.parameters())


def test_freeze_makes_a_fixed_reference() -> None:
    model = freeze(EchoModel(_config()))

    assert model.training is False
    assert not any(p.requires_grad for p in model.parameters())


def test_without_layers_matches_a_model_that_never_had_them(tmp_path) -> None:
    full = EchoModel(_config()).eval()
    full.save_pretrained(tmp_path)

    shallow = drop_layers(EchoModel.from_pretrained(tmp_path, train=True), [2, 3]).eval()

    input_ids, audio_mask, _, _ = _batch()

    with torch.no_grad():
        whole = full(input_ids, audio_mask).logits
        with without_layers(full, [2, 3]) as cut:
            assert len(cut.llm.layers) == 2
            bypassed = cut(input_ids, audio_mask).logits
        restored = full(input_ids, audio_mask).logits
        expected = shallow(input_ids, audio_mask).logits

    assert torch.equal(bypassed, expected)      # The same model, by another route
    assert not torch.allclose(whole, bypassed)
    assert torch.equal(whole, restored)         # And the original is back
    assert len(full.llm.layers) == 4


def test_drop_layers_leaves_a_model_that_describes_itself(tmp_path) -> None:
    model = drop_layers(EchoModel(_config()), [1])

    assert len(model.llm.layers) == 3
    assert [layer.self_attn.layer_idx for layer in model.llm.layers] == [0, 1, 2]
    assert model.config.llm_config.num_hidden_layers == 3
    assert model.echo_config.architecture.num_hidden_layers == 3

    # And it survives the round trip every later stage depends on. The config
    # carries no inline 'base', so the checkpoint's own depth is what counts.
    model.save_pretrained(tmp_path)
    saved = json.loads((tmp_path / "config.json").read_text())

    assert saved["llm_config"]["num_hidden_layers"] == 3
    assert saved["echo"]["architecture"]["num_hidden_layers"] == 3

    reloaded = build_model(EchoConfig(init_from=str(tmp_path)), train=True)
    assert len(reloaded.llm.layers) == 3


def test_surgery_rejects_indices_outside_the_model() -> None:
    with pytest.raises(ValueError, match="layer indices"):
        drop_layers(EchoModel(_config()), [9])


def test_guided_mixture_can_rule_out_the_mask_token() -> None:
    from echo.utils.cfg_utils import guided_log_probs

    torch.manual_seed(2)
    cond = torch.randn(1, NUM_CODEBOOKS, 4, AUDIO_VOCAB_SIZE)
    uncond = torch.randn(1, NUM_CODEBOOKS, 4, AUDIO_VOCAB_SIZE)
    mask_id = AUDIO_VOCAB_SIZE - 1

    plain = guided_log_probs(cond, uncond, 2.0)
    ruled = guided_log_probs(cond, uncond, 2.0, mask_id=mask_id)

    assert float(plain.exp()[..., mask_id].max()) > 0.0
    assert float(ruled.exp()[..., mask_id].max()) == 0.0

    # Still a distribution, and the order of the tokens that remain is untouched.
    assert torch.allclose(ruled.exp().sum(dim=-1), torch.ones(1, NUM_CODEBOOKS, 4))
    assert torch.equal(
        plain[..., :mask_id].argsort(dim=-1), ruled[..., :mask_id].argsort(dim=-1)
    )
