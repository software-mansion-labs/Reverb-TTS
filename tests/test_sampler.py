import pytest
import torch

from placeholder_tts.model import MASK, Network
from placeholder_tts.sampler import Batch, Prompt, SamplerConfig, sample


def _prompts(seed: int = 0) -> list:
    generator = torch.Generator().manual_seed(seed)
    return [Prompt(torch.randint(0, 64, (n,), generator=generator), torch.randint(0, 1024, (8, r), generator=generator),
                   frames, seed + i) for i, (n, r, frames) in enumerate([(12, 20, 30), (7, 9, 18)])]


def _takes(network: Network, config: SamplerConfig, seed: int = 0) -> list:
    prompts = _prompts(seed)
    with torch.inference_mode():
        return sample(Batch(network, prompts, config.polish_bias), [p.seed for p in prompts], config)


def test_every_slot_of_every_target_is_filled(network: Network) -> None:
    for take, frames in zip(_takes(network, SamplerConfig()), (30, 18)):
        assert take.shape == (8, frames)
        assert int(take.max()) < MASK and int(take.min()) >= 0


def test_the_same_seed_gives_the_same_take(network: Network) -> None:
    first, second = _takes(network, SamplerConfig()), _takes(network, SamplerConfig())
    assert all(torch.equal(a, b) for a, b in zip(first, second))
    other = _takes(network, SamplerConfig(), seed=1)
    assert not torch.equal(first[0], other[0])


@pytest.mark.parametrize("overrides", [{"steps": 1}, {"remasking": False}, {"steps": 16, "remask_ratio": 0.5},
                                       {"polish_bias": 0.0, "polish_guidance": 0.0}])
def test_sampler_settings_can_be_overridden(network: Network, overrides: dict) -> None:
    takes = _takes(network, SamplerConfig(**overrides))
    assert all(int(take.max()) < MASK for take in takes)


@pytest.mark.parametrize("bad", [{"steps": 0}, {"remask_ratio": 1.0}, {"t_shift": 0.0}, {"position_temperature": 0.0}])
def test_invalid_sampler_settings_are_rejected(bad: dict) -> None:
    with pytest.raises(ValueError):
        SamplerConfig(**bad)
