import pytest
import torch

from helpers import LLM_CONFIG, StubCodec, StubTokenizer
from placeholder_tts.model import Network
from placeholder_tts.tts import TTS


@pytest.fixture
def network() -> Network:
    torch.manual_seed(0)
    return Network(LLM_CONFIG, rank=8).eval()


@pytest.fixture
def tts(network: Network) -> TTS:
    return TTS(network, StubTokenizer(), StubCodec())
