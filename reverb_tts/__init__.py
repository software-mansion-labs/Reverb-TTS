"""
Reverb: voice-cloning text-to-speech in seven languages.

    from reverb_tts import TTS

    tts = TTS.from_pretrained()
    voice = tts.voice("reference.wav", "Exact transcript of the reference.", language="en")
    audio = tts.generate("Text to speak.", voice, language="en", seed=0)
"""

from importlib.metadata import PackageNotFoundError, version

from reverb_tts.sampler import SamplerConfig
from reverb_tts.text import LANGUAGES, prepare_text
from reverb_tts.tts import TTS, Voice

try:
    __version__ = version("reverb-tts")
except PackageNotFoundError:  # running from a source tree that isn't installed
    __version__ = "unknown"

__all__ = ["LANGUAGES", "SamplerConfig", "TTS", "Voice", "__version__", "prepare_text"]
