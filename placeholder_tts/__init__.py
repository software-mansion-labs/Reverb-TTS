"""
placeholder_tts: voice-cloning text-to-speech in seven languages.

    from placeholder_tts import TTS

    tts = TTS.from_pretrained()
    voice = tts.voice("reference.wav", "Exact transcript of the reference.", language="en")
    audio = tts.generate("Text to speak.", voice, language="en", seed=0)
"""

from placeholder_tts.sampler import SamplerConfig
from placeholder_tts.text import LANGUAGES, prepare_text
from placeholder_tts.tts import TTS, Voice

__all__ = ["LANGUAGES", "SamplerConfig", "TTS", "Voice", "prepare_text"]
