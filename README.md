# Reverb

**Reverb** is a voice-cloning text-to-speech model: give it a few seconds of someone speaking, and it speaks any text
in that voice. It speaks English, German, French, Spanish, Italian, Polish and Portuguese, and is small and fast enough
to run locally on desktop and mobile devices.

## Install

```bash
pip install .
```

Python 3.10 or later. The weights download from Hugging Face on first use; while the weights repository is private, log
in first (`hf auth login`) or set `HF_TOKEN`.

## Quick start

```python
import soundfile as sf
from reverb_tts import TTS

tts = TTS.from_pretrained()
voice = tts.voice("reference.wav", "The exact transcript of the reference recording.", language="en")
audio = tts.generate("Hello! This is my cloned voice.", voice, language="en")
sf.write("output.wav", audio, tts.sample_rate)
```

## Documentation

- [Overview](docs/overview.md): what the model is, how it was made, and how it scores.
- [Usage](docs/usage.md): every use case, from the quick start to sampler settings.
- [Reference](docs/reference.md): model variants and the repositories' layout.

## License and acknowledgments

- **Code:** Apache 2.0; see [LICENSE](LICENSE).
- **Model weights:** [CC BY-NC 4.0](https://creativecommons.org/licenses/by-nc/4.0/), for non-commercial use with
  attribution.

Reverb's weights were initialized from [OmniVoice](https://github.com/k2-fsa/OmniVoice) by the k2-fsa team, and its
target-length rule and prompt format come from OmniVoice's code. We thank its authors for releasing the model and code
openly. Reverb uses the [Higgs Audio v2 tokenizer](https://huggingface.co/eustlb/higgs-audio-v2-tokenizer) as its audio
codec, and [Verbalis](https://github.com/software-mansion-labs/Verbalis) prepares its text.
