# placeholder_tts

> `placeholder_tts` is a placeholder for the package's final name.

**placeholder_tts** is a **voice-cloning-first text-to-speech model**. Built on top of one of the leading open-weight
TTS models, it combines exceptional quality in mimicking any reference speaker, including prosody, emotions and voice
timbre, with cost efficiency and high performance, allowing it to run locally on mobile and desktop systems. It speaks
seven languages: English, German, French, Spanish, Italian, Polish and Portuguese.

## Install

```bash
pip install .
```

Python 3.10 or later. The weights are downloaded from Hugging Face on first use. While the weights repository is
private, log in first (`hf auth login`) or set `HF_TOKEN`.

## Quick start

```python
import soundfile as sf
from placeholder_tts import TTS

tts = TTS.from_pretrained()
voice = tts.voice("reference.wav", "The exact transcript of the reference recording.", language="en")
audio = tts.generate("I bought 3 tickets for $49, and we leave at 7:30 tomorrow.", voice, language="en", seed=0)
sf.write("output.wav", audio, tts.sample_rate)
```

## Usage

**Voices.** `tts.voice(audio, transcript, language)` takes a file path, or samples with their `sample_rate`. Use 3-10 s
of clean speech from one speaker, with a transcript that matches it exactly; `language` is the recording's. A voice
can be saved and loaded, so the recording needn't be encoded again:

```python
from placeholder_tts import Voice

voice.save("anna.npz")
voice = Voice.load("anna.npz")
```

**Generating.** `tts.generate(text, voice, language)` returns float32 samples at `tts.sample_rate` (24 kHz).

- `language` is the text's, and may differ from the voice's.
- `seed` makes a take reproducible; without one, each call draws a new take.
- `speed` scales the estimated length (`speed=1.2` speaks faster); `duration` sets it in seconds.
- A list of texts is generated in batches and returns a list, with one voice and language for all or one per text.
- A text estimated over 30 s is cut at sentence ends and spoken in pieces of about 15 s.

**Text.** Text is prepared with [Verbalis](https://github.com/software-mansion-labs/Verbalis) before it reaches the
model: numbers, dates, money, units, symbols and abbreviations are read out as words. Pass `prepare=False` for text
that is already prepared.

**Sampler.** Every sampler setting can be overridden per call, or for a loaded model:

```python
audio = tts.generate(text, voice, language="en", steps=12, remasking=False)

from placeholder_tts import SamplerConfig
tts = TTS.from_pretrained(sampler=SamplerConfig(steps=12, polish_guidance=0.5))
```

The settings and what they do are in [docs/pipeline.md](docs/pipeline.md#sampler-settings).

**Variants and devices.** `TTS.from_pretrained(variant=...)` picks the weights:

| Variant | Decoder weights | Download |
|---|---|---|
| `fp32` | fp32 | 1.19 GB |
| `int8` (default) | int8, one scale per output channel | 307 MB |
| `int8a8` | int8, with int8 activations | 307 MB |
| `int4` | int4, one scale per group of 32 inputs | 191 MB |

Quantized weights are expanded to the compute dtype on load, so they download smaller but run at full speed on any
device. The device is the first available of CUDA, MPS and the CPU (or `device=`); the dtype is bfloat16 on CUDA,
float16 on MPS and float32 on the CPU (or `dtype=`). `TTS.from_pretrained("path/to/folder")` loads local weights.

## How it works

[docs/pipeline.md](docs/pipeline.md) walks through the whole pipeline, from text and reference to audio.

The model is a masked-diffusion TTS model working on [Higgs audio codec](https://huggingface.co/eustlb/higgs-audio-v2-tokenizer)
tokens: it fills in a grid of codec tokens (8 codebooks per frame, 25 frames per second) over several passes,
conditioned on the text and on the reference speaker's audio. It was made from
[OmniVoice](https://github.com/k2-fsa/OmniVoice) in three stages:

1. **Pruning with distillation.** 4 of the 28 transformer layers are dropped, the FFN shrinks from 3072 to 2048, and the
   text vocabulary from 151,676 to 62,035 tokens. The pruned student is trained to match OmniVoice's predictions.
2. **Supervised retraining** on cleaned, selected subsets of open datasets in the seven languages, such as
   [Emilia](https://huggingface.co/datasets/amphion/Emilia-Dataset) and variants of YODAS2, such as
   [yodas2-sidon](https://huggingface.co/datasets/sarulab-speech/yodas2_sidon).
3. **Preference post-training with DPO.** The model generates several takes per prompt, a critic fitted on human
   preference rankings and a CTC intelligibility check pick winners and losers, and DPO trains on those pairs, in
   rounds.

Three changes make each generation cheap:

- **CFG self-distillation.** Classifier-free guidance normally runs every pass twice, with and without conditioning.
  The model learned the guided output directly, so each pass is one forward.
- **Prefix blocking.** The text and reference prefix never attends to the generated audio, so its keys and values are
  computed once and reused by every pass.
- **A two-stage sampler.** 8 passes decide codebooks 0-1, then two polish passes decide codebooks 2-7: 10 forwards in
  all, against OmniVoice's default of 32 guided passes (64 forwards).

| | OmniVoice | placeholder_tts |
|---|---|---|
| Parameters | 613M | **297M** |
| Weights | 2.45 GB (fp32) | 1.19 GB (fp32), 307 MB (int8), 191 MB (int4) |
| Forwards per generation | 64 (32 passes with CFG) | **10** |

## Samples

Each row pairs a reference recording with the model speaking new text in that voice. References are passed through
the same audio codec as the model's output. The samples come from a prototype of the model, before post-training.

| Voice | Language | Reference | Generated |
|---|---|---|---|
| Melissa | English | [reference](assets/samples/melissa_reference.mp4) | [generated](assets/samples/melissa_generated.mp4) |
| Shepherd | English | [reference](assets/samples/shepherd_reference.mp4) | [generated](assets/samples/shepherd_generated.mp4) |
| Jordan | English | [reference](assets/samples/jordan_reference.mp4) | [generated](assets/samples/jordan_generated.mp4) |
| Missler | German | [reference](assets/samples/missler_reference.mp4) | [generated](assets/samples/missler_generated.mp4) |
| Jean-Pierre | French | [reference](assets/samples/jean_pierre_reference.mp4) | [generated](assets/samples/jean_pierre_generated.mp4) |
| Corrado | Italian | [reference](assets/samples/corrado_reference.mp4) | [generated](assets/samples/corrado_generated.mp4) |
| Bruno | Portuguese | [reference](assets/samples/bruno_reference.mp4) | [generated](assets/samples/bruno_generated.mp4) |

## Tests

```bash
pip install ".[test]"
pytest
```

The tests use a small random model, so they need no download.

## License and acknowledgments

- **Code:** Apache 2.0; see [LICENSE](LICENSE).
- **Model weights:** [CC BY-NC 4.0](https://creativecommons.org/licenses/by-nc/4.0/), for non-commercial use with
  attribution.

The model's weights were initialized from [OmniVoice](https://github.com/k2-fsa/OmniVoice) by the k2-fsa team, and its
target-length rule and prompt format come from OmniVoice's code. We thank its authors for releasing the model and code
openly. The model uses the [Higgs Audio v2 tokenizer](https://huggingface.co/eustlb/higgs-audio-v2-tokenizer) as its
audio codec, and [Verbalis](https://github.com/software-mansion-labs/Verbalis) prepares its text.
