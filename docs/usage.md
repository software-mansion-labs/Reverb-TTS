# Usage

## Speak in a cloned voice

```python
import soundfile as sf
from reverb_tts import TTS

tts = TTS.from_pretrained()
voice = tts.voice("anna.wav", "The exact transcript of the recording.", language="en")
audio = tts.generate("Nice to meet you. I'll be your guide today.", voice, language="en")
sf.write("output.wav", audio, tts.sample_rate)
```

- **The reference:** 3-10 s of clean speech from one speaker (over 15 s makes worse takes), with a transcript that matches
  it word for word. `language` in `voice` is the recording's language.
- **The audio input:** a file path in any format and sample rate, or samples (`[samples]` or `[channels, samples]`) with
  `sample_rate=`.
- **The output:** float32 samples at `tts.sample_rate` (24 kHz), with silent edges trimmed (`trim=False` keeps them).

## Use a preset voice

Preset voices need no recording:

```python
tts.preset_voices()                       # ['de_female_standard', 'de_male_standard', ..., 'pt_male_standard']
voice = tts.preset_voice("en_male_narrator_1")
audio = tts.generate("Hello! This is a preset voice speaking.", voice, language="en")
```

Each name starts with the voice's language: `de`, `en`, `es`, `fr`, `it`, `pl` or `pt`. A voice can speak any of the
seven languages, though it sounds most natural in its own.

Presets come from the `voices/` folder of
[software-mansion/Reverb-TTS](https://huggingface.co/software-mansion/Reverb-TTS), or of the local folder the model was
loaded from. `Voice.load` reads a preset file directly.

## Reuse a voice

A voice holds the reference's codec codes, so it can be saved and loaded without encoding the recording again:

```python
from reverb_tts import Voice

voice.save("anna.npz")
voice = Voice.load("anna.npz")
```

## Speak another language

`language` in `generate` is the text's language. It can differ from the voice's:

```python
voice = tts.voice("anna.wav", "The exact transcript of the recording.", language="en")
audio = tts.generate("Dzień dobry, jak się masz?", voice, language="pl")
```

## Reproduce a take, or draw several

Without a seed, every call draws a new take. A seed makes it reproducible:

```python
audio = tts.generate(text, voice, language="en", seed=42)
takes = [tts.generate(text, voice, language="en", seed=s) for s in range(4)]  # four takes to choose from
```

## Control the length

The length is estimated from the text and the reference's speaking rate. The rate counts only the time the
reference spends speaking: its silent edges are left out and any pause over 0.5 s counts as 0.5 s, so a recording
with long pauses doesn't make the takes slow. The recording itself is used whole.
`pace="recording"` estimates from the whole recording instead, pauses included, as earlier versions did.

```python
audio = tts.generate(text, voice, language="en", speed=1.25)   # length = estimate / 1.25 = 0.8x: faster speech
audio = tts.generate(text, voice, language="en", speed=0.9)    # length = estimate / 0.9 = 1.11x: slower speech
audio = tts.generate(text, voice, language="en", duration=4.0) # length = 4 s, whatever the estimate
audio = tts.generate(text, voice, language="en", speed=1.0, pace="recording")  # the rate of the whole recording, pauses included
```

## Long texts

A text estimated over 30 s is cut at sentence ends into pieces of about 15 s, generated with consecutive seeds, and
joined. This happens automatically, unless `duration` is set.

## Many texts at once

A list of texts is generated in batches and returns a list. Give one voice and language for all, or one per text:

```python
audios = tts.generate(["First line.", "Second line."], voice, language="en")
audios = tts.generate(["Hello.", "Bonjour."], [anna, pierre], language=["en", "fr"])
```

## Text preparation

Text and transcripts are prepared with [Verbalis](https://github.com/software-mansion-labs/Verbalis) first: numbers,
dates, money, units, symbols and abbreviations become words ("$49" becomes "forty-nine dollars"). Pass `prepare=False`
for text that is already written out, in `generate` or in `voice`.

## Weights and hardware

```python
import torch

tts = TTS.from_pretrained(variant="int4")                  # fp32, fp16, int8 (default) or int4
tts = TTS.from_pretrained(device="cpu", dtype=torch.float32)
tts = TTS.from_pretrained("path/to/weights")                # a local folder
```

- **variant:** every variant is converted to `dtype` on load, so the smaller ones download faster but run at the same
  speed. See [Reference](reference.md#model-variants).
- **device:** the first available of CUDA, MPS and CPU by default.
- **dtype:** bfloat16 on CUDA (float16 without bf16 support), float16 on MPS, float32 on CPU by default.

## Sampler settings

Override any setting for one call, or for a loaded model:

```python
audio = tts.generate(text, voice, language="en", steps=12, remasking=False)

from reverb_tts import SamplerConfig
tts = TTS.from_pretrained(sampler=SamplerConfig(steps=12))
```

| Setting | Default | What it does |
|---|---|---|
| `steps` | 8 | main passes, which decide codebooks 0-1 (content and timing) |
| `remasking` | True | main passes withdraw earlier picks the model has come to doubt |
| `remask_ratio` | 0.25 | picks withdrawn per new pick |
| `t_shift` | 0.1 | warps the schedule: few picks early, most in the last passes |
| `codebook_penalty` | 5.0 | makes codebook 0 tend to be decided before codebook 1 |
| `position_temperature` | 5.0 | scales confidence against random noise; higher is more random |
| `polish_bias` | 1.0 | attention bias towards the reference in the polish passes |
| `polish_guidance` | 0.5 | weight of the second polish pass's change over the first (codebooks 4-7) |

The defaults are the tuned setup. In our own tuning runs on the MiniMax test set (Whisper large-v3 and OmniVoice's
evaluation scripts, so not comparable with the leaderboard's numbers in the overview), they scored 1.35% WER and
0.818 SIM-o, against 1.50% and 0.817 with `remasking=False`. A `polish_guidance` of 1.0 scored 1.34% and 0.816.
