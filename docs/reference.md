# Reference

## Model variants

Weights for the Python package (`TTS.from_pretrained(variant=...)`):

| Variant | Weights | Activations | Download |
|---|---|---|---|
| `fp32` | fp32 | full precision | 1.19 GB |
| `int8` (default) | int8, one scale per output channel | full precision | 307 MB |
| `int8a8` | int8, one scale per output channel | int8, one scale per token | 307 MB |
| `int4` | int4, one scale per group of 32 inputs | full precision | 191 MB |

In the quantized variants the decoder's linear layers hold the quantized weights, with fp16 scales. In automatic scoring
of generated speech, `int8` and `int8a8` showed no measurable loss against `fp32`, and `int4` a small one.

[ExecuTorch](https://pytorch.org/executorch) programs for on-device inference (Apple GPUs through MLX), with the codec
kept out of the program:

| Program | Size |
|---|---|
| MLX fp16 | 612 MB |
| MLX int8 | 352 MB |
| MLX int4 | 232 MB |

## Repository layout

This repository:

```
reverb_tts/
  tts.py        TTS and Voice: the public API
  text.py       text preparation, length estimate, long-text splitting
  audio.py      loading audio, trimming silence
  sampler.py    the sampler and its settings
  model.py      the network
tests/          tests on a small random model, no download needed
docs/
```

The weights on Hugging Face, [software-mansion/Reverb-TTS](https://huggingface.co/software-mansion/Reverb-TTS):

```
model/
  tokenizer.json, tokenizer_config.json, chat_template.jinja
  safetensors/<variant>/       config.json, model.safetensors: fp32, int8, int8a8, int4
  executorch/mlx/<precision>/  reverb_mlx_<precision>.pte: fp16, int8, int4
voices/                        preset voices: <name>.json
```

