# Reference

## Model variants

Weights for the Python package (`TTS.from_pretrained(variant=...)`):

| Variant | Weights | Activations | Download |
|---|---|---|---|
| `fp32` | fp32 | full precision | 1.19 GB |
| `int8` (default) | int8, one scale per output channel | full precision | 307 MB |
| `int8a8` | int8, one scale per output channel | int8, one scale per token | 307 MB |
| `int4` | int4, one scale per group of 32 inputs | full precision | 191 MB |

In the quantized variants the decoder's linear layers hold the quantized weights, with fp16 scales. On an internal test
set, `int8` showed no quality loss against `fp32`, and `int4` none that we could measure. The benchmarks use full
precision.

[ExecuTorch](https://pytorch.org/executorch) programs for on-device inference, with the codec kept out of the program:

| Program | Hardware | Size |
|---|---|---|
| MLX fp16 | Apple GPUs | 612 MB |
| MLX int8 | Apple GPUs | 352 MB |
| MLX int4 | Apple GPUs | 232 MB |
| Vulkan int4 | Android GPUs | 233 MB |
| QNN fp16 | Qualcomm NPUs (Snapdragon 8 Gen 3, SM8650) | 614 MB |

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
  executorch/vulkan/int4/      reverb_vulkan_int4.pte
  executorch/qnn/fp16/         reverb_qnn_fp16_sm8650.pte
voices/                        preset voices: <name>.json
assets/                        images for the model card
```

