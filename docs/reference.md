# Reference

## Model variants

### safetensors

PyTorch weights for the `reverb-tts` package, in four variants (`TTS.from_pretrained(variant=...)`):

| Variant | Weights | Activations | Download |
|---|---|---|---|
| `fp32` | fp32 | full precision | 1.19 GB |
| `fp16` | fp16 | full precision | 594 MB |
| `int8` (default) | int8, one scale per output channel | full precision | 307 MB |
| `int4` | int4, one scale per group of 32 inputs | full precision | 191 MB |

In the quantized variants the decoder's linear layers hold the quantized weights, with fp16 scales. On an internal test
set, `int8` showed no quality loss against `fp32`, and `int4` none that we could measure. The benchmarks use full
precision.

### ExecuTorch

[ExecuTorch](https://pytorch.org/executorch) programs for on-device inference, with the codec kept out of the program:

| Program | Hardware | Size |
|---|---|---|
| MLX fp16 | Apple GPUs | 612 MB |
| MLX int8 | Apple GPUs | 352 MB |
| MLX int4 | Apple GPUs | 232 MB |
| Vulkan int4 | Android GPUs | 233 MB |
| QNN int4 | Qualcomm NPUs, one program per chip: Snapdragon 8 Gen 1 (SM8450, v69), 8 Gen 2 (SM8550, v73), 8 Gen 3 (SM8650, v75), 8 Elite (SM8750, v79), 8 Elite Gen 5 (SM8850, v81) | 232–237 MB each |

### In the browser

[ruNNtime](https://docs.swmansion.com/runntime/zoo/getting-started/) runs Reverb on the GPU in the browser through
WebGPU, from its own copy of the weights.

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
  safetensors/<variant>/       config.json, model.safetensors: fp32, fp16, int8, int4
  executorch/mlx/<precision>/  reverb_mlx_<precision>.pte: fp16, int8, int4
  executorch/vulkan/int4/      reverb_vulkan_int4.pte
  executorch/qnn/int4/         reverb_qnn_int4_<hexagon>.pte: v69, v73, v75, v79, v81
  runntime/<precision>/        model.safetensors for ruNNtime: int8, int4
voices/                        preset voices: <name>.json
assets/                        images for the model card
```

