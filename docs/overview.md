# Overview

Reverb is a masked-diffusion text-to-speech model. It fills in a grid of
[Higgs audio codec](https://huggingface.co/eustlb/higgs-audio-v2-tokenizer) tokens (8 codebooks per frame, 25 frames per
second) over a few passes, conditioned on the text and on a reference recording of the voice to clone.

## How it was made

1. **Initialized from [OmniVoice](https://github.com/k2-fsa/OmniVoice)**, a 613M-parameter masked-diffusion TTS model.
2. **Pruned with distillation:** 4 of the 28 transformer layers removed, the FFN cut from 3072 to 2048, the text
   vocabulary from 151,676 to 62,035 tokens; the pruned model was trained to match OmniVoice's predictions.
3. **Retrained** on cleaned, selected subsets of open speech datasets in the seven languages.
4. **Preference post-training (RLHF-like):** the model generated several takes per prompt, a critic fitted on human
   preference rankings and a CTC intelligibility check picked the better and worse ones, and DPO trained on those pairs.

## Results

Quality on the [MiniMax Multilingual Test Set](https://huggingface.co/datasets/k2-fsa/TTS_eval_datasets) (the seven
languages, 700 sentences), scored with OmniVoice's evaluation scripts; WER and SIM-o are averaged over languages. Speed
is RTFx, seconds of speech generated per second: batch 32 on one RTX 5090, from the text and an already-encoded reference
to the waveform, each model in its default runtime.

| Model | Parameters | Weights (16-bit) | WER ↓ | SIM-o ↑ | RTFx ↑ |
|---|---|---|---|---|---|
| **Reverb** | **297M** | **0.59 GB** | **1.35%** | 0.816 | **164.9×** |
| [OmniVoice](https://huggingface.co/k2-fsa/OmniVoice) | 613M | 1.23 GB | 1.64% | **0.835** | 13.2× |
| [Qwen3-TTS 0.6B Base](https://huggingface.co/Qwen/Qwen3-TTS-12Hz-0.6B-Base) | 915M | 1.83 GB | 11.96%¹ | 0.780¹ | 3.3× |
| [Audio8-TTS-Realistic-Air](https://huggingface.co/Edge0/Audio8-TTS-Realistic-Air) | 601M | 1.20 GB | 1.97% | 0.750 | 4.2× |

¹ Over the six languages Qwen3-TTS supports (Polish excluded); the other models over all seven.

On an Apple M4 Pro (ExecuTorch MLX, median time from text to audio codes; speed relative to real time in brackets),
against OmniVoice at 16 guided passes:

| Reference + target speech | OmniVoice | Reverb int8 | Speed-up |
|---|---|---|---|
| 5 s + 5 s | 2,278 ms (2.2×) | **175 ms (28.5×)** | 13.0× |
| 5 s + 10 s | 3,388 ms (3.0×) | **300 ms (33.4×)** | 11.3× |
| 10 s + 5 s | 3,132 ms (1.6×) | **192 ms (26.1×)** | 16.3× |
| 10 s + 10 s | 4,157 ms (2.4×) | **317 ms (31.5×)** | 13.1× |
