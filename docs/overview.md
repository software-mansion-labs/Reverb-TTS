# Overview

Reverb is a masked-diffusion text-to-speech model. It fills in a grid of
[Higgs audio codec](https://huggingface.co/eustlb/higgs-audio-v2-tokenizer) tokens (8 codebooks per frame, 25 frames per
second) over a few passes, conditioned on the text and on a reference recording of the voice to clone.

## How it was made

1. **Initialized from [OmniVoice](https://github.com/k2-fsa/OmniVoice)**, a 612M-parameter masked-diffusion TTS model.
2. **Pruned with distillation:** 4 of the 28 transformer layers removed, the FFN cut from 3072 to 2048, the KV groups
   from 8 to 6, the text vocabulary from 151,676 to 62,035 tokens; the pruned model was trained to match OmniVoice's
   predictions.
3. **Retrained** on cleaned, selected subsets of open speech datasets in the seven languages.
4. **Preference post-training (RLHF-like):** the model generated several takes per prompt, a critic fitted on human
   preference rankings and a CTC intelligibility check picked the better and worse ones, and DPO trained on those pairs.

<picture>
  <source media="(prefers-color-scheme: dark)" srcset="assets/training_scheme_dark.png">
  <img src="assets/training_scheme.png" alt="How Reverb is made: OmniVoice is pruned and distilled, retrained on seven languages, then refined with iterative DPO against a critic fitted on human preference rankings.">
</picture>

## Results

We compare Reverb with well-known open-weight models on the
[Open TTS Leaderboard](https://huggingface.co/spaces/hf-audio/open_tts_leaderboard)'s MiniMax test set, in seven
languages: 100 sentences per language, one take each, every model cloning a reference recording. WER (Qwen3-ASR) and
speaker similarity (SIM-o, WavLM) are averaged over the languages.

| Model | Parameters | WER ↓ | SIM-o ↑ |
|:---|---:|---:|---:|
| **Reverb** | **319M** | **1.03%** | 0.817 |
| [OmniVoice](https://huggingface.co/k2-fsa/OmniVoice) | 634M | 1.13% | **0.833** |
| [VoxCPM2](https://huggingface.co/openbmb/VoxCPM2) | 2,336M | 1.20% | 0.812 |
| [Fish S2 Pro](https://huggingface.co/fishaudio/s2-pro) | 4.95B | 1.35% | 0.761 |
| [Audio8-TTS-Realistic](https://huggingface.co/Edge0/Audio8-TTS-Realistic-0.6b) | 732M | 1.46% | 0.764 |
| [MOSS-TTS](https://huggingface.co/OpenMOSS-Team/MOSS-TTS) | 8.49B | 1.84% | 0.774 |
| [Chatterbox](https://huggingface.co/ResembleAI/chatterbox) | 0.80B | 2.76% | 0.715 |
| [Higgs TTS 3](https://huggingface.co/bosonai/higgs-tts-3-4b) | 4.65B | 6.59% | 0.754 |

Generation speed, in × real time (seconds of audio per second of compute; higher is better), on MiniMax English
sentences. Timing starts from the text and an already-encoded reference voice. Every model runs its default settings.

| Hardware | Reverb | OmniVoice | Audio8 | Qwen3-TTS 0.6B | Qwen3-TTS 1.7B | VoxCPM2 | Fish S2 Pro |
|:---|---:|---:|---:|---:|---:|---:|---:|
| NVIDIA H200, batch 32 | **232.3×** | 45.5× | 11.2× | – | – | 1.9× | 0.40×² |
| NVIDIA RTX 5090, batch 32 | **164.9×** | 13.2× | 4.2× | 3.3× | 4.0× | 3.0×² | 0.18×² |
| Apple M4 Pro, batch 1 | **12.3×³** | 0.84× | 0.76× | 0.51× | 0.52× | 0.69× | – |
| iPhone 17, batch 1 | **20.6×** | 0.91× | 1.03× | 1.14× | 0.74× | – | – |

RTX 5090 and M4 Pro: PyTorch in bf16 on the GPU, to the waveform with codec decoding. iPhone 17: ExecuTorch MLX
programs in fp16 on the GPU, to the codec codes, without the decoder. H200: the Open TTS Leaderboard's default voice,
MiniMax test set, 7 languages.<br>
² No batched generation: sentences run one at a time.<br>
³ Reverb's ExecuTorch MLX program (fp16) with the Core ML codec decoder, as it ships: 21.6×.

<picture>
  <source media="(prefers-color-scheme: dark)" srcset="assets/frontier_size_speed_dark.png">
  <img src="assets/frontier_size_speed.png" alt="Speed against size on an RTX 5090, batch 32: every other model is bigger and slower than Reverb-TTS">
</picture>

### Time to codes on an Apple M4 Pro

ExecuTorch MLX, median time from the text to audio codes, without codec decoding (speed relative to real time in
brackets), against OmniVoice at 16 guided passes:

| Reference + target speech | OmniVoice | Reverb int8 | Speed-up |
|---|---|---|---|
| 5 s + 5 s | 2,278 ms (2.2×) | **175 ms (28.5×)** | 13.0× |
| 5 s + 10 s | 3,388 ms (3.0×) | **300 ms (33.4×)** | 11.3× |
| 10 s + 5 s | 3,132 ms (1.6×) | **192 ms (26.1×)** | 16.3× |
| 10 s + 10 s | 4,157 ms (2.4×) | **317 ms (31.5×)** | 13.1× |
