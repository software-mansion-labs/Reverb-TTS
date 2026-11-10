# How placeholder_tts generates speech

This is the whole inference pipeline, step by step, as `placeholder_tts` runs it. The code is short and follows the
same order: [text.py](../placeholder_tts/text.py), [audio.py](../placeholder_tts/audio.py),
[model.py](../placeholder_tts/model.py), [sampler.py](../placeholder_tts/sampler.py) and
[tts.py](../placeholder_tts/tts.py), which ties them together.

## The model

The model is a masked-diffusion text-to-speech model: OmniVoice pruned to 297M parameters and retrained. It speaks in
tokens of the [Higgs audio codec](https://huggingface.co/eustlb/higgs-audio-v2-tokenizer): 25 frames per second, each
frame 8 codebooks from coarse (0) to fine (7), each codebook 1024 codes plus a mask token. One bidirectional
transformer (a Qwen3 backbone) reads one sequence:

```
[ prompt + reference transcript + text | reference frames ][ target frames ]
                     prefix                                     target
```

The target starts fully masked and is filled in over several passes. A frame's input is the sum of its 8 codebook
embeddings; a masked slot has an embedding of its own.

## 1. Text

The text to speak and the reference transcript are both prepared with [Verbalis](https://github.com/software-mansion-labs/Verbalis):
numbers, dates, times, money, units, symbols, abbreviations and web addresses become words, and quotes, apostrophes,
dashes and ellipses take the forms the training transcripts use (`"`, `'`, ` - `, `...`). Brackets, which never
occurred in training text, are dropped. `prepare=False` skips this for text that is already prepared.

The model was trained on seven languages: English, German, French, Spanish, Italian, Polish and Portuguese (Brazilian).

## 2. Reference

The reference recording is mixed down to mono, resampled to 24 kHz with the windowed-sinc filter the training
references were resampled with, and zero-padded to whole codec frames (960 samples). Without the padding, the codec's
encoder pads its input by half a frame for about half of all lengths, and the codes come out misaligned. The Higgs
encoder then turns it into codes `[8, R]`.

A good reference is 3-10 s of clean speech from one speaker, with a transcript that matches the audio exactly: the
target's length is estimated from it. References over 15 s are known to make worse takes.

## 3. Prompt

```
<|denoise|><|lang_start|>{language}<|lang_end|><|instruct_start|>None<|instruct_end|><|text_start|>{reference transcript} {text}<|text_end|>
```

`language` is the text's language code. The reference may be in another language.

## 4. Target length

The length is estimated, not predicted: the text's spoken weight, at the rate the reference speaks its transcript.
Letters weigh 1, digits 3.5, punctuation and symbols 0.5, spaces 0.2 and combining marks 0. A text shorter than
50 frames gets relatively more time (the estimate is stretched by a cube root). `speed` divides the estimate, and
`duration` replaces it.

A text estimated over 30 s is cut at sentence ends into pieces of about 15 s. The pieces are generated with
consecutive seeds, trimmed, and joined.

## 5. Prefix

Two properties of the model keep a generation cheap:

- **Prefix blocking:** the prefix never attends to the target. It runs once, and its keys and values are cached; each
  pass then runs over the target frames only.
- **Distilled guidance:** classifier-free guidance was distilled into the weights, so each pass is a single forward,
  with no unconditional branch.

Several takes are generated together as a batch, padded on the right; padded keys are masked out.

## 6. Main passes: codebooks 0-1

`steps` passes decide codebooks 0 and 1, which carry the content and the timing.

- **Schedule.** With u = k / steps, the fraction of slots decided after pass k is t = s·u / (1 - (1 - s)·u), where
  s is `t_shift` (0.1). Pass k commits ⌈2·F·(t_{k+1} - t_k)⌉ of the 2·F slots, so early passes commit few and the last
  pass fills whatever is left.
- **Which slots.** Each masked slot scores (log p - `codebook_penalty`·c) / `position_temperature` + Gumbel noise,
  where log p is the log-probability of its most likely code and c its codebook (0 or 1). The highest scores are
  committed to their most likely code. The noise comes from a CPU generator seeded with the take's seed, so a seed
  gives the same noise on any device.
- **Remasking.** On every pass but the last, the model re-reads the codes committed earlier, and the ones it now finds
  least likely are masked again: `remask_ratio` as many as the pass committed.

## 7. Polish passes: codebooks 2-7

Two passes decide codebooks 2-7, which carry the voice's timbre and detail, with codebooks 0-1 complete. In both, every
attention score from the target to the reference frames gets `polish_bias` added, in every layer and head, so the
fine codebooks follow the reference more closely.

1. The first pass predicts codebooks 2-7 and commits codebooks 2-3.
2. The second pass, with codebooks 0-3 known, predicts codebooks 4-7 and commits the most likely code of
   lp₂ + `polish_guidance`·(lp₂ - lp₁), where lp₁ and lp₂ are the two passes' log-probabilities for codebooks 4-7.

## 8. Audio

The Higgs decoder turns the codes into 24 kHz audio. The last 200 ms are cut, since the final frames can hold a short
burst of noise. Then leading and trailing silence (10 ms windows under -50 dBFS) is trimmed, keeping 100 ms on each
side; a sound of at most 120 ms alone at either edge is a codec burst rather than speech, and goes too.

## Cost

One generation is 1 prefix pass and `steps` + 2 target passes: 10 target passes with the defaults, against
OmniVoice's default of 32 guided passes (64 forwards).

## Sampler settings

| Setting | Default | What it does |
|---|---|---|
| `steps` | 8 | main passes |
| `remasking` | True | whether main passes withdraw earlier commits |
| `remask_ratio` | 0.25 | commits withdrawn per new commit |
| `t_shift` | 0.1 | warps the schedule towards the last passes |
| `codebook_penalty` | 5.0 | makes codebook 0 tend to go before codebook 1 |
| `position_temperature` | 5.0 | scales confidence against the Gumbel noise; higher is more random |
| `polish_bias` | 1.0 | attention bias towards the reference in the polish passes |
| `polish_guidance` | 1.0 | weight of the second polish pass's change over the first |

Override any of them per call (`tts.generate(..., steps=12, remasking=False)`), or for a loaded model
(`TTS.from_pretrained(sampler=SamplerConfig(steps=12))`).

On the MiniMax multilingual test set (the seven languages, 700 texts, one take each, on CUDA in bf16), the defaults
scored 1.35% WER and 0.816 speaker similarity (SIM-o), against 1.50% and 0.817 with `remasking=False`.

## Weights

| Variant | Decoder projections | Activations | Download |
|---|---|---|---|
| `fp32` | fp32 | full precision | 1.19 GB |
| `int8` (default) | int8, one scale per output channel | full precision | 307 MB |
| `int8a8` | int8, one scale per output channel | int8, one scale per token | 307 MB |
| `int4` | int4, one scale per group of 32 inputs | full precision | 191 MB |

In the quantized variants, the text embedding table and the audio embeddings are int8 with one scale per row, and
the remaining weights are bf16. On load, every weight is expanded to the compute dtype: PyTorch has no int8 or int4
matrix product that runs the same on the CPU, MPS and CUDA. A quantized variant therefore downloads smaller but runs
at the same speed, in the same memory, as fp32 weights cast to that dtype. `int8a8` additionally rounds each token's
projection inputs to int8, as it was made to run.

The compute dtype is bfloat16 on CUDA (float16 without bf16 support), float16 on MPS and float32 on the CPU. Each
codebook's head rows are centred on load, which leaves the softmax unchanged and keeps the logits small enough for
16-bit floats.

## Reproducibility

The same inputs, seed, device and dtype give the same take. Takes generated in one batch share padding, which can
change 16-bit rounding: a text generated in a list may come out slightly different from the same text alone.
