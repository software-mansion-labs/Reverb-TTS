"""
The public API: load a model, make a voice from a reference recording, and speak text in that voice.

    tts = TTS.from_pretrained()
    voice = tts.voice("reference.wav", "Exact transcript of the reference.", language="en")
    audio = tts.generate("Text to speak.", voice, language="en", seed=0)  # float32 samples at tts.sample_rate
"""

import json
import logging
import random
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, List, Literal, Optional, Sequence, Tuple, TypeVar, Union

import numpy as np
import torch
from huggingface_hub import hf_hub_download, list_repo_files, snapshot_download
from transformers import HiggsAudioV2TokenizerModel, Qwen2Tokenizer

from reverb_tts.audio import AudioInput, load_audio, raise_quiet, speech_length, trim_silence
from reverb_tts.model import CODEBOOKS, MASK, Network
from reverb_tts.sampler import Batch, Prompt, SamplerConfig, sample
from reverb_tts.text import check_language, estimate_frames, prepare_text, split_text

logger = logging.getLogger(__name__)

REPOSITORY, REVISION = "software-mansion/Reverb-TTS", "v1.0.0"  # the released weights
PRESETS = REPOSITORY  # preset voices, as voices/<name>.json
CODEC, CODEC_REVISION = "eustlb/higgs-audio-v2-tokenizer", "0e3a9177c2991016b8e5e17d23aa474ee665bef2"
VARIANTS = ("fp32", "int8", "int8a8", "int4")
FRAME_RATE = 25              # codec frames per second
LONGEST, PIECE = 30.0, 15.0  # seconds: a text estimated longer than LONGEST is spoken in pieces of about PIECE
END_CUT = 0.2                # seconds cut from every take's end, where the codec can leave a short burst of noise
LONG_REFERENCE = 15.0        # seconds: references longer than this are known to hurt
PACES = ("speech", "recording")  # what the length estimate takes as the reference's length: see generate()

T = TypeVar("T")


@dataclass
class Voice:
    """
    A reference voice: the recording's codec codes [8, frames] and its transcript, prepared for the model.
    `speech_frames` is how many of those frames the speaker spends speaking (see `audio.speech_length`).
    """

    codes: torch.Tensor
    text: str
    language: str
    speech_frames: Optional[int] = None

    @property
    def duration(self) -> float:
        return self.codes.size(1) / FRAME_RATE

    @property
    def pace_frames(self) -> int:
        """
        The reference length the target's length is estimated from: the speech frames, at most every frame.
        """

        frames = self.codes.size(1)
        return frames if self.speech_frames is None else max(1, min(self.speech_frames, frames))

    def save(self, path: Union[str, Path]) -> None:
        """
        Writes the voice to an .npz file, so the recording needn't be encoded again.
        """

        extra = {} if self.speech_frames is None else {"speech_frames": self.speech_frames}
        np.savez(path, codes=self.codes.numpy().astype(np.int16), text=self.text, language=self.language, **extra)

    @classmethod
    def load(cls, path: Union[str, Path]) -> "Voice":
        """
        Reads a voice from an .npz file written by `save`, or from a preset .json file: {codes, transcript, language},
        the codes as 8 x frames integers, row-major, and the transcript already prepared, with an optional
        speech_frames.
        """

        if Path(path).suffix != ".json":
            data = np.load(path)
            speech = int(data["speech_frames"]) if "speech_frames" in data.files else None
            return cls(torch.from_numpy(data["codes"].astype(np.int64)), str(data["text"]), str(data["language"]),
                       speech)
        data = json.loads(Path(path).read_text())
        codes = torch.tensor(data["codes"], dtype=torch.long)
        if codes.numel() == 0 or codes.numel() % CODEBOOKS or not 0 <= int(codes.min()) <= int(codes.max()) < MASK:
            raise ValueError(f"{path}: expected codes as {CODEBOOKS} x frames integers below {MASK}")
        speech = data.get("speech_frames")
        if speech is not None and (not isinstance(speech, int) or speech < 1):
            raise ValueError(f"{path}: expected speech_frames as a positive integer")
        return cls(codes.view(CODEBOOKS, -1), data["transcript"], check_language(data["language"]), speech)


class TTS:
    """
    A loaded model with its tokenizer and audio codec. `sampler` holds the default sampler settings, `batch_frames`
    caps the target frames generated together, padding included, and `presets` is a local folder of preset voices
    (by default they come from Hugging Face).
    """

    def __init__(self, network: Network, tokenizer: Any, codec: Any, sampler: Optional[SamplerConfig] = None,
                 batch_frames: int = 8192, presets: Optional[Union[str, Path]] = None, token: Optional[str] = None) -> None:
        self.network, self.tokenizer, self.codec = network, tokenizer, codec
        self.sampler, self.batch_frames = sampler or SamplerConfig(), batch_frames
        self.presets, self.token = (Path(presets) if presets is not None else None), token

    @property
    def device(self) -> torch.device:
        return self.network.audio_heads.weight.device

    @property
    def sample_rate(self) -> int:
        return self.codec.config.sample_rate

    @classmethod
    def from_pretrained(
        cls,
        name_or_path: Union[str, Path] = REPOSITORY,
        *,
        variant: str = "int8",
        revision: Optional[str] = REVISION,
        token: Optional[str] = None,
        device: Optional[Union[str, torch.device]] = None,
        dtype: Optional[torch.dtype] = None,
        sampler: Optional[SamplerConfig] = None,
        codec: str = CODEC,
        codec_revision: Optional[str] = None,
    ) -> "TTS":
        """
        Loads a model from a Hugging Face repository, or from a local folder: a release (with
        model/safetensors/<variant>/) or one checkpoint (config.json and model.safetensors, with the tokenizer beside or
        above it).

        - variant: fp32, int8 (the default), int8a8 or int4. Quantized weights are expanded to `dtype` on load, so they
          download smaller but run at the same speed.
        - device: cuda, mps or cpu; by default the first one available.
        - dtype: by default bfloat16 on CUDA (float16 without bf16 support), float16 on MPS and float32 on the CPU.
        """

        weights, tokenizer = _resolve(name_or_path, variant, revision, token)
        device = _pick_device(device)
        network = Network.load(weights).to(device, dtype or _default_dtype(device))
        revision = codec_revision or (CODEC_REVISION if codec == CODEC else None)
        codec_model = HiggsAudioV2TokenizerModel.from_pretrained(codec, revision=revision).to(device).eval()
        local = Path(name_or_path).expanduser() / "voices"
        return cls(network, Qwen2Tokenizer.from_pretrained(tokenizer), codec_model, sampler,
                   presets=local if local.is_dir() else None, token=token)

    def preset_voices(self) -> List[str]:
        """
        The names of the preset voices.
        """

        if self.presets is not None:
            return sorted(p.stem for p in self.presets.glob("*.json"))
        files = list_repo_files(PRESETS, token=self.token)
        return sorted(Path(f).stem for f in files if f.startswith("voices/") and f.endswith(".json"))

    def preset_voice(self, name: str) -> Voice:
        """
        A preset voice, by name (see `preset_voices`): ready to use, with no recording to encode.
        """

        names = self.preset_voices()
        if name not in names:
            raise ValueError(f"unknown preset voice {name!r}, expected one of: {', '.join(names)}")
        if self.presets is not None:
            return Voice.load(self.presets / f"{name}.json")
        return Voice.load(hf_hub_download(PRESETS, f"voices/{name}.json", token=self.token))

    @torch.inference_mode()
    def voice(self, audio: AudioInput, text: str, language: str, *, sample_rate: Optional[int] = None,
              prepare: bool = True) -> Voice:
        """
        A voice from a reference recording - 3 to 10 s of clean speech from one speaker - and its exact transcript.

        - audio: a file path, or samples ([samples] or [channels, samples]) with their `sample_rate`.
        - language: the recording's language.
        - prepare: verbalize the transcript (turn off for a transcript that already is).
        """

        check_language(language)
        wav = raise_quiet(load_audio(audio, sample_rate, self.sample_rate))
        # Whole codec frames, as the training references were: otherwise the codes can come out misaligned.
        wav = torch.nn.functional.pad(wav, (0, -wav.numel() % self.codec.config.hop_length))
        codes = self.codec.encode(wav[None, None].to(self.device)).audio_codes[0].long().cpu()
        if codes.size(1) / FRAME_RATE > LONG_REFERENCE:
            logger.warning("the reference is %.1f s long; references over %.0f s tend to make worse takes",
                           codes.size(1) / FRAME_RATE, LONG_REFERENCE)
        # The recording stays whole; only the speaking time is measured, for the length estimate.
        speech = round(speech_length(wav.numpy(), self.sample_rate) * FRAME_RATE)
        return Voice(codes, prepare_text(text, language) if prepare else text, language, speech)

    @torch.inference_mode()
    def generate(
        self,
        text: Union[str, Sequence[str]],
        voice: Union[Voice, Sequence[Voice]],
        language: Union[str, Sequence[str]],
        *,
        seed: Optional[int] = None,
        speed: float = 1.0,
        duration: Optional[float] = None,
        pace: Literal["speech", "recording"] = "speech",
        prepare: bool = True,
        trim: bool = True,
        sampler: Optional[SamplerConfig] = None,
        **overrides: Any,
    ) -> Union[np.ndarray, List[np.ndarray]]:
        """
        Speaks `text` in `voice`. Returns float32 samples at `sample_rate`, or a list of them when `text` is a list;
        a list of texts is generated in batches, with one voice and language for all or one per text.

        - language: the text's language, one of LANGUAGES.
        - seed: the take's seed; consecutive seeds go to the pieces of a list or a long text. None draws one.
        - speed: a factor on the estimated length. duration: the length in seconds, in place of the estimate.
        - pace: the reference length the estimate follows. "speech" counts only the time the reference spends speaking
          (`Voice.speech_frames`: no silent edges, pauses capped at 0.5 s); "recording" counts the whole recording.
        - prepare: verbalize the text first (turn off for text that already is).
        - trim: cut the take's silent edges.
        - sampler, overrides: sampler settings for this call, e.g. steps=12 or remasking=False.

        A text estimated over 30 s is cut at sentence ends and spoken in pieces of about 15 s, unless `duration` is set.
        """

        single = isinstance(text, str)
        texts = [text] if single else list(text)
        voices, languages = _broadcast(voice, len(texts), "voice"), _broadcast(language, len(texts), "language")
        config = replace(sampler or self.sampler, **overrides)
        base = random.randrange(2**31) if seed is None else seed
        if pace not in PACES:
            raise ValueError(f"unknown pace {pace!r}, expected one of: {', '.join(PACES)}")

        prompts: List[Prompt] = []
        owners: List[int] = []
        for i, (words, speaker, lang) in enumerate(zip(texts, voices, languages)):
            check_language(lang)
            words = prepare_text(words, lang) if prepare else words
            reference_frames = speaker.pace_frames if pace == "speech" else speaker.codes.size(1)
            if duration is not None:
                pieces = [(words, max(1, round(duration * FRAME_RATE)))]
            else:
                parts = split_text(words, speaker.text, reference_frames, int(LONGEST * FRAME_RATE), int(PIECE * FRAME_RATE))
                pieces = [(part, max(1, round(estimate_frames(part, speaker.text, reference_frames) / speed)))
                          for part in parts]
            for part, frames in pieces:
                ids = self.tokenizer(_prompt(part, speaker.text, lang), return_tensors="pt").input_ids[0]
                prompts.append(Prompt(ids, speaker.codes, frames, base + len(prompts)))
                owners.append(i)

        takes: List[List[np.ndarray]] = [[] for _ in texts]
        for owner, codes in zip(owners, self._sample(prompts, config)):
            audio = self.codec.decode(codes[None].to(self.device)).audio_values[0, 0].float().cpu().numpy()
            audio = audio[: max(len(audio) - int(END_CUT * self.sample_rate), 0)]
            takes[owner].append(trim_silence(audio, self.sample_rate) if trim else audio)
        out = [np.concatenate(parts).astype(np.float32) for parts in takes]
        return out[0] if single else out

    def _sample(self, prompts: List[Prompt], config: SamplerConfig) -> List[torch.Tensor]:
        """
        Every prompt's codes, generated in batches of similar target lengths.
        """

        order = sorted(range(len(prompts)), key=lambda j: prompts[j].frames)
        codes: List[torch.Tensor] = [torch.empty(0)] * len(prompts)
        start = 0
        while start < len(order):
            # Sorted by length, so a batch's longest target is its last: count it once per take, as padded.
            end = start + 1
            while end < len(order) and (end + 1 - start) * prompts[order[end]].frames <= self.batch_frames:
                end += 1
            group = [prompts[j] for j in order[start:end]]
            batch = Batch(self.network, group, config.polish_bias)
            for j, take in zip(order[start:end], sample(batch, [p.seed for p in group], config)):
                codes[j] = take
            start = end
        return codes


def _prompt(text: str, reference_text: str, language: str) -> str:
    return (f"<|denoise|><|lang_start|>{language}<|lang_end|><|instruct_start|>None<|instruct_end|>"
            f"<|text_start|>{reference_text} {text}<|text_end|>")


def _broadcast(value: Union[T, Sequence[T]], n: int, name: str) -> List[T]:
    if isinstance(value, (str, Voice)):
        return [value] * n
    values = list(value)
    if len(values) != n:
        raise ValueError(f"got {len(values)} {name}s for {n} texts")
    return values


def _resolve(name_or_path: Union[str, Path], variant: str, revision: Optional[str],
             token: Optional[str]) -> Tuple[Path, Path]:
    """
    The variant's weights folder and the tokenizer's folder.
    """

    if variant not in VARIANTS:
        raise ValueError(f"unknown variant {variant!r}, expected one of: {', '.join(VARIANTS)}")
    root = Path(name_or_path).expanduser()
    if not root.exists():
        patterns = [f"model/safetensors/{variant}/*", "model/tokenizer*", "model/chat_template.jinja"]
        root = Path(snapshot_download(str(name_or_path), revision=revision, token=token, allow_patterns=patterns))
    folders = (root / "model" / "safetensors" / variant, root / "model" / variant, root / variant, root)
    weights = next((f for f in folders if (f / "config.json").is_file()), None)
    if weights is None:
        raise FileNotFoundError(f"no {variant} weights (config.json and model.safetensors) under {root}")
    folders = (weights, weights.parent, weights.parent.parent, root / "model")
    tokenizer = next((f for f in folders if (f / "tokenizer.json").is_file()), None)
    if tokenizer is None:
        raise FileNotFoundError(f"no tokenizer.json beside or above {weights}")
    return weights, tokenizer


def _pick_device(device: Optional[Union[str, torch.device]]) -> torch.device:
    if device is not None:
        return torch.device(device)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def _default_dtype(device: torch.device) -> torch.dtype:
    if device.type == "cuda":
        return torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    return torch.float16 if device.type == "mps" else torch.float32
