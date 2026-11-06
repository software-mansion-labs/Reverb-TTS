"""Checks a verbalized transcript against its audio: every span with alternative readings (verbalize_spans) is
scored with a CTC model that hears numbers as words, and replaced by an alternative that fits the audio better.

A span's score for an alternative is (CTC loss with the default - loss with the alternative) per character of the
default, the rest of the transcript held at its defaults; above THRESHOLD the alternative replaces the default.
"""
import re
from dataclasses import dataclass

import numpy as np
import torch
import torch.nn.functional as F
import torchaudio
from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor

from echo.preprocessing.verbalizer.core import Span, verbalize_spans

# MMS_FA does not fit: it emits blanks where numbers are spoken.
MODELS = {
    "en": "facebook/wav2vec2-base-960h",
    "pl": "jonatasgrosman/wav2vec2-large-xlsr-53-polish",
    "es": "jonatasgrosman/wav2vec2-large-xlsr-53-spanish",
    "de": "jonatasgrosman/wav2vec2-large-xlsr-53-german",
    "fr": "jonatasgrosman/wav2vec2-large-xlsr-53-french",
    "pt": "jonatasgrosman/wav2vec2-large-xlsr-53-portuguese",
    "it": "jonatasgrosman/wav2vec2-large-xlsr-53-italian",
}
THRESHOLD = 0.1
SAMPLE_RATE = 16000


@dataclass
class SpanCheck:
    span: Span
    scores: list[tuple[float, str]]  # (score, alternative), best first
    reading: str  # the default, or the alternative that replaced it


class CtcChecker:
    def __init__(self, lang: str, device: str = "cpu") -> None:
        self.lang, self.device = lang, device
        self.processor = Wav2Vec2Processor.from_pretrained(MODELS[lang])
        self.model = Wav2Vec2ForCTC.from_pretrained(MODELS[lang]).to(device).eval()
        self.vocab = self.processor.tokenizer.get_vocab()
        self.upper = "A" in self.vocab and "a" not in self.vocab

    def check(self, audio: np.ndarray, sample_rate: int, text: str) -> tuple[str, list[SpanCheck]]:
        """The verbalized transcript with replacements applied, and the check of every span with alternatives."""
        return self.check_spans(audio, sample_rate, verbalize_spans(text, self.lang))

    def check_spans(self, audio: np.ndarray, sample_rate: int, parts: list[str | Span]) -> tuple[str, list[SpanCheck]]:
        """check() on a transcript already split into text and spans: each span's default is the reading under
        test (the verbalizer's, or an old transcript's - see realign), its alternatives what it is tested against."""
        log_probs = self._log_probs(audio, sample_rate)
        defaults = [p.default if isinstance(p, Span) else p for p in parts]
        checks = []
        for j, part in enumerate(parts):
            if not isinstance(part, Span) or not part.alternatives:
                continue
            with_reading = lambda reading: "".join(defaults[:j] + [reading] + defaults[j + 1:])
            base = self._loss(log_probs, with_reading(part.default))
            chars = max(len(self._ids(part.default)), 1)
            scores = sorted(((base - self._loss(log_probs, with_reading(a))) / chars, a) for a in part.alternatives)[::-1]
            checks.append(SpanCheck(part, scores, scores[0][1] if scores[0][0] > THRESHOLD else part.default))
        readings = iter(c.reading for c in checks)
        out = [next(readings) if isinstance(p, Span) and p.alternatives else d for p, d in zip(parts, defaults)]
        return "".join(out), checks

    def aligned_span(self, audio: np.ndarray, sample_rate: int, text: str) -> tuple[float, float] | None:
        """Seconds from the start of the audio to the first and last aligned letter of the transcript; None when the
        transcript can't be aligned (nothing in the model's alphabet, or more letters than frames)."""
        log_probs = self._log_probs(audio, sample_rate)
        ids = self._ids(text)
        if not ids or len(ids) > log_probs.shape[0]:
            return None
        blank = self.processor.tokenizer.pad_token_id
        aligned, _ = torchaudio.functional.forced_align(log_probs[None], torch.tensor([ids]), blank=blank)
        frames = [t for t, token in enumerate(aligned[0].tolist()) if token not in (blank, self.vocab.get("|"))]
        if not frames:
            return None
        step = len(audio) / sample_rate / log_probs.shape[0]
        return frames[0] * step, (frames[-1] + 1) * step

    def _log_probs(self, audio: np.ndarray, sample_rate: int) -> torch.Tensor:
        x = torch.as_tensor(audio, dtype=torch.float32)
        if sample_rate != SAMPLE_RATE:
            x = torchaudio.functional.resample(x, sample_rate, SAMPLE_RATE)
        inputs = self.processor(x.numpy(), sampling_rate=SAMPLE_RATE, return_tensors="pt").input_values.to(self.device)
        with torch.no_grad():
            return self.model(inputs).logits.log_softmax(-1)[0].cpu()

    def _ids(self, text: str) -> list[int]:
        """The model's characters of a text, words joined by its "|" delimiter; hyphens split words."""
        text = re.sub(r"[-–—]", " ", text.upper() if self.upper else text.lower())
        words = ["".join(c for c in w if c in self.vocab and c != "|") for w in text.split()]
        return [self.vocab[c] for c in "|".join(w for w in words if w)]

    def _loss(self, log_probs: torch.Tensor, text: str) -> float:
        ids = self._ids(text)
        return F.ctc_loss(log_probs.unsqueeze(1), torch.tensor([ids]), torch.tensor([log_probs.shape[0]]),
                          torch.tensor([len(ids)]), blank=self.processor.tokenizer.pad_token_id, reduction="sum",
                          zero_infinity=True).item()
