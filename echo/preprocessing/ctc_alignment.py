"""Transcript/audio agreement from one MMS_FA forced alignment, with a wildcard for spans the model can't judge.

Ported from Resonance's tools/metrics/ctc_alignment_metric.py, which the YODAS2-sidon filter uses; scores and
thresholds are the same, so clips are judged alike in both repos. Two defects are reported:
    missing words    the audio holds less than the transcript: tokens crammed into too few frames, so maxrate
                     spikes while loss rises;
    redundant speech the audio holds more: unaligned frames whose posterior is not blank (gap_speech between
                     words, trail_speech after the last one).
The addition is the wildcard: a "*" word in the transcript aligns to MMS_FA's star token, which absorbs any audio.
MMS_FA emits blanks where numbers are spoken, so a number left as words fails the loss rule and one left out (or as
digits, which normalize() drops) fails the gap rule; as "*" it is judged by neither. The star column is appended
after the log-softmax, so the scores of transcripts without "*" equal the original metric's.
"""
import re

import numpy as np
import torch
import torchaudio

BUNDLE = torchaudio.pipelines.MMS_FA
DICTIONARY = BUNDLE.get_dict(star="*")
BLANK = 0
WILDCARD = "*"
# The YODAS2-sidon filter's bounds (Resonance configurations/quality_filtering_yodas2_sidon.yaml, ctc_max).
BOUNDS = {"loss": 0.7, "max_loss": 1.0, "maxrate": 30.0, "gap_speech": 0.06, "trail_speech": 0.05}
WINDOWS = (4, 6, 8)


class CtcAlignment:
    def __init__(self, device: str | None = None) -> None:
        self.device = device or ("cuda" if torch.cuda.is_available()
                                 else "mps" if torch.backends.mps.is_available() else "cpu")
        self._model = None
        self._uroman = None

    def evaluate(self, audio: np.ndarray, sample_rate: int, transcript: str) -> dict[str, float]:
        """loss, maxrate, gap_speech, trail_speech; higher is worse on every one."""
        tokens = tokenize(self._romanize(transcript))
        emission, duration = self._emission(audio, sample_rate)
        target = torch.tensor([[t for token in tokens for t in token]])
        loss = ctc_loss(emission, target)
        try:
            aligned, scores = torchaudio.functional.forced_align(emission, target, blank=BLANK)
        except RuntimeError:
            # More tokens than frames: missing words in the extreme (ctc_loss would score it 0.0).
            return {"loss": float("inf"), "maxrate": float(BUNDLE.sample_rate), "gap_speech": 0.0, "trail_speech": 0.0}
        spans = torchaudio.functional.merge_tokens(aligned[0], scores[0].exp())
        step = duration / emission.shape[1]
        words, cursor = [], 0
        for token in tokens:
            part = spans[cursor:cursor + len(token)]
            cursor += len(token)
            words.append({"start": part[0].start * step, "end": part[-1].end * step, "nch": len(token)})
        speech = (1 - emission[0, :, BLANK].exp()).numpy()
        starts, ends = [w["start"] for w in words], [w["end"] for w in words]
        return {
            "loss": loss,
            "maxrate": max_char_rate(words),
            "gap_speech": max((speech_mass(speech, step, end, start) for start, end in zip(starts[1:], ends[:-1])),
                              default=0.0),
            "trail_speech": speech_mass(speech, step, ends[-1], duration),
        }

    def _romanize(self, transcript: str) -> str:
        """uroman over non-ASCII text, MMS's own recipe ("Kålsva" would otherwise align as "k lsva")."""
        if transcript.isascii():
            return transcript
        if self._uroman is None:
            from uroman import Uroman
            self._uroman = Uroman()
        return self._uroman.romanize_string(transcript)

    def _emission(self, audio: np.ndarray, sample_rate: int) -> tuple[torch.Tensor, float]:
        if self._model is None:
            self._model = BUNDLE.get_model(with_star=True).to(self.device).eval()
        y = np.asarray(audio, dtype=np.float32)
        if y.ndim > 1:
            y = y.mean(axis=1)
        waveform = torch.from_numpy(y)[None].to(self.device)
        if sample_rate != BUNDLE.sample_rate:
            waveform = torchaudio.functional.resample(waveform, sample_rate, BUNDLE.sample_rate)
        with torch.inference_mode():
            emission, _ = self._model(waveform)
        return emission.cpu(), waveform.shape[1] / BUNDLE.sample_rate


def defects(scores: dict[str, float], bounds: dict[str, float] = BOUNDS) -> list[str]:
    """The defects the scores show: "missing words" (loss beyond max_loss, or beyond loss with a maxrate spike) and
    "redundant speech" (gap_speech or trail_speech beyond its bound)."""
    found = []
    if scores["loss"] > bounds["max_loss"] or scores["loss"] > bounds["loss"] and scores["maxrate"] > bounds["maxrate"]:
        found.append("missing words")
    if scores["gap_speech"] > bounds["gap_speech"] or scores["trail_speech"] > bounds["trail_speech"]:
        found.append("redundant speech")
    return found


def normalize(text: str) -> list[str]:
    """MMS's alphabet, lowercase Latin plus apostrophe, as words; a "*" stays a word of its own."""
    text = text.lower().replace("’", "'").replace(WILDCARD, f" {WILDCARD} ")
    return [w for w in re.sub(r"[^a-z'*]", " ", text).split() if w == WILDCARD or any(c.isalpha() for c in w)]


def tokenize(transcript: str) -> list[list[int]]:
    """MMS token ids per word, dropping characters the dictionary can't represent."""
    tokens = [[DICTIONARY[c] for c in word if DICTIONARY.get(c, BLANK) != BLANK] for word in normalize(transcript)]
    tokens = [t for t in tokens if t]
    if not tokens:
        raise ValueError(f"No alignable characters in transcript: {transcript!r}")
    return tokens


def ctc_loss(emission: torch.Tensor, target: torch.Tensor) -> float:
    """Per-token CTC loss of one target against one emission."""
    return torch.nn.functional.ctc_loss(
        emission.transpose(0, 1), target, torch.tensor([emission.shape[1]]), torch.tensor([target.shape[1]]),
        blank=BLANK, reduction="sum", zero_infinity=True,
    ).item() / target.shape[1]


def max_char_rate(words: list[dict]) -> float:
    """Highest character rate over a short window of words, in characters per second."""
    return max((sum(w["nch"] for w in words[i:i + n]) / max(words[i + n - 1]["end"] - words[i]["start"], 1e-3)
                for n in WINDOWS for i in range(len(words) - n + 1)), default=0.0)


def speech_mass(speech: np.ndarray, step: float, start: float, end: float) -> float:
    """Seconds of non-blank posterior inside an unaligned stretch."""
    lo, hi = int(round(start / step)), int(round(end / step))
    lo, hi = max(0, min(lo, len(speech))), max(0, min(hi, len(speech)))
    return float(speech[lo:hi].sum() * step) if hi > lo else 0.0
