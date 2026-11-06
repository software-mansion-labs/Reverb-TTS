"""Re-encode every clip's Higgs Audio v2 codec from its audio and rebuild the portion's codecs.zip.

    .venv/bin/python scripts/normalization/s8_codecs.py data/train/polish/YODAS2-sidon [--limit 200] [--dry]

Run it after the audio stages (s6_edges.py, s7_loudness.py), which leave the codecs out of date. Encoding matches
Resonance's finalize step, which made the existing codecs: eustlb/higgs-audio-v2-tokenizer at 24 kHz (another rate
is resampled with librosa's kaiser_best kernel), tokens stored as int16 [8, T] in codecs/<name>.npy.

Every clip with audio in a portion (audio/ or audio.zip next to its CSVs) is encoded; --limit encodes a random sample
of that many (seed 0). The new tokens are compared with the portion's current codecs.zip: same, a different number
of frames, or different tokens - a dry run (--dry) only reports this, which also checks rebuilt audio against codecs
made from the original. Otherwise, after the user answers Y, each codecs.zip is rebuilt with the new codecs (written
to codecs.zip.tmp, then moved over it); codecs of clips not encoded this run are kept. On a Mac the codec runs on the
CPU by default; --device mps works too (torch 2.8; Resonance's note that MPS can't allocate its 65536+ output
channels no longer holds) but is no faster one clip at a time. Use a CUDA box for whole datasets.

The comparison is exact only with the precision that made the codecs: the residual codebooks flip on tiny float
differences. torch runs CUDA convolutions in TF32 by default, which changes 3-10% of tokens (about 1% in codebook 0)
against true fp32, at the same length; with torch.backends.cudnn.allow_tf32 = False, CUDA and the CPU agree, and
reproduce the test set's codes. CPU and MPS agree to within 0.1%. Resonance's own encoder does the same.
"""
import argparse
import collections
import io
import os
import random
import sys
import time
import zipfile
from pathlib import Path

import numpy as np
import torch
import torchaudio
from transformers import AutoFeatureExtractor, HiggsAudioV2TokenizerModel

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from scripts.__style__ import Colors, print_info, print_section, print_test_title  # noqa: E402
from scripts.normalization.s1_characters import read, share  # noqa: E402
from scripts.normalization.s3_ctc_rescoring import THREADS, Audio, decoded  # noqa: E402

SKIP = ("_normalized", "_without_leftovers", "_ctc", "_readings", "_verbalized")
MODEL = "eustlb/higgs-audio-v2-tokenizer"
CODEBOOKS, FRAME_RATE = 8, 25
# librosa's kaiser_best, as Resonance's tools/codec/higgs.py resamples before encoding
RESAMPLE = {"lowpass_filter_width": 64, "rolloff": 0.9475937167399596, "resampling_method": "sinc_interp_kaiser",
            "beta": 14.769656459379392}
EXAMPLES = 6


class Codec:
    def __init__(self, device: str) -> None:
        self.feature_extractor = AutoFeatureExtractor.from_pretrained(MODEL)
        self.model = HiggsAudioV2TokenizerModel.from_pretrained(MODEL, device_map=device)
        self.rate = self.feature_extractor.sampling_rate

    def encode(self, audio: np.ndarray, sample_rate: int) -> np.ndarray:
        """int16 tokens [CODEBOOKS, T]."""
        y = torch.as_tensor(audio, dtype=torch.float32)
        if sample_rate != self.rate:
            y = torchaudio.functional.resample(y, sample_rate, self.rate, **RESAMPLE)
        with torch.inference_mode():
            inputs = self.feature_extractor(raw_audio=y.numpy(), sampling_rate=self.rate, return_tensors="pt")
            tokens = self.model.encode(inputs["input_values"].to(self.model.device)).audio_codes.squeeze(0)
        assert tokens.shape[0] == CODEBOOKS
        return tokens.cpu().numpy().astype(np.int16)


def existing(portion: Path) -> dict[str, np.ndarray]:
    """The portion's current codecs, by clip name."""
    if not (portion / "codecs.zip").exists():
        return {}
    with zipfile.ZipFile(portion / "codecs.zip") as z:
        return {n[len("codecs/"):-len(".npy")]: np.load(io.BytesIO(z.read(n))) for n in z.namelist() if n.endswith(".npy")}


def rebuild(portion: Path, new: dict[str, np.ndarray]) -> None:
    """codecs.zip with the new codecs swapped in; codecs of clips not in `new` are kept."""
    keep = {}
    if (portion / "codecs.zip").exists():
        with zipfile.ZipFile(portion / "codecs.zip") as z:
            keep = {n: z.read(n) for n in z.namelist() if n.endswith(".npy") and n[7:-4] not in new}
    with zipfile.ZipFile(portion / "codecs.zip.tmp", "w", zipfile.ZIP_DEFLATED, compresslevel=1) as out:
        out.writestr("codecs/", b"")
        for name, data in sorted(keep.items()):
            out.writestr(name, data)
        for name, tokens in sorted(new.items()):
            buffer = io.BytesIO()
            np.save(buffer, tokens)
            out.writestr(f"codecs/{name}.npy", buffer.getvalue())
    os.replace(portion / "codecs.zip.tmp", portion / "codecs.zip")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("root", type=Path, help="dataset root, e.g. data/train/polish/YODAS2-sidon")
    parser.add_argument("--limit", type=int, help="encode only this many randomly drawn clips (seed 0)")
    parser.add_argument("--dry", action="store_true", help="only compare with the current codecs; write nothing")
    parser.add_argument("--device", help="cuda, mps or cpu (default: cuda when available, else cpu)")
    args = parser.parse_args()
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")

    audio, clips = Audio(), {}
    for path in sorted(args.root.rglob("*.csv")):
        if path.stem.endswith(SKIP):
            continue
        fieldnames, rows = read(path)
        if "transcription" in fieldnames:
            for row in rows:
                clips.setdefault(row["name"], path.parent)
    if not clips:
        print(f"{Colors.WARNING}No CSV with a transcription column under {args.root}.{Colors.ENDC}")
        return
    names = sorted(n for n, portion in clips.items() if n in audio.index(portion))
    with_audio = len(names)
    if args.limit and args.limit < len(names):
        names = sorted(random.Random(0).sample(names, args.limit))

    print_test_title(f"Codecs: {args.root}" + (" (dry run)" if args.dry else ""))
    print_info("clips", f"{len(clips)} distinct, {with_audio} with audio, encoding {len(names)}")
    print_info("codec", f"{MODEL} on {device}")
    torch.set_num_threads(THREADS)
    codec, current, new, errors, t0 = Codec(device), {}, collections.defaultdict(dict), {}, time.time()
    for k, (name, result) in enumerate(zip(names, decoded([(clips[n], n) for n in names])), 1):
        portion = clips[name]
        if portion not in current:
            current[portion] = existing(portion)
        try:
            if isinstance(result, Exception):
                raise result
            y, sr, _ = result
            new[portion][name] = codec.encode(y, sr)
        except Exception as error:  # undecodable audio: its codec is kept, and counted
            errors[name] = repr(error)
        if k % 100 == 0 or k == len(names):
            print(f"  {k}/{len(names)} clips, {k / (time.time() - t0):.1f}/s", flush=True)

    total = collections.Counter()
    for portion, codecs in sorted(new.items()):
        old = current[portion]
        same = [n for n, t in codecs.items() if n in old and old[n].shape == t.shape and (old[n] == t).all()]
        longer = [n for n, t in codecs.items() if n in old and old[n].shape != t.shape]
        changed = [n for n, t in codecs.items() if n in old and old[n].shape == t.shape and not (old[n] == t).all()]
        missing = [n for n in codecs if n not in old]
        frames = sum(t.shape[1] for t in codecs.values())
        print_section(str(portion.relative_to(args.root)) or ".")
        print_info("encoded", f"{len(codecs)} clips, {frames / FRAME_RATE / 3600:.2f} h")
        print(f"    {Colors.OKCYAN}{'vs current codecs':22s}{Colors.ENDC} same {share(len(same), len(codecs))}, "
              f"other length {share(len(longer), len(codecs))}, other tokens {share(len(changed), len(codecs))}, "
              f"none yet {share(len(missing), len(codecs))}")
        for n in longer[:EXAMPLES]:
            print(f"      {n}: {old[n].shape[1]} -> {codecs[n].shape[1]} frames")
        for n in changed[:EXAMPLES]:
            print(f"      {n}: {(old[n] != codecs[n]).mean():.1%} of tokens differ "
                  f"(codebook 0: {(old[n][0] != codecs[n][0]).mean():.1%})")
        total.update({"encoded": len(codecs), "same": len(same), "longer": len(longer), "changed": len(changed)})
    if errors:
        print(f"\n{Colors.FAIL}{len(errors)} clips failed (codecs kept):{Colors.ENDC}")
        for name, error in list(errors.items())[:EXAMPLES]:
            print(f"  - {name}: {error[:80]}")

    if args.dry:
        print(f"\n{Colors.WARNING}Dry run: nothing written.{Colors.ENDC}")
        return
    answer = input(f"\n{Colors.BOLD}Rebuild codecs.zip in {len(new)} portion(s) with {total['encoded']} new codecs? "
                   f"[Y/N] {Colors.ENDC}").strip().upper()
    if answer != "Y":
        print(f"{Colors.WARNING}Nothing written.{Colors.ENDC}")
        return
    for portion, codecs in new.items():
        rebuild(portion, codecs)
        print(f"{Colors.OKGREEN}wrote {portion / 'codecs.zip'}{Colors.ENDC}")


if __name__ == "__main__":
    main()
