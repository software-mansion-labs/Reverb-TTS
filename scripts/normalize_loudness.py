import argparse
from pathlib import Path

import numpy as np
import torch
from pydub import AudioSegment
from torchaudio.functional import loudness

AUDIO_EXTENSIONS = {".mp3", ".wav", ".flac", ".ogg", ".m4a"}


def measure_lufs(audio: AudioSegment) -> float:
    samples = np.array(audio.get_array_of_samples(), dtype=np.float32)
    samples = samples.reshape(-1, audio.channels).T / (1 << (8 * audio.sample_width - 1))
    return loudness(torch.from_numpy(samples), audio.frame_rate).item()


def normalize(
    audio: AudioSegment, min_lufs: float, max_lufs: float, peak_ceiling: float
) -> tuple[AudioSegment, float, float]:
    lufs = measure_lufs(audio)
    gain = min(max(lufs, min_lufs), max_lufs) - lufs
    # Never push peaks above the ceiling, even if that leaves loudness below min_lufs.
    gain = min(gain, peak_ceiling - audio.max_dBFS)
    return audio.apply_gain(gain), lufs, gain


def main() -> None:
    parser = argparse.ArgumentParser(description="Clamp audio loudness (LUFS) into a range.")
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--min-lufs", type=float, default=-24.0)
    parser.add_argument("--max-lufs", type=float, default=-16.0)
    parser.add_argument("--peak-ceiling", type=float, default=-1.0, help="Max peak in dBFS.")
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    for path in sorted(args.input_dir.iterdir()):
        if path.suffix.lower() not in AUDIO_EXTENSIONS:
            continue
        audio = AudioSegment.from_file(path)
        normalized, lufs, gain = normalize(audio, args.min_lufs, args.max_lufs, args.peak_ceiling)
        normalized.export(args.output_dir / path.name, format=path.suffix.lstrip(".").lower())
        print(f"{path.name}: {lufs:.1f} LUFS, gain {gain:+.1f} dB")


if __name__ == "__main__":
    main()
