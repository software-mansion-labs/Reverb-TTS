"""Composes a pool of RL prompts, (speaker, target text) pairs, from the RL speakers and target pools
(knowledge/training/rl_data.md), for later rounds to draw from (s1_prompts.py).

    PYTHONPATH=. .venv/bin/python scripts/rl/s0_prompt_pool.py --shared-per-speaker 20 --individual-per-speaker 10 \
        --output data/rl/pool.csv [--exclude data/rl/prompts.csv] [--seed 0]

Every speaker of --speakers-pool gets --shared-per-speaker shared-pool targets: a CROSS_SHARE of them in a random other
language, the rest in the speaker's own. Every speaker with an individual pool also gets --individual-per-speaker
targets from it, on top. So the pool holds speakers x shared + speakers with an individual pool x individual prompts,
fewer only when a speaker runs out of targets. A speaker never gets the same text twice, nor a (speaker, text) pair
of an --exclude file, nor a reserved text (echo.rl.prompts.reserved_texts: evaluation texts and the texts the user
ranked). Shared texts are dealt from a shuffled deck per language, so every text is used about equally often. A
speaker with several references gets one of them at random for each prompt.

The output is pipe-separated, echo.rl.prompts.FIELDS, with reference codec paths relative to data/rl/ as in
--speakers-pool.
"""
import argparse
import random
from pathlib import Path

from echo.rl.prompts import FIELDS, RL_DATA, read_rows, reserved_texts, write_rows

CROSS_SHARE = 0.25


class Deck:
    """Shared texts of one language, dealt in shuffled rounds."""

    def __init__(self, rows: list[dict], rng: random.Random) -> None:
        self.rows, self.rng, self.queue = rows, rng, []

    def deal(self, taken: set[str]) -> dict | None:
        for _ in range(2 * len(self.rows)):
            if not self.queue:
                self.queue = self.rng.sample(self.rows, len(self.rows))
            row = self.queue.pop()
            if row["text"] not in taken:
                return row
        return None


def main() -> None:
    parser = argparse.ArgumentParser(description="Compose a pool of (speaker, target) prompts for RL.")
    parser.add_argument("--speakers-pool", type=Path, default=RL_DATA / "speakers_processed.csv",
                        help="The RL references, name|transcript|speaker_id|language|tokens, codec paths relative to data/rl/")
    parser.add_argument("--shared", type=Path, default=RL_DATA / "targets" / "shared.csv")
    parser.add_argument("--individual", type=Path, default=RL_DATA / "targets" / "individual.csv")
    parser.add_argument("--shared-per-speaker", type=int, required=True)
    parser.add_argument("--individual-per-speaker", type=int, required=True)
    parser.add_argument("--exclude", type=Path, action="append", default=[],
                        help="A prompts file whose (speaker, text) pairs the pool leaves out; repeatable")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    rng = random.Random(args.seed)
    references: dict[str, list[dict]] = {}
    for row in read_rows(args.speakers_pool):
        references.setdefault(row["speaker_id"], []).append(row)
    shared: dict[str, list[dict]] = {}
    for row in read_rows(args.shared):
        shared.setdefault(row["language"], []).append(row)
    individual: dict[str, list[dict]] = {}
    for row in read_rows(args.individual):
        individual.setdefault(row["speaker_id"], []).append(row)
    decks = {lang: Deck(rows, rng) for lang, rows in shared.items()}
    reserved = reserved_texts()
    excluded = {(r["speaker_id"], r["text"]) for f in args.exclude for r in read_rows(f)}

    prompts, short = [], []
    for speaker in sorted(references):
        language = references[speaker][0]["language"]
        taken = reserved | {text for s, text in excluded if s == speaker}
        cross = round(CROSS_SHARE * args.shared_per_speaker)
        others = [lang for lang in decks if lang != language]
        targets = []
        for i in range(args.shared_per_speaker):
            row = decks[rng.choice(others) if i < cross else language].deal(taken)
            if row is None:
                break
            taken.add(row["text"])
            targets.append((row, "shared", row["subcategory"]))
        own = [row for row in individual.get(speaker, []) if row["text"] not in taken]
        wanted = args.individual_per_speaker if speaker in individual else 0
        targets += [(row, "individual", "") for row in rng.sample(own, min(wanted, len(own)))]
        if len(targets) < args.shared_per_speaker + wanted:
            short.append(f"{speaker} ({len(targets)} of {args.shared_per_speaker + wanted})")
        for row, pool, subcategory in targets:
            reference = rng.choice(references[speaker])
            prompts.append(dict(reference=reference["name"], reference_transcript=reference["transcript"],
                                speaker_id=speaker, language=language, text=row["text"],
                                text_language=row["language"], pool=pool, subcategory=subcategory))

    rng.shuffle(prompts)
    write_rows(args.output, prompts, FIELDS)
    with_pool = sum(s in individual for s in references)
    expected = len(references) * args.shared_per_speaker + with_pool * args.individual_per_speaker
    cross = sum(p["pool"] == "shared" and p["text_language"] != p["language"] for p in prompts)
    print(f"{len(prompts)} prompts (of {len(references)} x {args.shared_per_speaker} + {with_pool} x "
          f"{args.individual_per_speaker} = {expected}) for {len(references)} speakers in {args.output}: "
          f"{sum(p['pool'] == 'shared' for p in prompts)} shared ({cross} cross-language), "
          f"{sum(p['pool'] == 'individual' for p in prompts)} individual", flush=True)
    if short:
        print(f"{len(short)} speakers ran out of targets: {', '.join(short)}", flush=True)


if __name__ == "__main__":
    main()
