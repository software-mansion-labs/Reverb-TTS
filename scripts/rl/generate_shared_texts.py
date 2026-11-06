"""Generates the shared pool of RL target texts (knowledge/training/rl_data.md) with Claude CLI agents. One reviewer
(Opus 5.5, medium effort) lives for the whole run as a single resumed session: it gets the plan first, then, for each
subcategory, the draft of a fresh writer (Haiku 5.5) with twice the texts needed, and keeps the half that is most
varied and closest to real-world speech.

    PYTHONPATH=. .venv/bin/python scripts/rl/generate_shared_texts.py --amount 1400 --out <pool>.csv

The output is pipe-separated, text|language|category|subcategory, with --amount rows; it grows after every subcategory.
"""
import argparse
import json
import math
import subprocess
import tempfile
import uuid
from dataclasses import dataclass
from pathlib import Path

# ---------- Constants ----------

CATEGORY_SHARES = {"general": 0.5, "expressive": 0.5}
LANGUAGE_SHARES = {"en": 0.25, "pl": 0.15, "de": 0.12, "es": 0.12, "fr": 0.12, "it": 0.12, "pt": 0.12}
LANGUAGES = {"en": "English", "pl": "Polish", "de": "German", "es": "Spanish", "fr": "French", "it": "Italian",
             "pt": "Portuguese"}
OVERSAMPLE = 2                              # the writer drafts this many times the texts the pool keeps
SECONDS, ENGLISH_WORDS = (3, 15), (8, 40)   # length of one text when spoken
WRITER = ("claude-haiku-5-5", None)         # model, effort
REVIEWER = ("claude-opus-5-5[1m]", "medium")   # 1M context: the session holds every draft of the run
RETRIES = 3                                 # per CLI call, per writer top-up and per reviewer correction
TIMEOUT = 1800                              # seconds per CLI call
PUNCTUATION = set(" .,!?;:'\"’‘“”«»„…-–—¿¡")   # any other non-letter (digit, symbol, bracket) can't be spoken

# ---------- Task and categories ----------


@dataclass(frozen=True)
class Subcategory:
    name: str
    category: str
    share: float        # within its category
    description: str
    example: str


SUBCATEGORIES = (
    Subcategory("narrative", "general", 0.25,
                "Storytelling as in audiobooks, novels, short stories, fairy tales, memoirs or a friend recounting what "
                "happened: past-tense narration of events, scenes and characters, sometimes with a short line of "
                "reported dialogue.",
                "By the time the ferry reached the island, the fog had lifted and the harbour was already full of "
                "fishing boats."),
    Subcategory("descriptive", "general", 0.20,
                "Factual explanation as in documentaries, encyclopedias, museum audio guides, lessons or science "
                "podcasts: what something is, how it works, where it comes from. Animals, places, history, science, "
                "technology, food, art, everyday objects.",
                "The red fox is a small predator found across Europe, Asia and North America, known for its bushy tail "
                "and sharp hearing."),
    Subcategory("instructions", "general", 0.20,
                "Step-by-step guidance as from a voice assistant, car navigation, a recipe, a manual, customer support, "
                "or a fitness or software tutorial: ordered steps, calm imperatives, sequencing words.",
                "First, unplug the device and wait ten seconds. Then hold the power button until the light turns blue."),
    Subcategory("long_sentences", "general", 0.15,
                "One or two long sentences with subordinate and nested clauses, asides and contrasts, as in literary "
                "prose, essays, lectures or formal speeches, which a reader has to phrase and breathe through. These "
                "texts sit at the long end of the length range.",
                "Although the report, which had been delayed twice, finally arrived on Friday, nobody in the office had "
                "time to read it before the meeting."),
    Subcategory("numbers", "general", 0.10,
                "Texts built around numbers: prices, times, dates, years, phone numbers, distances, measurements, "
                "percentages, scores, statistics, ages. Every number is written out in words, in the form a speaker "
                "says it, with the right grammatical case, gender, and ordinal or cardinal form.",
                "The flight leaves at a quarter past seven on the twenty-third of March, and it costs about four "
                "hundred euros."),
    Subcategory("lists", "general", 0.05,
                "Enumerations of several items, as in shopping lists, menus, agendas, packing lists, ingredients, "
                "options or rankings, with the items separated by commas and closed by 'and' or 'or'.",
                "We'll need flour, eggs, butter, two lemons, a pinch of salt, and a little sugar for the glaze."),
    Subcategory("tongue_twisters", "general", 0.05,
                "Texts dense with words that are hard to say: tongue-twisters, consonant clusters, long compound words, "
                "foreign and place names, loanwords and technical terms, in sentences that still make sense.",
                "The sixth sheikh's sixth sheep is sick, said the vet from Worcestershire."),
    Subcategory("casual", "expressive", 0.25,
                "Everyday spoken lines between friends, family, colleagues or strangers: chatting, sharing news, making "
                "plans, complaining, small talk. The rhythm is spontaneous, with fillers, hedges, contractions, false "
                "starts and self-corrections, the way people really talk.",
                "Yeah, no, I mean, I get it, but honestly, it's just not worth the hassle right now."),
    Subcategory("structural_mixed", "expressive", 0.24,
                "An ordinary conversational line, mostly statements, with a question, an exclamation, or sometimes "
                "both, inserted where they come naturally.",
                "I finally fixed the sink this morning. Took me the whole morning! Have you ever tried to get under one "
                "of those old cabinets?"),
    Subcategory("structural_questions", "expressive", 0.10,
                "Texts carried by questions: yes/no and wh- questions, tag questions, rhetorical questions, chains of "
                "follow-up questions, or a statement followed by a question.",
                "Wait, you actually said yes? And what did she say after that?"),
    Subcategory("structural_exclamations", "expressive", 0.06,
                "Texts carried by exclamations: surprise, delight, alarm, warnings, praise, cheering; short bursts and "
                "full exclamatory sentences.",
                "That's incredible! We actually won! I can't believe it!"),
    Subcategory("commands", "expressive", 0.10,
                "Commands, urgent instructions, warnings, pep talks, sales pitches, coaching and arguments meant to "
                "convince: imperatives with force and rhetorical push.",
                "Stop. Look at me. You're going to walk in there and tell them the truth, right now."),
    Subcategory("emotion_anger", "expressive", 0.05,
                "Anger, irritation or exasperation: complaints, arguments, confrontations, venting; emphatic, clipped "
                "delivery.",
                "I told you three times! Three! And you still went ahead and did it anyway!"),
    Subcategory("emotion_joy", "expressive", 0.05,
                "Joy and excitement: good news, celebrations, surprises, enthusiasm; bright, fast, rising energy.",
                "Guess what? She's coming home tomorrow! I've been waiting for this all year!"),
    Subcategory("emotion_sadness", "expressive", 0.05,
                "Sadness and tenderness: grief, regret, nostalgia, consolation, affection; soft, slow, falling, with "
                "long pauses.",
                "I kept his letters, you know. Every single one. I just never had the courage to read them again."),
    Subcategory("emotion_fear", "expressive", 0.05,
                "Fear and tension: suspense, worry, urgency, hushed warnings; hesitant, broken phrasing, close to a "
                "whisper.",
                "Did you hear that? Something's outside. Don't move... just, please, stay quiet."),
    Subcategory("emotion_sarcasm", "expressive", 0.05,
                "Sarcasm, irony, deadpan humour and playful teasing between friends: what the speaker means is the "
                "opposite of, or more than, what the words say.",
                "Oh, brilliant idea. Let's leave the keys inside the car. What could possibly go wrong?"),
)
for _category in CATEGORY_SHARES:
    assert math.isclose(sum(s.share for s in SUBCATEGORIES if s.category == _category), 1.0), _category

# ---------- Agent roles and prompts ----------

WRITER_ROLE = f"""You write target texts for Echo, a multilingual voice-cloning text-to-speech model. Each text is what
one speaker says aloud, and any voice may read it: a man or a woman, young or old, calm or theatrical. The texts teach
the model to speak real-world material well, so each must read like something a real person would actually say or read
aloud, written by a native speaker of its language.

Rules for every text:
1. Write each language's texts from scratch for that language; they are not translations of each other. Use names,
   places, food, customs and situations natural to its speakers.
2. A text takes {SECONDS[0]} to {SECONDS[1]} seconds to say at a natural pace (in English, about {ENGLISH_WORDS[0]} to
   {ENGLISH_WORDS[1]} words). Spread the lengths over that range unless the subcategory asks for long or short texts.
3. Write everything as it is spoken: numbers, dates, times, amounts and units in words, in their correct grammatical
   form. No digits, symbols (%, €, &, /, +), abbreviations read differently from how they are written, or emojis.
4. Only the words the speaker says: no stage directions, sound effects, speaker labels, brackets or markup.
5. Punctuate as a careful transcriber would: commas where the speaker pauses, question and exclamation marks where the
   intonation calls for them, ellipses and dashes only for real hesitations or breaks.
6. Avoid first-person forms that mark the speaker's gender (Polish "byłem" or "byłam", French "je suis allé" or
   "allée"), since any voice may read the text.
7. Make the texts as varied as possible: no two share a topic, situation, opening phrase or sentence pattern. Cover
   different settings (home, work, travel, school, health, sport, nature, the city, shopping, technology, culture),
   registers and kinds of speakers.

Answer only with the structured output: for each language, the list of its texts."""

REVIEWER_ROLE = f"""You review a pool of target texts for the reinforcement-learning stage of Echo, a multilingual
({', '.join(LANGUAGES.values())}) voice-cloning text-to-speech model. In that stage, cloned voices of every kind read
these texts and a reward judges how well they are spoken, so the pool must look like the real-world material a speech
model meets: audiobooks, the news, podcasts, voice assistants, games, films and everyday conversation. Half the pool is
general texts and half is expressive ones, each split into subcategories with fixed shares.

Writer agents draft the pool one subcategory at a time, each drafting about {OVERSAMPLE} times the texts the pool needs.
For every batch you keep exactly the number of texts given for each language, choosing them so the pool as a whole is
as good as possible:
- Real-world: keep texts a real person would actually say or read aloud. Reject texts that are artificial, contrived,
  stilted, overwritten, unidiomatic or ungrammatical, that read like translations, that are in the wrong language, or
  that don't fit their subcategory.
- Variety: within a subcategory and language, the kept texts cover as many topics, situations, sentence structures,
  lengths and registers as possible. Reject near-duplicates, texts that repeat the idea, situation or pattern of
  another text in different words, including texts you kept earlier in this run.
- Speakability: reject texts with digits, symbols, stage directions or anything else that isn't the words a speaker
  says, and texts whose first-person forms mark the speaker's gender.
When fewer texts are good than you must keep, keep the least bad.

This conversation lasts the whole run: you get the plan first, then every batch in turn. Remember what you kept, so the
pool stays varied across subcategories too. Answer only with the structured output."""


def language_counts(counts: dict[str, int]) -> str:
    return ", ".join(f"{LANGUAGES[lang]} ({lang}) {n}" for lang, n in counts.items())


def writer_task(sub: Subcategory, counts: dict[str, int]) -> str:
    return (f'Write texts for the subcategory "{sub.name}" ({sub.category} texts).\n\n{sub.description}\n\n'
            f'An example in English, for the style only (don\'t translate it or reuse its topic): "{sub.example}"\n\n'
            f"Texts to write per language: {language_counts(counts)}.")


def writer_top_up(missing: dict[str, int], dropped: int) -> str:
    reason = f"{dropped} of your texts were dropped for breaking the rules or repeating another text. " if dropped else ""
    return (f"{reason}Some texts are still missing: {language_counts(missing)}. Write only those, each different from "
            "every text you have written so far.")


def reviewer_plan(amount: int, counts: dict[str, dict[str, int]]) -> str:
    rows = [f"| {s.name} | {s.category} | {sum(counts[s.name].values())} | "
            + " | ".join(str(counts[s.name][lang]) for lang in LANGUAGES) + " |" for s in SUBCATEGORIES]
    return (f"The plan for this run: a pool of {amount} texts from {len(SUBCATEGORIES)} subcategories, kept per "
            "language as below.\n\n| subcategory | category | kept | " + " | ".join(LANGUAGES) + " |\n|"
            + "---|" * (3 + len(LANGUAGES)) + "\n" + "\n".join(rows) + "\n\n"
            "For each subcategory you'll get the writer's draft, numbered per language, and you'll pick the texts to "
            "keep by their numbers. Before the first batch, note in a few lines what you'll watch for across the run.")


def reviewer_batch(sub: Subcategory, step: int, candidates: dict[str, list[str]], counts: dict[str, int],
                   done: list[str]) -> str:
    progress = f"Done so far: {', '.join(done)}." if done else "This is the first batch."
    remaining = [s.name for s in SUBCATEGORIES[step:]]
    lists = "\n\n".join(f"## {LANGUAGES[lang]} ({lang})\n" + "\n".join(f"{i}. {text}" for i, text in enumerate(texts, 1))
                        for lang, texts in candidates.items() if lang in counts)
    return (f'Batch {step} of {len(SUBCATEGORIES)}: "{sub.name}" ({sub.category} texts). {sub.description}\n'
            f"{progress} Still to come: {', '.join(remaining) or 'nothing'}.\n\n"
            f"Keep exactly: {language_counts(counts)}.\n\n{lists}")


def reviewer_correction(problems: list[str]) -> str:
    return (f"Your selection doesn't match the counts: {'; '.join(problems)}. Answer again with the complete selection "
            "for every language.")


def texts_schema(languages: list[str]) -> dict:
    return {"type": "object", "required": languages,
            "properties": {lang: {"type": "array", "items": {"type": "string"}} for lang in languages}}


def selection_schema(languages: list[str]) -> dict:
    keep = {"type": "object", "required": languages,
            "properties": {lang: {"type": "array", "items": {"type": "integer"}} for lang in languages}}
    rejected = {"type": "string", "description": "One or two lines on the main reasons for the rejections."}
    return {"type": "object", "required": ["keep", "rejected"], "properties": {"keep": keep, "rejected": rejected}}


NOTES_SCHEMA = {"type": "object", "required": ["notes"], "properties": {"notes": {"type": "string"}}}

# ---------- Claude CLI agents ----------


class Agent:
    """One Claude CLI conversation, resumed on every ask(), with no tools, settings, MCP servers or CLAUDE.md."""

    spent = 0.0     # USD over all agents, as the CLI reports it

    def __init__(self, name: str, model: str, effort: str | None, role: str, workdir: Path) -> None:
        self.name, self.workdir, self.session, self.started = name, workdir, str(uuid.uuid4()), False
        self.args = ["claude", "-p", "--model", model, "--system-prompt", role, "--tools", "", "--setting-sources", "",
                     "--strict-mcp-config", "--disable-slash-commands", "--output-format", "json"]
        self.args += ["--effort", effort] if effort else []

    def ask(self, prompt: str, schema: dict) -> dict:
        for _ in range(RETRIES):
            session = ["--resume", self.session] if self.started else ["--session-id", self.session]
            try:
                done = subprocess.run(self.args + session + ["--json-schema", json.dumps(schema)], input=prompt,
                                      capture_output=True, text=True, cwd=self.workdir, timeout=TIMEOUT)
                reply, error = json.loads(done.stdout or "{}"), done.stderr
            except (subprocess.TimeoutExpired, json.JSONDecodeError) as e:
                reply, error = {}, str(e)
            if "structured_output" in reply and not reply.get("is_error"):
                self.started = True
                Agent.spent += reply.get("total_cost_usd", 0.0)
                return reply["structured_output"]
            print(f"    {self.name}: call failed ({(error or reply.get('result') or '').strip()[-200:]}), retrying",
                  flush=True)
            if not self.started:    # the failed call may have created the session, so a retry starts a new one
                self.session = str(uuid.uuid4())
        raise RuntimeError(f"{self.name}: no valid answer in {RETRIES} attempts")

# ---------- Main ----------


def apportion(total: int, shares: dict[str, float]) -> dict[str, int]:
    """Largest-remainder rounding: whole counts in proportion to the shares, adding up to total."""
    if not total:
        return dict.fromkeys(shares, 0)
    exact = {key: total * share / sum(shares.values()) for key, share in shares.items()}
    counts = {key: math.floor(value) for key, value in exact.items()}
    for key in sorted(exact, key=lambda key: counts[key] - exact[key])[:total - sum(counts.values())]:
        counts[key] += 1
    return counts


def plan(amount: int) -> dict[str, dict[str, int]]:
    """Texts kept per subcategory and language. Each subcategory's languages are apportioned to what the languages
    still lack after the subcategories before it, so rounding never favours the same languages."""
    totals = apportion(amount, {s.name: CATEGORY_SHARES[s.category] * s.share for s in SUBCATEGORIES})
    counts, given, running = {}, dict.fromkeys(LANGUAGE_SHARES, 0), 0
    for name, total in totals.items():
        running += total
        counts[name] = apportion(total, {lang: max(share * running - given[lang], 0.0)
                                         for lang, share in LANGUAGE_SHARES.items()})
        given = {lang: n + counts[name][lang] for lang, n in given.items()}
    return counts


def spoken(text: str) -> bool:
    return bool(text) and all(c.isalpha() or c in PUNCTUATION for c in text)


def draft(writer: Agent, sub: Subcategory, counts: dict[str, int], seen: set[str]) -> tuple[dict[str, list[str]], int]:
    """The writer's texts per language and how many were dropped (unspeakable, or repeating a text seen earlier in the
    run). Missing texts are asked for again, up to RETRIES times."""
    texts, dropped, missing = {lang: [] for lang in counts}, 0, dict(counts)
    prompt = writer_task(sub, counts)
    for _ in range(RETRIES):
        reply, before = writer.ask(prompt, texts_schema(list(missing))), dropped
        for lang in missing:
            for text in reply[lang]:
                text = " ".join(text.split())
                if spoken(text) and text.casefold() not in seen:
                    texts[lang].append(text)
                    seen.add(text.casefold())
                else:
                    dropped += 1
        missing = {lang: n - len(texts[lang]) for lang, n in counts.items() if len(texts[lang]) < n}
        if not missing:
            break
        prompt = writer_top_up(missing, dropped - before)
    return texts, dropped


def review(reviewer: Agent, sub: Subcategory, step: int, candidates: dict[str, list[str]], counts: dict[str, int],
           done: list[str]) -> dict[str, list[str]]:
    """The texts the reviewer keeps per language. A selection off the counts is sent back up to RETRIES times, then
    taken as it is: valid numbers only, in the reviewer's order, cut to the count."""
    prompt = reviewer_batch(sub, step, candidates, counts, done)
    for _ in range(RETRIES):
        reply = reviewer.ask(prompt, selection_schema(list(counts)))
        chosen, problems = {}, []
        for lang, n in counts.items():
            numbers = list(dict.fromkeys(i for i in reply["keep"][lang] if 1 <= i <= len(candidates[lang])))
            chosen[lang] = [candidates[lang][i - 1] for i in numbers[:n]]
            if len(numbers) != n:
                problems.append(f"{LANGUAGES[lang]} ({lang}) needs {n} distinct numbers from 1 to "
                                f"{len(candidates[lang])}, you gave {len(numbers)}")
        if not problems:
            break
        prompt = reviewer_correction(problems)
    print(f"    rejected: {reply['rejected']}" + (f"\n    still off the counts: {'; '.join(problems)}" if problems else ""),
          flush=True)
    return chosen


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate the shared pool of RL target texts.")
    parser.add_argument("--amount", type=int, required=True, help="Texts in the pool.")
    parser.add_argument("--out", type=Path, required=True, help="Output CSV: text|language|category|subcategory.")
    args = parser.parse_args()

    counts = plan(args.amount)
    workdir = Path(tempfile.mkdtemp(prefix="echo_rl_texts_"))    # outside the repo, so the agents find no CLAUDE.md
    reviewer = Agent("reviewer", *REVIEWER, REVIEWER_ROLE, workdir)
    print(f"Reviewer's notes on the plan:\n{reviewer.ask(reviewer_plan(args.amount, counts), NOTES_SCHEMA)['notes']}",
          flush=True)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    seen, done, total = set(), [], 0
    with open(args.out, "w") as out:
        out.write("text|language|category|subcategory\n")
        for step, sub in enumerate(SUBCATEGORIES, 1):
            wanted = {lang: n for lang, n in counts[sub.name].items() if n}
            if not wanted:
                continue
            writer = Agent(f"writer {sub.name}", *WRITER, WRITER_ROLE, workdir)
            candidates, dropped = draft(writer, sub, {lang: OVERSAMPLE * n for lang, n in wanted.items()}, seen)
            print(f"[{step}/{len(SUBCATEGORIES)}] {sub.name}: {sum(map(len, candidates.values()))} drafted "
                  f"({dropped} dropped), reviewing", flush=True)
            keep = {lang: min(n, len(candidates[lang])) for lang, n in wanted.items()}
            chosen = review(reviewer, sub, step, candidates, keep, done)
            for lang, texts in chosen.items():
                out.writelines(f"{text}|{lang}|{sub.category}|{sub.name}\n" for text in texts)
            out.flush()
            kept = sum(map(len, chosen.values()))
            total += kept
            done.append(f"{sub.name} ({kept} kept)")
            print(f"    kept {kept}, {total}/{args.amount} in the pool, ${Agent.spent:.2f} spent", flush=True)
    print(f"DONE: {total} texts in {args.out}, ${Agent.spent:.2f} spent", flush=True)


if __name__ == "__main__":
    main()
