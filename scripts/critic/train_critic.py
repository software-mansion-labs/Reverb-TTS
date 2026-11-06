"""Trains a pairwise critic on synthetic s8 vs sN pairs and tracks how it agrees with the user's blind rankings.

    cd /workspace/Echo && PYTHONPATH=. /venv/main/bin/python scripts/critic/train_critic.py --pairs /workspace/pairs/s8_s32_reencoded \
        --preferences /workspace/pairs/preferences --out <dir>

The critic: Echo's backbone (runs/final/fp32, every weight trainable) reads each take as the model reads a
generation, [prompt + reference text + text | reference frames][take frames] with the prefix blocked from the take,
and the take's states are mean-pooled. The two takes' vectors, concatenated as [A, B], go through an MLP with one
hidden layer, whose 2 logits say whether A or B is better. The stronger sampler's take is the better one; which slot
it takes is random.

Evaluation, each counted over both orders of a pair (a pair is right when P(better wins), averaged over A/B and
B/A, is over a half):
    - synthetic: --eval-share of the synthetic pairs, held out by prompt;
    - tracking: the user's pairs of a --tracking-share of their targets, every --eval-every steps;
    - at the end, all of the user's pairs: all of them, across samplers (with the "more steps wins" rule beside
      it) and within one sampler.

With --fold k the critic trains on the user's pairs instead (no --pairs): the user's targets are split into --folds
folds, the pairs of fold k are held out for tracking and the final report (by tier gap: AC, BC, AB), and the rest
are trained on. Running every fold gives each pair one held-out score (fold_<k>.npz in --out).
"""
import argparse
import csv
import math
import random
import time
from pathlib import Path

import numpy as np
import torch
from transformers import Qwen2Tokenizer

from echo.rl.sampler import CODES, CODEBOOKS, Echo


def read(path: Path) -> list[dict]:
    with open(path, encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f, delimiter="|", quoting=csv.QUOTE_NONE))


def steps_of(sampler: str) -> int:
    return int(sampler.split("-")[0][1:])


class Critic(torch.nn.Module):
    def __init__(self, checkpoint: Path) -> None:
        super().__init__()
        echo = Echo(checkpoint)
        self.llm, self.audio_embeddings = echo.llm, echo.audio_embeddings
        hidden = self.llm.config.hidden_size
        self.head = torch.nn.Sequential(torch.nn.Linear(2 * hidden, hidden), torch.nn.GELU(), torch.nn.Linear(hidden, 2))

    def encode(self, takes: list[dict]) -> torch.Tensor:
        """Each take's mean-pooled state over its frames: [len(takes), hidden]."""
        device = self.head[0].weight.device
        offsets = CODES * torch.arange(CODEBOOKS, device=device)[:, None]
        sequences, prefixes, totals = [], [], []
        for t in takes:
            audio = torch.cat([t["reference"], t["codes"]], 1).to(device)
            sequences.append(torch.cat([self.llm.embed_tokens(t["ids"].to(device)),
                                        self.audio_embeddings(audio + offsets).sum(0)]))
            prefixes.append(t["ids"].numel() + t["reference"].size(1))
            totals.append(sequences[-1].size(0))
        x = torch.nn.utils.rnn.pad_sequence(sequences, batch_first=True)
        L = x.size(1)
        prefix, total = torch.tensor(prefixes, device=device), torch.tensor(totals, device=device)
        position = torch.arange(L, device=device)
        # Prefix queries see the prefix; take queries see everything; padded queries see key 0 (else NaN).
        reach = torch.where(position[None] < prefix[:, None], prefix[:, None], total[:, None])          # [B, L]
        allowed = position[None, None] < reach[:, :, None]                                              # [B, L, L]
        allowed[..., 0] = True
        states = self.llm(inputs_embeds=x, attention_mask=allowed[:, None], position_ids=position[None].expand(len(takes), -1)).last_hidden_state
        frames = (position[None] >= prefix[:, None]) & (position[None] < total[:, None])
        return (states * frames[..., None]).sum(1) / frames.sum(1, keepdim=True)

    def forward(self, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        return self.head(torch.cat([a, b], -1))


class Data:
    """Takes and their prompts, tokenized once."""

    def __init__(self, tokenizer: Qwen2Tokenizer) -> None:
        self.tokenizer, self.prompts, self.codes = tokenizer, {}, {}

    def prompt(self, row: dict, root: Path) -> dict:
        key = (row["reference"], row["text"])
        if key not in self.prompts:
            text = (f"<|denoise|><|lang_start|>{row['text_language']}<|lang_end|><|instruct_start|>None<|instruct_end|>"
                    f"<|text_start|>{row['reference_transcript']} {row['text']}<|text_end|>")
            self.prompts[key] = dict(ids=self.tokenizer(text, return_tensors="pt").input_ids[0],
                                     reference=torch.from_numpy(np.load(root / row["reference"]).astype(np.int64)))
        return self.prompts[key]

    def take(self, prompt: dict, path: Path) -> dict:
        return dict(prompt, codes=torch.from_numpy(np.load(path).astype(np.int64)))


def batches(pairs: list[dict], budget: int, rng: random.Random, shuffle: bool) -> list[list[dict]]:
    """Pairs grouped by length into batches of at most `budget` tokens (both takes, padding included)."""
    pairs = sorted(pairs, key=lambda p: p["length"])
    out, current = [], []
    for p in pairs:
        if current and 2 * (len(current) + 1) * p["length"] > budget:
            out.append(current)
            current = []
        current.append(p)
    out += [current] if current else []
    if shuffle:
        rng.shuffle(out)
    return out


@torch.no_grad()
def evaluate(critic: Critic, pairs: list[dict], budget: int) -> np.ndarray:
    """P(better wins) per pair, averaged over both orders."""
    critic.eval()
    probability = {}
    for batch in batches(pairs, budget, random.Random(0), False):
        with torch.autocast("cuda", torch.bfloat16):
            states = critic.encode([t for p in batch for t in (p["better"], p["worse"])]).float()
        good, bad = states[0::2], states[1::2]
        with torch.autocast("cuda", torch.bfloat16):
            first = critic(good, bad).float().softmax(-1)[:, 0]
            second = critic(bad, good).float().softmax(-1)[:, 1]
        for p, x in zip(batch, ((first + second) / 2).tolist()):
            probability[id(p)] = x
    critic.train()
    return np.array([probability[id(p)] for p in pairs])


def main() -> None:
    parser = argparse.ArgumentParser(description="Train a pairwise critic on synthetic sampler pairs.")
    parser.add_argument("--pairs", type=Path, help="Re-encoded generate_pairs.py output (not with --fold).")
    parser.add_argument("--preferences", type=Path, required=True, help="preference_pairs.py output.")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, default=Path("runs/final/fp32"))
    parser.add_argument("--strong", type=int, default=32, help="Main passes of the stronger sampler in --pairs.")
    parser.add_argument("--epochs", type=float, default=1.0)
    parser.add_argument("--batch-tokens", type=int, default=8192)
    parser.add_argument("--lr", type=float, default=1e-5, help="Backbone learning rate; the head's is 10x.")
    parser.add_argument("--eval-share", type=float, default=0.05)
    parser.add_argument("--tracking-share", type=float, default=0.25)
    parser.add_argument("--eval-every", type=int, default=250)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--fold", type=int, help="Train on the user's pairs outside this fold of their targets.")
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--init", type=Path, help="Start from this critic.pt instead of Echo's weights alone.")
    args = parser.parse_args()

    rng = random.Random(args.seed)
    torch.manual_seed(args.seed)
    tokenizer = Qwen2Tokenizer.from_pretrained(args.checkpoint)
    data = Data(tokenizer)
    data_root = Path("data/rl")

    synthetic = []
    for row in read(args.pairs / "pairs.csv") if args.fold is None else []:
        prompt = data.prompt(row, data_root)
        synthetic.append(dict(row=row["row"], better=data.take(prompt, args.pairs / row[f"codes_s{args.strong}"]),
                              worse=data.take(prompt, args.pairs / row["codes_s8"]),
                              length=prompt["ids"].numel() + prompt["reference"].size(1) + int(row["frames"])))
    rng.shuffle(synthetic)
    held = int(args.eval_share * len(synthetic))
    synthetic_eval, train = synthetic[:held], synthetic[held:]

    preferences = []
    for row in read(args.preferences / "pairs.csv"):
        prompt = data.prompt(row, data_root)
        better, worse = (data.take(prompt, args.preferences / row[c]) for c in ("codes_better", "codes_worse"))
        length = prompt["ids"].numel() + prompt["reference"].size(1) + max(better["codes"].size(1), worse["codes"].size(1))
        preferences.append(dict(target=row["target"], better=better, worse=worse, length=length,
                                same=row["same_sampler"] == "1", tiers=row["tiers"], key=f"{row['better']}>{row['worse']}",
                                steps=(steps_of(row["better"].split(":")[0]), steps_of(row["worse"].split(":")[0]))))
    targets = sorted({p["target"] for p in preferences})
    if args.fold is None:
        tracked = set(random.Random(args.seed + 1).sample(targets, round(args.tracking_share * len(targets))))
    else:
        random.Random(args.seed + 1).shuffle(targets)
        tracked = set(targets[args.fold:: args.folds])
        train = [p for p in preferences if p["target"] not in tracked]
    tracking = [p for p in preferences if p["target"] in tracked]
    print(f"synthetic: {len(train)} train, {len(synthetic_eval)} eval; preferences: {len(preferences)} pairs "
          f"({sum(p['same'] for p in preferences)} same-sampler), tracking {len(tracking)} from {len(tracked)} targets",
          flush=True)

    critic = Critic(args.checkpoint).cuda().train()
    if args.init:
        critic.load_state_dict(torch.load(args.init, map_location="cuda"))
    head = list(critic.head.parameters())
    backbone = [p for n, p in critic.named_parameters() if not n.startswith("head.")]
    optimizer = torch.optim.AdamW([{"params": backbone, "lr": args.lr}, {"params": head, "lr": 10 * args.lr}],
                                  weight_decay=0.01)
    plan = [b for _ in range(math.ceil(args.epochs)) for b in batches(train, args.batch_tokens, rng, True)]
    plan = plan[: round(args.epochs * len(plan) / math.ceil(args.epochs))]
    warmup = max(1, len(plan) // 20)
    schedule = torch.optim.lr_scheduler.LambdaLR(
        optimizer, lambda s: min((s + 1) / warmup, max(0.0, (len(plan) - s) / (len(plan) - warmup))))

    def report(step: int) -> None:
        s, t = evaluate(critic, synthetic_eval, args.batch_tokens), evaluate(critic, tracking, args.batch_tokens)
        same = np.array([p["same"] for p in tracking])
        ab = np.array([p["tiers"] == "AB" for p in tracking])
        print(f"[eval step {step}] synthetic acc {np.mean(s > 0.5) if len(s) else math.nan:.3f} | user (tracking) acc "
              f"{np.mean(t > 0.5):.3f}: cross-sampler {np.mean(t[~same] > 0.5):.3f}, same-sampler "
              f"{np.mean(t[same] > 0.5):.3f}, A vs B {np.mean(t[ab] > 0.5):.3f}", flush=True)

    report(0)
    start, losses, hits = time.time(), [], []
    for step, batch in enumerate(plan, 1):
        flip = torch.tensor([rng.random() < 0.5 for _ in batch], device="cuda")   # True: the better take is B
        with torch.autocast("cuda", torch.bfloat16):
            states = critic.encode([t for p in batch for t in (p["better"], p["worse"])])
            good, bad = states[0::2], states[1::2]
            a, b = torch.where(flip[:, None], bad, good), torch.where(flip[:, None], good, bad)
            logits = critic(a, b).float()
        label = flip.long()                                                          # 0: A is better, 1: B is better
        loss = torch.nn.functional.cross_entropy(logits, label)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(critic.parameters(), 1.0)
        optimizer.step()
        schedule.step()
        losses.append(loss.item())
        hits.append((logits.argmax(-1) == label).float().mean().item())
        if step % 50 == 0:
            print(f"step {step}/{len(plan)}: loss {np.mean(losses[-50:]):.4f}, acc {np.mean(hits[-50:]):.3f}, "
                  f"{(time.time() - start) / step:.2f} s/step", flush=True)
        if step % args.eval_every == 0 and step < len(plan):
            report(step)
    report(len(plan))

    if args.fold is not None:
        p = evaluate(critic, tracking, args.batch_tokens)
        same, tiers = np.array([x["same"] for x in tracking]), np.array([x["tiers"] for x in tracking])
        for gap in ("AC", "BC", "AB"):
            m = tiers == gap
            print(f"FINAL fold {args.fold} {gap}: all {np.mean(p[m] > 0.5):.3f} ({m.sum()}), same-sampler "
                  f"{np.mean(p[m & same] > 0.5):.3f} ({(m & same).sum()})", flush=True)
        args.out.mkdir(parents=True, exist_ok=True)
        np.savez(args.out / f"fold_{args.fold}.npz", probability=p, target=[x["target"] for x in tracking],
                 key=[x["key"] for x in tracking], tiers=tiers, same=same)
        print(f"DONE: fold {args.fold} in {args.out}", flush=True)
        return

    p = evaluate(critic, preferences, args.batch_tokens)
    same = np.array([x["same"] for x in preferences])
    rule = np.array([x["steps"][0] > x["steps"][1] for x in preferences])
    tie = np.array([x["steps"][0] == x["steps"][1] for x in preferences])
    cross = ~same & ~tie
    print(f"FINAL all user pairs ({len(p)}): critic acc {np.mean(p > 0.5):.3f}", flush=True)
    print(f"FINAL across samplers with different steps ({cross.sum()}): critic acc {np.mean(p[cross] > 0.5):.3f}, "
          f"'more steps wins' rule {np.mean(rule[cross]):.3f}", flush=True)
    print(f"FINAL within one sampler ({same.sum()}): critic acc {np.mean(p[same] > 0.5):.3f}", flush=True)
    args.out.mkdir(parents=True, exist_ok=True)
    torch.save(critic.state_dict(), args.out / "critic.pt")
    np.save(args.out / "preference_probabilities.npy", p)
    print(f"DONE: critic in {args.out / 'critic.pt'}", flush=True)


if __name__ == "__main__":
    main()
