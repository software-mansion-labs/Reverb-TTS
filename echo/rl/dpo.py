"""
DPO on a round's pairs (knowledge/training/rl_dpo.md). A take's score is the mean log-ratio (policy - reference) of
its codebook 0-1 tokens over several masks: mask i masks each codebook at the share the deployed sampler has before
pass (i + a random offset) mod 8. Winner and loser share their masks when they are as long. A pair's loss is

    -log sigmoid(beta x margin) + kl x KL(reference || policy) + winner_anchor x winner score^2

over the masked slots, times the pair's weight. The reference is the starting model.
"""

import json
import math
import random
import time

from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch

from safetensors.torch import save_file
from transformers import Qwen2Tokenizer

from echo.rl.config import RLConfig
from echo.rl.groups import read_prompts, read_take
from echo.rl.pairs import Pair, load_pairs
from echo.rl.prompts import model_inputs
from echo.rl.sampler import CODEBOOKS, CODES, MASK, Echo, checkpoint_weights
from echo.rl.tracking import Tracker

# Masked share of codebooks 0 and 1 before main pass k, s8 without remasking (3,000 round-2 takes)
SHARES = torch.tensor([[1.0, 0.976, 0.946, 0.907, 0.854, 0.776, 0.651, 0.412],
                       [1.0, 0.992, 0.981, 0.967, 0.948, 0.918, 0.863, 0.736]])

PROBE_PAIRS = 32   # Pairs per kind in each probe set


def masks(frames: int, count: int, rng: random.Random) -> List[torch.Tensor]:
    """
    `count` [2, F] masks, mask i at pass (i + offset) mod 8's shares, each with at least one masked slot.
    """

    g = torch.Generator().manual_seed(rng.randrange(2**31))
    offset = rng.randrange(8)
    out = []
    for i in range(count):
        m = torch.rand(2, frames, generator=g) < SHARES[:, (i + offset) % 8, None]
        m[0, torch.randint(frames, (1,), generator=g)] = True
        out.append(m)
    return out


class Scorer:
    """
    Pair terms of a policy against its frozen reference, the takes and prompts read from their groups directories.
    """

    def __init__(self, policy: Echo, reference: Echo, config: RLConfig, tokenizer: Qwen2Tokenizer) -> None:
        self.policy, self.reference, self.train, self.tokenizer = policy, reference, config.train, tokenizer
        self.rows: Dict[str, Dict[str, dict]] = {}
        self.prompts: Dict[Tuple[str, str], dict] = {}

    def prompt(self, groups: str, prompt_id: str) -> dict:
        if (groups, prompt_id) not in self.prompts:
            if groups not in self.rows:
                self.rows[groups] = {r["id"]: r for r in read_prompts(Path(groups))}
            self.prompts[(groups, prompt_id)] = model_inputs([self.rows[groups][prompt_id]], self.tokenizer)[0]
        return self.prompts[(groups, prompt_id)]

    def codes(self, pair: Pair) -> Dict[str, torch.Tensor]:
        return {k: torch.from_numpy(read_take(Path(pair.groups), k).astype(np.int64)) for k in (pair.winner, pair.loser)}

    def masks(self, codes: Dict[str, torch.Tensor], pair: Pair, rng: random.Random) -> Dict[str, List[torch.Tensor]]:
        """
        Shared masks when both takes have the same frame count, independent ones otherwise.
        """

        def count(frames: int) -> int:
            return self.train.short_masks if frames < self.train.short_frames else self.train.masks

        fw, fl = codes[pair.winner].size(1), codes[pair.loser].size(1)
        if fw == fl:
            shared = masks(fw, count(fw), rng)
            return {pair.winner: shared, pair.loser: shared}
        return {pair.winner: masks(fw, count(fw), rng), pair.loser: masks(fl, count(fl), rng)}

    def terms(self, pair: Pair, codes: Dict[str, torch.Tensor], ms: Dict[str, List[torch.Tensor]], grad: bool) -> tuple:
        """
        (margin, winner score, loser score, KL per slot) of one pair; policy and reference share the masks.
        """

        p = self.prompt(pair.groups, pair.prompt)
        win, lose = pair.winner, pair.loser
        items = [(p, codes[k], m) for k in (win, lose) for m in ms[k]]
        with torch.no_grad():
            ref_lp = forward(self.reference, items, grad=False)
        new_lp = forward(self.policy, items, grad=grad)
        score, kl = {win: [], lose: []}, 0.0
        for j, ((_, c, m), lp, lr_) in enumerate(zip(items, new_lp, ref_lp)):
            kl = kl + (lr_.exp() * (lr_ - lp)).sum(-1).mean()
            tokens = c[:2][m].cuda()[:, None]
            score[win if j < len(ms[win]) else lose].append((lp.gather(-1, tokens) - lr_.gather(-1, tokens))[:, 0].mean())
        w, lo = torch.stack(score[win]).mean(), torch.stack(score[lose]).mean()
        return w - lo, w, lo, kl / len(items)


def forward(model: Echo, items: List[tuple], grad: bool) -> List[torch.Tensor]:
    """
    Per (prompt, codes [8, F], mask [2, F]): [n masked, 1024] log-probabilities at the masked codebook 0-1 slots, in
    the generation layout (prefix blocked from the target; codebooks 2-7 masked).
    """

    offsets = CODES * torch.arange(CODEBOOKS, device="cuda")[:, None]
    embeds, prefixes, totals = [], [], []
    for p, codes, mask, *_ in items:
        state = torch.full_like(codes, MASK)
        state[:2] = torch.where(mask, MASK, codes[:2])
        audio = torch.cat([p["reference"], state], 1).cuda()
        embeds.append(torch.cat([model.llm.embed_tokens(p["ids"].cuda()), model.audio_embeddings(audio + offsets).sum(0)]))
        prefixes.append(p["ids"].numel() + p["reference"].size(1))
        totals.append(embeds[-1].size(0))
    x = torch.nn.utils.rnn.pad_sequence(embeds, batch_first=True)
    position = torch.arange(x.size(1), device="cuda")
    prefix, total = torch.tensor(prefixes, device="cuda"), torch.tensor(totals, device="cuda")
    reach = torch.where(position[None] < prefix[:, None], prefix[:, None], total[:, None])
    allowed = position[None, None] < reach[:, :, None]
    allowed[..., 0] = True
    res = []
    with torch.set_grad_enabled(grad), torch.autocast("cuda", dtype=torch.bfloat16):
        hidden = model.llm(inputs_embeds=x, attention_mask=allowed[:, None],
                           position_ids=position[None].expand(len(items), -1)).last_hidden_state
        heads = model.audio_heads.weight[: 2 * CODES]
        for j, (p, codes, mask, *_) in enumerate(items):
            h = hidden[j, prefixes[j]:totals[j]]
            logits = (h @ heads.T).float().view(-1, 2, CODES).transpose(0, 1)[..., :MASK]     # [2, F, 1024]
            res.append(logits.log_softmax(-1)[mask.cuda()])
    return res


class Probe:
    """
    Pair margins under fixed masks, on PROBE_PAIRS pairs per kind of the held-out probe groups and of the training
    pairs: whether training still moves held-out pairs, or only its own.
    """

    def __init__(self, scorer: Scorer, pairs: List[Pair], config: RLConfig, out: Path) -> None:
        rng = random.Random(123)

        def pick(ps: List[Pair]) -> List[Pair]:
            return [p for k in ("ctc", "critic")
                    for p in rng.sample([q for q in ps if q.kind == k], min(PROBE_PAIRS, sum(q.kind == k for q in ps)))]

        self.scorer, self.sets = scorer, {}
        if config.train.probe_groups is not None:
            self.sets["held"] = pick(load_pairs(Path(config.train.probe_groups), config.pairs))
        self.sets["train"] = pick(pairs)
        self.masks = {}
        for name, ps in self.sets.items():
            for n, p in enumerate(ps):
                self.masks[(name, n)] = scorer.masks(scorer.codes(p), p, random.Random(10_000 + n + 5000 * (name == "train")))
        print("probe pairs: " + ", ".join(f"{name} {len(ps)}" for name, ps in self.sets.items()), flush=True)
        self.log = open(out / "probe.jsonl", "a")

    @torch.no_grad()
    def __call__(self, step: int) -> Dict[str, float]:
        policy = self.scorer.policy
        policy.eval()
        metrics = {}
        for name, ps in self.sets.items():
            res = []
            for n, p in enumerate(ps):
                margin, w, lo, kl = self.scorer.terms(p, self.scorer.codes(p), self.masks[(name, n)], False)
                res.append((p.kind, float(margin), float(w), float(lo), float(kl)))
            for kind in ("ctc", "critic"):
                r = np.array([x[1:] for x in res if x[0] == kind])
                if not len(r):
                    continue
                row = dict(margin=float(r[:, 0].mean()), acc=float((r[:, 0] > 0).mean()), w=float(r[:, 1].mean()),
                           lo=float(r[:, 2].mean()), kl=float(r[:, 3].mean()))
                self.log.write(json.dumps(dict(step=step, set=name, kind=kind, n=len(r), **row)) + "\n")
                metrics.update({f"probe/{name}/{kind}/{k}": v for k, v in row.items()})
        self.log.flush()
        policy.train()
        return metrics


def save(policy: Echo, start: Path, target: Path) -> None:
    """
    The policy's weights in the starting checkpoint's layout, with every other file of it.
    """

    target.mkdir(exist_ok=True)
    save_file(checkpoint_weights(policy, start), target / "model.safetensors")
    for f in start.iterdir():
        if f.name != "model.safetensors" and f.is_file() and not f.name.endswith(".pt"):
            (target / f.name).write_bytes(f.read_bytes())
    print(f"saved {target}", flush=True)


def train(config: RLConfig, pairs: List[Pair], start: Path, out: Path, seed: int = 0, max_steps: Optional[int] = None,
          tracker: Optional[Tracker] = None) -> None:
    """
    One DPO run over `pairs` from the `start` checkpoint, into out/model, with train.log and probe.jsonl beside it.
    """

    t = config.train
    out.mkdir(parents=True, exist_ok=True)
    policy = Echo(start).to("cuda").train()
    policy.llm.gradient_checkpointing_enable()
    reference = Echo(start).to("cuda").eval()
    for q in reference.parameters():
        q.requires_grad_(False)
    scorer = Scorer(policy, reference, config, Qwen2Tokenizer.from_pretrained(start))
    opt = torch.optim.AdamW(policy.parameters(), lr=t.lr, weight_decay=0.0)
    steps = t.epochs * math.ceil(len(pairs) / t.pairs_per_step)
    schedule = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: (s + 1) / t.warmup if s < t.warmup else t.lr_min + (
        1 - t.lr_min) * 0.5 * (1 + math.cos(math.pi * (s - t.warmup) / max(1, steps - t.warmup))))
    probe = Probe(scorer, pairs, config, out)

    def report(step: int) -> None:
        metrics = probe(step)
        if tracker is not None:
            tracker.log(metrics, step)

    kinds = {k: sum(p.kind == k for p in pairs) for k in ("ctc", "critic")}
    log = open(out / "train.log", "a")
    log.write(f"{config.to_dict()}; start {start}; seed {seed}; {len(pairs)} pairs {kinds}; {steps} steps\n")
    rng, step, begin = random.Random(seed), 0, time.time()
    report(0)
    for epoch in range(t.epochs):
        order = list(pairs)
        random.Random(1000 * seed + epoch).shuffle(order)
        for i in range(0, len(order), t.pairs_per_step):
            chunk = order[i:i + t.pairs_per_step]
            opt.zero_grad(set_to_none=True)
            stats = {k: [0.0, 0] for k in ("loss", "acc", "margin", "kl", "dw", "dl", "acc_ctc", "acc_critic")}
            for p in chunk:
                codes = scorer.codes(p)
                margin, w, lo, kl = scorer.terms(p, codes, scorer.masks(codes, p, rng), True)
                loss = p.weight * (-torch.nn.functional.logsigmoid(t.beta * margin) + t.kl * kl
                                   + t.winner_anchor * w.square()) / len(chunk)
                loss.backward()
                for k, v in (("loss", float(loss) * len(chunk)), ("acc", float(margin > 0)), ("margin", float(margin)),
                             ("kl", float(kl)), ("dw", float(w)), ("dl", float(lo)), (f"acc_{p.kind}", float(margin > 0))):
                    stats[k][0] += v
                    stats[k][1] += 1
            norm = torch.nn.utils.clip_grad_norm_(policy.parameters(), 1.0)
            if torch.isfinite(norm):
                opt.step()
            schedule.step()
            step += 1
            m_ = {k: v[0] / v[1] if v[1] else float("nan") for k, v in stats.items()}
            line = (f"epoch {epoch + 1} step {step}/{steps}: lr {schedule.get_last_lr()[0]:.2e}, loss {m_['loss']:.4f}, "
                    f"acc {m_['acc']:.2f} (CTC {m_['acc_ctc']:.2f}, critic {m_['acc_critic']:.2f}), margin {m_['margin']:+.4f}, "
                    f"KL/slot {m_['kl']:.4f}, winner {m_['dw']:+.4f}, loser {m_['dl']:+.4f}, norm {float(norm):.1f}, "
                    f"{time.time() - begin:.0f}s")
            log.write(line + "\n")
            log.flush()
            if tracker is not None:
                tracker.log({"train/loss": m_["loss"], "train/acc": m_["acc"], "train/acc_ctc": m_["acc_ctc"],
                             "train/acc_critic": m_["acc_critic"], "train/margin": m_["margin"], "train/kl": m_["kl"],
                             "train/winner": m_["dw"], "train/loser": m_["dl"], "train/grad_norm": float(norm),
                             "learning_rate": schedule.get_last_lr()[0]}, step)
            if step % 10 == 0:
                print(line, flush=True)
            if step % t.probe_every == 0:
                report(step)
            if step in t.save_at:
                save(policy, start, out / f"model_s{step}")
            if max_steps and step >= max_steps:
                break
        if max_steps and step >= max_steps:
            break
    report(step)
    save(policy, start, out / "model")
