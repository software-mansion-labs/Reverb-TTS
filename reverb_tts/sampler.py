"""
The sampler: `steps` main passes decide codebooks 0-1, then two polish passes decide codebooks 2-7 (docs/pipeline.md).
Takes are generated in batches: each prompt's prefix runs once, and every pass runs over the target frames only.
"""

import math
from dataclasses import dataclass
from typing import List, Sequence

import torch
from transformers import DynamicCache

from reverb_tts.model import CODEBOOKS, CODES, MASK, Network


@dataclass(frozen=True)
class SamplerConfig:
    """
    How takes are sampled. The defaults are the tuned setup; any field can be overridden.
    """

    steps: int = 8                      # main passes, which decide codebooks 0-1: content and timing
    remasking: bool = True              # whether main passes withdraw earlier commits the model has come to doubt
    remask_ratio: float = 0.25          # commits withdrawn per new commit, on every main pass but the last
    t_shift: float = 0.1                # warps the commit schedule: few commits early, most in the last passes
    codebook_penalty: float = 5.0       # per codebook index, so codebook 0 tends to be committed before codebook 1
    position_temperature: float = 5.0   # scales confidence and penalty against the Gumbel noise
    polish_bias: float = 1.0            # attention bias towards the reference frames, in the polish passes only
    polish_guidance: float = 1.0        # codebooks 4-7: weight of (second polish pass - first polish pass)

    def __post_init__(self) -> None:
        if self.steps < 1:
            raise ValueError(f"steps must be at least 1, got {self.steps}")
        if not 0 <= self.remask_ratio < 1:
            raise ValueError(f"remask_ratio must be in [0, 1), got {self.remask_ratio}")
        if not 0 < self.t_shift <= 1:
            raise ValueError(f"t_shift must be in (0, 1], got {self.t_shift}")
        if self.position_temperature <= 0:
            raise ValueError(f"position_temperature must be positive, got {self.position_temperature}")


@dataclass
class Prompt:
    """
    One take to generate: the prompt's text token ids, the reference's codes [8, R], the target length and a seed.
    """

    ids: torch.Tensor
    reference: torch.Tensor
    frames: int
    seed: int


class Batch:
    """
    Prompts padded together on the right: prefix [B, P] (text, then reference frames) and target [B, F].
    """

    def __init__(self, network: Network, prompts: Sequence[Prompt], polish_bias: float) -> None:
        self.network, self.n = network, len(prompts)
        self.device, dtype = network.audio_heads.weight.device, network.audio_heads.weight.dtype
        offsets = CODES * torch.arange(CODEBOOKS, device=self.device)[:, None]
        prefixes = [torch.cat([network.llm.embed_tokens(p.ids.to(self.device)),
                               network.audio_embeddings(p.reference.to(self.device) + offsets).sum(0)])
                    for p in prompts]
        lengths = torch.tensor([x.size(0) for x in prefixes], device=self.device)
        references = torch.tensor([p.reference.size(1) for p in prompts], device=self.device)
        self.frames = [p.frames for p in prompts]
        P, self.F = int(lengths.max()), max(self.frames)

        # The prefix never attends to the target, so it runs once and its keys and values are cached.
        keys = torch.arange(P, device=self.device)
        valid = keys[None] < lengths[:, None]                                                         # [B, P]
        mask = torch.where(valid[:, None, None], 0.0, -math.inf).expand(-1, 1, P, -1).clone()
        mask[..., 0] = 0.0  # padded queries need one key, or their softmax is NaN
        self.cache = DynamicCache()
        network.llm(inputs_embeds=torch.nn.utils.rnn.pad_sequence(prefixes, batch_first=True),
                    attention_mask=mask.to(dtype), position_ids=keys[None].expand(self.n, -1), past_key_values=self.cache)

        slots = torch.arange(self.F, device=self.device)
        self.target = slots[None] < torch.tensor(self.frames, device=self.device)[:, None]            # [B, F]
        attend = torch.cat([valid, self.target], 1)                                                    # [B, P + F]
        plain = torch.where(attend[:, None, None], 0.0, -math.inf).expand(-1, 1, self.F, -1).clone()
        on_reference = (keys[None] >= (lengths - references)[:, None]) & valid                        # [B, P]
        polish = plain.clone()
        polish[..., :P] += polish_bias * on_reference[:, None, None].to(polish.dtype)
        self.masks = {False: plain.to(dtype), True: polish.to(dtype)}
        self.positions = lengths[:, None] + slots[None]

    def log_probs(self, tokens: torch.Tensor, codebooks: range, polish: bool = False) -> torch.Tensor:
        """
        One pass over the targets: [B, 8, F] tokens -> [B, len(codebooks), F, 1024] fp32 log-probabilities.
        """

        network = self.network
        offsets = CODES * torch.arange(CODEBOOKS, device=self.device)[None, :, None]
        hidden = network.llm(inputs_embeds=network.audio_embeddings(tokens + offsets).sum(1),
                             attention_mask=self.masks[polish], position_ids=self.positions,
                             past_key_values=self.cache).last_hidden_state
        self.cache.crop(-self.F)  # drop this pass's target keys and values, keep the prefix's
        weight = network.audio_heads.weight[codebooks.start * CODES: codebooks.stop * CODES]
        logits = (hidden @ weight.T).float().view(self.n, self.F, len(codebooks), CODES).transpose(1, 2)
        return logits.log_softmax(-1)[..., :MASK]  # a masked slot is never a prediction


def rank(score: torch.Tensor, descending: bool) -> torch.Tensor:
    """
    Each entry's rank within its row.
    """

    order = score.argsort(-1, descending=descending)
    return torch.empty_like(order).scatter_(-1, order, torch.arange(score.size(-1), device=score.device).expand_as(order))


def sample(batch: Batch, seeds: Sequence[int], config: SamplerConfig) -> List[torch.Tensor]:
    """
    Fills each prompt's fully masked target. Returns one [8, frames] code tensor per take, on the CPU.
    """

    B, F, device = batch.n, batch.F, batch.device
    tokens = torch.full((B, CODEBOOKS, F), MASK, device=device)
    open_ = batch.target[:, None].expand(-1, 2, -1)                                                    # [B, 2, F]
    penalty = config.codebook_penalty * torch.arange(2, device=device)[None, :, None]
    generators = [torch.Generator().manual_seed(seed) for seed in seeds]  # on the CPU: the same noise on any device
    t = torch.linspace(0, 1, config.steps + 1)
    t = (config.t_shift * t / (1 - (1 - config.t_shift) * t)).tolist()

    for step in range(config.steps):
        lp = batch.log_probs(tokens, range(0, 2))
        confidence, prediction = lp.max(-1)
        held = tokens[:, :2]
        masked, revealed = (held == MASK) & open_, (held != MASK) & open_
        left, last = masked.flatten(1).sum(1), step == config.steps - 1
        # This pass's share of the 2 * frames slots; the last pass fills whatever is left.
        quota = torch.tensor([math.ceil(2 * f * (t[step + 1] - t[step])) for f in batch.frames], device=device)
        k = left if last else torch.minimum(left, quota)
        noise = torch.stack([torch.nn.functional.pad(torch.rand((2, f), generator=g), (0, F - f))
                             for g, f in zip(generators, batch.frames)]).to(device)
        gumbel = -torch.log(-torch.log(noise + 1e-10) + 1e-10)
        score = ((confidence - penalty) / config.position_temperature + gumbel).masked_fill(~masked, -math.inf).flatten(1)
        chosen = ((rank(score, True) < k[:, None]) & score.isfinite()).view_as(masked)
        updated = torch.where(chosen, prediction, held)
        if config.remasking and not last:
            # Withdraw the held codes the model now finds least likely, remask_ratio as many as this pass committed.
            likely = lp.gather(-1, held.clamp(max=MASK - 1).unsqueeze(-1)).squeeze(-1)
            counts = torch.tensor([round(config.remask_ratio * x) for x in k.tolist()], device=device)
            count = torch.minimum(counts, revealed.flatten(1).sum(1))
            doubted = rank(likely.masked_fill(~revealed, math.inf).flatten(1), False) < count[:, None]
            updated = updated.masked_fill(doubted.view_as(revealed) & revealed, MASK)
        tokens[:, :2] = updated

    # Polish: codebooks 2-3, then 4-7 guided against the first polish pass's view of them.
    first = batch.log_probs(tokens, range(2, 8), polish=True)
    tokens[:, 2:4] = first[:, :2].argmax(-1)
    second = batch.log_probs(tokens, range(4, 8), polish=True)
    tokens[:, 4:] = (second + config.polish_guidance * (second - first[:, 2:])).argmax(-1)
    return [tokens[i, :, :f].cpu() for i, f in enumerate(batch.frames)]
