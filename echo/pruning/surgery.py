import copy
import itertools
import torch

from contextlib import contextmanager
from torch import nn
from typing import Iterable, Iterator, List, Optional, Set, Tuple

from echo.model.model import EchoModel, FactorizedEmbedding

# Ridge added to the kept neurons' Gram before the refit, relative to its mean diagonal.
# It only has to keep the solve well conditioned - near-dead neurons make the Gram singular.
RIDGE = 1e-3

# Byte-level BPE starts from one token per byte - kept whole, so any text still tokenizes after vocabulary pruning.
BYTE_TOKENS = 256


# Layer pruning - temporary
@contextmanager
def without_layers(
    model: EchoModel,
    indices: Iterable[int],   # Layers to be dropped (indicated by their current IDs, starting from 0 up to num_layers - 1)
) -> Iterator[EchoModel]:
    """
    Makes the model behave as if those layers were not there, and puts them back on the way out. 

    It's a context manager, which makes it elegant and reversible.

    Forward passes only: the configs still claim the original depth, and
    'generate' sizes its KV cache from them. Use 'drop_layers' before generating.
    """

    layers = model.llm.layers   # Create a backup
    model.llm.layers = _survivors(model, indices)

    try:
        yield model
    finally:
        model.llm.layers = layers   # Restore the backup


# Layer pruning - gradual
@contextmanager
def faded_layer(
    model: EchoModel,
    index: int,      # Layer to fade (its current ID)
    alpha: float,    # Scale of the layer's updates to the residual stream - 1 leaves it whole, 0 removes it
) -> Iterator[EchoModel]:
    """
    Scales a layer's attention and FFN outputs by 'alpha', and takes the scaling off on the way out.

    At alpha 0 the layer passes its input through untouched - the same model as 'without_layers'.
    """

    layer = model.llm.layers[index]

    def scale(module: nn.Module, args: tuple, output: object) -> object:
        # The attention returns its weights beside the output.
        return (output[0] * alpha, *output[1:]) if isinstance(output, tuple) else output * alpha

    handles = [layer.self_attn.register_forward_hook(scale), layer.mlp.register_forward_hook(scale)]

    try:
        yield model
    finally:
        for handle in handles:
            handle.remove()


class LayerFade:
    """
    Fades one layer out over a training run: its outputs scaled by alpha, from 1 down to 0, with the layer frozen.

    The removal's damage grows about as (1 - alpha)^power, so alpha = 1 - t^(1/power) spreads it evenly over the fade.
    With a 'budget', the schedule only advances while the smoothed loss stays within it of where it started - a layer
    the rest of the model is slow to cover for fades slower, rather than turning into a cut. At alpha 0 the layer passes its
    input through untouched, and 'drop_layers' removes it without changing the model.
    """

    def __init__(
        self,
        model: EchoModel,
        index: int,      # Layer to fade (its current ID)
        steps: int,                # Advances the fade takes - held ones come on top
        power: float,              # Exponent of the damage curve the schedule evens out
        budget: Optional[float],   # Loss the fade may add over its starting level before it is held - None never holds
    ) -> None:
        layer = model.llm.layers[index]
        # Frozen whole, norms included: a layer that could learn would grow its outputs against the falling alpha.
        layer.requires_grad_(False)

        self.index, self.steps, self.power, self.budget = index, steps, power, budget
        self.progress, self.alpha = 0, 1.0
        self.start: Optional[float] = None
        self.level: Optional[float] = None
        self.handles = [layer.self_attn.register_forward_hook(self._scale), layer.mlp.register_forward_hook(self._scale)]

    def _scale(self, module: nn.Module, args: tuple, output: object) -> object:
        return (output[0] * self.alpha, *output[1:]) if isinstance(output, tuple) else output * self.alpha

    def update(self, loss: float) -> None:
        """
        Moves the fade along after an optimizer step, given that step's loss (the same on every rank).
        """

        # About the last 50 steps: one batch is too noisy to hold the fade on.
        self.level = loss if self.level is None else 0.98 * self.level + 0.02 * loss
        if self.start is None:
            self.start = self.level

        if self.progress < self.steps and (self.budget is None or self.level <= self.start + self.budget):
            self.progress += 1

        self.alpha = 1.0 - (self.progress / self.steps) ** (1.0 / self.power)

    @property
    def done(self) -> bool:
        return self.progress >= self.steps


# Layer pruning - permament
def drop_layers(
    model: EchoModel,
    indices: Iterable[int],   # Layers to be dropped (indicated by their current IDs, starting from 0 up to num_layers - 1)
) -> EchoModel:
    """
    Removes layers for good, leaving a model that describes itself correctly and
    can be saved as an ordinary checkpoint.

    In comparision to the `without_layers` context manager, this one is irreversible.
    """

    architecture = model.echo_config.architecture
    kept = architecture.kept_layers or list(range(len(model.llm.layers)))
    dropped = set(indices)

    model.llm.layers = _survivors(model, indices)
    remaining = len(model.llm.layers)
    architecture.kept_layers = [layer for index, layer in enumerate(kept) if index not in dropped]
    if architecture.kept_kv_heads is not None:
        architecture.kept_kv_heads = [heads for index, heads in enumerate(architecture.kept_kv_heads) if index not in dropped]

    # Attention blocks carry the index the KV cache is keyed by. A model whose
    # indices are left stale trains happily, and breaks on its first generation.
    for index, layer in enumerate(model.llm.layers):
        layer.self_attn.layer_idx = index
        if hasattr(layer, "layer_idx"):
            layer.layer_idx = index

    for config in (model.llm.config, model.config.llm_config):
        config.num_hidden_layers = remaining
        if getattr(config, "layer_types", None):
            config.layer_types = list(config.layer_types)[:remaining]

    # The echo sidecar travels with the checkpoint, so it has to agree as well.
    architecture.num_hidden_layers = remaining
    model.config.echo = model.echo_config.to_dict(include_base=False)

    return model


# FFN pruning - permanent
def shrink_ffn(
    model: EchoModel,
    size: int,                   # FFN neurons every layer keeps
    grams: List[torch.Tensor],   # [I, I] per layer: Gram of the down_proj inputs over calibration tokens (echo.pruning.calibration)
) -> EchoModel:
    """
    Keeps, in every layer, the neurons that put the most energy into the layer's output,
    and refits down_proj so the kept ones reproduce the full layer's output as closely as possible.

    A neuron's energy is E[a_j^2] * ||W_down[:, j]||^2. The refit is the least-squares solution
    W' = W G[:, K] G[K, K]^-1, in closed form - it changes weights only, never the architecture.
    """

    current = model.llm.config.intermediate_size
    if not 0 < size <= current:
        raise ValueError(f"FFN size must lie in (0, {current}], got {size}")

    for layer, gram in zip(model.llm.layers, grams):
        _shrink_mlp(layer.mlp, size, gram)

    for config in (model.llm.config, model.config.llm_config):
        config.intermediate_size = size

    model.echo_config.architecture.intermediate_size = size
    model.config.echo = model.echo_config.to_dict(include_base=False)

    return model


# FFN pruning - temporary
@contextmanager
def shrunk_ffn(
    model: EchoModel,
    size: int,
    grams: List[torch.Tensor],
) -> Iterator[EchoModel]:
    """
    Makes the model behave as if 'shrink_ffn' had been applied, and puts the full FFNs back on the way out.
    """

    mlps = [layer.mlp for layer in model.llm.layers]
    backup = [(mlp.gate_proj, mlp.up_proj, mlp.down_proj, mlp.intermediate_size) for mlp in mlps]
    sizes = [config.intermediate_size for config in (model.llm.config, model.config.llm_config)]
    architecture, echo = model.echo_config.architecture.intermediate_size, model.config.echo

    try:
        yield shrink_ffn(model, size, grams)
    finally:
        for mlp, (gate, up, down, width) in zip(mlps, backup):
            mlp.gate_proj, mlp.up_proj, mlp.down_proj, mlp.intermediate_size = gate, up, down, width
        for config, width in zip((model.llm.config, model.config.llm_config), sizes):
            config.intermediate_size = width
        model.echo_config.architecture.intermediate_size, model.config.echo = architecture, echo


# Attention pruning - permanent
def drop_kv_groups(
    model: EchoModel,
    kv_heads: int,               # KV heads every layer keeps - the query heads sharing them go with them
    grams: List[torch.Tensor],   # [H*D, H*D] per layer: Gram of the o_proj inputs over calibration tokens (echo.pruning.calibration)
) -> EchoModel:
    """
    Removes whole KV groups - a KV head and the query heads that share it - so every KV head still serves as many
    query heads as before: grouped attention allows no other cut (16 query heads don't divide over 6 KV heads).

    Per layer it keeps the groups that let o_proj best make up for the rest, and refits o_proj by least squares,
    W' = W G[:, K] G[K, K]^-1 - the same refit as down_proj's in shrink_ffn. Without it the cut costs 6x as much.
    """

    config = model.llm.config
    current = config.num_key_value_heads
    head = getattr(config, "head_dim", None) or config.hidden_size // config.num_attention_heads
    per_group = config.num_attention_heads // current
    if not 0 < kv_heads <= current:
        raise ValueError(f"KV heads must lie in (0, {current}], got {kv_heads}")

    architecture = model.echo_config.architecture
    original = architecture.kept_kv_heads or [list(range(current)) for _ in model.llm.layers]
    kept = []

    for layer, gram, heads in zip(model.llm.layers, grams, original):
        groups = _keep_kv_groups(layer.self_attn, kv_heads, gram, head, per_group)
        kept.append([heads[group] for group in groups])

    for llm_config in (model.llm.config, model.config.llm_config):
        llm_config.num_attention_heads = kv_heads * per_group
        llm_config.num_key_value_heads = kv_heads
        # Stated outright: derived from the hidden size, it would change with the head count.
        llm_config.head_dim = head

    architecture.num_attention_heads, architecture.num_key_value_heads = kv_heads * per_group, kv_heads
    architecture.kept_kv_heads = kept
    model.config.echo = model.echo_config.to_dict(include_base=False)

    return model


# Vocabulary pruning - the rows a pruned text tokenizer keeps
def vocab_rows(
    tokenizer_json: dict,   # The tokenizer.json of the model, parsed
    used: Iterable[int],    # Token ids the text actually needs, e.g. every id the training transcripts tokenize to
) -> List[int]:
    """
    The byte tokens, the used ids closed under BPE merge ancestry, then every added (special) token - in their new order.

    A BPE merge's result is reachable only when both its parts are, so the closure tokenizes the text it was built
    from exactly as before; anything else falls back to smaller pieces, down to bytes.
    """

    model = tokenizer_json["model"]
    vocab = model["vocab"]
    parents = {}
    for merge in model["merges"]:
        left, right = merge if isinstance(merge, list) else merge.split(" ")
        parents[vocab[left + right]] = (vocab[left], vocab[right])

    keep: Set[int] = set(range(BYTE_TOKENS))
    stack = [token for token in used if token < len(vocab)]
    while stack:
        token = stack.pop()
        if token not in keep:
            keep.add(token)
            stack.extend(parents.get(token, ()))

    return sorted(keep) + sorted(entry["id"] for entry in tokenizer_json["added_tokens"])


# Vocabulary pruning - the tokenizer
def prune_tokenizer(
    tokenizer_json: dict,
    rows: List[int],   # From vocab_rows: old ids, in their new order
) -> dict:
    """
    The tokenizer.json of the pruned vocabulary: ids renumbered by their place in 'rows', merges whose result is gone dropped.
    """

    pruned = copy.deepcopy(tokenizer_json)
    model, new_id = pruned["model"], {old: index for index, old in enumerate(rows)}
    vocab = model["vocab"]

    model["merges"] = [
        merge for merge in model["merges"]
        if vocab["".join(merge if isinstance(merge, list) else merge.split(" "))] in new_id
    ]
    model["vocab"] = {token: new_id[old] for token, old in vocab.items() if old in new_id}
    for entry in pruned["added_tokens"]:
        entry["id"] = new_id[entry["id"]]

    return pruned


# Vocabulary pruning - permanent
def prune_vocab(
    model: EchoModel,
    rows: List[int],   # From vocab_rows: the text embedding rows kept, by old id, in their new order
) -> EchoModel:
    """
    Keeps only the given rows of the text embedding, renumbered by their place in 'rows' - lossless for any text the
    pruned tokenizer (prune_tokenizer) produces. The text embedding is untied: the audio heads are the only output.
    """

    embedding = model.llm.embed_tokens
    if not isinstance(embedding, nn.Embedding):
        raise ValueError("prune the vocabulary before factorizing the embedding")

    pruned = nn.Embedding(len(rows), embedding.embedding_dim, device=embedding.weight.device, dtype=embedding.weight.dtype)
    pruned.weight = nn.Parameter(embedding.weight.detach()[torch.tensor(rows, device=embedding.weight.device)].contiguous())
    model.llm.embed_tokens = pruned

    new_id = {old: index for index, old in enumerate(rows)}
    scopes = {id(scope): scope for scope in (model.config, model.config.llm_config, model.llm.config)}
    for scope in scopes.values():
        for key in ("bos_token_id", "eos_token_id", "pad_token_id"):
            if getattr(scope, key, None) in new_id:
                setattr(scope, key, new_id[getattr(scope, key)])
    for config in {id(c): c for c in (model.llm.config, model.config.llm_config)}.values():
        config.vocab_size = len(rows)

    model.echo_config.architecture.vocab_size = len(rows)
    model.config.echo = model.echo_config.to_dict(include_base=False)

    return model


# Embedding factorization - permanent
def factorize_embedding(
    model: EchoModel,
    rank: int,   # Rank of the factorization: the table becomes [V, rank], then a rank -> H projection
) -> EchoModel:
    """
    Replaces the text embedding by its best rank-r approximation (truncated SVD, the singular values split evenly
    between the two factors).
    """

    embedding = model.llm.embed_tokens
    if not isinstance(embedding, nn.Embedding):
        raise ValueError("the text embedding is already factorized")
    weight = embedding.weight.detach()
    if not 0 < rank <= min(weight.shape):
        raise ValueError(f"rank must lie in (0, {min(weight.shape)}], got {rank}")

    u, s, vh = torch.linalg.svd(weight.float(), full_matrices=False)
    root = s[:rank].sqrt()
    factorized = FactorizedEmbedding(*weight.shape, rank).to(device=weight.device, dtype=weight.dtype)
    factorized.table.weight = nn.Parameter((u[:, :rank] * root).to(weight.dtype).contiguous())
    factorized.project.weight = nn.Parameter((vh[:rank].T * root).to(weight.dtype).contiguous())
    model.llm.embed_tokens = factorized

    model.echo_config.architecture.embedding_rank = rank
    model.config.echo = model.echo_config.to_dict(include_base=False)

    return model


def _keep_kv_groups(attention: nn.Module, count: int, gram: torch.Tensor, head: int, per_group: int) -> List[int]:
    """
    Keeps the 'count' KV groups of one attention block whose o_proj refit leaves the least of its output unexplained.
    """

    weight = attention.o_proj.weight.detach().float()
    gram = gram.to(device=weight.device, dtype=torch.float32)
    groups = attention.k_proj.weight.size(0) // head
    total = (weight @ gram @ weight.T).trace()

    def dims(kept: Iterable[int], width: int) -> torch.Tensor:
        return torch.cat([torch.arange(group * width, (group + 1) * width, device=weight.device) for group in kept])

    def residual(kept: Tuple[int, ...]) -> float:
        # E||W a - W' a_K||^2, the part of the attention output the kept heads can't give back.
        keep = dims(kept, per_group * head)
        refit = _refit(weight, gram, keep)
        return float(total - 2 * (refit @ gram[keep] @ weight.T).trace() + (refit @ gram[keep][:, keep] @ refit.T).trace())

    kept = list(min(itertools.combinations(range(groups), count), key=residual))
    queries, keys = dims(kept, per_group * head), dims(kept, head)

    attention.q_proj = _linear(attention.q_proj.weight.detach()[queries])
    attention.k_proj = _linear(attention.k_proj.weight.detach()[keys])
    attention.v_proj = _linear(attention.v_proj.weight.detach()[keys])
    attention.o_proj = _linear(_refit(weight, gram, queries).to(attention.o_proj.weight.dtype))

    return kept


def _refit(weight: torch.Tensor, gram: torch.Tensor, keep: torch.Tensor) -> torch.Tensor:
    """
    The least-squares W' = W G[:, K] G[K, K]^-1: from the kept inputs alone, the closest to what all of them gave.
    """

    kept = gram[keep][:, keep]
    kept = kept + RIDGE * kept.diagonal().mean() * torch.eye(len(keep), device=kept.device)

    return torch.linalg.solve(kept, gram[keep] @ weight.T).T


def _shrink_mlp(mlp: nn.Module, size: int, gram: torch.Tensor) -> None:
    down = mlp.down_proj.weight
    weight = down.detach().float()
    gram = gram.to(device=down.device, dtype=torch.float32)

    energy = gram.diagonal() * weight.pow(2).sum(dim=0)
    keep = energy.argsort(descending=True)[:size].sort().values

    kept = gram[keep][:, keep]
    kept = kept + RIDGE * kept.diagonal().mean() * torch.eye(size, device=kept.device)
    refit = torch.linalg.solve(kept, gram[keep] @ weight.T).T

    # New modules rather than new weights, so the temporary variant can put the old ones back.
    mlp.gate_proj = _linear(mlp.gate_proj.weight.detach()[keep])
    mlp.up_proj = _linear(mlp.up_proj.weight.detach()[keep])
    mlp.down_proj = _linear(refit.to(down.dtype))
    mlp.intermediate_size = size


def _linear(weight: torch.Tensor) -> nn.Linear:
    """
    A bias-free linear layer holding 'weight', as every projection of a Qwen3 MLP is.
    """

    linear = nn.Linear(weight.size(1), weight.size(0), bias=False, device=weight.device, dtype=weight.dtype)
    linear.weight = nn.Parameter(weight.contiguous())

    return linear


def _survivors(
    model: EchoModel,
    indices: Iterable[int]  # Layers to be dropped (indicated by their current IDs, starting from 0 up to num_layers - 1)
) -> nn.ModuleList:
    """
    The model's layers, minus the ones asked for.
    """

    dropped: Set[int] = set(indices)
    depth = len(model.llm.layers)

    if any(not 0 <= index < depth for index in dropped):
        raise ValueError(f"layer indices must lie in [0, {depth}), got {sorted(dropped)}")

    return nn.ModuleList(
        [layer for index, layer in enumerate(model.llm.layers) if index not in dropped]
    )
