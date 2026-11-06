"""Applies one pruning step - the FFN width, the KV groups, one layer, the vocabulary or the embedding's rank - and writes
the pruned checkpoint with a report.

With --choose-only, picks the layer and writes the report alone, leaving the removal to a fade (stage_2_pruning.py --fade-layer).
"""

import argparse
import csv
import json
import shutil
from pathlib import Path
from typing import Dict, Iterator, List, Tuple

import torch
from transformers import AutoTokenizer

from echo.model.model import EchoModel
from echo.pruning.calibration import (
    CalibrationInput,
    branch_kl,
    calibration_inputs,
    calibration_loader,
    attention_grams,
    ffn_grams,
    load_student,
    load_teacher,
)
from echo.pruning.search import layer_costs, next_layer
from echo.data.constants import PROMPT_TEMPLATE, TEXT_TEMPLATE
from echo.pruning.surgery import drop_kv_groups, drop_layers, factorize_embedding, prune_tokenizer, prune_vocab, shrink_ffn, vocab_rows
from echo.training.checkpoint import TOKENIZER_FILES
from echo.training.config import EchoTrainingConfig
from echo.utils.distributed_utils import setup
from inspect_depth import print_choice

CONFIG = "echo/configs/training/train.json"
REPORT = "pruning_report.json"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=CONFIG, help="Pruning stage's .json config")
    parser.add_argument("--checkpoint", type=Path, required=True, help="The model to prune")
    parser.add_argument("--output", type=Path, required=True, help="Directory the pruned checkpoint is written to")
    step = parser.add_mutually_exclusive_group(required=True)
    step.add_argument("--ffn", action="store_true", help="Shrinks every FFN to the config's pruning.ffn_size")
    step.add_argument("--kv-groups", action="store_true", help="Drops KV groups down to the config's pruning.kv_heads per layer")
    step.add_argument("--drop-layer", help="'auto' for the cheapest layer within the limits, or a layer index")
    step.add_argument("--vocab", action="store_true", help="Keeps the text tokens the training transcripts need, and rewrites the tokenizer")
    step.add_argument("--embedding", action="store_true", help="Factorizes the text embedding to the config's pruning.embedding_rank")
    parser.add_argument("--choose-only", action="store_true", help="Writes only the report of the chosen layer, for a fade to remove it")
    args = parser.parse_args()

    config = EchoTrainingConfig.from_json(args.config)
    if args.vocab:
        # The checkpoint's own tokenizer: the stage's config already names the pruned one this step writes.
        config.preprocessing.tokenizer = str(args.checkpoint)
    device = setup()
    model = load_student(config, args.checkpoint, device)
    # Every measurement is taken on a scoring set kept apart from the one the FFN refit is fitted on.
    loader, batches = calibration_loader(config), config.pruning.calibration_batches
    inputs = calibration_inputs(loader, batches, device, skip=batches)
    if args.vocab:
        prune_vocabulary(config, args, model, inputs)
        return
    teacher = load_teacher(config, device)
    before = branch_kl(model, teacher, inputs)
    if args.ffn:
        size = config.pruning.ffn_size
        if size is None:
            parser.error("--ffn needs pruning.ffn_size in the config")
        report = {"operation": "ffn", "from": model.llm.config.intermediate_size, "to": size}
        shrink_ffn(model, size, ffn_grams(model, calibration_inputs(loader, batches, device)))
    elif args.kv_groups:
        heads = config.pruning.kv_heads
        if heads is None:
            parser.error("--kv-groups needs pruning.kv_heads in the config")
        report = {"operation": "kv_groups", "from": model.llm.config.num_key_value_heads, "to": heads}
        drop_kv_groups(model, heads, attention_grams(model, calibration_inputs(loader, batches, device)))
        report["kept_kv_heads"] = model.echo_config.architecture.kept_kv_heads
    elif args.embedding:
        rank = config.pruning.embedding_rank
        if rank is None:
            parser.error("--embedding needs pruning.embedding_rank in the config")
        report = {"operation": "embedding", "from": model.llm.config.hidden_size, "to": rank}
        factorize_embedding(model, rank)
    else:
        costs = layer_costs(model, teacher, inputs, config.pruning.protected_layers)
        if args.drop_layer != "auto" and int(args.drop_layer) not in costs:
            parser.error(f"layer {args.drop_layer} is protected or outside the model")
        candidates = costs if args.drop_layer == "auto" else {int(args.drop_layer): costs[int(args.drop_layer)]}
        choice = next_layer(candidates, before)

        kept = model.echo_config.architecture.kept_layers or list(range(len(model.llm.layers)))
        print_choice(choice, kept)

        report = {"operation": "drop_layer", "layer": choice.layer, "original_layer": kept[choice.layer], "costs": costs}
        if args.choose_only:
            args.output.mkdir(parents=True, exist_ok=True)
            (args.output / REPORT).write_text(json.dumps({"source": str(args.checkpoint), **report}, indent=2))
            print(f"chose layer {choice.layer}, report written to {args.output}")
            return
        drop_layers(model, [choice.layer])

    after = branch_kl(model, teacher, inputs)
    report["kl"] = after
    report["dkl"] = {key: after[key] - before[key] for key in after}
    print(f"{report['operation']}: dKL cond {report['dkl']['cond']:+.4f}  uncond {report['dkl']['uncond']:+.4f}")

    args.output.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(args.output)
    for name in TOKENIZER_FILES:
        if (args.checkpoint / name).is_file():
            shutil.copy2(args.checkpoint / name, args.output / name)
    (args.output / REPORT).write_text(json.dumps({"source": str(args.checkpoint), **report}, indent=2))
    print(f"written to {args.output}")


def prune_vocabulary(config: EchoTrainingConfig, args: argparse.Namespace, model: EchoModel, inputs: List[CalibrationInput]) -> None:
    """
    Keeps the text tokens the training transcripts tokenize to (and their BPE ancestry, the bytes and every special
    token), rewrites the tokenizer to match, and checks both: every transcript tokenizes the same after the id
    remap, and the logits are unchanged on remapped inputs.
    """

    source = AutoTokenizer.from_pretrained(args.checkpoint)
    with open(config.data.train_manifest, newline="") as f:
        rows = list(csv.DictReader(f, delimiter="|", quoting=csv.QUOTE_NONE))
    texts = [TEXT_TEMPLATE.format(text=row["transcript"]) for row in rows]
    languages = sorted({row["language"] for row in rows} | {"None"})
    prompts = [denoise + PROMPT_TEMPLATE.format(language=language) for language in languages for denoise in ("", "<|denoise|>")]
    texts += prompts
    print(f"tokenizing {len(texts):,} transcripts", flush=True)
    used = {token for chunk in _chunks(texts) for ids in source(chunk).input_ids for token in ids}

    tokenizer_json = json.loads((args.checkpoint / "tokenizer.json").read_text())
    kept = vocab_rows(tokenizer_json, used)
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "tokenizer.json").write_text(json.dumps(prune_tokenizer(tokenizer_json, kept), ensure_ascii=False))
    for name in TOKENIZER_FILES:
        if name != "tokenizer.json" and (args.checkpoint / name).is_file():
            shutil.copy2(args.checkpoint / name, args.output / name)

    new_id, pruned = {old: index for index, old in enumerate(kept)}, AutoTokenizer.from_pretrained(args.output)
    changed = sum(
        new != [new_id[token] for token in old]
        for chunk in _chunks(texts)
        for old, new in zip(source(chunk).input_ids, pruned(chunk).input_ids)
    )
    print(f"vocabulary {len(tokenizer_json['model']['vocab']) + len(tokenizer_json['added_tokens']):,} -> {len(kept):,}; "
          f"texts tokenized differently: {changed}", flush=True)
    if changed:
        raise SystemExit("the pruned tokenizer changes the tokenization of the training transcripts")

    # Dev rows whose text holds a dropped token would tokenize differently, so only the fully kept ones are compared.
    remap = torch.full((model.llm.embed_tokens.num_embeddings,), -1, dtype=torch.long)
    remap[torch.tensor(kept)] = torch.arange(len(kept))
    remapped = [_remapped(item.cond, remap) for item in inputs]
    with torch.no_grad():
        before = [model(**{k: v for k, v in item.cond.items() if k != "labels"}).logits for item in inputs]
        prune_vocab(model, kept)
        after = [model(**cond).logits for cond, _ in remapped]
    gaps = [float((a[rows] - b[rows]).abs().max()) for a, b, (_, rows) in zip(after, before, remapped) if rows.any()]
    checked = sum(int(rows.sum()) for _, rows in remapped)
    gap = max(gaps, default=float("nan"))
    print(f"largest logit change on remapped inputs: {gap:.2e} ({checked} rows checked)", flush=True)

    model.save_pretrained(args.output)
    report = {"source": str(args.checkpoint), "operation": "vocab", "from": len(remap), "to": len(kept), "logit_gap": gap}
    (args.output / REPORT).write_text(json.dumps(report, indent=2))
    print(f"written to {args.output}")


def _chunks(texts: List[str], size: int = 50_000) -> Iterator[List[str]]:
    for start in range(0, len(texts), size):
        yield texts[start:start + size]


def _remapped(inputs: Dict[str, torch.Tensor], remap: torch.Tensor) -> Tuple[Dict[str, torch.Tensor], torch.Tensor]:
    """
    The forward inputs with the text ids renumbered by 'remap' (audio positions left as they are), and the rows
    whose text holds only kept tokens.
    """

    inputs = {key: value for key, value in inputs.items() if key != "labels"}
    ids, text = inputs["input_ids"], ~inputs["audio_mask"].unsqueeze(1).expand_as(inputs["input_ids"])
    mapped = remap.to(ids.device)[ids.clamp(min=0, max=len(remap) - 1)]
    rows = ~(text & (mapped < 0)).flatten(1).any(1)
    inputs["input_ids"] = torch.where(text, mapped.clamp(min=0), ids)
    return inputs, rows


if __name__ == "__main__":
    main()
