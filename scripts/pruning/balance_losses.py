"""
Suggests a KL weight for a mixed CE + KL run, from how the two gradients compare on a CE-trained student.

CE stays the main loss: the suggested weight makes the KL gradient a chosen fraction of the CE gradient,
so KL only steers where CE has little left to say.
"""

import argparse
from pathlib import Path
from typing import List

import torch

from echo import EchoConfig, build_model, freeze
from echo.data.dataset import EchoDataset
from echo.data.inputs import build_model_inputs
from echo.data.loader import EchoDataLoader
from echo.data.preprocessor import EchoPreprocessor
from echo.training.config import EchoTrainingConfig
from echo.training.step import distillation_step
from echo.utils.distributed_utils import setup

CONFIG = "echo/configs/training/train.json"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=CONFIG, help="The mixed run's .json config")
    parser.add_argument("--init", type=Path, required=True, help="The CE-trained student checkpoint")
    parser.add_argument("--batches", type=int, default=4, help="Training batches the gradients are measured on")
    parser.add_argument("--fraction", type=float, default=0.2, help="Wanted size of the KL gradient, relative to the CE one")
    args = parser.parse_args()

    config = EchoTrainingConfig.from_json(args.config)
    if not config.distilling:
        parser.error("the config needs a teacher and a positive loss kl to measure the KL term")

    device = setup()
    student_config = EchoConfig.from_json(config.models.student)
    student_config.init_from = str(args.init)
    student = build_model(student_config, train=True, device=device)
    if config.optim.gradient_checkpointing:
        student.llm.gradient_checkpointing_enable({"use_reentrant": False})
    teacher = freeze(build_model(config.models.teacher, train=True, device=device))
    weights = torch.tensor(student.normalized_audio_codebook_weights, device=device)
    params = [p for p in student.parameters() if p.requires_grad]

    # Training batches exactly as training sees them: masked, with the unconditional share dropped in.
    loader = EchoDataLoader(
        EchoDataset(config.data.train_manifest, **config.dataset_kwargs),
        EchoPreprocessor(**config.preprocessor_kwargs),
        num_workers=0,
        **config.sampler_kwargs,
    )

    ce_norms: List[float] = []
    kl_norms: List[float] = []
    cosines: List[float] = []

    for index, batch in enumerate(loader):
        if index >= args.batches:
            break

        inputs = build_model_inputs(batch, batch["mask"], batch["polish"], device=device)
        terms = distillation_step(student, teacher, inputs, config, weights)
        ce = torch.autograd.grad(terms["ce"], params, retain_graph=True, allow_unused=True)
        kl = torch.autograd.grad(terms["kl"], params, allow_unused=True)

        pairs = [(a, b) for a, b in zip(ce, kl) if a is not None and b is not None]
        dot = sum(float((a * b).sum()) for a, b in pairs)
        ce_norm = sum(float(a.pow(2).sum()) for a, _ in pairs) ** 0.5
        kl_norm = sum(float(b.pow(2).sum()) for _, b in pairs) ** 0.5

        ce_norms.append(ce_norm)
        kl_norms.append(kl_norm)
        cosines.append(dot / max(ce_norm * kl_norm, 1e-12))
        print(f"batch {index}: CE {float(terms['ce'].detach()):.4f}  KL {float(terms['kl'].detach()):.4f}  "
              f"|grad CE| {ce_norm:.4f}  |grad KL| {kl_norm:.4f}  cos {cosines[-1]:+.3f}")

        # Two full sets of gradients and a graph: gone before the next batch builds its own.
        del terms, ce, kl, pairs

    ce_mean, kl_mean = sum(ce_norms) / len(ce_norms), sum(kl_norms) / len(kl_norms)
    suggested = args.fraction * config.loss.main.ce * ce_mean / max(kl_mean, 1e-12)
    print(f"\nmean |grad CE| {ce_mean:.4f}  |grad KL| {kl_mean:.4f}  cos {sum(cosines) / len(cosines):+.3f}")
    print(f"suggested loss.main.kl: {suggested:.4g} (with loss.main.ce {config.loss.main.ce}, "
          f"KL gradient at {args.fraction:.0%} of CE's)")


if __name__ == "__main__":
    main()
