import logging
import torch

from contextlib import nullcontext
from dataclasses import dataclass, field
from pathlib import Path
from torch.nn.parallel import DistributedDataParallel
from typing import Any, ContextManager, Dict, Iterator, List, Optional

from echo.data.inputs import build_model_inputs
from echo.data.loader import EchoDataLoader
from echo.eval.single_step import evaluate_single_step
from echo.model.model import EchoModel
from echo.pruning.surgery import LayerFade
from echo.training.builder import build_optimizer, build_schedule
from echo.training.checkpoint import load_checkpoint, save_checkpoint
from echo.training.config import EchoTrainingConfig
from echo.training.meter import Meter
from echo.training.step import distillation_step
from echo.training.tracking import MLflowTracker
from echo.training.types import StepFn, Student
from echo.utils.distributed_utils import barrier, is_main, reduce_mean, world
from echo.utils.seed_utils import seed_everything

logger = logging.getLogger(__name__)


@dataclass
class Trainer:
    """
    The loop every stage shares.

    It's a complete training framework, parametrized only with a specific step function
    which allows it to cover different training scenarios (plain CE runs, distillations, CFG).
    """

    config: EchoTrainingConfig
    student: Student
    train_loader: EchoDataLoader
    loss_fn: StepFn = distillation_step
    teacher: Optional[EchoModel] = None
    dev_loader: Optional[EchoDataLoader] = None    # Eval only - training does not touch it
    test_loader: Optional[EchoDataLoader] = None   # Eval only - training does not touch it
    reference: Optional[EchoModel] = None          # Eval only - scored against in place of a teacher, when there is none
    tracker: Optional[MLflowTracker] = None        # Rank zero's only - the curves go to MLflow as well
    fade: Optional[LayerFade] = None               # A layer fading out over the run, attached before the wrapper

    step: int = field(init=False, default=0)
    epoch: int = field(init=False, default=0)

    # Set when the loader rolls over, and read by the next report: an epoch ends
    # in the middle of an optimizer step, and a checkpoint wants a finished one.
    checkpoint_requested: bool = field(init=False, default=False)

    def __post_init__(self) -> None:
        self.rank, self.world_size = world()

        # Every rank draws its own splits and masks; the weights do not need the
        # same treatment, since the wrapper below hands them out from rank zero.
        seed_everything(self.config.meta.seed + self.rank)

        if self.world_size > 1:
            self.student = DistributedDataParallel(
                self.student,
                device_ids=[self.rank],
                # The only buffer here is a table of codebook offsets, constant
                # and identical everywhere. Syncing it costs a collective per
                # forward and buys nothing.
                broadcast_buffers=False,
            )

        self.device = next(self.student.parameters()).device
        self.weights = torch.tensor(self.config.codebook_weights, device=self.device)

        # Only the backbone holds enough activations to be worth recomputing,
        # and it is the only part that takes the flag.
        if self.config.optim.gradient_checkpointing:
            _unwrap(self.student).llm.gradient_checkpointing_enable(
                {"use_reentrant": False}
            )

        # The length of the run, in this rank's steps. Every rank takes the same
        # number, and each one is a batch on every card.
        self.total_steps = self.config.total_steps(len(self.train_loader))

        self.optimizer = build_optimizer(self.student, self.config)
        self.schedule = build_schedule(self.optimizer, self.config, self.total_steps)

        # bf16 carries the range of fp32, so only fp16 needs its gradients scaled.
        self.autocast = self.device.type == "cuda"
        self.scaler = torch.amp.GradScaler(
            self.device.type, enabled=self.config.optim.mixed_precision == "fp16"
        )

        self.batches = self._batches()

        if self.config.checkpoint.resume is not None:
            resume_point = load_checkpoint(
                self.config.checkpoint.resume,
                _unwrap(self.student),
                self.optimizer,
                self.schedule,
            )
            self.step, self.epoch = resume_point["step"], resume_point["epoch"]
            self.train_loader.set_epoch(self.epoch)
            logger.info("Resumed at step %d of epoch %d", self.step, self.epoch)

        # Started once the step is known, so a resumed run's rate counts only
        # its own steps.
        self.meter = Meter(self.effective_step)

    def train(self, steps: Optional[int] = None) -> Dict[str, float]:
        """
        Runs for 'steps' optimizer steps, or to the end of the schedule.
        """

        last = self.step + (steps if steps is not None else self.total_steps - self.step)
        reported: Dict[str, float] = {}

        # A resumed run has taken steps already, so only a fresh one has a start to score.
        if self.config.evaluation.eval_at_start and self.step == 0:
            self._evaluate_all()

        self.student.train()

        while self.step < last:
            terms = self._accumulate()
            self._apply()
            # Every rank moves the fade on the same loss, so alpha stays the same everywhere.
            if self.fade is not None:
                self.fade.update(reduce_mean(float(terms["loss"])))
            self.step += 1

            self.meter.add(terms)
            reported = self._log_evaluate_save()

        return reported

    @property
    def effective_step(self) -> int:
        """
        Steps as work rather than as iterations.

        One optimizer step is a batch on every rank, so a run on four cards has
        four times the batches behind it - which is what makes a log line
        comparable between runs of different widths.
        """

        return self.step * self.world_size

    def _accumulate(self) -> Dict[str, torch.Tensor]:
        """
        One optimizer step's worth of micro batches, backward as they go.
        """

        accumulation = self.config.optim.gradient_accumulation_steps
        totals: Dict[str, torch.Tensor] = {}

        for micro in range(accumulation):
            batch = next(self.batches)
            inputs = build_model_inputs(batch, batch["mask"], batch["polish"], device=self.device)

            # The gradient exchange belongs to the backward, and the casting to
            # the forward alone.
            with self._syncing(micro == accumulation - 1):
                with torch.autocast(
                    self.device.type, dtype=self.config.dtype, enabled=self.autocast
                ):
                    terms = self.loss_fn(
                        self.student, self.teacher, inputs, self.config, self.weights
                    )

                self.scaler.scale(terms["loss"] / accumulation).backward()

            # Detached, so no graph is held, but left on the device: reading a
            # number back would make the CPU wait for the GPU every micro batch,
            # and nothing needs the value until a line is logged.
            for key, value in terms.items():
                held = value.detach() / accumulation
                totals[key] = held if key not in totals else totals[key] + held

        return totals

    def _syncing(self, last: bool) -> ContextManager[None]:
        """
        Whether this micro batch is the one that exchanges gradients.

        Every other one keeps them to itself, so the ranks talk once per
        optimizer step rather than once per micro batch.
        """

        if last or self.world_size == 1:
            return nullcontext()

        return self.student.no_sync()

    def _apply(self) -> None:
        """
        Clips what was accumulated, steps, and moves the rate along.
        """

        if self.config.optim.max_grad_norm > 0.0:
            self.scaler.unscale_(self.optimizer)
            torch.nn.utils.clip_grad_norm_(
                self.student.parameters(), self.config.optim.max_grad_norm
            )

        self.scaler.step(self.optimizer)
        self.scaler.update()
        self.optimizer.zero_grad(set_to_none=True)
        self.schedule.step()

    def _log_evaluate_save(self) -> Dict[str, float]:
        """
        Logs, evaluates and checkpoints, each on its own cadence.
        """

        evaluation = self.config.evaluation
        checkpoint = self.config.checkpoint
        reported: Dict[str, float] = {}

        if self.step % self.config.logging.log_every_steps == 0:
            reported = {
                key: reduce_mean(value)
                for key, value in self.meter.report(self.effective_step).items()
            }
            reported["learning_rate"] = self.schedule.get_last_lr()[0]
            if self.fade is not None:
                reported["fade_alpha"] = self.fade.alpha

            if is_main():
                logger.info(
                    "step %s/%s | epoch %s | %s",
                    f"{self.effective_step:,}",
                    f"{self.total_steps * self.world_size:,}",
                    self._epoch(),
                    _line(reported),
                )

                # MLFlow logging
                if self.tracker is not None:
                    self.tracker.log_train({**reported, "epoch": self.epoch + 1}, self.effective_step)

        if self.step % evaluation.eval_every_steps == 0:
            self._evaluate_all()

        # The checkpoint is one rank's work, and the others wait rather than run
        # into the next exchange without it.
        scheduled = checkpoint.save_every_steps > 0 and self.step % checkpoint.save_every_steps == 0

        if scheduled or self.checkpoint_requested:
            self.checkpoint_requested = False

            if is_main():
                self.save()

            barrier()

        return reported

    def _evaluate_all(self) -> None:
        """
        Evaluates on every enabled held out set.

        The evaluations are one rank's work, and the others wait rather than
        run into the next exchange without it.
        """

        evaluation = self.config.evaluation

        if is_main():
            self._evaluate(self.dev_loader if evaluation.eval_on_dev else None, "dev")
            self._evaluate(self.test_loader if evaluation.eval_on_test else None, "test")

        barrier()

    def _evaluate(self, loader: Optional[EchoDataLoader], name: str) -> None:
        """
        Scores a single step over one of the held out sets, if there is one.
        """

        if loader is None:
            return

        # The model itself, never the wrapper: a forward through it carries
        # collectives of its own - the buffers it keeps in step, the gradients
        # it expects - and one rank entering those alone waits for three that
        # never arrive.
        model = _unwrap(self.student)

        logger.info("step %d | %s", self.step, name)
        # With no teacher, the evaluation's comparison columns are taken against the reference instead.
        against = self.teacher if self.teacher is not None else self.reference

        results = evaluate_single_step(
            model,
            against,
            loader,
            max_batches=self.config.evaluation.eval_batches,
            per_language=True,
            polish_share=self.config.masking.polish_share,
            polish_weights=self.config.masking.polish_weights,
            # A CFG-distilling stage is scored against the mixture at its final guidance, wherever the ramp stands.
            guidance_scale=self.config.guidance.scale if self.config.guiding else None,
        )

        # MLFlow logging
        if self.tracker is not None:
            self.tracker.log_eval(name, results, self.effective_step, against is not None)

        model.train()

    def save(self) -> Path:
        """
        Writes a checkpoint of where the run stands.
        """

        model = _unwrap(self.student)

        return save_checkpoint(
            self.config.checkpoint.output_dir,
            model,
            self.optimizer,
            self.schedule,
            step=self.step,
            epoch=self.epoch,
            keep_last_n=self.config.checkpoint.keep_last_n_checkpoints,
            tokenizer_from=model.echo_config.init_from,
        )

    def _epoch(self) -> str:
        """
        Which epoch the run is in, out of how many when that is known.
        """

        epochs = self.config.schedule.epochs

        return f"{self.epoch + 1}/{epochs}" if epochs else f"{self.epoch + 1}"

    def _batches(self) -> Iterator[Dict[str, List[Any]]]:
        """
        The training data, over and over, in a fresh order every epoch.
        """

        while True:
            if is_main():
                logger.info(
                    "epoch %s begins - %s batches on each of %d rank(s)",
                    self._epoch(),
                    f"{len(self.train_loader):,}",
                    self.world_size,
                )

            yield from self.train_loader

            if is_main():
                logger.info("epoch %s ends at step %s", self._epoch(), f"{self.effective_step:,}")

            self.epoch += 1
            self.checkpoint_requested = self.config.checkpoint.save_at_epoch_end
            self.train_loader.set_epoch(self.epoch)


def _line(reported: Dict[str, float]) -> str:
    return "  ".join(f"{key} {value:.4g}" for key, value in reported.items())


def _unwrap(model: Student) -> EchoModel:
    return getattr(model, "module", model)
