import torch

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

from echo.data.constants import DEFAULT_TOKENIZER
from echo.data.sampler import BatchPacking
from echo.eval.constants import PREFIX_RATIO
from echo.masking.audio_masking import codebook_weights
from echo.masking.constants import (
    DEFAULT_MIN_PASS_SHARE,
    DEFAULT_POLISH_SHARE,
    DEFAULT_POLISH_WEIGHTS,
    DEFAULT_TAIL_MASKED,
    MAIN_PATH_SAMPLERS,
)
from echo.training.meta import MIXED_PRECISION_DTYPES
from echo.training.types import AnnealShape, LRSchedulerType, MixedPrecision
from echo.utils.config_utils import (
    build_config,
    check,
    check_choice,
    check_range,
    read_json,
    write_json,
)


# Subconfig - training stage metadata
@dataclass
class EchoTrainingMeta:
    version: int = 1
    name: str = "training"
    notes: str = ""
    seed: int = 1234   # Seeds the training run itself - the data order has its own in 'data' 


# Subconfig - the models a stage trains, and distils from
@dataclass
class ModelsConfig:
    student: Optional[str] = None   # Path to the student's model config
    teacher: Optional[str] = None   # Path to the teacher's model config, None disables distillation


# Subconfig - manifests and batching
@dataclass
class DataConfig:
    train_manifest: Optional[str] = None
    dev_manifest: Optional[str] = None
    test_manifest: Optional[str] = None
    data_root: Optional[str] = None   # Root the codec paths are relative to, defaults to each manifest's directory
    max_batch_size: int = 64
    max_batch_tokens: int = 8192
    max_pad_ratio: float = 0.3                 # Only for "greedy" packing
    packing: BatchPacking = "greedy"           # "sorted": length-sorted chunks, ~4% padding instead of ~31% - more labels per step
    packing_chunk: int = 2048                  # Utterances sorted together, only for "sorted" packing
    num_workers: int = 4
    seed: int = 42


# Subconfig - the prefix / target split drawn per sample
@dataclass
class PrefixSplitConfig:
    prefix_min_dur: float = 1.0
    prefix_min_ratio: float = 0.0
    prefix_max_ratio: float = 0.5
    prefix_clamp: bool = False   # True: clamp short draws up to prefix_min_dur, instead of drawing above it


# Subconfig - turning raw samples into model inputs
@dataclass
class PreprocessingConfig:
    tokenizer: str = DEFAULT_TOKENIZER
    drop_prefix_ratio: float = 0.1   # Share of a batch trained without any prefix
    prefix_split: PrefixSplitConfig = field(default_factory=PrefixSplitConfig)


# Subconfig - masking, replaying the states the two-stage sampler visits
@dataclass
class MaskingConfig:
    polish_share: float = DEFAULT_POLISH_SHARE   # Share of samples at the polish state, scored on codebooks 2-7; the rest are main-path, scored on 0-1
    polish_weights: Tuple[float, ...] = DEFAULT_POLISH_WEIGHTS   # Loss weights of codebooks 2-7 within the polish samples
    stage_shares: Optional[Tuple[float, ...]] = None   # The v0.0.2 schedule: shares of the (early, middle, penultimate, final, tail) main-path draws; None replays the traced samplers below
    tail_masked: float = DEFAULT_TAIL_MASKED   # Least masked fraction of codebook 0 the tail reaches (v0.0.2 schedule)
    samplers: Optional[Dict[str, float]] = None   # Weights of the traced deployed samplers (MAIN_PATH_SAMPLERS) whose passes are replayed; None weighs all equally
    min_pass_share: float = DEFAULT_MIN_PASS_SHARE   # Least probability of any pass of a replayed sampler; the rest follow its decision shares


# Subconfig - the loss terms of one sample path
@dataclass
class PathLossConfig:
    ce: float = 0.0   # Cross entropy against the true tokens, 0 disables it
    kl: float = 1.0   # KL against the teacher, 0 disables it

    # Share of the scored tokens the CE is taken on: those where the student trails the teacher most.
    # 1.0 keeps every token; 0.5 keeps the worse half. Below 1.0 it needs the teacher as the reference.
    ce_selection: float = 1.0


# Subconfig - what the student is trained against
# Each sample path states its own terms:
# - main: the main-path samples, scored on codebooks 0-1
# - polish: the polish samples, scored on codebooks 2-7
# Examples:
# - KL-only:            main {ce 0, kl 1}, polish {ce 0, kl 1}
# - CE main, CE+KL pol: main {ce 1, kl 0}, polish {ce 1, kl 1, ce_selection 0.5}
@dataclass
class LossConfig:
    main: PathLossConfig = field(default_factory=PathLossConfig)
    polish: PathLossConfig = field(default_factory=PathLossConfig)

    # Divides the logits of both models before the KL's softmax, and the loss is
    # multiplied back by T^2 so its gradients keep their size.
    # At 1.0 the student copies the teacher's distribution as it stands. This is
    # also the only value a CFG teacher allows, since its target is already a
    # normalized log-prob vector - dividing it would rescale a distribution.
    # Above 1.0 (1.5 to 2.0 in practice) both sides flatten, so the student also
    # learns how the teacher ranks the tokens it finds unlikely instead of just
    # its top choice. Worth reaching for when the teacher is very confident and
    # the student's KL stops falling.
    temperature: float = 1.0

    # KL(student || teacher) instead of KL(teacher || student): mode-seeking, so a student that can't match the
    # teacher everywhere keeps its peaks sharp rather than spreading out to cover the teacher's tail.
    reverse_kl: bool = False

    @property
    def paths(self) -> Tuple[PathLossConfig, PathLossConfig]:
        return self.main, self.polish


# Subconfig - optimizer
@dataclass
class OptimConfig:
    learning_rate: float = 3e-5
    weight_decay: float = 0.01
    max_grad_norm: float = 1.0
    gradient_accumulation_steps: int = 1
    mixed_precision: MixedPrecision = "bf16"

    # Recomputes the backbone's activations instead of holding them, which costs
    # around a third of the step's time and takes most of the memory a batch
    # needs - the difference between a batch of 8k tokens and one of 32k.
    gradient_checkpointing: bool = False


# Subconfig - how long training runs, and at what rate
@dataclass
class ScheduleConfig:
    steps: Optional[int] = None    # Exactly one of steps / epochs is set
    epochs: Optional[int] = None
    lr_scheduler_type: LRSchedulerType = "cosine"
    warmup_ratio: float = 0.03
    min_lr_ratio: float = 0.1   # Share of the peak rate the decay ends at. For example: with lrearning_rate=3e-5 and min_lr_ratio=0.1, we end up with lr=3e-6 at the end.
    hold_ratio: float = 0.0     # Share of the run the peak rate is held for after the warmup - the decay spans the rest
    warmup_start_ratio: float = 0.0   # Share of the peak the warmup climbs from, after a short ramp up to it from zero


# Subconfig - CFG distillation: the student learns, in one pass, the guided mixture its teacher makes of two
@dataclass
class GuidanceConfig:
    scale: float = 0.0                   # The guidance w the mixture is distilled at, 0 disables CFG distillation
    anneal_shape: AnnealShape = "power"  # How w rises from 0 to 'scale': at once ('constant'), linearly, or as a power
    anneal_ratio: float = 0.25           # Share of the run the rise takes - w stays at 'scale' afterwards
    anneal_power: float = 0.5            # Exponent of the 'power' rise. The target moves away from the conditional
                                         # distribution as ~w^2, so 0.5 makes it move at an even pace

    def at(self, step: int, total_steps: int) -> float:
        """
        The guidance w of the given step.
        """

        if self.anneal_shape == "constant":
            return self.scale

        progress = min(1.0, step / max(self.anneal_ratio * total_steps, 1.0))

        return self.scale * (progress if self.anneal_shape == "linear" else progress ** self.anneal_power)


# Subconfig - logging
@dataclass
class LoggingConfig:
    log_every_steps: int = 100


# Subconfig - evaluation
@dataclass
class EvaluationConfig:
    eval_every_steps: int = 1000
    eval_batches: int = 50
    eval_at_start: bool = True     # Also evaluates before the first step, to see the loss training starts from
    eval_on_dev: bool = True       # Sweeps a single step over the dev set
    eval_on_test: bool = True      # And over the test set, which no training ever sees
    prefix_ratio: float = PREFIX_RATIO   # Pins the split an evaluation's preprocessor draws, so two checkpoints are scored alike


# Subconfig - checkpoints
@dataclass
class CheckpointConfig:
    output_dir: str = "runs/echo"
    resume: Optional[str] = None   # Checkpoint the run continues from, None starts a fresh one
    save_every_steps: int = 5000   # 0 leaves the cadence to 'save_at_epoch_end' alone
    save_at_epoch_end: bool = False   # A checkpoint whenever the data has been seen through once
    keep_last_n_checkpoints: int = 3


# Subconfig - the pruning phase: what the next step removes, and how much damage it may do
# Damage is the conditional branch's KL(teacher || model) (echo.pruning.search).
@dataclass
class PruningConfig:
    reference: Optional[str] = None       # Model config the evaluations of a run without a teacher compare against
    calibration_batches: int = 4          # Dev batches the FFN statistics are fitted on - the next as many score every decision
    ffn_size: Optional[int] = None        # FFN neurons per layer the width step leaves
    kv_heads: Optional[int] = None        # KV heads per layer the attention step leaves, each with its query heads
    embedding_rank: Optional[int] = None  # Rank the text embedding is factorized to
    protected_layers: List[int] = field(default_factory=lambda: [0, -1])   # Never removed - negative indices count from the end
    layers_to_prune: int = 0              # Layers the depth loop removes, the cheapest each round
    fade_steps: int = 0                   # Steps a layer fades out over before its removal (echo.pruning.surgery.LayerFade)
    fade_power: float = 2.0               # Exponent of the removal's damage curve, (1 - alpha)^power, the fade evens out
    fade_budget: Optional[float] = None   # Loss a fade may add over its starting level before it is held - None never holds


# Subconfig - the MLflow run the training reports to
@dataclass
class TrackingConfig:
    enabled: bool = False
    tracking_uri: str = "https://mlflow.swmansion.com"
    experiment_id: str = "3"
    username: Optional[str] = None
    password: Optional[str] = None


@dataclass
class EchoTrainingConfig:
    """
    Everything one training stage needs, loaded from a single .json file.
    """

    meta: EchoTrainingMeta = field(default_factory=EchoTrainingMeta)
    models: ModelsConfig = field(default_factory=ModelsConfig)
    data: DataConfig = field(default_factory=DataConfig)
    preprocessing: PreprocessingConfig = field(default_factory=PreprocessingConfig)
    masking: MaskingConfig = field(default_factory=MaskingConfig)
    loss: LossConfig = field(default_factory=LossConfig)
    optim: OptimConfig = field(default_factory=OptimConfig)
    schedule: ScheduleConfig = field(default_factory=ScheduleConfig)
    guidance: GuidanceConfig = field(default_factory=GuidanceConfig)
    logging: LoggingConfig = field(default_factory=LoggingConfig)
    evaluation: EvaluationConfig = field(default_factory=EvaluationConfig)
    checkpoint: CheckpointConfig = field(default_factory=CheckpointConfig)
    tracking: TrackingConfig = field(default_factory=TrackingConfig)
    pruning: PruningConfig = field(default_factory=PruningConfig)

    def __post_init__(self) -> None:
        self.validate()

    @classmethod
    def from_dict(cls, data: Dict[str, Any], *, strict: bool = False) -> "EchoTrainingConfig":
        return build_config(cls, data, strict=strict)

    @classmethod
    def from_json(
        cls, json_path: Union[str, Path], *, strict: bool = False
    ) -> "EchoTrainingConfig":
        return cls.from_dict(read_json(json_path), strict=strict)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    def to_json(self, json_path: Union[str, Path]) -> None:
        write_json(json_path, self.to_dict())

    def validate(self) -> None:
        check(self.data.train_manifest is not None, "data.train_manifest is required")
        check_range("data.max_pad_ratio", self.data.max_pad_ratio, 0, 1, closed=False)

        split = self.preprocessing.prefix_split
        check(
            split.prefix_min_dur >= 0.0,
            "preprocessing.prefix_split.prefix_min_dur must not be negative, "
            f"got {split.prefix_min_dur}",
        )
        check_range(
            "preprocessing.prefix_split.prefix_min_ratio", split.prefix_min_ratio, 0, 1
        )
        check_range(
            "preprocessing.prefix_split.prefix_max_ratio", split.prefix_max_ratio, 0, 1
        )
        check(
            split.prefix_min_ratio <= split.prefix_max_ratio,
            f"preprocessing.prefix_split.prefix_min_ratio ({split.prefix_min_ratio}) "
            f"exceeds prefix_max_ratio ({split.prefix_max_ratio})",
        )
        check_range(
            "preprocessing.drop_prefix_ratio",
            self.preprocessing.drop_prefix_ratio,
            0,
            1,
        )

        masking = self.masking
        check_range("masking.polish_share", masking.polish_share, 0, 1)
        check(
            len(masking.polish_weights) == len(DEFAULT_POLISH_WEIGHTS) and all(w > 0.0 for w in masking.polish_weights),
            f"masking.polish_weights needs {len(DEFAULT_POLISH_WEIGHTS)} positive weights, got {masking.polish_weights}",
        )
        check(
            masking.stage_shares is None
            or (
                len(masking.stage_shares) == 5
                and all(share >= 0.0 for share in masking.stage_shares)
                and abs(sum(masking.stage_shares) - 1.0) < 1e-6
            ),
            f"masking.stage_shares needs 5 non-negative shares summing to 1, got {masking.stage_shares}",
        )
        check_range("masking.tail_masked", masking.tail_masked, 0, 1, closed=False)
        check(
            masking.samplers is None
            or (
                set(masking.samplers) <= set(MAIN_PATH_SAMPLERS)
                and all(w >= 0.0 for w in masking.samplers.values())
                and sum(masking.samplers.values()) > 0.0
            ),
            f"masking.samplers needs non-negative weights of {sorted(MAIN_PATH_SAMPLERS)}, got {masking.samplers}",
        )
        check(
            0.0 <= masking.min_pass_share * max(len(v["decisions"]) for v in MAIN_PATH_SAMPLERS.values()) < 1.0,
            f"masking.min_pass_share has to leave the passes a share above it, got {masking.min_pass_share}",
        )

        loss = self.loss
        for name, path in (("main", loss.main), ("polish", loss.polish)):
            check(path.ce >= 0.0 and path.kl >= 0.0, f"loss.{name} weights must not be negative, got ce {path.ce}, kl {path.kl}")
            check(0.0 < path.ce_selection <= 1.0, f"loss.{name}.ce_selection must lie in (0, 1], got {path.ce_selection}")
            if path.ce_selection < 1.0:
                check(path.ce > 0.0, f"loss.{name}.ce_selection selects the tokens of a CE term: loss.{name}.ce must be positive")
                check(self.models.teacher is not None, f"loss.{name}.ce_selection needs models.teacher to select tokens against")
        check(any(path.ce > 0.0 or path.kl > 0.0 for path in loss.paths), "loss needs a positive weight on at least one term")
        check(
            all(path.kl <= 0.0 for path in loss.paths) or self.models.teacher is not None,
            "a positive loss kl needs models.teacher to distil from",
        )
        check(loss.temperature > 0.0, f"loss.temperature must be positive, got {loss.temperature}")

        check_choice("data.packing", self.data.packing, BatchPacking)
        check(
            self.pruning.layers_to_prune >= 0,
            f"pruning.layers_to_prune must not be negative, got {self.pruning.layers_to_prune}",
        )
        check(self.pruning.kv_heads is None or self.pruning.kv_heads > 0, f"pruning.kv_heads must be positive, got {self.pruning.kv_heads}")
        check(self.pruning.embedding_rank is None or self.pruning.embedding_rank > 0, f"pruning.embedding_rank must be positive, got {self.pruning.embedding_rank}")
        check(self.pruning.fade_steps >= 0, f"pruning.fade_steps must not be negative, got {self.pruning.fade_steps}")
        check(self.pruning.fade_power > 0.0, f"pruning.fade_power must be positive, got {self.pruning.fade_power}")
        check(self.pruning.fade_budget is None or self.pruning.fade_budget > 0.0, f"pruning.fade_budget must be positive or null, got {self.pruning.fade_budget}")
        check(self.data.packing_chunk > 0, f"data.packing_chunk must be positive, got {self.data.packing_chunk}")

        check_choice(
            "optim.mixed_precision", self.optim.mixed_precision, MixedPrecision
        )

        check(
            (self.schedule.steps is None) != (self.schedule.epochs is None),
            "schedule needs exactly one of 'steps' or 'epochs'",
        )
        check_choice(
            "schedule.lr_scheduler_type",
            self.schedule.lr_scheduler_type,
            LRSchedulerType,
        )
        check_range("schedule.min_lr_ratio", self.schedule.min_lr_ratio, 0, 1)
        check(0.0 <= self.schedule.hold_ratio < 1.0, f"schedule.hold_ratio must lie in [0, 1), got {self.schedule.hold_ratio}")
        check(0.0 <= self.schedule.warmup_start_ratio < 1.0, f"schedule.warmup_start_ratio must lie in [0, 1), got {self.schedule.warmup_start_ratio}")

        guidance = self.guidance
        check(guidance.scale >= 0.0, f"guidance.scale must not be negative, got {guidance.scale}")
        check_choice("guidance.anneal_shape", guidance.anneal_shape, AnnealShape)
        check(
            guidance.anneal_shape == "constant" or 0.0 < guidance.anneal_ratio <= 1.0,
            f"guidance.anneal_ratio must lie in (0, 1], got {guidance.anneal_ratio}",
        )
        check(guidance.anneal_power > 0.0, f"guidance.anneal_power must be positive, got {guidance.anneal_power}")
        if self.guiding:
            check(
                self.models.teacher is not None and any(path.kl > 0.0 for path in self.loss.paths),
                "guidance distils a teacher's mixture: it needs models.teacher and a positive loss kl",
            )
            check(
                self.loss.temperature == 1.0,
                "a guided teacher hands over a normalized distribution, so loss.temperature has to be 1.0",
            )
            check(all(path.ce_selection == 1.0 for path in self.loss.paths), "loss ce_selection is not defined for guidance distillation")

        check_range(
            "evaluation.prefix_ratio", self.evaluation.prefix_ratio, 0, 1, closed=False
        )

    @property
    def distilling(self) -> bool:
        """
        Whether a teacher is loaded and actually contributes to the loss.
        """

        return self.models.teacher is not None and any(path.kl > 0.0 for path in self.loss.paths)

    @property
    def guiding(self) -> bool:
        """
        Whether the stage distils the guided (CFG) mixture rather than the plain conditional pass.
        """

        return self.guidance.scale > 0.0

    @property
    def selecting(self) -> bool:
        """
        Whether the CE is taken on a selection of tokens, chosen against the teacher.
        """

        return self.models.teacher is not None and any(path.ce_selection < 1.0 for path in self.loss.paths)

    @property
    def dtype(self) -> torch.dtype:
        """
        The dtype the mixed precision setting asks for.
        """

        return MIXED_PRECISION_DTYPES[self.optim.mixed_precision]

    @property
    def dataset_kwargs(self) -> Dict[str, Any]:
        """
        Arguments :class:`EchoDataset` takes, minus the manifest itself.
        """

        return {"data_root": self.data.data_root}

    @property
    def sampler_kwargs(self) -> Dict[str, Any]:
        """
        Arguments :class:`EchoBatchSampler` takes, minus the dataset and the ranks.
        """

        return {
            "max_batch_size": self.data.max_batch_size,
            "max_batch_tokens": self.data.max_batch_tokens,
            "max_pad_ratio": self.data.max_pad_ratio,
            "seed": self.data.seed,
            "packing": self.data.packing,
            "packing_chunk": self.data.packing_chunk,
        }

    @property
    def preprocessor_kwargs(self) -> Dict[str, Any]:
        """
        Arguments :class:`EchoPreprocessor` takes.
        """

        return {
            "tokenizer": self.preprocessing.tokenizer,
            "drop_prefix_ratio": self.preprocessing.drop_prefix_ratio,
            "polish_share": self.masking.polish_share,
            "masking_kwargs": self.masking_kwargs,
            **asdict(self.preprocessing.prefix_split),
        }

    @property
    def eval_preprocessor_kwargs(self) -> Dict[str, Any]:
        """
        Arguments the preprocessor of an evaluation loader takes.

        It is the training one with the split pinned: training draws a prefix
        per sample, which would score every checkpoint on something else.
        """

        return {
            **self.preprocessor_kwargs,
            "prefix_min_ratio": self.evaluation.prefix_ratio,
            "prefix_max_ratio": self.evaluation.prefix_ratio,
        }

    @property
    def masking_kwargs(self) -> Dict[str, Any]:
        """
        Arguments the main-path mask builder takes.
        """

        return {
            "stage_shares": self.masking.stage_shares,
            "tail_masked": self.masking.tail_masked,
            "samplers": self.masking.samplers,
            "min_pass_share": self.masking.min_pass_share,
        }

    @property
    def codebook_weights(self) -> List[float]:
        """
        Loss weight of every codebook: the main-path and the polish samples split the loss as they split the batch.
        """

        return codebook_weights(self.masking.polish_share, self.masking.polish_weights)

    def warmup_steps(self, total_steps: int) -> int:
        """
        Steps the scheduler spends warming up, given the run's length.
        """

        return int(self.schedule.warmup_ratio * total_steps)

    def total_steps(self, batches_per_epoch: Optional[int] = None) -> int:
        """
        Total optimizer steps the run takes.
        """

        if self.schedule.steps is not None:
            return self.schedule.steps
        if batches_per_epoch is None:
            raise ValueError("a config set in epochs needs batches_per_epoch to resolve steps")

        batches = self.schedule.epochs * batches_per_epoch

        return batches // self.optim.gradient_accumulation_steps
