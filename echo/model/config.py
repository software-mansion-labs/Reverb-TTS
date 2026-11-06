import copy

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Union
from transformers import AutoConfig

from omnivoice.models.omnivoice import (
    OmniVoiceConfig,
    OmniVoiceGenerationConfig,
    _resolve_model_path,
)
from echo.masking.constants import MAIN_CODEBOOKS
from echo.model.types import EchoCFGMode
from echo.utils.config_utils import (
    build_config,
    check,
    check_choice,
    read_json,
    write_json,
)


# Subconfig - Echo Meta
# General purpose metadata container
@dataclass
class EchoMeta:
    version: int = 1
    name: str = "echo"


# Subconfig - Prefix Blocking
# In a prefix blocking mode, forward attention connections (prefix -> target) can be disabled.
@dataclass
class PrefixBlockingConfig:
    enabled: bool = False


# Subconfig - Architecture (Pruning)
# Describes the loaded checkpoint's shape rather than changing it: pruning happens in echo/pruning, and a
# pruned checkpoint carries its own shape. None inherits the checkpoint's value, anything else must match it.
@dataclass
class ArchitectureConfig:
    num_hidden_layers: Optional[int] = None
    hidden_size: Optional[int] = None
    intermediate_size: Optional[int] = None
    num_attention_heads: Optional[int] = None
    num_key_value_heads: Optional[int] = None
    vocab_size: Optional[int] = None          # Text tokens the embedding holds - fewer after vocabulary pruning
    embedding_rank: Optional[int] = None      # The text embedding as a rank-r factorization (a V x r table, then r -> H), None for a full table
    kept_layers: Optional[List[int]] = None   # Provenance: the original indices of the layers still present
    kept_kv_heads: Optional[List[List[int]]] = None   # Provenance: per layer, the original indices of the KV heads still present


# Subconfig - CFG
# The available modes are:
# - "default": standard mode, forward() runs the conditional pass, and generate() runs mixed one
# - "teacher": CFG-distilation mode, forward() runs a single mixed pass, and generate() stays the same as default
# - "student": no CFG at all - forward() runs conditional pass only, and generate() does not guide too.
@dataclass
class CFGConfig:
    mode: EchoCFGMode = "default"
    guidance_scale: float = 2.0

    @property
    def unconditional_pass(self) -> bool:
        """
        Whether 'forward' also runs the unconditional branch.
        """

        return self.mode == "teacher"

    @property
    def generation_scale(self) -> float:
        """
        Guidance the decoder runs with - a distilled student already holds it.
        """

        return 0.0 if self.mode == "student" else self.guidance_scale


@dataclass
class EchoConfig:
    """
    Everything one model needs, loaded from a single .json file.
    """

    # Subconfig - general metadata
    meta: EchoMeta = field(default_factory=EchoMeta)

    # Subconfig - prefix blocking
    prefix_blocking: PrefixBlockingConfig = field(default_factory=PrefixBlockingConfig)

    # Subconfig - pruning
    architecture: ArchitectureConfig = field(default_factory=ArchitectureConfig)

    # Subconfig - CFG
    cfg: CFGConfig = field(default_factory=CFGConfig)

    # Initialization paths
    base: Optional[Dict[str, Any]] = None   # An inline OmniVoiceConfig dump - most explicit
    # OmniVoiceConfig.from_pretrained(path), example: 'k2-fsa/OmniVoice' - typical, loads ready weights
    # Custom checkpoint can also be loaded (ex. "init_from": "runs/p1-prefix-blocking/checkpoint-40000"), must be in a HF repo style (model + config.json)
    init_from: Optional[str] = None
    llm_name_or_path: Optional[str] = None   # A bare HF LLM (e.g. Qwen/Qwen3-0.6B) - only special cases

    def __post_init__(self) -> None:
        self.validate()

    @classmethod
    def from_dict(cls, data: Dict[str, Any], *, strict: bool = False) -> "EchoConfig":
        return build_config(cls, data, strict=strict)

    @classmethod
    def from_json(
        cls, json_path: Union[str, Path], *, strict: bool = False
    ) -> "EchoConfig":
        return cls.from_dict(read_json(json_path), strict=strict)

    @classmethod
    def from_omnivoice_config(cls, config: OmniVoiceConfig) -> "EchoConfig":
        return cls.from_dict(getattr(config, "echo", None) or {})

    def to_dict(self, *, include_base: bool = True) -> Dict[str, Any]:
        data = asdict(self)
        if not include_base:
            data.pop("base")

        return data

    def to_json(
        self, json_path: Union[str, Path], *, include_base: bool = True
    ) -> None:
        write_json(json_path, self.to_dict(include_base=include_base))

    def validate(self) -> None:
        check_choice("cfg.mode", self.cfg.mode, EchoCFGMode)
        if self.architecture.hidden_size is not None:
            raise NotImplementedError("architecture.hidden_size is not implemented yet")

    def resolve_base_config(self) -> OmniVoiceConfig:
        if self.base is not None:
            return OmniVoiceConfig(**copy.deepcopy(self.base))
        if self.init_from is not None:
            return OmniVoiceConfig.from_pretrained(_resolve_model_path(self.init_from))
        if self.llm_name_or_path is not None:
            llm_config = AutoConfig.from_pretrained(
                _resolve_model_path(self.llm_name_or_path)
            )
            return OmniVoiceConfig(llm_config=llm_config)

        raise ValueError(
            "EchoConfig needs one of 'base', 'init_from' or 'llm_name_or_path'"
        )

    def to_omnivoice_config(self) -> OmniVoiceConfig:
        config = self.resolve_base_config()
        architecture = self.architecture
        inherited = (getattr(config, "echo", None) or {}).get("architecture") or {}

        for knob in ("num_hidden_layers", "intermediate_size", "num_attention_heads", "num_key_value_heads", "vocab_size"):
            actual = getattr(config.llm_config, knob)
            wanted = getattr(architecture, knob)
            check(
                wanted is None or wanted == actual,
                f"architecture.{knob}={wanted} does not match the base model's {actual} - "
                "prune with echo.pruning instead",
            )
            setattr(architecture, knob, actual)

        if architecture.kept_layers is None:
            architecture.kept_layers = inherited.get("kept_layers")
        if architecture.kept_kv_heads is None:
            architecture.kept_kv_heads = inherited.get("kept_kv_heads")
        check(
            architecture.embedding_rank in (None, inherited.get("embedding_rank")),
            f"architecture.embedding_rank={architecture.embedding_rank} does not match the base model's "
            f"{inherited.get('embedding_rank')} - factorize with echo.pruning instead",
        )
        architecture.embedding_rank = inherited.get("embedding_rank")
        config.echo = self.to_dict(include_base=False)

        return config

    def default_generation_config(self, **overrides: Any) -> OmniVoiceGenerationConfig:
        # The two-stage sampler: 8 passes fill codebooks 0-1, remasking half of each pass's commits there, then one
        # polish pass fills the rest.
        params: Dict[str, Any] = {
            "prefix_blocked": self.prefix_blocking.enabled,
            "guidance_scale": self.cfg.generation_scale,
            "num_step": 8,
            "remask_ratio": 0.5,
            "remask_codebooks": MAIN_CODEBOOKS,
            "main_codebooks": MAIN_CODEBOOKS,
            "polish_codebooks_from": MAIN_CODEBOOKS,
            "polish_ratio": 1.0,
        }
        params.update(overrides)

        return OmniVoiceGenerationConfig.from_dict(params)
