import json
import logging

from dataclasses import fields, is_dataclass
from pathlib import Path
from typing import Any, Dict, Type, TypeVar, Union, get_args

logger = logging.getLogger(__name__)

Config = TypeVar("Config")


def filter_known_keys(
    config_cls: Type[Any],
    data: Dict[str, Any],
    *,
    strict: bool = False,
    where: str = "config",
) -> Dict[str, Any]:
    """
    Keeps only the keys that are actual fields of ``config_cls``.

    Unknown keys raise when ``strict``, and are warned about and dropped otherwise.
    """

    valid = {f.name for f in fields(config_cls)}
    unknown = sorted(set(data) - valid)
    if unknown:
        message = f"unknown key(s) in {where}: {', '.join(unknown)}"
        if strict:
            raise ValueError(message)
        logger.warning("Ignoring %s", message)

    return {key: value for key, value in data.items() if key in valid}


def build_config(
    config_cls: Type[Config],
    data: Dict[str, Any],
    *,
    strict: bool = False,
    where: str = "",   # Dotted path of the section being built, empty for the config itself
) -> Config:
    """
    Builds a config dataclass from a dict, and each of its sections from its own dict.
    """

    values = filter_known_keys(config_cls, data, strict=strict, where=where or "config")

    for section in fields(config_cls):
        value = values.get(section.name)
        if is_dataclass(section.type) and isinstance(value, dict):
            path = f"{where}.{section.name}" if where else section.name
            values[section.name] = build_config(
                section.type, value, strict=strict, where=path
            )

    return config_cls(**values)


def read_json(path: Union[str, Path]) -> Dict[str, Any]:
    """
    The dict a .json file holds.
    """

    with open(path, "r") as f:
        return json.load(f)


def write_json(path: Union[str, Path], data: Dict[str, Any]) -> None:
    """
    Writes a dict as indented .json.
    """

    Path(path).write_text(json.dumps(data, indent=4) + "\n")


def check(condition: bool, message: str) -> None:
    """
    Rejects a value: raises the message as a ValueError unless the condition holds.
    """

    if not condition:
        raise ValueError(message)


def check_choice(
    name: str,
    value: Any,
    choices: Any,   # The Literal type listing the allowed values
) -> None:
    """
    Rejects a value its Literal type does not list.
    """

    allowed = get_args(choices)
    check(value in allowed, f"{name} must be one of {allowed}, got {value!r}")


def check_range(
    name: str,
    value: float,
    low: float,
    high: float,
    *,
    closed: bool = True,   # False leaves 'high' itself out
) -> None:
    """
    Rejects a value outside [low, high].
    """

    inside = low <= value <= high if closed else low <= value < high
    bracket = "]" if closed else ")"
    check(inside, f"{name} must lie in [{low}, {high}{bracket}, got {value}")
