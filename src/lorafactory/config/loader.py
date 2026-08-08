"""Config overlay loader: extends + overrides_allowed + canonical hash."""

from dataclasses import dataclass
from pathlib import Path

import yaml

from .hashing import config_hash

_META_KEYS = {"adapter_id", "extends", "overrides_allowed", "budget_exempt"}


class ConfigViolation(Exception):
    """An overlay touched a section it did not declare in overrides_allowed."""


@dataclass(frozen=True)
class ResolvedConfig:
    data: dict
    sha256: str


def _load_yaml(path: Path) -> dict:
    with open(path) as f:
        return yaml.safe_load(f) or {}


def resolve(path) -> ResolvedConfig:
    path = Path(path)
    overlay = _load_yaml(path)

    extends = overlay.get("extends")
    if extends:
        extends_path = Path(extends)
        if not extends_path.is_absolute():
            extends_path = (path.parent / extends_path).resolve()
        base = _load_yaml(extends_path)
    else:
        base = {}

    merged = dict(base)
    overrides_allowed = overlay.get("overrides_allowed", [])
    adapter_id = overlay.get("adapter_id")

    for key, value in overlay.items():
        if key in _META_KEYS:
            continue
        if key not in overrides_allowed:
            raise ConfigViolation(
                f"adapter '{adapter_id}': section '{key}' not declared in "
                f"overrides_allowed"
            )
        merged[key] = value

    merged.pop("extends", None)
    if adapter_id is not None:
        merged["adapter_id"] = adapter_id
    if "budget_exempt" in overlay:
        merged["budget_exempt"] = overlay["budget_exempt"]

    return ResolvedConfig(data=merged, sha256=config_hash(merged))
