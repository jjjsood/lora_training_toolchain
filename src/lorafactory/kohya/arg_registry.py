"""Registry of kohya argparse dests @ sd-scripts v0.11.1 — defence against silent typos.

    from lorafactory.kohya.arg_registry import (
        load_registry, validate_keys, UnknownKohyaKeyError)
    dests = load_registry("sd3")      # frozenset[str]; also "flux"
    validate_keys(["seed", ...], "sd3")   # raises UnknownKohyaKeyError

The committed JSON snapshots (kohya_args_sd3.json / kohya_args_flux.json,
sitting next to this module) are the real argparse dests of
sd3_train_network.py / flux_train_network.py at tag v0.11.1, regenerated via
tools/dump_kohya_args.py from a shallow clone of kohya-ss/sd-scripts. kohya
silently ignores unknown `--config_file` keys, so this registry is how a
typo'd budget/network key becomes OUR error instead of a silently-dropped one.
"""

from __future__ import annotations

import json
from functools import cache
from pathlib import Path

_REGISTRY_DIR = Path(__file__).resolve().parent
_ENGINES = {
    "sd3": _REGISTRY_DIR / "kohya_args_sd3.json",
    "flux": _REGISTRY_DIR / "kohya_args_flux.json",
}


class UnknownKohyaKeyError(Exception):
    """A key was validated against the registry and is not a known kohya dest."""


@cache
def load_registry(engine: str) -> frozenset[str]:
    """Load the frozenset of known argparse dests for `engine` ("sd3" | "flux")."""
    try:
        path = _ENGINES[engine]
    except KeyError:
        raise ValueError(
            f"unknown kohya engine {engine!r}; expected one of {sorted(_ENGINES)}"
        ) from None
    with open(path) as f:
        dests = json.load(f)
    return frozenset(dests)


def validate_keys(keys, engine: str) -> None:
    """Raise UnknownKohyaKeyError if any of `keys` is not a known dest for `engine`."""
    dests = load_registry(engine)
    unknown = sorted(set(keys) - dests)
    if unknown:
        raise UnknownKohyaKeyError(
            f"unknown kohya {engine} argparse dest(s): {', '.join(unknown)}"
        )
