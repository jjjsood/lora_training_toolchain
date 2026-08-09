"""Config target dict -> kohya `network_args` list (sd-scripts v0.11.1, pinned
commit 6721028).

    from lorafactory.kohya.network_args import build_network_args, TargetSpecError
    spec = build_network_args(config_dict)   # config_dict == ResolvedConfig.data
    spec.network_module   # "networks.lora_sd3" | "networks.lora_flux"
    spec.network_dim      # int
    spec.network_alpha    # int
    spec.network_args     # list[str] of "key=value"
    spec.flags            # dict of extra kohya keys (e.g. network_train_text_encoder_only)

Kohya bridge facts, SD3 (`networks.lora_sd3`): targeting goes through the
`network_args` list of "key=value" strings — `train_block_indices`, per-type
`context_*_dim` / `x_*_dim` overrides, `context_mod_dim`/`x_mod_dim` always
zeroed since the spec vocabulary has no adaLN concept. kohya itself does not
bounds-check `train_block_indices`, so we enforce 0..23 ourselves.

Kohya bridge facts, FLUX (`networks.lora_flux`): double- and single-stream
blocks are indexed independently via `train_double_block_indices` (0..18) and
`train_single_block_indices` (0..37) — each an inclusive `lo-hi` range or
`none`. There is no `train_block_indices` in lora_flux.py; we never emit it.
Per-type dims are `img_attn_dim`/`txt_attn_dim` and `img_mlp_dim`/
`txt_mlp_dim` (never `context_*`/`x_*`, which is SD3-only vocabulary);
`img_mod_dim`/`txt_mod_dim`/`single_mod_dim` are always zeroed. The
single-stream block fuses qkv+mlp into `linear1`/`linear2` (kohya source
comment: "SingleStreamBlock is not supported because of combined qkv"), so a
`blocks_single` range cannot exclude just `attn` or just `mlp` — that
combination is rejected as a `TargetSpecError`. kohya does not bounds-check
either FLUX index list, so we enforce 0..18 / 0..37 ourselves.

Neither `split_qkv` nor `train_t5xxl` is ever emitted, for either arch.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from lorafactory.constants import (
    FLUX_MAX_DOUBLE_BLOCK_INDEX,
    FLUX_MAX_SINGLE_BLOCK_INDEX,
    SD3_MAX_BLOCK_INDEX,
)

MIN_BLOCK = 0
MAX_BLOCK = SD3_MAX_BLOCK_INDEX

_MODULE_TYPES = ("attn", "mlp")


class TargetSpecError(Exception):
    """A `target` section describes a targeting spec kohya cannot express safely."""


@dataclass(frozen=True)
class NetworkSpec:
    network_module: str
    network_dim: int
    network_alpha: int
    network_args: list[str]
    flags: dict = field(default_factory=dict)


def _network_module(arch: str) -> str:
    return "networks.lora_flux" if arch == "flux" else "networks.lora_sd3"


def _te_only_spec(config: dict, target: dict) -> NetworkSpec:
    arch = config.get("model", {}).get("arch", "sd3")
    return NetworkSpec(
        network_module=_network_module(arch),
        network_dim=int(target["rank"]),
        network_alpha=int(target["alpha"]),
        network_args=[],
        flags={"network_train_text_encoder_only": True},
    )


def _block_index_arg(key: str, pair: list[int] | None, max_index: int) -> str:
    """Render one FLUX `train_*_block_indices` arg, `none` when unset."""
    if pair is None:
        return f"{key}=none"
    lo, hi = pair
    if not (MIN_BLOCK <= lo <= max_index) or not (MIN_BLOCK <= hi <= max_index):
        raise TargetSpecError(f"{key} {[lo, hi]} out of bounds {MIN_BLOCK}..{max_index}")
    if lo > hi:
        raise TargetSpecError(f"{key} {[lo, hi]} is not a valid ascending range")
    return f"{key}={lo}-{hi}"


def _sd3_transformer_args(target: dict, module_classes: list[str]) -> list[str]:
    """SD3 (`networks.lora_sd3`) transformer targeting. Unchanged emission."""
    lo, hi = target.get("blocks", [MIN_BLOCK, MAX_BLOCK])
    if not (MIN_BLOCK <= lo <= MAX_BLOCK) or not (MIN_BLOCK <= hi <= MAX_BLOCK):
        raise TargetSpecError(
            f"target.blocks {[lo, hi]} out of bounds {MIN_BLOCK}..{MAX_BLOCK}"
        )
    if lo > hi:
        raise TargetSpecError(f"target.blocks {[lo, hi]} is not a valid ascending range")

    args = [f"train_block_indices={lo}-{hi}"]
    for module_type in _MODULE_TYPES:
        if module_type not in module_classes:
            args.append(f"context_{module_type}_dim=0")
            args.append(f"x_{module_type}_dim=0")
    args += ["context_mod_dim=0", "x_mod_dim=0"]
    return args


def _flux_transformer_args(target: dict, module_classes: list[str]) -> list[str]:
    """FLUX (`networks.lora_flux`) transformer targeting.

    Double- and single-stream blocks are indexed independently; there is no
    combined `train_block_indices` in lora_flux.py, so we never emit it.
    """
    blocks_double = target.get("blocks_double")
    blocks_single = target.get("blocks_single")

    if blocks_single is not None:
        missing_classes = [c for c in _MODULE_TYPES if c not in module_classes]
        if missing_classes:
            raise TargetSpecError(
                "FLUX single-stream blocks fuse attn+mlp (linear1/linear2); "
                "per-class exclusion is inexpressible — drop blocks_single or "
                "target both classes"
            )

    args = [
        _block_index_arg(
            "train_double_block_indices", blocks_double, FLUX_MAX_DOUBLE_BLOCK_INDEX
        ),
        _block_index_arg(
            "train_single_block_indices", blocks_single, FLUX_MAX_SINGLE_BLOCK_INDEX
        ),
    ]
    for module_type in _MODULE_TYPES:
        if module_type not in module_classes:
            args.append(f"img_{module_type}_dim=0")
            args.append(f"txt_{module_type}_dim=0")
    args += ["img_mod_dim=0", "txt_mod_dim=0", "single_mod_dim=0"]
    return args


def build_network_args(config: dict) -> NetworkSpec:
    """Build the kohya network spec for a resolved adapter config."""
    target = config["target"]
    scope = target.get("scope")

    if scope == "text_encoders":
        return _te_only_spec(config, target)

    module_classes = target.get("module_classes") or []
    if not module_classes:
        raise TargetSpecError("target.module_classes must not be empty")

    arch = config.get("model", {}).get("arch", "sd3")
    if arch == "flux":
        args = _flux_transformer_args(target, module_classes)
    else:
        args = _sd3_transformer_args(target, module_classes)

    return NetworkSpec(
        network_module=_network_module(arch),
        network_dim=int(target["rank"]),
        network_alpha=int(target["alpha"]),
        network_args=args,
        flags={},
    )
