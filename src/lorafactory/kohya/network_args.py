"""Config target dict -> kohya `network_args` list (sd-scripts v0.11.1 vocabulary).

    from lorafactory.kohya.network_args import build_network_args, TargetSpecError
    spec = build_network_args(config_dict)   # config_dict == ResolvedConfig.data
    spec.network_module   # "networks.lora_sd3" | "networks.lora_flux"
    spec.network_dim      # int
    spec.network_alpha    # int
    spec.network_args     # list[str] of "key=value"
    spec.flags            # dict of extra kohya keys (e.g. network_train_text_encoder_only)

Kohya bridge facts: SD3 targeting goes through the
`network_args` list of "key=value" strings understood by `networks.lora_sd3`
(`train_block_indices`, per-type `*_attn_dim` / `*_mlp_dim` overrides,
`context_mod_dim`/`x_mod_dim` always zeroed since the spec vocabulary has no
adaLN concept). FLUX (`networks.lora_flux`) instead always zeroes
`img_mod_dim`/`txt_mod_dim`/`single_mod_dim`. Neither `split_qkv` nor
`train_t5xxl` is ever emitted. kohya itself does not bounds-check
`train_block_indices`, so we enforce 0..23 ourselves.
"""

from __future__ import annotations

from dataclasses import dataclass, field

MIN_BLOCK = 0
MAX_BLOCK = 23

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


def build_network_args(config: dict) -> NetworkSpec:
    """Build the kohya network spec for a resolved adapter config."""
    target = config["target"]
    scope = target.get("scope")

    if scope == "text_encoders":
        return _te_only_spec(config, target)

    module_classes = target.get("module_classes") or []
    if not module_classes:
        raise TargetSpecError("target.module_classes must not be empty")

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

    arch = config.get("model", {}).get("arch", "sd3")
    if arch == "flux":
        args += ["img_mod_dim=0", "txt_mod_dim=0", "single_mod_dim=0"]
    else:
        args += ["context_mod_dim=0", "x_mod_dim=0"]

    return NetworkSpec(
        network_module=_network_module(arch),
        network_dim=int(target["rank"]),
        network_alpha=int(target["alpha"]),
        network_args=args,
        flags={},
    )
