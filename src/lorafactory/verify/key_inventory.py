"""Key-inventory verifier (README §4's second life).

`expected_module_names(target)` builds the full expected module-name set
straight from the pinned constants — never introspected from a live model —
so a renamed/missing module surfaces as a loud diff, not a quietly narrower
adapter. `verify(sd, target)` compares a converted state dict's *modules*
(one entry per `lora_A`/`lora_B` pair) against that set.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from lorafactory.constants import (
    CLIP_ATTN_LEAVES,
    CLIP_G_NUM_LAYERS,
    CLIP_L_NUM_LAYERS,
    CLIP_MLP_LEAVES,
    SD3_ATTN_LEAVES,
    SD3_BLOCK23_ABSENT,
    SD3_CONTEXT_PRE_ONLY_BLOCK,
    SD3_MLP_LEAVES,
)
from lorafactory.convert.keymap import DiffusersKey, UnconvertibleKeyError, parse_diffusers_lora_key

_TE_ENCODER_INFO = {
    "clip_l": ("text_encoder", CLIP_L_NUM_LAYERS),
    "clip_g": ("text_encoder_2", CLIP_G_NUM_LAYERS),
}
_CLIP_LEAVES = (*CLIP_ATTN_LEAVES, *CLIP_MLP_LEAVES)


@dataclass(frozen=True)
class VerifyReport:
    ok: bool
    unexpected: set[str] = field(default_factory=set)
    missing: set[str] = field(default_factory=set)
    rank_mismatches: dict[str, int] = field(default_factory=dict)


def expected_module_names(target: dict) -> frozenset[str]:
    """Full set of expected diffusers module names (no `.lora_A`/`.lora_B` suffix)."""
    scope = target["scope"]

    if scope == "transformer":
        lo, hi = target["blocks"]
        classes = set(target["module_classes"])
        leaves: list[str] = []
        if "attn" in classes:
            leaves.extend(SD3_ATTN_LEAVES)
        if "mlp" in classes:
            leaves.extend(SD3_MLP_LEAVES)
        names = set()
        for block in range(lo, hi + 1):
            for leaf in leaves:
                if block == SD3_CONTEXT_PRE_ONLY_BLOCK and leaf in SD3_BLOCK23_ABSENT:
                    continue
                names.add(f"transformer.transformer_blocks.{block}.{leaf}")
        return frozenset(names)

    if scope == "text_encoders":
        names = set()
        for encoder in target["te_encoders"]:
            prefix, num_layers = _TE_ENCODER_INFO[encoder]
            for layer in range(num_layers):
                for leaf in _CLIP_LEAVES:
                    names.add(f"{prefix}.text_model.encoder.layers.{layer}.{leaf}")
        return frozenset(names)

    raise ValueError(f"unknown target scope: {scope!r}")


def _full_module_name(parsed: DiffusersKey) -> str:
    if parsed.kind == "transformer":
        return f"transformer.transformer_blocks.{parsed.block}.{parsed.module}"
    prefix = "text_encoder" if parsed.te_index == 1 else "text_encoder_2"
    return f"{prefix}.{parsed.module}"


def verify(sd: dict, target: dict) -> VerifyReport:
    """Compare a converted state dict's module inventory against `target`."""
    expected = expected_module_names(target)
    expected_rank = target["rank"]

    found_ranks: dict[str, int] = {}
    unexpected: set[str] = set()

    for key, tensor in sd.items():
        try:
            parsed = parse_diffusers_lora_key(key)
        except UnconvertibleKeyError:
            unexpected.add(key)
            continue
        if parsed.part != "lora_A":
            continue
        found_ranks[_full_module_name(parsed)] = int(tensor.shape[0])

    for module_name in found_ranks:
        if module_name not in expected:
            unexpected.add(module_name)

    missing = expected - found_ranks.keys()
    rank_mismatches = {
        name: rank
        for name, rank in found_ranks.items()
        if name in expected and rank != expected_rank
    }

    ok = not unexpected and not missing and not rank_mismatches
    return VerifyReport(
        ok=ok, unexpected=unexpected, missing=missing, rank_mismatches=rank_mismatches,
    )
