"""Vocabulary verification, CPU: the pinned FLUX leaf constants against a real
`FluxTransformer2DModel` module tree (mirrors `tests/test_load_smoke.py:20-34`'s
tiny-model pattern).

This is the anchor for `lorafactory.constants.FLUX_DOUBLE_*` / `FLUX_SINGLE_*`
(README §4: pinned, never discovered) — a permanent cheap regression test that
would fail loudly if an upstream diffusers rename ever moved a leaf, rather than
letting `key_inventory`/`keymap` quietly agree with themselves on a vocabulary
that no longer matches the real model.
"""

from __future__ import annotations

import torch
from torch import nn

from lorafactory.constants import (
    FLUX_DOUBLE_ATTN_LEAVES,
    FLUX_DOUBLE_MLP_LEAVES,
    FLUX_SINGLE_ATTN_LEAVES,
    FLUX_SINGLE_MLP_LEAVES,
)

NUM_LAYERS = 2
NUM_SINGLE_LAYERS = 2

PINNED_DOUBLE = frozenset((*FLUX_DOUBLE_ATTN_LEAVES, *FLUX_DOUBLE_MLP_LEAVES))
PINNED_SINGLE = frozenset((*FLUX_SINGLE_ATTN_LEAVES, *FLUX_SINGLE_MLP_LEAVES))


def tiny_flux_transformer():
    from diffusers import FluxTransformer2DModel

    torch.manual_seed(0)
    return FluxTransformer2DModel(
        patch_size=1,
        in_channels=8,
        out_channels=8,
        num_layers=NUM_LAYERS,
        num_single_layers=NUM_SINGLE_LAYERS,
        attention_head_dim=8,
        num_attention_heads=4,
        joint_attention_dim=32,
        pooled_projection_dim=32,
        guidance_embeds=False,
    )


def _linear_leaves(named_modules: dict, prefix: str, *, allow) -> set[str]:
    """Linear-typed submodules directly under `prefix`, filtered by `allow`.

    `allow(relative_name) -> bool` excludes the AdaLN-modulation `norm*.linear`
    family, which is a real `nn.Linear` too but is never a LoRA target (same
    exclusion SD3 already relies on — see test_keymap.py's rejection of
    `norm1.linear`).
    """
    out = set()
    for name, module in named_modules.items():
        if not name.startswith(prefix) or not isinstance(module, nn.Linear):
            continue
        rel = name[len(prefix):]
        if allow(rel):
            out.add(rel)
    return out


def test_flux_double_block_leaves_match_pinned_vocabulary():
    model = tiny_flux_transformer()
    named = dict(model.named_modules())

    for block in range(NUM_LAYERS):
        prefix = f"transformer_blocks.{block}."
        leaves = _linear_leaves(
            named, prefix,
            allow=lambda rel: rel.startswith("attn.") or rel.startswith("ff"),
        )
        assert leaves == PINNED_DOUBLE, (
            f"double block {block}: real leaves {sorted(leaves)} != "
            f"pinned {sorted(PINNED_DOUBLE)}"
        )


def test_flux_double_block_leaves_are_uniform_no_pre_only_quirk():
    """Unlike SD3 block 23, FLUX double blocks have no context-pre-only leaf
    drop on the last block — every double block carries the full 12 leaves."""
    model = tiny_flux_transformer()
    named = dict(model.named_modules())

    last = NUM_LAYERS - 1
    prefix = f"transformer_blocks.{last}."
    leaves = _linear_leaves(
        named, prefix,
        allow=lambda rel: rel.startswith("attn.") or rel.startswith("ff"),
    )
    assert leaves == PINNED_DOUBLE, (
        f"last double block ({last}) dropped leaves: "
        f"missing {sorted(PINNED_DOUBLE - leaves)}"
    )


def test_flux_single_block_leaves_match_pinned_vocabulary():
    model = tiny_flux_transformer()
    named = dict(model.named_modules())

    for block in range(NUM_SINGLE_LAYERS):
        prefix = f"single_transformer_blocks.{block}."
        leaves = _linear_leaves(
            named, prefix,
            allow=lambda rel: not rel.startswith("norm"),
        )
        assert leaves == PINNED_SINGLE, (
            f"single block {block}: real leaves {sorted(leaves)} != "
            f"pinned {sorted(PINNED_SINGLE)}"
        )


def test_flux_single_block_has_no_attn_to_out_or_ff():
    """Single-stream blocks fuse attention output into `proj_out` — there is
    no separate `attn.to_out.0`, and no `ff.*`/`ff_context.*` family at all
    (that's a double-stream-only structure)."""
    model = tiny_flux_transformer()
    named = dict(model.named_modules())

    for block in range(NUM_SINGLE_LAYERS):
        prefix = f"single_transformer_blocks.{block}."
        names = [n[len(prefix):] for n in named if n.startswith(prefix)]
        assert "attn.to_out.0" not in names
        assert not any(n.startswith("ff") for n in names)
