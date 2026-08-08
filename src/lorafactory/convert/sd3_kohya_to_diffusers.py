"""SD3 kohya -> diffusers/PEFT LoRA converter (no public converter exists).

Each kohya module (`lora_down` + `lora_up` + `alpha`) is fused into a single
PEFT `lora_A` / `lora_B` pair with the `alpha/rank` scale folded into the
weights — split as `sqrt(scale)` on each side, mirroring diffusers' own
`_convert_to_ai_toolkit` — so no `.alpha` keys survive in the output and
`lora_B @ lora_A == (alpha/rank) * lora_up @ lora_down` exactly (up to fp32
rounding). The fused qkv projection is split into three per-projection
modules: `lora_A` (== scaled lora_down) is shared/replicated across the
split, `lora_B` (== scaled lora_up) is chunked along dim 0.
"""

from __future__ import annotations

import math
import re
from typing import Any

import torch

from lorafactory.constants import (
    CLIP_ATTN_LEAVES,
    CLIP_MLP_LEAVES,
    SD3_CONTEXT_PRE_ONLY_BLOCK,
)
from lorafactory.convert.keymap import KohyaKey, UnconvertibleKeyError, parse_kohya_sd3_key

_X_QKV_SPLIT = ("attn.to_q", "attn.to_k", "attn.to_v")
_CONTEXT_QKV_SPLIT = ("attn.add_q_proj", "attn.add_k_proj", "attn.add_v_proj")

_X_DIRECT_MODULE = {
    "attn_proj": "attn.to_out.0",
    "mlp_fc1": "ff.net.0.proj",
    "mlp_fc2": "ff.net.2",
}
_CONTEXT_DIRECT_MODULE = {
    "attn_proj": "attn.to_add_out",
    "mlp_fc1": "ff_context.net.0.proj",
    "mlp_fc2": "ff_context.net.2",
}
# These context leaves do not exist on the context-pre-only last block (23).
_CONTEXT_ABSENT_LEAVES = frozenset(_CONTEXT_DIRECT_MODULE)

_TE_PREFIX = {1: "text_encoder", 2: "text_encoder_2"}
# lora_te3 is the T5 encoder; diffusers has no LoRA path for it.
_TE_INDEX_T5 = 3
_TE_LEAF_TO_DOTTED = {
    leaf.replace(".", "_"): leaf for leaf in (*CLIP_ATTN_LEAVES, *CLIP_MLP_LEAVES)
}
_TE_LEAF_RE = re.compile(r"^text_model_encoder_layers_(?P<layer>\d+)_(?P<mod>.+)$")

_PART_SUFFIX = {
    "lora_down": ".lora_down.weight",
    "lora_up": ".lora_up.weight",
    "alpha": ".alpha",
}


def convert(kohya_sd: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    """Convert a kohya SD3 LoRA state dict into diffusers/PEFT layout."""
    modules: dict[str, dict[str, Any]] = {}
    for key, tensor in kohya_sd.items():
        parsed = parse_kohya_sd3_key(key)  # raises UnconvertibleKeyError on garbage

        if parsed.te_index == _TE_INDEX_T5:
            raise UnconvertibleKeyError(f"lora_te3 (T5) has no diffusers LoRA path: {key!r}")

        if (
            parsed.kind == "transformer"
            and parsed.stream == "context"
            and parsed.block == SD3_CONTEXT_PRE_ONLY_BLOCK
            and parsed.leaf in _CONTEXT_ABSENT_LEAVES
        ):
            raise UnconvertibleKeyError(
                f"context {parsed.leaf} is absent on context-pre-only block "
                f"{SD3_CONTEXT_PRE_ONLY_BLOCK}: {key!r}"
            )

        prefix = key[: -len(_PART_SUFFIX[parsed.part])]
        entry = modules.setdefault(prefix, {"parsed": parsed})
        entry[parsed.part] = tensor

    out: dict[str, torch.Tensor] = {}
    for entry in modules.values():
        _emit_module(out, entry)
    return out


def _emit_module(out: dict[str, torch.Tensor], entry: dict[str, Any]) -> None:
    parsed: KohyaKey = entry["parsed"]
    down = entry["lora_down"].float()
    up = entry["lora_up"].float()
    rank = down.shape[0]
    alpha = float(entry["alpha"]) if "alpha" in entry else float(rank)
    sqrt_scale = math.sqrt(alpha / rank)
    lora_a = down * sqrt_scale
    lora_b = up * sqrt_scale

    if parsed.kind == "transformer":
        block_prefix = f"transformer.transformer_blocks.{parsed.block}"
        if parsed.leaf == "attn_qkv":
            split = _X_QKV_SPLIT if parsed.stream == "x" else _CONTEXT_QKV_SPLIT
            chunk_size = lora_b.shape[0] // 3
            for j, module_name in enumerate(split):
                chunk = lora_b[j * chunk_size : (j + 1) * chunk_size, :]
                _write(out, f"{block_prefix}.{module_name}", lora_a.clone(), chunk)
        else:
            table = _X_DIRECT_MODULE if parsed.stream == "x" else _CONTEXT_DIRECT_MODULE
            _write(out, f"{block_prefix}.{table[parsed.leaf]}", lora_a, lora_b)
        return

    # kind == "te"
    m = _TE_LEAF_RE.fullmatch(parsed.leaf)
    if not m or m.group("mod") not in _TE_LEAF_TO_DOTTED:
        raise UnconvertibleKeyError(f"unrecognised text-encoder leaf: {parsed.leaf!r}")
    dotted = _TE_LEAF_TO_DOTTED[m.group("mod")]
    te_prefix = _TE_PREFIX[parsed.te_index]
    module_name = f"{te_prefix}.text_model.encoder.layers.{m.group('layer')}.{dotted}"
    _write(out, module_name, lora_a, lora_b)


def _write(
    out: dict[str, torch.Tensor], module_name: str, lora_a: torch.Tensor, lora_b: torch.Tensor
) -> None:
    out[f"{module_name}.lora_A.weight"] = lora_a
    out[f"{module_name}.lora_B.weight"] = lora_b
