"""Anchored key grammars for SD3 LoRA state dicts, kohya and diffusers/PEFT.

Every pattern here is matched with `re.fullmatch` against a fixed vocabulary of
block indices / module leaves pulled from `lorafactory.constants` — never a
substring search. That is what keeps `blocks_1` from ever matching inside
`blocks_11`: the block-index group is bounded on both sides by literal
underscores that are part of the fullmatch, so a partial digit match cannot
leave a dangling remainder that a substring search would silently accept.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from lorafactory.constants import (
    CLIP_ATTN_LEAVES,
    CLIP_MLP_LEAVES,
    SD3_ATTN_LEAVES,
    SD3_MLP_LEAVES,
    SD3_NUM_BLOCKS,
)


class UnconvertibleKeyError(ValueError):
    """A state-dict key does not fit the pinned kohya or diffusers grammar."""


@dataclass(frozen=True)
class KohyaKey:
    kind: str  # "transformer" | "te"
    block: int | None
    stream: str | None  # "x" | "context" | None
    leaf: str
    part: str  # "lora_down" | "lora_up" | "alpha"
    te_index: int | None = None


@dataclass(frozen=True)
class DiffusersKey:
    kind: str  # "transformer" | "te"
    block: int | None
    module: str
    part: str  # "lora_A" | "lora_B"
    te_index: int | None = None


_PART_SUFFIX_MAP = {
    "lora_down.weight": "lora_down",
    "lora_up.weight": "lora_up",
    "alpha": "alpha",
}
_PART_ALT = r"lora_down\.weight|lora_up\.weight|alpha"

_KOHYA_TRANSFORMER_RE = re.compile(
    r"^lora_unet_joint_blocks_(?P<block>\d+)_(?P<stream>x|context)_block_"
    r"(?P<leaf>attn_qkv|attn_proj|mlp_fc1|mlp_fc2)"
    rf"\.(?P<part>{_PART_ALT})$"
)

_KOHYA_TE_RE = re.compile(
    r"^lora_te(?P<te_index>[123])_(?P<leaf>.+)"
    rf"\.(?P<part>{_PART_ALT})$"
)


def parse_kohya_sd3_key(key: str) -> KohyaKey:
    """Parse a kohya-layout SD3 LoRA key, transformer or text-encoder."""
    m = _KOHYA_TRANSFORMER_RE.fullmatch(key)
    if m:
        return KohyaKey(
            kind="transformer",
            block=int(m.group("block")),
            stream=m.group("stream"),
            leaf=m.group("leaf"),
            part=_PART_SUFFIX_MAP[m.group("part")],
        )
    m = _KOHYA_TE_RE.fullmatch(key)
    if m:
        return KohyaKey(
            kind="te",
            block=None,
            stream=None,
            leaf=m.group("leaf"),
            part=_PART_SUFFIX_MAP[m.group("part")],
            te_index=int(m.group("te_index")),
        )
    raise UnconvertibleKeyError(f"unparseable kohya SD3 key: {key!r}")


_SD3_MODULE_LEAVES = frozenset((*SD3_ATTN_LEAVES, *SD3_MLP_LEAVES))
_CLIP_LEAVES = frozenset((*CLIP_ATTN_LEAVES, *CLIP_MLP_LEAVES))

_DIFF_PART_ALT = r"lora_A|lora_B"

_DIFF_TRANSFORMER_RE = re.compile(
    r"^transformer\.transformer_blocks\.(?P<block>\d+)\."
    rf"(?P<module>.+)\.(?P<part>{_DIFF_PART_ALT})\.weight$"
)

_DIFF_TE_RE = re.compile(
    r"^text_encoder(?P<suffix>|_2)\."
    r"(?P<module>text_model\.encoder\.layers\.(?P<layer>\d+)\.(?P<leaf>.+))"
    rf"\.(?P<part>{_DIFF_PART_ALT})\.weight$"
)


def parse_diffusers_lora_key(key: str) -> DiffusersKey:
    """Parse a diffusers/PEFT-layout SD3 LoRA key, transformer or text-encoder."""
    m = _DIFF_TRANSFORMER_RE.fullmatch(key)
    if m:
        block = int(m.group("block"))
        module = m.group("module")
        if block < SD3_NUM_BLOCKS and module in _SD3_MODULE_LEAVES:
            return DiffusersKey(
                kind="transformer", block=block, module=module, part=m.group("part"),
            )
        raise UnconvertibleKeyError(f"unparseable diffusers key: {key!r}")

    m = _DIFF_TE_RE.fullmatch(key)
    if m:
        leaf = m.group("leaf")
        if leaf in _CLIP_LEAVES:
            te_index = 2 if m.group("suffix") == "_2" else 1
            return DiffusersKey(
                kind="te", block=None, module=m.group("module"),
                part=m.group("part"), te_index=te_index,
            )
        raise UnconvertibleKeyError(f"unparseable diffusers key: {key!r}")

    raise UnconvertibleKeyError(f"unparseable diffusers key: {key!r}")
