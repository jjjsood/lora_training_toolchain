"""Pure state-dict introspection of a community LoRA checkpoint (CAP-28).

API pinned:
    from lorafactory.survey import introspect
    info = introspect.introspect(state_dict)
    info.layout        # "kohya" | "peft"
    info.arch          # "sd3" | "flux" | "unknown"
    info.ranks         # set of per-module ranks, e.g. {4, 32}
    info.module_count  # number of LoRA modules
    info.modules       # sorted tuple of module names

No model loading, no network, no GPU: the checkpoint is described from its
key names and tensor shapes alone, before anything is put on a device.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field

from lorafactory.constants import SD3_MAX_BLOCK_INDEX

KOHYA_DOWN = ".lora_down.weight"
KOHYA_UP = ".lora_up.weight"
PEFT_A = ".lora_A.weight"
PEFT_B = ".lora_B.weight"

# kohya module-name prefixes (README §4 / T3 constants.py block bounds).
KOHYA_SD3_JOINT_PREFIX = "lora_unet_joint_blocks_"
KOHYA_FLUX_DOUBLE_PREFIX = "lora_unet_double_blocks_"
KOHYA_FLUX_SINGLE_PREFIX = "lora_unet_single_blocks_"

# diffusers/PEFT key prefixes (see convert/keymap.py parse_flux_diffusers_lora_key).
DIFFUSERS_FLUX_SINGLE_PREFIX = "transformer.single_transformer_blocks."
DIFFUSERS_TRANSFORMER_BLOCKS_PREFIX = "transformer.transformer_blocks."


class IntrospectionError(ValueError):
    """The state dict holds no recognisable LoRA modules, or mixes layouts."""


@dataclass(frozen=True)
class SurveyInfo:
    """What a checkpoint is, as read off its keys and shapes."""

    layout: str
    ranks: frozenset[int]
    modules: tuple[str, ...] = field(default_factory=tuple)
    alphas: frozenset[float] = frozenset()
    arch: str = "unknown"

    @property
    def module_count(self) -> int:
        return len(self.modules)


def _rank_of(tensor) -> int:
    shape = tuple(tensor.shape)
    if len(shape) < 1:
        raise IntrospectionError("LoRA down/A weight is not a matrix")
    return int(shape[0])


def detect_arch(state_dict: Mapping) -> str:
    """Guess the base architecture ("sd3" | "flux" | "unknown") from key names alone.

    Both families use kohya (``lora_unet_*``) or diffusers/PEFT
    (``transformer.*``) module-name conventions; the block-family token in
    the key tells them apart:

    - kohya: ``lora_unet_joint_blocks_`` is SD3-only (MMDiT joint blocks);
      ``lora_unet_double_blocks_``/``lora_unet_single_blocks_`` is FLUX-only
      (its two-stream DiT has no joint-block naming).
    - diffusers/PEFT: ``transformer.single_transformer_blocks.`` is FLUX-only
      (SD3 has no single-stream blocks). ``transformer.transformer_blocks.``
      alone is genuinely ambiguous — both SD3's single-stream MMDiT and
      FLUX's double-stream blocks use that prefix — so it is disambiguated
      by the highest block index seen: SD3 has exactly
      ``SD3_MAX_BLOCK_INDEX + 1`` (24) blocks, indices 0-23, so any index
      above that bound cannot be SD3 and is reported as flux. In the common
      case (index within 0-23, which also covers FLUX's own 19 double-stream
      blocks, 0-18) this heuristic cannot tell the two apart and defaults to
      sd3 — callers that need certainty here should cross-check against
      ``base_arch`` (see screen.py) rather than trust this alone.

    A state dict with none of the above key shapes returns "unknown".
    """
    keys = list(state_dict)

    if any(key.startswith(KOHYA_SD3_JOINT_PREFIX) for key in keys):
        return "sd3"
    if any(
        key.startswith(KOHYA_FLUX_DOUBLE_PREFIX) or key.startswith(KOHYA_FLUX_SINGLE_PREFIX)
        for key in keys
    ):
        return "flux"
    if any(key.startswith(DIFFUSERS_FLUX_SINGLE_PREFIX) for key in keys):
        return "flux"

    block_indices = []
    for key in keys:
        if not key.startswith(DIFFUSERS_TRANSFORMER_BLOCKS_PREFIX):
            continue
        index_str = key[len(DIFFUSERS_TRANSFORMER_BLOCKS_PREFIX):].split(".", 1)[0]
        if index_str.isdigit():
            block_indices.append(int(index_str))
    if block_indices:
        return "flux" if max(block_indices) > SD3_MAX_BLOCK_INDEX else "sd3"

    return "unknown"


def introspect(state_dict: Mapping) -> SurveyInfo:
    """Describe ``state_dict``: layout, per-module ranks, touched modules.

    Raises IntrospectionError if no LoRA module is present, or if the two
    layouts are mixed in one checkpoint (which no loader would accept).
    """
    kohya: dict[str, int] = {}
    peft: dict[str, int] = {}

    for key, tensor in state_dict.items():
        if key.endswith(KOHYA_DOWN):
            kohya[key[: -len(KOHYA_DOWN)]] = _rank_of(tensor)
        elif key.endswith(PEFT_A):
            peft[key[: -len(PEFT_A)]] = _rank_of(tensor)

    if kohya and peft:
        raise IntrospectionError(
            "mixed LoRA layouts: found both kohya (lora_down) and "
            "peft (lora_A) modules in one state dict"
        )

    if kohya:
        layout, modules = "kohya", kohya
        up_suffix = KOHYA_UP
    elif peft:
        layout, modules = "peft", peft
        up_suffix = PEFT_B
    else:
        raise IntrospectionError(
            "no LoRA modules found: no key ends in "
            f"{KOHYA_DOWN!r} or {PEFT_A!r}"
        )

    missing = [name for name in modules if f"{name}{up_suffix}" not in state_dict]
    if missing:
        raise IntrospectionError(
            f"{len(missing)} module(s) have no {up_suffix} pair, "
            f"first: {missing[0]!r}"
        )

    alphas = {
        float(state_dict[f"{name}.alpha"])
        for name in modules
        if f"{name}.alpha" in state_dict
    }

    return SurveyInfo(
        layout=layout,
        ranks=frozenset(modules.values()),
        modules=tuple(sorted(modules)),
        alphas=frozenset(alphas),
        arch=detect_arch(state_dict),
    )
