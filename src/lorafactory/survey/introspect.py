"""Pure state-dict introspection of a community LoRA checkpoint (CAP-28).

API pinned:
    from lorafactory.survey import introspect
    info = introspect.introspect(state_dict)
    info.layout        # "kohya" | "peft"
    info.ranks         # set of per-module ranks, e.g. {4, 32}
    info.module_count  # number of LoRA modules
    info.modules       # sorted tuple of module names

No model loading, no network, no GPU: the checkpoint is described from its
key names and tensor shapes alone, before anything is put on a device.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field

KOHYA_DOWN = ".lora_down.weight"
KOHYA_UP = ".lora_up.weight"
PEFT_A = ".lora_A.weight"
PEFT_B = ".lora_B.weight"


class IntrospectionError(ValueError):
    """The state dict holds no recognisable LoRA modules, or mixes layouts."""


@dataclass(frozen=True)
class SurveyInfo:
    """What a checkpoint is, as read off its keys and shapes."""

    layout: str
    ranks: frozenset[int]
    modules: tuple[str, ...] = field(default_factory=tuple)
    alphas: frozenset[float] = frozenset()

    @property
    def module_count(self) -> int:
        return len(self.modules)


def _rank_of(tensor) -> int:
    shape = tuple(tensor.shape)
    if len(shape) < 1:
        raise IntrospectionError("LoRA down/A weight is not a matrix")
    return int(shape[0])


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
    )
