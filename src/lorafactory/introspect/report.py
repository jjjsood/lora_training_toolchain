"""Per-module introspection of a LoRA checkpoint, as a CSV plus its config.

API pinned:
    from lorafactory.introspect.report import (
        detect_lora_layout, introspect_checkpoint, write_csv, write_config_json,
    )
    layout = detect_lora_layout(state_dict)            # "kohya" | "peft"
    report = introspect_checkpoint(path, base_cache=None, k=10, tau=0.5)
    write_csv(report, csv_path, adapter_id)
    write_config_json(report, json_path)

CSV header (exactly, in this order):
    adapter, block, module, rank, frob_norm, effective_rank,
    top_sigma_1 .. top_sigma_k, n_intruders, intruder_score

k truncates every direction-valued statistic, including the intruder count:
`n_intruders` is computed over the adapter's *top-k* left singular directions
(`min(k, r)` of them), not all r. The config JSON records that as
`"adapter_directions": "top_k"`, and carries `intruder_modules_matched` /
`intruder_modules_unmatched` so a partly-wrong base-key mapping is visible as a
count rather than as a scatter of blank cells.

Alpha, applied exactly once
---------------------------
A kohya module stores `lora_down`, `lora_up` and `alpha`, and the effective
update is (alpha/rank) * up @ down — the scale lives outside the weights. A
converted diffusers/PEFT module has no `.alpha` key at all because
`convert/sd3_kohya_to_diffusers.py` folds sqrt(alpha/rank) into each factor.
So the scale must be applied to a kohya checkpoint and must *not* be applied to
a PEFT one; applying it to both would report a converted checkpoint's norms
(alpha/rank) times too large. `detect_lora_layout` is what decides, per
checkpoint, which of the two worlds we are in — that is the whole guard.

The scale is applied to one factor only (`up *= alpha/rank`), never to both:
scaling both would square it.
"""

from __future__ import annotations

import csv
import json
import re
from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field
from pathlib import Path

import torch
from safetensors import safe_open

from lorafactory.introspect.base_cache import BaseSubspaceCache
from lorafactory.introspect.linalg import (
    effective_rank,
    intruder_stats,
    lora_frobenius_norm,
    lora_singular_values,
)
from lorafactory.survey.introspect import (
    KOHYA_DOWN,
    KOHYA_UP,
    PEFT_A,
    PEFT_B,
    IntrospectionError,
)

_SUFFIXES = {"kohya": (KOHYA_DOWN, KOHYA_UP), "peft": (PEFT_A, PEFT_B)}

#: Which adapter directions the intruder count is computed over. Recorded in the
#: config JSON so a thesis figure can cite the choice rather than assume it.
ADAPTER_DIRECTIONS = "top_k"

# Block index of a module name, in either spelling:
#   transformer.transformer_blocks.5.attn.to_q
#   lora_unet_joint_blocks_5_x_block_attn_qkv
#   double_blocks.5..., single_blocks.5...
_BLOCK_RE = re.compile(
    r"(?:joint_blocks|transformer_blocks|double_blocks|single_blocks|blocks|layers)[._](\d+)"
)


@dataclass(frozen=True)
class Settings:
    """The knobs a whole introspection run shares: where the base subspaces
    come from, how many directions to look at, and how much overlap still
    counts as "the base already had this direction"."""

    base_cache: BaseSubspaceCache | None = None
    k: int = 10
    tau: float = 0.5


@dataclass(frozen=True)
class ModuleStats:
    """One row: everything measurable about a single LoRA module."""

    module: str
    block: int | None
    rank: int
    frob_norm: float
    effective_rank: float
    top_sigma: tuple[float, ...]
    n_intruders: int | None = None
    intruder_score: float | None = None

    @property
    def intruder_matched(self) -> bool:
        """True iff a base subspace was found and compared for this module."""
        return self.n_intruders is not None


@dataclass(frozen=True)
class IntrospectionReport:
    """A whole checkpoint's rows plus the settings they were produced under."""

    checkpoint: str
    layout: str
    k: int
    tau: float
    base_revision: str = ""
    base_checkpoint: str = ""
    base_used: bool = False
    rows: tuple[ModuleStats, ...] = field(default_factory=tuple)

    @property
    def has_intruder_stats(self) -> bool:
        return any(row.intruder_matched for row in self.rows)

    @property
    def intruder_matched(self) -> int:
        """Modules whose base subspace was found — an intruder count was computed."""
        return sum(1 for row in self.rows if row.intruder_matched)

    @property
    def intruder_unmatched(self) -> int:
        """Modules the intruder path was attempted for and could not resolve.

        Zero when no base was supplied at all (nothing was attempted). A
        non-zero count with a base supplied is the signal that separates "this
        architecture has no pinned mapping" from "the mapping has a typo":
        both leave cells blank, only the counts say how many.
        """
        if not self.base_used:
            return 0
        return len(self.rows) - self.intruder_matched

    def unmatched_modules(self, limit: int | None = None) -> tuple[str, ...]:
        """Names of the modules with no base subspace, for a diagnostic message."""
        if not self.base_used:
            return ()
        names = tuple(row.module for row in self.rows if not row.intruder_matched)
        return names if limit is None else names[:limit]


def detect_lora_layout(sd: Mapping | Iterable[str]) -> str:
    """"kohya" or "peft", decided from key names alone (survey's suffixes).

    Raises IntrospectionError on a checkpoint that is neither, or that mixes
    both — no loader would accept a mixed one, and guessing which half to trust
    would silently mis-apply alpha.
    """
    keys = list(sd)
    has_kohya = any(key.endswith(KOHYA_DOWN) for key in keys)
    has_peft = any(key.endswith(PEFT_A) for key in keys)
    if has_kohya and has_peft:
        raise IntrospectionError(
            "mixed LoRA layouts: found both kohya (lora_down) and peft (lora_A) modules"
        )
    if has_kohya:
        return "kohya"
    if has_peft:
        return "peft"
    raise IntrospectionError(
        f"no LoRA modules found: no key ends in {KOHYA_DOWN!r} or {PEFT_A!r}"
    )


def block_of(module_name: str) -> int | None:
    """The block index a module sits in, or None when the name has no block."""
    match = _BLOCK_RE.search(module_name)
    return int(match.group(1)) if match else None


def module_stats(
    module_name: str,
    down: torch.Tensor,
    up: torch.Tensor,
    settings: Settings | None = None,
) -> ModuleStats:
    """Measure one module from its (already alpha-scaled) factors."""
    settings = settings or Settings()
    down = down.float()
    up = up.float()
    svd = lora_singular_values(down, up)
    sigma = svd.sigma
    k = settings.k

    n_intruders: int | None = None
    intruder_score: float | None = None
    if settings.base_cache is not None:
        base_topk = settings.base_cache.top_k_subspace(module_name, k)
        # A base tensor of the wrong height means the mapping does not describe
        # this checkpoint's architecture: leave the columns empty rather than
        # comparing directions that live in different spaces.
        if base_topk is not None and base_topk.shape[0] == svd.u_vectors.shape[0]:
            # Score the adapter's top-k directions, not all r of them: the same
            # truncation that governs top_sigma_1..k. Scoring all r against the
            # base's top-k would let a rank-64 adapter's 50 lowest-energy
            # directions — which carry almost none of ΔW — dominate the headline
            # number, and they are exactly the ones most likely to look novel.
            n_intruders, intruder_score = intruder_stats(
                svd.u_vectors[:, :k], base_topk, settings.tau
            )

    return ModuleStats(
        module=module_name,
        block=block_of(module_name),
        rank=int(down.shape[0]),
        frob_norm=lora_frobenius_norm(down, up),
        effective_rank=effective_rank(sigma),
        top_sigma=tuple(float(value) for value in sigma[:k]),
        n_intruders=n_intruders,
        intruder_score=intruder_score,
    )


def introspect_checkpoint(
    ckpt_path: Path | str,
    *,
    base_cache: BaseSubspaceCache | None = None,
    k: int = 10,
    tau: float = 0.5,
) -> IntrospectionReport:
    """Measure every LoRA module in a safetensors checkpoint, on CPU.

    Tensors are pulled one module at a time through `safe_open`, so peak memory
    is two factors plus one rank-sized core — a 100-module checkpoint is never
    resident in full, and ΔW is never formed at all.
    """
    ckpt_path = Path(ckpt_path)
    settings = Settings(base_cache=base_cache, k=k, tau=tau)
    rows: list[ModuleStats] = []

    with safe_open(str(ckpt_path), framework="pt") as f:
        keys = set(f.keys())
        layout = detect_lora_layout(keys)
        down_suffix, up_suffix = _SUFFIXES[layout]
        names = sorted(key[: -len(down_suffix)] for key in keys if key.endswith(down_suffix))

        for name in names:
            up_key = f"{name}{up_suffix}"
            if up_key not in keys:
                raise IntrospectionError(f"module {name!r} has no {up_suffix} pair")
            down = f.get_tensor(f"{name}{down_suffix}").float()
            up = f.get_tensor(up_key).float()

            if layout == "kohya":
                rank = down.shape[0]
                alpha_key = f"{name}.alpha"
                # kohya's own default is alpha == rank, i.e. scale 1.
                alpha = float(f.get_tensor(alpha_key)) if alpha_key in keys else float(rank)
                up = up * (alpha / rank)

            rows.append(module_stats(name, down, up, settings))

    return IntrospectionReport(
        checkpoint=str(ckpt_path),
        layout=layout,
        k=k,
        tau=tau,
        base_revision=base_cache.model_revision if base_cache is not None else "",
        base_checkpoint=(
            str(base_cache.base_checkpoint)
            if base_cache is not None and base_cache.base_checkpoint is not None
            else ""
        ),
        base_used=base_cache is not None,
        rows=tuple(rows),
    )


def csv_header(k: int) -> list[str]:
    """The pinned column list for a given k."""
    return [
        "adapter",
        "block",
        "module",
        "rank",
        "frob_norm",
        "effective_rank",
        *[f"top_sigma_{i}" for i in range(1, k + 1)],
        "n_intruders",
        "intruder_score",
    ]


def _blank_if_none(value) -> object:
    return "" if value is None else value


def write_csv(report: IntrospectionReport, path: Path | str, adapter_id: str) -> None:
    """Write the per-module CSV.

    A module of rank < k leaves the surplus `top_sigma_*` cells empty rather
    than writing zeros: a zero would read as "this direction exists and carries
    no energy", which is a different claim from "this direction does not exist".
    Intruder columns are likewise empty when no base subspace was available.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    k = report.k
    with open(path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(csv_header(k))
        for row in report.rows:
            sigma = list(row.top_sigma[:k])
            sigma += [""] * (k - len(sigma))
            writer.writerow(
                [
                    adapter_id,
                    _blank_if_none(row.block),
                    row.module,
                    row.rank,
                    row.frob_norm,
                    row.effective_rank,
                    *sigma,
                    _blank_if_none(row.n_intruders),
                    _blank_if_none(row.intruder_score),
                ]
            )


def write_config_json(report: IntrospectionReport, path: Path | str) -> None:
    """Record what the CSV next to it was produced with.

    Without this the CSV is uninterpretable a year later: tau and k change what
    `n_intruders` means, and an empty intruder column has two very different
    causes (no base checkpoint vs. an architecture with no pinned mapping).
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "adapter_directions": ADAPTER_DIRECTIONS,
        "base_checkpoint": report.base_checkpoint,
        "base_revision": report.base_revision,
        "checkpoint": report.checkpoint,
        "intruder_modules_matched": report.intruder_matched,
        "intruder_modules_unmatched": report.intruder_unmatched,
        "intruder_stats": report.has_intruder_stats,
        "k": report.k,
        "layout": report.layout,
        "module_count": len(report.rows),
        "tau": report.tau,
    }
    path.write_text(json.dumps(payload, indent=2, sort_keys=True))
