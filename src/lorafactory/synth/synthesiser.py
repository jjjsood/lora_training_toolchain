"""Norm-matched random adapter synthesiser — CAP-27 / README acceptance #6.

API pinned:
    from lorafactory.synth.synthesiser import synthesise
    out_sd, recipe = synthesise(reference_sd, seed=..., tolerance=1e-4)
    # out_sd: same module names & ranks as reference, Gaussian, rescaled so
    #         per-module ||B@A||_F matches the reference within tolerance
    # recipe: {"seed": int, "distribution": "gaussian",
    #          "modules": {name: {"target_norm", "achieved_norm", "rank"}}}
"""

from __future__ import annotations

import hashlib
from collections.abc import Mapping

import torch

from lorafactory.synth.norms import lora_frobenius_norm

_A_SUFFIX = ".lora_A.weight"
_B_SUFFIX = ".lora_B.weight"


def _module_names(sd: Mapping[str, torch.Tensor]) -> list[str]:
    """Sorted, insertion-order-independent set of module base names."""
    return sorted({k.rsplit(".lora_", 1)[0] for k in sd})


def _module_seed(global_seed: int, name: str) -> int:
    """Deterministic per-module seed derived from (global seed, module
    name). Depends only on the pair, never on dict/insertion order."""
    digest = hashlib.sha256(f"{global_seed}:{name}".encode()).digest()
    # Keep within the range accepted by torch.Generator.manual_seed.
    return int.from_bytes(digest[:8], "big") % (2**63 - 1)


def synthesise(
    reference_sd: Mapping[str, torch.Tensor],
    seed: int,
    tolerance: float = 1e-4,
) -> tuple[dict, dict]:
    """Synthesise a random Gaussian LoRA state dict whose per-module
    ||B @ A||_F matches ``reference_sd`` within relative ``tolerance``.

    Module names and ranks are taken from ``reference_sd``. Each module's
    A, B pair is drawn from its own generator seeded from
    (``seed``, module name), so the result is independent of the dict's
    insertion order and reproducible for a fixed ``seed``.
    """
    names = _module_names(reference_sd)

    out_sd: dict = {}
    modules_recipe: dict = {}

    for name in names:
        a_key = f"{name}{_A_SUFFIX}"
        b_key = f"{name}{_B_SUFFIX}"
        ref_a = reference_sd[a_key]
        ref_b = reference_sd[b_key]
        rank = ref_a.shape[0]

        target_norm = lora_frobenius_norm(ref_a.float(), ref_b.float())

        gen = torch.Generator().manual_seed(_module_seed(seed, name))
        a = torch.randn(tuple(ref_a.shape), generator=gen, dtype=ref_a.dtype)
        b = torch.randn(tuple(ref_b.shape), generator=gen, dtype=ref_b.dtype)

        raw_norm = lora_frobenius_norm(a.float(), b.float())
        if raw_norm > 0:
            scale = (target_norm / raw_norm) ** 0.5
        else:
            scale = 0.0
        a = a * scale
        b = b * scale

        achieved_norm = lora_frobenius_norm(a.float(), b.float())

        out_sd[a_key] = a
        out_sd[b_key] = b
        modules_recipe[name] = {
            "target_norm": target_norm,
            "achieved_norm": achieved_norm,
            "rank": int(rank),
        }

    recipe = {
        "seed": seed,
        "distribution": "gaussian",
        "modules": modules_recipe,
    }
    return out_sd, recipe
