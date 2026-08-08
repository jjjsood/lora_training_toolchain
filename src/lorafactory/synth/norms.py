"""Frobenius norm of an unmaterialised LoRA delta.

API pinned:
    from lorafactory.synth.norms import lora_frobenius_norm
    n = lora_frobenius_norm(A, B)   # == ||B @ A||_F, ΔW never materialised
"""

from __future__ import annotations

import torch


def lora_frobenius_norm(A: torch.Tensor, B: torch.Tensor) -> float:
    """||B @ A||_F computed without materialising the (out_dim, in_dim)
    delta matrix.

    lora_frobenius_norm(A, B) = sqrt(tr((A @ A^T) @ (B^T @ B)))
    """
    aat = A @ A.T
    btb = B.T @ B
    value = torch.trace(aat @ btb)
    # Guard against tiny negative values from floating-point round-off.
    value = torch.clamp(value, min=0.0)
    return float(torch.sqrt(value))
