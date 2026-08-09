"""Frobenius norm of an unmaterialised LoRA delta.

Re-export shim: the implementation moved to `lorafactory.introspect.linalg`,
where it sits next to the rest of the ΔW-free linear algebra (SVD via QR,
effective rank). The synthesiser and its tests keep importing it from here, so
the pinned API below is unchanged.

API pinned:
    from lorafactory.synth.norms import lora_frobenius_norm
    n = lora_frobenius_norm(A, B)   # == ||B @ A||_F, ΔW never materialised
"""

from __future__ import annotations

from lorafactory.introspect.linalg import lora_frobenius_norm

__all__ = ["lora_frobenius_norm"]
