"""Linear algebra over LoRA factors — ΔW is never materialised.

API pinned:
    from lorafactory.introspect.linalg import (
        lora_frobenius_norm, lora_singular_values, effective_rank, intruder_stats,
    )

    n = lora_frobenius_norm(A, B)          # == ||B @ A||_F
    svd = lora_singular_values(A, B)       # svd.sigma / .u_vectors / .v_vectors
    er = effective_rank(svd.sigma)         # exp(-sum p_i log p_i), p_i = s_i^2/sum s^2
    n_int, score = intruder_stats(svd.u_vectors, base_topk, tau=0.5)

Factor convention throughout the package: ``A`` (``lora_down`` / ``lora_A``) is
(rank, in_dim), ``B`` (``lora_up`` / ``lora_B``) is (out_dim, rank), and the
update is ΔW = B @ A of shape (out_dim, in_dim). That product is the one matrix
this module must never build: an SD3 ff layer would be 6144x1536 floats per
module, and the whole point of the thesis instrument is that a checkpoint can be
characterised on a laptop CPU. Everything below therefore works through the
r x r Gram matrix or the r x r QR core, which are rank-sized, not layer-sized.
"""

from __future__ import annotations

from typing import NamedTuple

import torch

_MATRIX_NDIM = 2


class FactorShapeError(ValueError):
    """A down/up pair does not form a (out_dim, r) @ (r, in_dim) product."""


def _as_matrix(tensor: torch.Tensor, name: str) -> torch.Tensor:
    matrix = torch.as_tensor(tensor)
    if matrix.ndim != _MATRIX_NDIM:
        raise FactorShapeError(f"{name} must be a 2-D matrix, got shape {tuple(matrix.shape)}")
    return matrix.float()


def _checked_pair(A: torch.Tensor, B: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    a = _as_matrix(A, "A (lora_down/lora_A)")
    b = _as_matrix(B, "B (lora_up/lora_B)")
    if b.shape[1] != a.shape[0]:
        raise FactorShapeError(
            f"inner dimensions disagree: B is {tuple(b.shape)}, A is {tuple(a.shape)}"
        )
    return a, b


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


class LoraSVD(NamedTuple):
    """Thin SVD of ΔW = B @ A, obtained without forming ΔW.

    sigma:     (m,) singular values, descending.
    u_vectors: (out_dim, m) left singular vectors, one per column.
    v_vectors: (in_dim, m) right singular vectors, one per column.
    """

    sigma: torch.Tensor
    u_vectors: torch.Tensor
    v_vectors: torch.Tensor


def lora_singular_values(down: torch.Tensor, up: torch.Tensor) -> LoraSVD:
    """Exact thin SVD of ΔW = ``up`` @ ``down`` via two rank-sized QRs.

    B = Q_B R_B and A^T = Q_A R_A (reduced), so

        ΔW = B A = Q_B (R_B R_A^T) Q_A^T,

    and an SVD of the r x r core R_B R_A^T = U Σ V^T gives ΔW = (Q_B U) Σ (Q_A V)^T.
    The singular values are therefore exact, not approximate, and the largest
    matrix ever allocated is Q_B (out_dim x r) — never out_dim x in_dim.

    Rank-deficient factors are fine: LAPACK's Householder QR returns an
    orthonormal Q whatever R does, so the zero singular values come out as
    zeros rather than as garbage directions.
    """
    a, b = _checked_pair(down, up)
    q_b, r_b = torch.linalg.qr(b, mode="reduced")
    q_a, r_a = torch.linalg.qr(a.T, mode="reduced")
    u_core, sigma, vh_core = torch.linalg.svd(r_b @ r_a.T, full_matrices=False)
    return LoraSVD(sigma=sigma, u_vectors=q_b @ u_core, v_vectors=q_a @ vh_core.T)


def effective_rank(sigma: torch.Tensor) -> float:
    """Roy & Vetterli effective rank: exp(-Σ p_i log p_i), p_i = σ_i²/Σσ².

    r uniform singular values give exactly r; a single direction gives 1; an
    all-zero spectrum gives 0 (there is no direction to spend entropy on).
    """
    values = torch.as_tensor(sigma, dtype=torch.float64).flatten()
    energy = float(torch.sum(values * values))
    if energy <= 0.0:
        return 0.0
    p = (values * values) / energy
    p = p[p > 0]
    entropy = -torch.sum(p * torch.log(p))
    return float(torch.exp(entropy))


def intruder_stats(
    u_vectors: torch.Tensor,
    base_topk: torch.Tensor,
    tau: float = 0.5,
) -> tuple[int, float]:
    """Count adapter directions that miss the base model's top-k subspace.

    Direction i is an *intruder* iff max_j |⟨u_i, u⁰_j⟩| < ``tau``, where the
    u⁰_j are the columns of ``base_topk`` (the base weight's top-k left
    singular vectors). Both argument matrices are expected to have orthonormal
    columns living in the same output space, which is what
    ``lora_singular_values`` and ``BaseSubspaceCache`` produce.

    Returns ``(n_intruders, intruder_score)`` where the score is the fraction
    n_intruders / n_directions — a per-module number in [0, 1] that stays
    comparable across ranks, so an r=4 and an r=64 adapter can sit in the same
    plot.
    """
    u = _as_matrix(u_vectors, "u_vectors")
    base = _as_matrix(base_topk, "base_topk")
    if base.shape[0] != u.shape[0]:
        raise FactorShapeError(
            f"u_vectors live in R^{u.shape[0]} but base_topk lives in R^{base.shape[0]}"
        )
    n_directions = u.shape[1]
    if n_directions == 0:
        return 0, 0.0
    if base.shape[1] == 0:
        # No base subspace at all: every adapter direction is new by definition.
        return n_directions, 1.0

    best_overlap = (u.T @ base).abs().max(dim=1).values
    n_intruders = int((best_overlap < tau).sum())
    return n_intruders, n_intruders / n_directions
