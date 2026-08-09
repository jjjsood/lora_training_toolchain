"""Extended weight introspection — per-module spectra of a LoRA checkpoint.

API pinned:
    from lorafactory.introspect.linalg import (
        lora_frobenius_norm, lora_singular_values, effective_rank, intruder_stats)
    svd = lora_singular_values(down, up)   # .sigma / .u_vectors / .v_vectors
    er  = effective_rank(svd.sigma)
    n_intruders, score = intruder_stats(svd.u_vectors, base_topk, tau)

    from lorafactory.introspect.base_cache import BaseSubspaceCache, load_base_weight
    from lorafactory.introspect.report import (
        detect_lora_layout, introspect_checkpoint, write_csv, write_config_json)

Every numeric expectation below is an analytic value or a dense `torch.linalg`
reference computed on a deliberately tiny matrix. The library never materialises
ΔW = B @ A; these tests do, because that is the only independent check of the
QR/Gram identities the library relies on.
"""

import csv
import json
import math

import pytest
import torch
from click.testing import CliRunner
from safetensors.torch import save_file

from conftest import MATRIX, delta_w_kohya, delta_w_peft, kohya_module, peft_module
from lorafactory.cli import cli
from lorafactory.convert.sd3_kohya_to_diffusers import convert
from lorafactory.introspect.base_cache import (
    SD3_BASE_KEY_MAP,
    BaseKeyRef,
    BaseSubspaceCache,
    base_key_for,
    load_base_weight,
)
from lorafactory.introspect.linalg import (
    FactorShapeError,
    effective_rank,
    intruder_stats,
    lora_frobenius_norm,
    lora_singular_values,
)
from lorafactory.introspect.report import (
    csv_header,
    detect_lora_layout,
    introspect_checkpoint,
    write_config_json,
    write_csv,
)
from lorafactory.survey.introspect import IntrospectionError

REL_TOL = 1e-5


def close(actual: float, expected: float, tol: float = REL_TOL) -> bool:
    return abs(actual - expected) <= tol * max(abs(expected), 1.0)


def write_checkpoint(path, sd):
    save_file({k: v.contiguous() for k, v in sd.items()}, str(path))
    return path


def random_factors(rank, out_dim, in_dim, seed=0):
    g = torch.Generator().manual_seed(seed)
    return (torch.randn(rank, in_dim, generator=g), torch.randn(out_dim, rank, generator=g))


def orthonormal(dim, cols, seed=0):
    """A (dim, cols) matrix with orthonormal columns."""
    g = torch.Generator().manual_seed(seed)
    q, _ = torch.linalg.qr(torch.randn(dim, cols, generator=g), mode="reduced")
    return q


# --------------------------------------------------------------------------
# linalg: norms, singular values, singular vectors
# --------------------------------------------------------------------------


def test_frobenius_norm_matches_the_dense_delta():
    down, up = random_factors(4, 12, 7, seed=11)
    dense = torch.linalg.matrix_norm(up @ down, ord="fro").item()
    assert close(lora_frobenius_norm(down, up), dense, 1e-4)


def test_singular_values_match_dense_svd():
    down, up = random_factors(5, 16, 9, seed=12)
    dense_sigma = torch.linalg.svdvals(up @ down)
    sigma = lora_singular_values(down, up).sigma
    assert sigma.shape == (5,)
    assert torch.allclose(sigma, dense_sigma[:5], rtol=1e-5, atol=1e-6)
    # Descending, as every consumer (top_sigma_1..k) assumes.
    assert torch.all(sigma[:-1] >= sigma[1:] - 1e-9)


def test_singular_vectors_reconstruct_the_delta():
    down, up = random_factors(4, 10, 6, seed=13)
    svd = lora_singular_values(down, up)
    rebuilt = svd.u_vectors @ torch.diag(svd.sigma) @ svd.v_vectors.T
    assert torch.allclose(rebuilt, up @ down, atol=1e-5)
    # Both vector sets are orthonormal — intruder_stats assumes unit columns.
    eye = torch.eye(4)
    assert torch.allclose(svd.u_vectors.T @ svd.u_vectors, eye, atol=1e-5)
    assert torch.allclose(svd.v_vectors.T @ svd.v_vectors, eye, atol=1e-5)


def test_singular_values_of_a_rank_deficient_pair():
    """A duplicated factor row makes ΔW rank 2 while r == 4: the surplus
    singular values must come out as zeros, not as noise directions."""
    down, up = random_factors(4, 10, 6, seed=14)
    down[2] = down[0]
    down[3] = down[1]
    up[:, 2] = up[:, 0]
    up[:, 3] = up[:, 1]
    sigma = lora_singular_values(down, up).sigma
    dense = torch.linalg.svdvals(up @ down)[:4]
    assert torch.allclose(sigma, dense, atol=1e-4)
    assert sigma[2] < 1e-4 and sigma[3] < 1e-4


def test_singular_values_when_rank_exceeds_a_dimension():
    """r > out_dim: the thin SVD has min(out, in, r) values, and the QR core
    is no longer square — the dense reference still has to agree."""
    down, up = random_factors(8, 3, 5, seed=15)
    svd = lora_singular_values(down, up)
    dense = torch.linalg.svdvals(up @ down)
    assert svd.sigma.shape[0] == 3
    assert torch.allclose(svd.sigma, dense, atol=1e-4)
    rebuilt = svd.u_vectors @ torch.diag(svd.sigma) @ svd.v_vectors.T
    assert torch.allclose(rebuilt, up @ down, atol=1e-4)


def test_zero_factors_give_zero_spectrum():
    down = torch.zeros(3, 5)
    up = torch.zeros(7, 3)
    svd = lora_singular_values(down, up)
    assert torch.all(svd.sigma == 0)
    assert lora_frobenius_norm(down, up) == 0.0
    assert effective_rank(svd.sigma) == 0.0


def test_mismatched_factors_are_rejected():
    with pytest.raises(FactorShapeError):
        lora_singular_values(torch.randn(4, 6), torch.randn(8, 5))
    with pytest.raises(FactorShapeError):
        lora_singular_values(torch.randn(4), torch.randn(8, 4))


# --------------------------------------------------------------------------
# linalg: effective rank
# --------------------------------------------------------------------------


@pytest.mark.parametrize("rank", [1, 2, 5, 16])
def test_effective_rank_of_a_uniform_spectrum_is_the_rank(rank):
    sigma = torch.full((rank,), 0.37)
    assert close(effective_rank(sigma), float(rank), 1e-6)


def test_effective_rank_of_a_single_direction_is_one():
    sigma = torch.tensor([4.0, 0.0, 0.0, 0.0])
    assert close(effective_rank(sigma), 1.0, 1e-6)
    # A lone value, no padding at all — the log(0) edge must not appear.
    assert close(effective_rank(torch.tensor([2.5])), 1.0, 1e-6)


def test_effective_rank_is_scale_invariant_and_bounded():
    sigma = torch.tensor([3.0, 2.0, 1.0, 0.5])
    er = effective_rank(sigma)
    assert close(effective_rank(sigma * 100.0), er, 1e-6)
    assert 1.0 < er < 4.0


def test_effective_rank_matches_the_closed_form():
    """Two directions with energies 4:1 -> exp(-(0.8 ln0.8 + 0.2 ln0.2))."""
    sigma = torch.tensor([2.0, 1.0])
    p = torch.tensor([0.8, 0.2])
    expected = math.exp(float(-(p * torch.log(p)).sum()))
    assert close(effective_rank(sigma), expected, 1e-6)


# --------------------------------------------------------------------------
# linalg: intruder dimensions
# --------------------------------------------------------------------------


def test_directions_inside_the_base_subspace_are_not_intruders():
    q = orthonormal(24, 8, seed=21)
    base_topk = q[:, :6]
    n_intruders, score = intruder_stats(q[:, :4], base_topk, 0.5)
    assert (n_intruders, score) == (0, 0.0)


def test_random_gaussian_directions_in_high_dimension_are_all_intruders():
    """Two independent random subspaces of R^512 overlap by ~1/sqrt(512)."""
    u = orthonormal(512, 8, seed=22)
    base_topk = orthonormal(512, 10, seed=23)
    n_intruders, score = intruder_stats(u, base_topk, 0.5)
    assert n_intruders == 8
    assert score == 1.0


def test_intruder_count_splits_a_mixed_set():
    q = orthonormal(32, 8, seed=24)
    base_topk = q[:, :2]
    # Columns 0,1 sit exactly in the base subspace; 2..4 are orthogonal to it.
    n_intruders, score = intruder_stats(q[:, :5], base_topk, 0.5)
    assert n_intruders == 3
    assert close(score, 3 / 5, 1e-9)


def test_tau_decides_the_borderline_case():
    q = orthonormal(16, 4, seed=25)
    mixed = ((q[:, 0] + q[:, 1]) / math.sqrt(2.0)).reshape(-1, 1)  # overlap 1/sqrt2
    assert intruder_stats(mixed, q[:, :1], 0.5)[0] == 0
    assert intruder_stats(mixed, q[:, :1], 0.9)[0] == 1


def test_intruder_stats_rejects_a_dimension_mismatch():
    with pytest.raises(FactorShapeError):
        intruder_stats(orthonormal(16, 4, seed=26), orthonormal(8, 4, seed=27), 0.5)


# --------------------------------------------------------------------------
# layout detection and the alpha double-apply guard
# --------------------------------------------------------------------------


def test_detect_lora_layout():
    assert detect_lora_layout(kohya_module("m", 2, 4, 4, alpha=2)) == "kohya"
    assert detect_lora_layout(peft_module("m", 2, 4, 4)) == "peft"
    with pytest.raises(IntrospectionError):
        detect_lora_layout({**kohya_module("a", 2, 4, 4, alpha=2), **peft_module("b", 2, 4, 4)})
    with pytest.raises(IntrospectionError):
        detect_lora_layout({"model.diffusion_model.x.weight": torch.zeros(2, 2)})


def test_kohya_alpha_scaling_is_applied(tmp_path):
    prefix = "lora_unet_joint_blocks_0_x_block_attn_proj"
    sd = kohya_module(prefix, rank=4, out_dim=8, in_dim=8, alpha=8.0, seed=31)
    path = write_checkpoint(tmp_path / "kohya.safetensors", sd)

    row = introspect_checkpoint(path).rows[0]
    scaled = torch.linalg.matrix_norm(delta_w_kohya(sd, prefix), ord="fro").item()
    unscaled = torch.linalg.matrix_norm(
        sd[f"{prefix}.lora_up.weight"] @ sd[f"{prefix}.lora_down.weight"], ord="fro"
    ).item()

    assert close(row.frob_norm, scaled, 1e-4)
    assert not close(row.frob_norm, unscaled, 1e-3), "alpha/rank was not applied"
    assert close(scaled, 2.0 * unscaled, 1e-4), "fixture sanity: alpha/rank == 2"
    # The scale must land on the spectrum too, not just on the norm.
    dense_sigma = torch.linalg.svdvals(delta_w_kohya(sd, prefix))[:4]
    assert torch.allclose(torch.tensor(row.top_sigma), dense_sigma, atol=1e-4)


def test_kohya_without_an_alpha_key_uses_scale_one(tmp_path):
    prefix = "lora_unet_joint_blocks_1_x_block_attn_proj"
    sd = kohya_module(prefix, rank=4, out_dim=8, in_dim=8, alpha=8.0, seed=32)
    del sd[f"{prefix}.alpha"]
    path = write_checkpoint(tmp_path / "noalpha.safetensors", sd)

    row = introspect_checkpoint(path).rows[0]
    raw = torch.linalg.matrix_norm(
        sd[f"{prefix}.lora_up.weight"] @ sd[f"{prefix}.lora_down.weight"], ord="fro"
    ).item()
    assert close(row.frob_norm, raw, 1e-4)


def test_peft_layout_is_not_scaled(tmp_path):
    name = "transformer.transformer_blocks.0.attn.to_out.0"
    sd = peft_module(name, rank=4, out_dim=8, in_dim=8, seed=33)
    path = write_checkpoint(tmp_path / "peft.safetensors", sd)

    row = introspect_checkpoint(path).rows[0]
    dense = torch.linalg.matrix_norm(delta_w_peft(sd, name), ord="fro").item()
    assert close(row.frob_norm, dense, 1e-4)


def test_converted_checkpoint_is_not_double_scaled(tmp_path):
    """The other direction of the guard: a kohya module with alpha != rank and
    its converted PEFT twin describe the same ΔW, so they must introspect to
    the same norm. Scaling the converted one again would multiply it by 2."""
    prefix = "lora_unet_joint_blocks_2_x_block_attn_proj"
    kohya_sd = kohya_module(prefix, rank=4, out_dim=8, in_dim=8, alpha=8.0, seed=34)
    peft_sd = convert(kohya_sd)
    assert not any(key.endswith(".alpha") for key in peft_sd), "converter folds alpha"

    kohya_row = introspect_checkpoint(
        write_checkpoint(tmp_path / "k.safetensors", kohya_sd)
    ).rows[0]
    peft_row = introspect_checkpoint(
        write_checkpoint(tmp_path / "p.safetensors", peft_sd)
    ).rows[0]

    assert close(peft_row.frob_norm, kohya_row.frob_norm, 1e-4)
    assert torch.allclose(
        torch.tensor(peft_row.top_sigma), torch.tensor(kohya_row.top_sigma), atol=1e-4
    )


# --------------------------------------------------------------------------
# base subspace cache
# --------------------------------------------------------------------------


def base_checkpoint_with(path, key, weight):
    save_file({key: weight.contiguous()}, str(path))
    return path


def projector(u):
    return u @ u.T


def test_base_key_map_covers_the_sd3_vocabulary():
    from lorafactory.constants import SD3_ATTN_LEAVES, SD3_BLOCK23_ABSENT, SD3_MLP_LEAVES

    for block in range(24):
        for leaf in (*SD3_ATTN_LEAVES, *SD3_MLP_LEAVES):
            name = f"transformer.transformer_blocks.{block}.{leaf}"
            if block == 23 and leaf in SD3_BLOCK23_ABSENT:
                assert base_key_for(name) is None
                continue
            assert base_key_for(name) is not None, name

    # q/k/v are three slices of one fused base tensor, in order.
    refs = [
        SD3_BASE_KEY_MAP[f"transformer.transformer_blocks.3.attn.{leaf}"]
        for leaf in ("to_q", "to_k", "to_v")
    ]
    assert {ref.key for ref in refs} == {
        "model.diffusion_model.joint_blocks.3.x_block.attn.qkv.weight"
    }
    assert [ref.part for ref in refs] == [0, 1, 2]
    assert all(ref.parts == 3 for ref in refs)

    # The kohya spelling keeps qkv whole.
    kohya_ref = SD3_BASE_KEY_MAP["lora_unet_joint_blocks_3_context_block_attn_qkv"]
    assert kohya_ref == BaseKeyRef(
        "model.diffusion_model.joint_blocks.3.context_block.attn.qkv.weight"
    )
    assert base_key_for("transformer.transformer_blocks.0.attn.to_q.nonsense") is None


def test_load_base_weight_slices_a_fused_projection(tmp_path):
    g = torch.Generator().manual_seed(41)
    fused = torch.randn(12, 4, generator=g)
    key = "model.diffusion_model.joint_blocks.0.x_block.attn.qkv.weight"
    path = base_checkpoint_with(tmp_path / "base.safetensors", key, fused)

    whole = load_base_weight(path, key)
    assert torch.allclose(whole, fused)
    for part in range(3):
        sliced = load_base_weight(path, BaseKeyRef(key, part=part, parts=3))
        assert torch.allclose(sliced, fused[part * 4 : (part + 1) * 4, :])
    assert load_base_weight(path, "no.such.key") is None


def test_base_subspace_cache_roundtrip_and_hit(tmp_path):
    module = "transformer.transformer_blocks.0.attn.to_out.0"
    key = "model.diffusion_model.joint_blocks.0.x_block.attn.proj.weight"
    g = torch.Generator().manual_seed(42)
    weight = torch.randn(12, 12, generator=g)
    base = base_checkpoint_with(tmp_path / "base.safetensors", key, weight)

    cache = BaseSubspaceCache(tmp_path, "rev-abc", base_checkpoint=base)
    first = cache.top_k_subspace(module, 4)
    assert first.shape == (12, 4)
    assert (cache.misses, cache.hits) == (1, 0)

    expected = torch.linalg.svd(weight, full_matrices=False)[0][:, :4]
    assert torch.allclose(projector(first), projector(expected), atol=1e-5)

    cached_path = tmp_path / ".cache" / "introspect" / "rev-abc" / f"{module}.k4.safetensors"
    assert cached_path.exists()

    second = cache.top_k_subspace(module, 4)
    assert (cache.misses, cache.hits) == (1, 1)
    assert torch.allclose(second, first)

    # A cache-only instance (no base checkpoint) still answers from disk...
    offline = BaseSubspaceCache(tmp_path, "rev-abc")
    assert torch.allclose(offline.top_k_subspace(module, 4), first)
    assert offline.hits == 1
    # ...but not for a different revision, a different k, or an unmapped module.
    assert BaseSubspaceCache(tmp_path, "rev-other").top_k_subspace(module, 4) is None
    assert offline.top_k_subspace(module, 6) is None
    assert cache.top_k_subspace("some.flux.module", 4) is None


# --------------------------------------------------------------------------
# intruder statistic end to end
# --------------------------------------------------------------------------


def aligned_lora(base_dirs, in_dim, seed):
    """Factors whose ΔW has exactly ``base_dirs``' columns as left vectors.

    ΔW = Q diag(c) R with c descending and R row-orthonormal is already an SVD,
    so the module's u_vectors are Q's columns — no rotation to reason about.
    """
    rank = base_dirs.shape[1]
    row_space = orthonormal(in_dim, rank, seed=seed).T
    scales = torch.tensor([float(rank - i) for i in range(rank)])
    return torch.diag(scales) @ row_space, base_dirs.clone()


def sd3_base_with_subspace(path, key, u_columns, in_dim, seed):
    """A base weight whose top-k left singular vectors are ``u_columns``."""
    k = u_columns.shape[1]
    right = orthonormal(in_dim, k, seed=seed)
    scales = torch.tensor([float(100 - i) for i in range(k)])
    weight = u_columns @ torch.diag(scales) @ right.T
    return base_checkpoint_with(path, key, weight)


def test_lora_aligned_with_the_base_subspace_has_no_intruders(tmp_path):
    module = "transformer.transformer_blocks.0.attn.to_out.0"
    key = "model.diffusion_model.joint_blocks.0.x_block.attn.proj.weight"
    dim = 32
    q = orthonormal(dim, 6, seed=51)

    base = sd3_base_with_subspace(tmp_path / "base.safetensors", key, q, dim, seed=52)
    down, up = aligned_lora(q[:, :4], dim, seed=53)
    ckpt = write_checkpoint(
        tmp_path / "aligned.safetensors",
        {f"{module}.lora_A.weight": down, f"{module}.lora_B.weight": up},
    )

    cache = BaseSubspaceCache(tmp_path, "rev-aligned", base_checkpoint=base)
    report = introspect_checkpoint(ckpt, base_cache=cache, k=6, tau=0.5)
    row = report.rows[0]
    assert row.n_intruders == 0
    assert row.intruder_score == 0.0
    assert report.has_intruder_stats


def test_random_lora_against_a_random_base_is_all_intruders(tmp_path):
    module = "transformer.transformer_blocks.0.attn.to_out.0"
    key = "model.diffusion_model.joint_blocks.0.x_block.attn.proj.weight"
    dim = 512
    g = torch.Generator().manual_seed(54)

    base = base_checkpoint_with(
        tmp_path / "base.safetensors", key, torch.randn(dim, dim, generator=g)
    )
    down = torch.randn(8, dim, generator=g)
    up = torch.randn(dim, 8, generator=g)
    ckpt = write_checkpoint(
        tmp_path / "random.safetensors",
        {f"{module}.lora_A.weight": down, f"{module}.lora_B.weight": up},
    )

    cache = BaseSubspaceCache(tmp_path, "rev-random", base_checkpoint=base)
    row = introspect_checkpoint(ckpt, base_cache=cache, k=10, tau=0.5).rows[0]
    assert row.n_intruders == 8
    assert row.intruder_score == 1.0


def test_intruder_columns_are_empty_without_a_base(tmp_path):
    name = "transformer.transformer_blocks.4.attn.to_q"
    ckpt = write_checkpoint(
        tmp_path / "a.safetensors", peft_module(name, 4, 8, 8, seed=55)
    )
    report = introspect_checkpoint(ckpt, k=5)
    row = report.rows[0]
    assert row.n_intruders is None and row.intruder_score is None
    assert report.has_intruder_stats is False
    # ... and the other three statistics are still there.
    assert row.frob_norm > 0 and row.effective_rank > 0 and len(row.top_sigma) == 4


# --------------------------------------------------------------------------
# report: rows, CSV, JSON config
# --------------------------------------------------------------------------


def mixed_rank_checkpoint(path):
    sd = {}
    sd.update(peft_module("transformer.transformer_blocks.0.attn.to_q", 2, 8, 8, seed=61))
    sd.update(peft_module("transformer.transformer_blocks.7.ff.net.2", 4, 8, 16, seed=62))
    return write_checkpoint(path, sd), sd


def test_rows_carry_block_module_and_rank(tmp_path):
    ckpt, _ = mixed_rank_checkpoint(tmp_path / "mixed.safetensors")
    report = introspect_checkpoint(ckpt, k=3)
    assert report.layout == "peft"
    assert [row.block for row in report.rows] == [0, 7]
    assert [row.rank for row in report.rows] == [2, 4]
    assert [len(row.top_sigma) for row in report.rows] == [2, 3]


def test_block_index_is_read_from_kohya_names_too(tmp_path):
    sd = kohya_module(
        "lora_unet_joint_blocks_11_context_block_mlp_fc1", 2, 8, 8, alpha=2.0, seed=63
    )
    row = introspect_checkpoint(write_checkpoint(tmp_path / "k.safetensors", sd)).rows[0]
    assert row.block == 11


def test_csv_columns_and_top_sigma_padding(tmp_path):
    ckpt, _ = mixed_rank_checkpoint(tmp_path / "mixed.safetensors")
    report = introspect_checkpoint(ckpt, k=3)
    out = tmp_path / "introspect.csv"
    write_csv(report, out, "L-A")

    with open(out, newline="") as f:
        rows = list(csv.reader(f))

    assert rows[0] == [
        "adapter", "block", "module", "rank", "frob_norm", "effective_rank",
        "top_sigma_1", "top_sigma_2", "top_sigma_3", "n_intruders", "intruder_score",
    ]
    assert rows[0] == csv_header(3)
    assert len(rows) == 3

    rank2, rank4 = rows[1], rows[2]
    assert rank2[0] == "L-A" and rank2[1] == "0" and rank2[3] == "2"
    # rank 2 < k 3: the surplus column is empty, not a zero.
    assert rank2[8] == ""
    assert rank4[8] != ""
    # No base supplied -> both intruder cells empty on every row.
    assert rank2[-2:] == ["", ""]
    assert rank4[-2:] == ["", ""]

    sigmas = [float(value) for value in rank4[6:9]]
    assert sigmas == sorted(sigmas, reverse=True)
    # The rank-2 row's spectrum is complete (2 <= k), so its energy must be the
    # whole Frobenius norm; the rank-4 row's is truncated and must fall short.
    full = [float(value) for value in rank2[6:8]]
    assert close(float(rank2[4]), math.sqrt(sum(s * s for s in full)), 1e-4)
    assert math.sqrt(sum(s * s for s in sigmas)) < float(rank4[4])


def test_csv_records_intruder_columns_when_a_base_is_supplied(tmp_path):
    module = "transformer.transformer_blocks.0.attn.to_out.0"
    key = "model.diffusion_model.joint_blocks.0.x_block.attn.proj.weight"
    dim = 32
    q = orthonormal(dim, 6, seed=64)
    base = sd3_base_with_subspace(tmp_path / "base.safetensors", key, q, dim, seed=65)
    down, up = aligned_lora(q[:, :4], dim, seed=66)
    ckpt = write_checkpoint(
        tmp_path / "aligned.safetensors",
        {f"{module}.lora_A.weight": down, f"{module}.lora_B.weight": up},
    )

    cache = BaseSubspaceCache(tmp_path, "rev-csv", base_checkpoint=base)
    report = introspect_checkpoint(ckpt, base_cache=cache, k=6, tau=0.5)
    out = tmp_path / "with_base.csv"
    write_csv(report, out, "L-A")
    row = list(csv.reader(out.open(newline="")))[1]
    assert row[-2] == "0" and float(row[-1]) == 0.0


def test_config_json_records_k_tau_and_base_revision(tmp_path):
    ckpt, _ = mixed_rank_checkpoint(tmp_path / "mixed.safetensors")
    cache = BaseSubspaceCache(tmp_path, "sha-1234", base_checkpoint=None)
    report = introspect_checkpoint(ckpt, base_cache=cache, k=7, tau=0.42)

    out = tmp_path / "introspect.config.json"
    write_config_json(report, out)
    payload = json.loads(out.read_text())

    assert payload["k"] == 7
    assert payload["tau"] == 0.42
    assert payload["base_revision"] == "sha-1234"
    assert payload["layout"] == "peft"
    assert payload["module_count"] == 2
    # No base weights were reachable, so the empty intruder columns are recorded.
    assert payload["intruder_stats"] is False
    assert out.read_text() == json.dumps(payload, indent=2, sort_keys=True)


def test_checkpoint_without_lora_modules_is_rejected(tmp_path):
    path = write_checkpoint(tmp_path / "plain.safetensors", {"some.weight": torch.zeros(2, 2)})
    with pytest.raises(IntrospectionError):
        introspect_checkpoint(path)


# --------------------------------------------------------------------------
# L-A cross-check: an attn-only adapter must produce no ff/mlp rows
# --------------------------------------------------------------------------

L_A_BLOCKS = (0, 5, 23)


def l_a_kohya_checkpoint():
    """An L-A-shaped kohya SD3 adapter: attention leaves only, rank 16."""
    sd = {}
    for block in L_A_BLOCKS:
        for stream in ("x", "context"):
            for leaf, out_dim in (("attn_qkv", 24), ("attn_proj", 8)):
                if block == 23 and stream == "context" and leaf == "attn_proj":
                    continue  # context-pre-only block has no output projection
                name = f"lora_unet_joint_blocks_{block}_{stream}_block_{leaf}"
                sd.update(kohya_module(name, 16, out_dim, 8, alpha=16.0, seed=block + 1))
    return sd


def test_l_a_style_checkpoint_has_no_ff_or_mlp_rows(tmp_path):
    sd = l_a_kohya_checkpoint()
    report = introspect_checkpoint(write_checkpoint(tmp_path / "L-A.safetensors", sd))

    assert report.layout == "kohya"
    assert len(report.rows) == 11  # 3 blocks x 2 streams x 2 leaves, minus block 23 context proj
    offenders = [r.module for r in report.rows if "mlp" in r.module or "ff" in r.module]
    assert offenders == [], f"attn-only adapter reported non-attn modules: {offenders}"
    assert {row.block for row in report.rows} == set(L_A_BLOCKS)
    assert {row.rank for row in report.rows} == {16}


def test_l_a_modules_are_contained_in_the_verify_keys_inventory(tmp_path):
    """Cross-check against the other half of the toolchain: every module the
    introspector reports for a converted L-A adapter must be one `verify-keys`
    expects for L-A. Two independent name derivations, one answer."""
    from lorafactory.config.loader import resolve
    from lorafactory.verify.key_inventory import expected_module_names

    peft_sd = convert(l_a_kohya_checkpoint())
    report = introspect_checkpoint(write_checkpoint(tmp_path / "conv.safetensors", peft_sd))

    expected = expected_module_names(resolve(MATRIX / "L-A.yaml").data["target"])
    reported = {row.module for row in report.rows}
    assert reported, "converted L-A checkpoint produced no rows"
    assert reported <= expected, f"not in the L-A inventory: {sorted(reported - expected)}"
    assert not any("ff" in name for name in reported)


# --------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------


def test_cli_introspect_writes_csv_and_config(tmp_path):
    ckpt, _ = mixed_rank_checkpoint(tmp_path / "adapter.safetensors")
    out_dir = tmp_path / "out"
    result = CliRunner().invoke(
        cli,
        ["introspect", str(ckpt), "--out", str(out_dir), "--k", "4", "--tau", "0.6",
         "--adapter-id", "L-A"],
    )
    assert result.exit_code == 0, result.output

    csv_path = out_dir / "introspect.csv"
    config_path = out_dir / "introspect.config.json"
    assert csv_path.exists() and config_path.exists()

    rows = list(csv.reader(csv_path.open(newline="")))
    assert rows[0] == csv_header(4)
    assert all(row[0] == "L-A" for row in rows[1:])

    payload = json.loads(config_path.read_text())
    assert payload["k"] == 4 and payload["tau"] == 0.6


def test_cli_introspect_defaults_the_adapter_id_to_the_stem(tmp_path):
    ckpt, _ = mixed_rank_checkpoint(tmp_path / "L-R64.safetensors")
    out_dir = tmp_path / "out"
    result = CliRunner().invoke(cli, ["introspect", str(ckpt), "--out", str(out_dir)])
    assert result.exit_code == 0, result.output
    rows = list(csv.reader((out_dir / "introspect.csv").open(newline="")))
    assert {row[0] for row in rows[1:]} == {"L-R64"}


def test_cli_introspect_fails_loudly_on_a_non_lora_checkpoint(tmp_path):
    path = write_checkpoint(tmp_path / "plain.safetensors", {"w": torch.zeros(2, 2)})
    result = CliRunner().invoke(cli, ["introspect", str(path), "--out", str(tmp_path / "o")])
    assert result.exit_code != 0
    assert "no LoRA modules" in result.output
