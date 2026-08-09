"""Config → kohya network spec. Exact per-adapter network_args.

API pinned:
    from lorafactory.kohya.network_args import build_network_args, TargetSpecError
    spec = build_network_args(config_dict)   # config_dict == ResolvedConfig.data
    spec.network_module   # "networks.lora_sd3" | "networks.lora_flux"
    spec.network_dim      # int
    spec.network_alpha    # int
    spec.network_args     # list[str] of "key=value"
    spec.flags            # dict of extra kohya keys (e.g. network_train_text_encoder_only)
"""

import copy

import pytest

from conftest import MATRIX
from lorafactory.config.loader import resolve
from lorafactory.kohya.network_args import TargetSpecError, build_network_args

MOD_OFF = {"context_mod_dim=0", "x_mod_dim=0"}


def spec_for(adapter_id):
    return build_network_args(resolve(MATRIX / f"{adapter_id}.yaml").data)


@pytest.mark.parametrize("aid", ["L-F", "O-F"])
def test_full_adapters(aid):
    s = spec_for(aid)
    assert s.network_module == "networks.lora_sd3"
    assert (s.network_dim, s.network_alpha) == (16, 16)
    assert set(s.network_args) == MOD_OFF | {"train_block_indices=0-23"}


@pytest.mark.parametrize("aid", ["L-A", "O-A"])
def test_attention_only(aid):
    s = spec_for(aid)
    assert set(s.network_args) == MOD_OFF | {
        "context_mlp_dim=0", "x_mlp_dim=0", "train_block_indices=0-23"}


@pytest.mark.parametrize("aid", ["L-M", "O-M"])
def test_mlp_only(aid):
    s = spec_for(aid)
    assert set(s.network_args) == MOD_OFF | {
        "context_attn_dim=0", "x_attn_dim=0", "train_block_indices=0-23"}


def test_block_ranges():
    assert "train_block_indices=0-7" in spec_for("L-E").network_args
    assert "train_block_indices=16-23" in spec_for("L-L").network_args


def test_ranks():
    for aid, rank in (("L-R4", 4), ("L-R64", 64)):
        s = spec_for(aid)
        assert (s.network_dim, s.network_alpha) == (rank, rank)
        assert "train_block_indices=0-23" in s.network_args


def test_te_only_adapter():
    s = spec_for("L-T")
    assert s.network_module == "networks.lora_sd3"
    assert s.flags.get("network_train_text_encoder_only") is True
    assert not any(a.startswith("train_t5xxl") for a in s.network_args)


def test_flux_adapter():
    s = spec_for("F-F")
    assert s.network_module == "networks.lora_flux"
    assert {"img_mod_dim=0", "txt_mod_dim=0", "single_mod_dim=0"} <= set(s.network_args)
    assert not any(a.startswith("train_t5xxl") for a in s.network_args)


def test_flux_emits_both_stream_indices():
    """Real F-F config: blocks_double=[0,18] / blocks_single=[0,37]."""
    s = spec_for("F-F")
    assert set(s.network_args) == {
        "train_double_block_indices=0-18",
        "train_single_block_indices=0-37",
        "img_mod_dim=0", "txt_mod_dim=0", "single_mod_dim=0",
    }


def test_flux_never_emits_train_block_indices():
    for a in spec_for("F-F").network_args:
        assert not a.startswith("train_block_indices")


def test_flux_absent_stream_emits_none():
    cfg = copy.deepcopy(resolve(MATRIX / "F-F.yaml").data)
    cfg["target"] = {**cfg["target"], "blocks_double": None, "blocks_single": None}
    s = build_network_args(cfg)
    assert "train_double_block_indices=none" in s.network_args
    assert "train_single_block_indices=none" in s.network_args


def test_flux_double_only_leaves_single_none():
    cfg = copy.deepcopy(resolve(MATRIX / "F-F.yaml").data)
    cfg["target"] = {**cfg["target"], "blocks_single": None}
    s = build_network_args(cfg)
    assert "train_double_block_indices=0-18" in s.network_args
    assert "train_single_block_indices=none" in s.network_args


def test_flux_attn_only_exclusion_dims():
    cfg = copy.deepcopy(resolve(MATRIX / "F-F.yaml").data)
    cfg["target"] = {**cfg["target"], "module_classes": ["attn"], "blocks_single": None}
    s = build_network_args(cfg)
    assert {"img_mlp_dim=0", "txt_mlp_dim=0"} <= set(s.network_args)
    assert not any("attn_dim" in a for a in s.network_args)


def test_flux_mlp_only_exclusion_dims():
    cfg = copy.deepcopy(resolve(MATRIX / "F-F.yaml").data)
    cfg["target"] = {**cfg["target"], "module_classes": ["mlp"], "blocks_single": None}
    s = build_network_args(cfg)
    assert {"img_attn_dim=0", "txt_attn_dim=0"} <= set(s.network_args)
    assert not any("mlp_dim" in a for a in s.network_args)


def test_flux_never_emits_context_or_x_dims():
    cfg = copy.deepcopy(resolve(MATRIX / "F-F.yaml").data)
    cfg["target"] = {**cfg["target"], "module_classes": ["attn"], "blocks_single": None}
    s = build_network_args(cfg)
    assert not any(a.startswith("context_") or a.startswith("x_") for a in s.network_args)


def test_flux_single_stream_partial_classes_rejected():
    cfg = copy.deepcopy(resolve(MATRIX / "F-F.yaml").data)
    cfg["target"] = {**cfg["target"], "module_classes": ["attn"]}
    with pytest.raises(TargetSpecError):
        build_network_args(cfg)


def test_flux_mod_dims_always_zeroed():
    s = spec_for("F-F")
    assert {"img_mod_dim=0", "txt_mod_dim=0", "single_mod_dim=0"} <= set(s.network_args)


def test_flux_double_block_out_of_range_rejected():
    cfg = copy.deepcopy(resolve(MATRIX / "F-F.yaml").data)
    cfg["target"] = {**cfg["target"], "blocks_double": [0, 19]}
    with pytest.raises(TargetSpecError):
        build_network_args(cfg)


def test_flux_single_block_out_of_range_rejected():
    cfg = copy.deepcopy(resolve(MATRIX / "F-F.yaml").data)
    cfg["target"] = {**cfg["target"], "blocks_single": [0, 38]}
    with pytest.raises(TargetSpecError):
        build_network_args(cfg)


def test_flux_double_block_descending_rejected():
    cfg = copy.deepcopy(resolve(MATRIX / "F-F.yaml").data)
    cfg["target"] = {**cfg["target"], "blocks_double": [18, 0]}
    with pytest.raises(TargetSpecError):
        build_network_args(cfg)


def test_flux_single_block_descending_rejected():
    cfg = copy.deepcopy(resolve(MATRIX / "F-F.yaml").data)
    cfg["target"] = {**cfg["target"], "blocks_single": [37, 0]}
    with pytest.raises(TargetSpecError):
        build_network_args(cfg)


def test_sd3_descending_blocks_rejected():
    """Existing gap: network_args.py:76-77 (lo > hi) was never exercised."""
    cfg = copy.deepcopy(resolve(MATRIX / "L-E.yaml").data)
    cfg["target"]["blocks"] = [7, 0]
    with pytest.raises(TargetSpecError):
        build_network_args(cfg)


def test_never_split_qkv():
    for aid in ("L-F", "L-A", "L-E", "F-F"):
        assert not any("split_qkv" in a for a in spec_for(aid).network_args)


def test_out_of_range_blocks_rejected():
    cfg = copy.deepcopy(resolve(MATRIX / "L-E.yaml").data)
    cfg["target"]["blocks"] = [0, 24]
    with pytest.raises(TargetSpecError):
        build_network_args(cfg)


def test_empty_module_classes_rejected():
    cfg = copy.deepcopy(resolve(MATRIX / "L-F.yaml").data)
    cfg["target"]["module_classes"] = []
    with pytest.raises((TargetSpecError, ValueError)):
        build_network_args(cfg)
