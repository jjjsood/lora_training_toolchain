"""Key-inventory verifier — anchored, README §4's second life.

API pinned:
    from lorafactory.verify.key_inventory import expected_module_names, verify
    names = expected_module_names(target_dict)   # frozenset[str], full prefixes
    report = verify(state_dict, target_dict)     # rank taken from target["rank"]
    report.ok               # bool
    report.unexpected       # set[str] module names present but not expected
    report.missing          # set[str] expected but absent
    report.rank_mismatches  # dict[str, int] module -> actual rank

target_dict examples (== config["target"]):
    {"scope": "transformer", "blocks": [0, 23],
     "module_classes": ["attn", "mlp"], "rank": 16, "alpha": 16}
    {"scope": "text_encoders", "te_encoders": ["clip_l", "clip_g"],
     "rank": 16, "alpha": 16}
"""

import torch

from conftest import peft_module
from lorafactory.verify.key_inventory import expected_module_names, verify


def t(blocks, classes, rank=16):
    return {"scope": "transformer", "blocks": list(blocks),
            "module_classes": list(classes), "rank": rank, "alpha": rank}


TE_TARGET = {"scope": "text_encoders", "te_encoders": ["clip_l", "clip_g"],
             "rank": 16, "alpha": 16}


def build_sd(names, rank=16):
    sd = {}
    for i, n in enumerate(names):
        sd.update(peft_module(n, rank, 8, 8, seed=i))
    return sd


# ---- expected inventory counts (block-23 exceptions included) ----

def test_expected_counts():
    assert len(expected_module_names(t([0, 23], ["attn", "mlp"]))) == 285
    assert len(expected_module_names(t([0, 23], ["attn"]))) == 191
    assert len(expected_module_names(t([0, 23], ["mlp"]))) == 94
    assert len(expected_module_names(t([0, 7], ["attn", "mlp"]))) == 96
    assert len(expected_module_names(t([16, 23], ["attn", "mlp"]))) == 93
    # CLIP-L 12 layers + CLIP-G 32 layers, 6 leaves each
    assert len(expected_module_names(TE_TARGET)) == 264


def test_expected_names_block23_exceptions():
    names = expected_module_names(t([16, 23], ["attn", "mlp"]))
    assert "transformer.transformer_blocks.23.attn.add_q_proj" in names
    assert "transformer.transformer_blocks.23.attn.to_add_out" not in names
    assert "transformer.transformer_blocks.23.ff.net.2" in names
    assert "transformer.transformer_blocks.23.ff_context.net.2" not in names
    assert "transformer.transformer_blocks.22.attn.to_add_out" in names


def test_te_expected_names():
    names = expected_module_names(TE_TARGET)
    assert "text_encoder.text_model.encoder.layers.0.self_attn.q_proj" in names
    assert "text_encoder_2.text_model.encoder.layers.31.mlp.fc2" in names
    assert not any(n.startswith("transformer.") for n in names)


# ---- verification ----

def test_exact_match_passes():
    target = t([0, 7], ["attn", "mlp"])
    sd = build_sd(expected_module_names(target))
    r = verify(sd, target)
    assert r.ok and not r.unexpected and not r.missing and not r.rank_mismatches


def test_out_of_range_block_flagged():
    """The blocks.1 vs blocks.11 trap: block 11 must NOT pass for range 0-7."""
    target = t([0, 7], ["attn", "mlp"])
    names = set(expected_module_names(target))
    names.add("transformer.transformer_blocks.11.attn.to_q")
    r = verify(build_sd(names), target)
    assert not r.ok
    assert "transformer.transformer_blocks.11.attn.to_q" in r.unexpected


def test_wrong_module_class_flagged():
    target = t([0, 23], ["attn"])
    names = set(expected_module_names(target))
    names.add("transformer.transformer_blocks.0.ff.net.2")
    r = verify(build_sd(names), target)
    assert not r.ok
    assert "transformer.transformer_blocks.0.ff.net.2" in r.unexpected


def test_missing_module_is_hard_failure():
    """Zero/partial match must fail — a silent null adapter is the worst case."""
    target = t([0, 7], ["attn", "mlp"])
    names = set(expected_module_names(target))
    names.discard("transformer.transformer_blocks.3.attn.to_q")
    r = verify(build_sd(names), target)
    assert not r.ok
    assert "transformer.transformer_blocks.3.attn.to_q" in r.missing


def test_rank_mismatch_flagged():
    target = t([0, 7], ["attn", "mlp"], rank=16)
    names = sorted(expected_module_names(target))
    sd = build_sd(names[:-1], rank=16)
    bad = names[-1]
    sd.update(peft_module(bad, 4, 8, 8))  # rank 4 instead of 16
    r = verify(sd, target)
    assert not r.ok
    assert r.rank_mismatches.get(bad) == 4


def test_te_only_negative_control():
    sd = build_sd(expected_module_names(TE_TARGET))
    r = verify(sd, TE_TARGET)
    assert r.ok
    sd.update(peft_module("transformer.transformer_blocks.0.attn.to_q", 16, 8, 8))
    r2 = verify(sd, TE_TARGET)
    assert not r2.ok
    assert "transformer.transformer_blocks.0.attn.to_q" in r2.unexpected


def test_verify_accepts_alpha_keys_absent():
    """Converted checkpoints have no alpha keys; verifier must not require them."""
    target = t([0, 7], ["mlp"])
    sd = build_sd(expected_module_names(target))
    assert verify(sd, target).ok


def test_verify_rejects_unparseable_key():
    target = t([0, 7], ["attn", "mlp"])
    sd = build_sd(expected_module_names(target))
    sd["transformer.transformer_blocks.0.norm1.linear.lora_A.weight"] = torch.zeros(2, 2)
    r = verify(sd, target)
    assert not r.ok


# ---- SD3 default-arch unchanged ----

def test_sd3_default_arch_matches_explicit_arch():
    target = t([0, 23], ["attn", "mlp"])
    assert expected_module_names(target) == expected_module_names(target, arch="sd3")

    sd = build_sd(expected_module_names(target))
    assert verify(sd, target) == verify(sd, target, arch="sd3")
    assert verify(sd, target).ok


# ---- FLUX ----

def tf(blocks_double=None, blocks_single=None, classes=("attn", "mlp"), rank=16):
    return {
        "scope": "transformer",
        "blocks_double": list(blocks_double) if blocks_double is not None else None,
        "blocks_single": list(blocks_single) if blocks_single is not None else None,
        "module_classes": list(classes),
        "rank": rank,
        "alpha": rank,
    }


def build_flux_sd(names, rank=16):
    sd = {}
    for i, n in enumerate(names):
        sd.update(peft_module(n, rank, 8, 8, seed=i))
    return sd


def test_flux_full_coverage_count_is_exactly_pinned():
    target = tf(blocks_double=(0, 18), blocks_single=(0, 37))
    # 19 double blocks x 12 leaves + 38 single blocks x 5 leaves.
    assert len(expected_module_names(target, arch="flux")) == 19 * 12 + 38 * 5 == 418


def test_flux_range_subset_counts():
    assert len(expected_module_names(
        tf(blocks_double=(0, 3)), arch="flux")) == 4 * 12
    assert len(expected_module_names(
        tf(blocks_single=(0, 4)), arch="flux")) == 5 * 5
    assert len(expected_module_names(
        tf(blocks_double=(0, 3), blocks_single=(0, 4)), arch="flux")) == 4 * 12 + 5 * 5


def test_flux_single_stream_class_filtering():
    attn_only = expected_module_names(tf(blocks_single=(0, 37), classes=("attn",)), arch="flux")
    mlp_only = expected_module_names(tf(blocks_single=(0, 37), classes=("mlp",)), arch="flux")
    assert len(attn_only) == 38 * 3
    assert len(mlp_only) == 38 * 2
    assert "transformer.single_transformer_blocks.0.attn.to_q" in attn_only
    assert "transformer.single_transformer_blocks.0.proj_mlp" not in attn_only
    assert "transformer.single_transformer_blocks.0.proj_mlp" in mlp_only
    assert "transformer.single_transformer_blocks.0.attn.to_q" not in mlp_only


def test_flux_double_grammar_never_leaks_into_single_namespace():
    """A double-only target must not expect/accept single-block keys, and
    the blocks_11/blocks_37 substring trap must not silently pass either."""
    target = tf(blocks_double=(0, 3))
    names = set(expected_module_names(target, arch="flux"))
    names.add("transformer.single_transformer_blocks.0.attn.to_q")
    names.add("transformer.transformer_blocks.11.attn.to_q")
    r = verify(build_flux_sd(names), target, arch="flux")
    assert not r.ok
    assert "transformer.single_transformer_blocks.0.attn.to_q" in r.unexpected
    assert "transformer.transformer_blocks.11.attn.to_q" in r.unexpected


def test_flux_clean_synthetic_checkpoint_verifies():
    target = tf(blocks_double=(0, 18), blocks_single=(0, 37))
    sd = build_flux_sd(expected_module_names(target, arch="flux"), rank=target["rank"])
    r = verify(sd, target, arch="flux")
    assert r.ok and not r.unexpected and not r.missing and not r.rank_mismatches


def test_flux_mutations_report_exactly_three_findings():
    # blocks_single stops at 36 (not 37) so single-block 37 is valid FLUX
    # grammar yet outside this target's requested range — the "extra" case.
    target = tf(blocks_double=(0, 18), blocks_single=(0, 36), rank=16)
    names = sorted(expected_module_names(target, arch="flux"))

    # Drop one (missing), keep the rest at the right rank.
    missing_name = names[0]
    kept = names[1:]
    sd = build_flux_sd(kept, rank=16)

    # Corrupt one existing module's rank.
    bad_rank_name = kept[0]
    sd.update(peft_module(bad_rank_name, 4, 8, 8))

    # Add one extra/unexpected module (valid grammar, out of requested range).
    extra_name = "transformer.single_transformer_blocks.37.attn.to_q"
    sd.update(peft_module(extra_name, 16, 8, 8))

    r = verify(sd, target, arch="flux")
    assert not r.ok
    assert r.missing == {missing_name}
    assert r.rank_mismatches == {bad_rank_name: 4}
    assert r.unexpected == {extra_name}
