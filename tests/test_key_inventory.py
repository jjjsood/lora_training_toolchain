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
