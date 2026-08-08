"""Norm-matched random adapter synthesiser — CAP-27 / README acceptance #6.

API pinned:
    from lorafactory.synth.norms import lora_frobenius_norm
    n = lora_frobenius_norm(A, B)   # == ||B @ A||_F, ΔW never materialised

    from lorafactory.synth.synthesiser import synthesise
    out_sd, recipe = synthesise(reference_sd, seed=..., tolerance=1e-4)
    # out_sd: same module names & ranks as reference, Gaussian, rescaled so
    #         per-module ||B@A||_F matches the reference within tolerance
    # recipe: {"seed": int, "distribution": "gaussian",
    #          "modules": {name: {"target_norm", "achieved_norm", "rank"}}}
"""

import torch

from conftest import peft_module
from lorafactory.synth.norms import lora_frobenius_norm
from lorafactory.synth.synthesiser import synthesise


def module_names(sd):
    return {k.rsplit(".lora_", 1)[0] for k in sd}


def make_reference():
    sd = {}
    sd.update(peft_module("transformer.transformer_blocks.0.attn.to_q", 4, 16, 16, seed=1))
    sd.update(peft_module("transformer.transformer_blocks.5.ff.net.2", 4, 16, 64, seed=2))
    sd.update(peft_module("transformer.transformer_blocks.23.attn.add_q_proj", 4, 16, 16, seed=3))
    return sd


def test_norm_identity():
    g = torch.Generator().manual_seed(7)
    a = torch.randn(4, 32, generator=g)
    b = torch.randn(16, 4, generator=g)
    direct = torch.linalg.matrix_norm(b @ a, ord="fro").item()
    assert abs(lora_frobenius_norm(a, b) - direct) < 1e-4 * max(direct, 1.0)


def test_synthesised_norms_match_reference():
    ref = make_reference()
    out, recipe = synthesise(ref, seed=20260808)
    assert module_names(out) == module_names(ref)
    for name in module_names(ref):
        ra = ref[f"{name}.lora_A.weight"]
        rb = ref[f"{name}.lora_B.weight"]
        oa = out[f"{name}.lora_A.weight"]
        ob = out[f"{name}.lora_B.weight"]
        assert oa.shape == ra.shape and ob.shape == rb.shape
        target = lora_frobenius_norm(ra.float(), rb.float())
        achieved = lora_frobenius_norm(oa.float(), ob.float())
        assert abs(achieved - target) <= 1e-4 * max(target, 1.0)
        assert not torch.allclose(oa.float(), ra.float()), "must be random, not copied"


def test_recipe_contents():
    ref = make_reference()
    _, recipe = synthesise(ref, seed=99)
    assert recipe["seed"] == 99
    assert recipe["distribution"] == "gaussian"
    assert set(recipe["modules"]) == module_names(ref)
    for entry in recipe["modules"].values():
        assert {"target_norm", "achieved_norm", "rank"} <= set(entry)


def test_deterministic_for_same_seed():
    ref = make_reference()
    out1, _ = synthesise(ref, seed=5)
    out2, _ = synthesise(ref, seed=5)
    for k in out1:
        assert torch.equal(out1[k], out2[k])
    out3, _ = synthesise(ref, seed=6)
    assert any(not torch.equal(out1[k], out3[k]) for k in out1)


def test_module_seed_independent_of_dict_order():
    ref = make_reference()
    reordered = dict(reversed(list(ref.items())))
    out1, _ = synthesise(ref, seed=5)
    out2, _ = synthesise(reordered, seed=5)
    for k in out1:
        assert torch.equal(out1[k], out2[k])
