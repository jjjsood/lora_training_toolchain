"""survey/introspect.py + survey/screen.py — CAP-28 community-checkpoint survey.

Introspection is pure state-dict analysis, so it is fully checkable on CPU: given someone
else's downloaded LoRA, say what layout it is in, what rank it uses, and which
modules it touches — before any of it is loaded onto a GPU.
"""

import pytest

from conftest import CONFIGS, kohya_module, peft_module
from lorafactory.survey import introspect, screen

SURVEY_SD3 = CONFIGS / "survey" / "sd3_community.yaml"
SURVEY_FLUX = CONFIGS / "survey" / "flux_community.yaml"


def kohya_sd():
    sd = {}
    for i in (0, 1):
        sd.update(kohya_module(
            f"lora_unet_joint_blocks_{i}_x_block_attn_qkv", 8, 96, 32, 8.0, seed=i))
    return sd


def peft_sd():
    sd = {}
    for i in (0, 1):
        sd.update(peft_module(
            f"transformer.transformer_blocks.{i}.attn.to_q", 16, 32, 32, seed=i))
    return sd


def test_introspect_detects_kohya_layout():
    info = introspect.introspect(kohya_sd())
    assert info.layout == "kohya"


def test_introspect_detects_peft_layout():
    info = introspect.introspect(peft_sd())
    assert info.layout == "peft"


def test_introspect_reports_rank_and_module_count():
    info = introspect.introspect(peft_sd())
    assert info.ranks == {16}
    assert info.module_count == 2


def test_introspect_reports_mixed_ranks():
    """A checkpoint with per-module ranks is legal but must not be summarised
    as a single rank — the matched-budget comparison depends on knowing."""
    sd = {}
    sd.update(peft_module("transformer.transformer_blocks.0.attn.to_q", 4, 32, 32))
    sd.update(peft_module("transformer.transformer_blocks.1.attn.to_q", 32, 32, 32))
    assert introspect.introspect(sd).ranks == {4, 32}


def test_introspect_lists_touched_modules():
    info = introspect.introspect(peft_sd())
    assert "transformer.transformer_blocks.0.attn.to_q" in info.modules


def test_introspect_rejects_a_non_lora_state_dict():
    import torch
    with pytest.raises(introspect.IntrospectionError):
        introspect.introspect({"model.weight": torch.zeros(4, 4)})


def test_screen_plan_lists_every_survey_slot():
    plan = screen.screen_plan(SURVEY_SD3)
    slots = {entry["slot"] for entry in plan}
    assert slots == {"style_1", "style_2", "character_1", "concept_1"}
    for entry in plan:
        # Each slot must carry the gate it will be screened against, or a
        # checkpoint could be accepted without ever passing E_img.
        assert entry["gate_config"]
        assert entry["status"]


def test_screen_plan_resolves_the_gate_config_path():
    plan = screen.screen_plan(SURVEY_SD3)
    gate_path = plan[0]["gate_config"]
    assert gate_path.exists(), f"gate config not resolved: {gate_path}"


def test_flux_survey_plan_loads_too():
    assert screen.screen_plan(SURVEY_FLUX)
