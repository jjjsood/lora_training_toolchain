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


def test_flux_survey_slots_are_cf1_cf2_pending_acquisition():
    plan = screen.screen_plan(SURVEY_FLUX)
    slots = {entry["slot"] for entry in plan}
    assert slots == {"CF-1", "CF-2"}
    for entry in plan:
        assert entry["status"] == "pending_acquisition"
        assert entry["base_arch"] == "flux"
        assert entry["gate_config"].name == "e_img_flux.yaml"


# --- detect_arch: one state dict per named layout -------------------------

def kohya_flux_sd():
    sd = {}
    sd.update(kohya_module("lora_unet_double_blocks_0_img_attn_qkv", 8, 96, 32, 8.0, seed=0))
    sd.update(kohya_module("lora_unet_single_blocks_0_linear1", 8, 96, 32, 8.0, seed=1))
    return sd


def diffusers_flux_sd():
    sd = {}
    sd.update(peft_module("transformer.transformer_blocks.0.attn.to_q", 16, 32, 32, seed=0))
    sd.update(peft_module("transformer.single_transformer_blocks.0.attn.to_q", 16, 32, 32, seed=1))
    return sd


def unknown_sd():
    # lora_down-suffixed keys whose prefix matches none of the known
    # kohya/diffusers block-family conventions.
    sd = {}
    sd.update(kohya_module("lora_unet_resblocks_0_attn", 8, 96, 32, 8.0, seed=0))
    return sd


def test_detect_arch_flux_kohya():
    assert introspect.detect_arch(kohya_flux_sd()) == "flux"


def test_detect_arch_flux_diffusers():
    assert introspect.detect_arch(diffusers_flux_sd()) == "flux"


def test_detect_arch_sd3_kohya():
    assert introspect.detect_arch(kohya_sd()) == "sd3"


def test_detect_arch_unknown():
    assert introspect.detect_arch(unknown_sd()) == "unknown"


def test_detect_arch_ambiguous_transformer_blocks_defaults_to_sd3():
    """`transformer.transformer_blocks.` alone, with every block index within
    SD3's real range (0-23), cannot be told apart from FLUX's own
    double-stream blocks (0-18) by key shape alone; the heuristic defaults
    to sd3 rather than guessing flux."""
    sd = peft_module("transformer.transformer_blocks.5.attn.to_q", 16, 32, 32)
    assert introspect.detect_arch(sd) == "sd3"


def test_detect_arch_ambiguous_transformer_blocks_above_sd3_bound_is_flux():
    """A block index above SD3's max (23) cannot be genuine SD3, so the
    ambiguous `transformer_blocks` prefix is reported as flux."""
    sd = peft_module("transformer.transformer_blocks.24.attn.to_q", 16, 32, 32)
    assert introspect.detect_arch(sd) == "flux"


def test_introspect_reports_arch():
    assert introspect.introspect(kohya_flux_sd()).arch == "flux"
    assert introspect.introspect(kohya_sd()).arch == "sd3"


# --- screen.screen_target: detected-arch vs base_arch mismatch ------------

def test_screen_target_flags_arch_mismatch():
    entry = {"slot": "CF-1", "base_arch": "flux", "status": "pending_acquisition"}
    info = introspect.introspect(kohya_sd())  # sd3 checkpoint mistakenly filed under flux
    result = screen.screen_target(entry, info)
    assert result["detected_arch"] == "sd3"
    assert result["arch_mismatch"] is True


def test_screen_target_no_mismatch_when_arch_matches():
    entry = {"slot": "style_1", "base_arch": "sd3", "status": "pending_acquisition"}
    info = introspect.introspect(kohya_sd())
    result = screen.screen_target(entry, info)
    assert result["detected_arch"] == "sd3"
    assert result["arch_mismatch"] is False


def test_screen_target_no_mismatch_when_arch_unknown():
    entry = {"slot": "CF-1", "base_arch": "flux", "status": "pending_acquisition"}
    info = introspect.introspect(unknown_sd())
    result = screen.screen_target(entry, info)
    assert result["detected_arch"] == "unknown"
    assert result["arch_mismatch"] is False
