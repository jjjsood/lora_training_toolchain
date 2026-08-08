"""SD3 kohya→diffusers converter — the correctness-critical core.

API pinned:
    from lorafactory.convert.sd3_kohya_to_diffusers import convert
    converted = convert(kohya_state_dict)   # dict[str, torch.Tensor]

Contract:
- Output keys are diffusers/PEFT layout with `transformer.` / `text_encoder.` /
  `text_encoder_2.` prefixes, lora_A/lora_B naming, NO alpha keys.
- alpha/rank scaling is folded into the weights: for every module,
  lora_B @ lora_A == (alpha/rank) * lora_up @ lora_down  (fp32, atol 1e-5).
- Fused qkv is split into to_q/to_k/to_v (x stream) resp.
  add_q_proj/add_k_proj/add_v_proj (context stream): lora_up chunked in 3
  along dim 0, lora_A shared/replicated.
- Block 23 (context_pre_only): context proj / context mlp keys in the INPUT
  are an error. lora_te3 (T5) is an error. adaLN keys are an error.
"""

import pytest
import torch

from conftest import delta_w_kohya, delta_w_peft, kohya_module
from lorafactory.convert.keymap import UnconvertibleKeyError, parse_diffusers_lora_key
from lorafactory.convert.sd3_kohya_to_diffusers import convert

H = 8          # tiny hidden dim
RANK = 2
ALPHA = 4.0    # alpha != rank so the folding actually matters


def full_block(i: int, pre_only: bool = False, seed: int = 0) -> dict:
    p = f"lora_unet_joint_blocks_{i}"
    sd = {}
    sd.update(kohya_module(f"{p}_x_block_attn_qkv", RANK, 3 * H, H, ALPHA, seed))
    sd.update(kohya_module(f"{p}_x_block_attn_proj", RANK, H, H, ALPHA, seed + 1))
    sd.update(kohya_module(f"{p}_x_block_mlp_fc1", RANK, 4 * H, H, ALPHA, seed + 2))
    sd.update(kohya_module(f"{p}_x_block_mlp_fc2", RANK, H, 4 * H, ALPHA, seed + 3))
    sd.update(kohya_module(f"{p}_context_block_attn_qkv", RANK, 3 * H, H, ALPHA, seed + 4))
    if not pre_only:
        sd.update(kohya_module(f"{p}_context_block_attn_proj", RANK, H, H, ALPHA, seed + 5))
        sd.update(kohya_module(f"{p}_context_block_mlp_fc1", RANK, 4 * H, H, ALPHA, seed + 6))
        sd.update(kohya_module(f"{p}_context_block_mlp_fc2", RANK, H, 4 * H, ALPHA, seed + 7))
    return sd


def test_all_output_keys_are_valid_diffusers_keys():
    sd = full_block(0)
    out = convert(sd)
    assert out, "converter returned empty dict"
    for key in out:
        parse_diffusers_lora_key(key)  # raises on any invalid key
    assert not any(k.endswith(".alpha") for k in out)


def test_qkv_split_math_x_stream():
    sd = full_block(0)
    out = convert(sd)
    dw = delta_w_kohya(sd, "lora_unet_joint_blocks_0_x_block_attn_qkv")
    prefix = "transformer.transformer_blocks.0"
    for j, leaf in enumerate(("attn.to_q", "attn.to_k", "attn.to_v")):
        got = delta_w_peft(out, f"{prefix}.{leaf}")
        expect = dw[j * H:(j + 1) * H, :]
        torch.testing.assert_close(got, expect, atol=1e-5, rtol=1e-5)
        rank = out[f"{prefix}.{leaf}.lora_A.weight"].shape[0]
        assert rank == RANK, "split qkv must keep the per-projection rank"


def test_qkv_split_math_context_stream():
    sd = full_block(0)
    out = convert(sd)
    dw = delta_w_kohya(sd, "lora_unet_joint_blocks_0_context_block_attn_qkv")
    prefix = "transformer.transformer_blocks.0"
    for j, leaf in enumerate(("attn.add_q_proj", "attn.add_k_proj",
                              "attn.add_v_proj")):
        got = delta_w_peft(out, f"{prefix}.{leaf}")
        torch.testing.assert_close(got, dw[j * H:(j + 1) * H, :],
                                   atol=1e-5, rtol=1e-5)


@pytest.mark.parametrize("kohya_leaf,diff_leaf", [
    ("x_block_attn_proj", "attn.to_out.0"),
    ("context_block_attn_proj", "attn.to_add_out"),
    ("x_block_mlp_fc1", "ff.net.0.proj"),
    ("x_block_mlp_fc2", "ff.net.2"),
    ("context_block_mlp_fc1", "ff_context.net.0.proj"),
    ("context_block_mlp_fc2", "ff_context.net.2"),
])
def test_direct_module_mapping_and_alpha_folding(kohya_leaf, diff_leaf):
    sd = full_block(3)
    out = convert(sd)
    dw = delta_w_kohya(sd, f"lora_unet_joint_blocks_3_{kohya_leaf}")
    got = delta_w_peft(out, f"transformer.transformer_blocks.3.{diff_leaf}")
    torch.testing.assert_close(got, dw, atol=1e-5, rtol=1e-5)


def test_block_index_survives_two_digit():
    sd = full_block(17)
    out = convert(sd)
    assert "transformer.transformer_blocks.17.attn.to_q.lora_A.weight" in out
    assert not any(".transformer_blocks.1." in k for k in out)


def test_block23_pre_only_input_accepted():
    out = convert(full_block(23, pre_only=True))
    p = "transformer.transformer_blocks.23"
    assert f"{p}.attn.add_q_proj.lora_A.weight" in out
    assert f"{p}.attn.to_add_out.lora_A.weight" not in out
    assert not any("ff_context" in k and ".23." in k for k in out)


def test_block23_context_proj_input_rejected():
    with pytest.raises(UnconvertibleKeyError):
        convert(full_block(23, pre_only=False))


def test_te1_te2_mapping():
    sd = {}
    sd.update(kohya_module(
        "lora_te1_text_model_encoder_layers_0_self_attn_q_proj",
        RANK, H, H, ALPHA))
    sd.update(kohya_module(
        "lora_te2_text_model_encoder_layers_31_mlp_fc2", RANK, H, 4 * H, ALPHA))
    out = convert(sd)
    k1 = "text_encoder.text_model.encoder.layers.0.self_attn.q_proj"
    k2 = "text_encoder_2.text_model.encoder.layers.31.mlp.fc2"
    torch.testing.assert_close(
        delta_w_peft(out, k1),
        delta_w_kohya(sd, "lora_te1_text_model_encoder_layers_0_self_attn_q_proj"),
        atol=1e-5, rtol=1e-5)
    assert f"{k2}.lora_B.weight" in out


def test_te3_rejected():
    sd = kohya_module(
        "lora_te3_encoder_block_0_layer_0_SelfAttention_q", RANK, H, H, ALPHA)
    with pytest.raises(UnconvertibleKeyError):
        convert(sd)


def test_adaln_rejected():
    sd = kohya_module(
        "lora_unet_joint_blocks_0_x_block_adaLN_modulation_1",
        RANK, 6 * H, H, ALPHA)
    with pytest.raises(UnconvertibleKeyError):
        convert(sd)
