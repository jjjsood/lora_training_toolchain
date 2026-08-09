"""Anchored key grammars. The blocks.1 / blocks.11 substring defect must be
structurally impossible (fullmatch parsing, never substring search).

API pinned:
    from lorafactory.convert.keymap import (
        parse_kohya_sd3_key, parse_diffusers_lora_key, UnconvertibleKeyError)

parse_kohya_sd3_key(key) -> object with fields:
    kind: "transformer" | "te"
    block: int | None          (transformer only)
    stream: "x" | "context" | None
    leaf: "attn_qkv" | "attn_proj" | "mlp_fc1" | "mlp_fc2" | str (te leaf path)
    part: "lora_down" | "lora_up" | "alpha"
    te_index: int | None       (1=CLIP-L, 2=CLIP-G, 3=T5)

parse_diffusers_lora_key(key) -> object with fields:
    kind: "transformer" | "te"
    block: int | None
    module: str                ("attn.to_q", "ff.net.0.proj", te module path)
    part: "lora_A" | "lora_B"
    te_index: int | None
"""

import pytest

from lorafactory.convert.keymap import (
    UnconvertibleKeyError,
    parse_diffusers_lora_key,
    parse_flux_diffusers_lora_key,
    parse_kohya_sd3_key,
)


def test_kohya_transformer_keys_parse():
    k = parse_kohya_sd3_key(
        "lora_unet_joint_blocks_0_x_block_attn_qkv.lora_down.weight")
    assert (k.kind, k.block, k.stream, k.leaf, k.part) == (
        "transformer", 0, "x", "attn_qkv", "lora_down")

    k = parse_kohya_sd3_key(
        "lora_unet_joint_blocks_23_context_block_attn_qkv.alpha")
    assert (k.block, k.stream, k.leaf, k.part) == (23, "context", "attn_qkv", "alpha")

    k = parse_kohya_sd3_key(
        "lora_unet_joint_blocks_11_x_block_mlp_fc2.lora_up.weight")
    assert (k.block, k.leaf) == (11, "mlp_fc2")


def test_kohya_block_index_is_anchored():
    one = parse_kohya_sd3_key(
        "lora_unet_joint_blocks_1_x_block_attn_proj.lora_up.weight")
    eleven = parse_kohya_sd3_key(
        "lora_unet_joint_blocks_11_x_block_attn_proj.lora_up.weight")
    assert one.block == 1
    assert eleven.block == 11


def test_kohya_te_keys_parse():
    k = parse_kohya_sd3_key(
        "lora_te1_text_model_encoder_layers_0_self_attn_q_proj.lora_down.weight")
    assert (k.kind, k.te_index, k.part) == ("te", 1, "lora_down")
    k = parse_kohya_sd3_key(
        "lora_te2_text_model_encoder_layers_31_mlp_fc1.alpha")
    assert (k.te_index, k.part) == (2, "alpha")
    k = parse_kohya_sd3_key(
        "lora_te3_encoder_block_0_layer_0_SelfAttention_q.lora_up.weight")
    assert k.te_index == 3


@pytest.mark.parametrize("bad", [
    "lora_unet_joint_blocks_0_x_block_adaLN_modulation_1.lora_down.weight",
    "lora_unet_final_layer_adaLN_modulation_1.lora_down.weight",
    "lora_unet_joint_blocks_x_block_attn_qkv.lora_down.weight",
    "lora_unet_joint_blocks_0_y_block_attn_qkv.lora_down.weight",
    "lora_unet_joint_blocks_0_x_block_attn_qkv.weird.weight",
    "transformer.transformer_blocks.0.attn.to_q.lora_A.weight",
    "",
])
def test_kohya_unknown_keys_rejected(bad):
    with pytest.raises(UnconvertibleKeyError):
        parse_kohya_sd3_key(bad)


def test_diffusers_transformer_keys_parse():
    k = parse_diffusers_lora_key(
        "transformer.transformer_blocks.11.attn.to_q.lora_A.weight")
    assert (k.kind, k.block, k.module, k.part) == (
        "transformer", 11, "attn.to_q", "lora_A")
    k = parse_diffusers_lora_key(
        "transformer.transformer_blocks.0.attn.to_out.0.lora_B.weight")
    assert (k.block, k.module, k.part) == (0, "attn.to_out.0", "lora_B")
    k = parse_diffusers_lora_key(
        "transformer.transformer_blocks.5.ff_context.net.0.proj.lora_A.weight")
    assert k.module == "ff_context.net.0.proj"


def test_diffusers_te_keys_parse():
    k = parse_diffusers_lora_key(
        "text_encoder.text_model.encoder.layers.0.self_attn.q_proj.lora_A.weight")
    assert (k.kind, k.te_index) == ("te", 1)
    k = parse_diffusers_lora_key(
        "text_encoder_2.text_model.encoder.layers.31.mlp.fc2.lora_B.weight")
    assert k.te_index == 2


@pytest.mark.parametrize("bad", [
    "transformer.transformer_blocks.0.attn.to_qq.lora_A.weight",
    "transformer.transformer_blocks.0.norm1.linear.lora_A.weight",
    "transformer.transformer_blocks.24.attn.to_q.lora_A.weight",
    "transformer.transformer_blocks.0.attn.to_q.alpha",
    "lora_unet_joint_blocks_0_x_block_attn_qkv.lora_down.weight",
])
def test_diffusers_unknown_keys_rejected(bad):
    with pytest.raises(UnconvertibleKeyError):
        parse_diffusers_lora_key(bad)


# ---- FLUX diffusers/PEFT grammar ----

def test_flux_double_block_keys_parse():
    k = parse_flux_diffusers_lora_key(
        "transformer.transformer_blocks.11.attn.to_q.lora_A.weight")
    assert (k.stream, k.block, k.module, k.part) == (
        "double", 11, "attn.to_q", "lora_A")
    k = parse_flux_diffusers_lora_key(
        "transformer.transformer_blocks.0.attn.to_out.0.lora_B.weight")
    assert (k.stream, k.block, k.module, k.part) == (
        "double", 0, "attn.to_out.0", "lora_B")
    k = parse_flux_diffusers_lora_key(
        "transformer.transformer_blocks.18.ff_context.net.0.proj.lora_A.weight")
    assert (k.stream, k.block, k.module) == ("double", 18, "ff_context.net.0.proj")


def test_flux_single_block_keys_parse():
    k = parse_flux_diffusers_lora_key(
        "transformer.single_transformer_blocks.0.attn.to_q.lora_A.weight")
    assert (k.stream, k.block, k.module, k.part) == (
        "single", 0, "attn.to_q", "lora_A")
    k = parse_flux_diffusers_lora_key(
        "transformer.single_transformer_blocks.37.proj_out.lora_B.weight")
    assert (k.stream, k.block, k.module) == ("single", 37, "proj_out")


def test_flux_double_grammar_cannot_swallow_single_namespace():
    """`transformer_blocks` is a substring of `single_transformer_blocks` —
    a key in the single namespace must come back tagged `stream="single"`,
    never be mistaken for a double-block key with a mangled block index."""
    k = parse_flux_diffusers_lora_key(
        "transformer.single_transformer_blocks.5.attn.to_q.lora_A.weight")
    assert k.stream == "single"
    assert k.block == 5
    assert k.module == "attn.to_q"

    # A single-only leaf (no attn.to_out.0/ff.* on single blocks) must not be
    # accepted by falling through to the double grammar.
    with pytest.raises(UnconvertibleKeyError):
        parse_flux_diffusers_lora_key(
            "transformer.single_transformer_blocks.5.attn.to_out.0.lora_A.weight")


def test_flux_block_index_is_anchored():
    """blocks_1 vs blocks_11 (double, <=18) and blocks_3 vs blocks_37 (single,
    <=37) must not be confused by a substring match."""
    one = parse_flux_diffusers_lora_key(
        "transformer.transformer_blocks.1.attn.to_q.lora_A.weight")
    eleven = parse_flux_diffusers_lora_key(
        "transformer.transformer_blocks.11.attn.to_q.lora_A.weight")
    assert (one.block, eleven.block) == (1, 11)

    three = parse_flux_diffusers_lora_key(
        "transformer.single_transformer_blocks.3.attn.to_q.lora_A.weight")
    thirty_seven = parse_flux_diffusers_lora_key(
        "transformer.single_transformer_blocks.37.attn.to_q.lora_A.weight")
    assert (three.block, thirty_seven.block) == (3, 37)

    with pytest.raises(UnconvertibleKeyError):
        parse_flux_diffusers_lora_key(
            "transformer.transformer_blocks.19.attn.to_q.lora_A.weight")
    with pytest.raises(UnconvertibleKeyError):
        parse_flux_diffusers_lora_key(
            "transformer.single_transformer_blocks.38.attn.to_q.lora_A.weight")


@pytest.mark.parametrize("bad", [
    # double-block leaf that doesn't exist on single blocks
    "transformer.single_transformer_blocks.0.ff.net.2.lora_A.weight",
    # single-only leaf on the double namespace
    "transformer.transformer_blocks.0.proj_mlp.lora_A.weight",
    "transformer.transformer_blocks.0.attn.to_qq.lora_A.weight",
    "transformer.transformer_blocks.0.norm1.linear.lora_A.weight",
    "transformer.transformer_blocks.19.attn.to_q.lora_A.weight",
    "transformer.single_transformer_blocks.38.attn.to_q.lora_A.weight",
    "transformer.transformer_blocks.0.attn.to_q.alpha",
    "lora_unet_joint_blocks_0_x_block_attn_qkv.lora_down.weight",
    "text_encoder.text_model.encoder.layers.0.self_attn.q_proj.lora_A.weight",
])
def test_flux_unknown_keys_rejected(bad):
    with pytest.raises(UnconvertibleKeyError):
        parse_flux_diffusers_lora_key(bad)
