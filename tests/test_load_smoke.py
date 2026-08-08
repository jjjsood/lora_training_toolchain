"""Load smoke test, CPU part: converted adapter attaches via PEFT to a tiny
SD3Transformer2DModel and a forward pass succeeds. This is the same code
path diffusers 0.39 `load_lora_weights` uses under the hood
(`load_lora_adapter` with prefix="transformer").
"""

import torch

from conftest import kohya_module
from lorafactory.convert.sd3_kohya_to_diffusers import convert

NUM_LAYERS = 2
HEADS = 4
HEAD_DIM = 8
HIDDEN = HEADS * HEAD_DIM  # 32
RANK = 2
ALPHA = 2.0


def tiny_transformer():
    from diffusers import SD3Transformer2DModel
    torch.manual_seed(0)
    return SD3Transformer2DModel(
        sample_size=32,
        patch_size=1,
        in_channels=4,
        num_layers=NUM_LAYERS,
        attention_head_dim=HEAD_DIM,
        num_attention_heads=HEADS,
        caption_projection_dim=HIDDEN,
        joint_attention_dim=HIDDEN,
        pooled_projection_dim=64,
        out_channels=4,
    )


def kohya_sd_for_tiny_model():
    sd = {}
    for i in range(NUM_LAYERS):
        pre_only = i == NUM_LAYERS - 1
        p = f"lora_unet_joint_blocks_{i}"
        sd.update(kohya_module(f"{p}_x_block_attn_qkv", RANK, 3 * HIDDEN, HIDDEN,
                               ALPHA, seed=10 * i))
        sd.update(kohya_module(f"{p}_x_block_attn_proj", RANK, HIDDEN, HIDDEN,
                               ALPHA, seed=10 * i + 1))
        sd.update(kohya_module(f"{p}_x_block_mlp_fc1", RANK, 4 * HIDDEN, HIDDEN,
                               ALPHA, seed=10 * i + 2))
        sd.update(kohya_module(f"{p}_x_block_mlp_fc2", RANK, HIDDEN, 4 * HIDDEN,
                               ALPHA, seed=10 * i + 3))
        sd.update(kohya_module(f"{p}_context_block_attn_qkv", RANK, 3 * HIDDEN,
                               HIDDEN, ALPHA, seed=10 * i + 4))
        if not pre_only:
            sd.update(kohya_module(f"{p}_context_block_attn_proj", RANK, HIDDEN,
                                   HIDDEN, ALPHA, seed=10 * i + 5))
            sd.update(kohya_module(f"{p}_context_block_mlp_fc1", RANK, 4 * HIDDEN,
                                   HIDDEN, ALPHA, seed=10 * i + 6))
            sd.update(kohya_module(f"{p}_context_block_mlp_fc2", RANK, HIDDEN,
                                   4 * HIDDEN, ALPHA, seed=10 * i + 7))
    return sd


def test_convert_attach_forward():
    model = tiny_transformer()
    converted = convert(kohya_sd_for_tiny_model())
    model.load_lora_adapter(converted, prefix="transformer")

    peft_layers = [n for n, _ in model.named_modules() if "lora_A" in n]
    assert peft_layers, "no LoRA layers attached"

    hidden = torch.randn(1, 4, 32, 32)
    encoder = torch.randn(1, 8, HIDDEN)
    pooled = torch.randn(1, 64)
    timestep = torch.tensor([1.0])
    with torch.no_grad():
        out = model(hidden_states=hidden, encoder_hidden_states=encoder,
                    pooled_projections=pooled, timestep=timestep).sample
    assert out.shape == (1, 4, 32, 32)
    assert torch.isfinite(out).all()
