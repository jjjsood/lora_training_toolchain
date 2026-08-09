"""FLUX load-path decision experiment (WS4/T4): does diffusers 0.39's own
kohya→diffusers conversion (`_convert_kohya_flux_lora_to_diffusers`, wired
into `FluxLoraLoaderMixin.lora_state_dict`) actually consume a kohya-layout
FLUX LoRA end to end, or does this repo need its own converter (as it does
for SD3, which has no diffusers-side equivalent)?

This is the decision authority for T4 — run, not assumed.

Result: GREEN, with one caveat found by actually running it (not by reading
the source). `_convert_to_ai_toolkit_cat` in diffusers 0.39's
`lora_conversion_utils.py` splits the single-stream fused `linear1` module
(kohya `lora_unet_single_blocks_{N}_linear1`, which fuses attn q/k/v + the
mlp-in projection) with a *hardcoded* `dims=[3072, 3072, 3072, 12288]` —
real FLUX.1's `inner_dim`/`mlp_hidden_dim`, not derived from the tensors it
is fed (`lora_conversion_utils.py:436`, `assert sum(dims) == up_weight.shape[0]`).
Verified directly: feeding it a tiny (non-3072) width for `linear1` raises
`AssertionError` before conversion even completes; every other one of the 13
kohya FLUX module leaves (10 double-stream + `linear2`/`modulation_lin` on
single-stream) is dimension-agnostic and converts fine at any width.

Consequence for this test: the "tiny model, full block count" pattern used
for SD3 (tests/test_load_smoke.py) is run here almost unchanged, covering 12
of 13 leaves through a real tiny `FluxTransformer2DModel` — PEFT attach +
forward pass, the strongest CPU check available. `linear1` is verified
separately, feeding the real conversion function real FLUX.1 dims (3072 /
12288) — those tensors are trivially small (a LoRA module, not a full model:
rank x 3072 and 3072ish x rank), so this stays CPU-cheap while proving the
one dimension-sensitive leaf converts correctly for the only shape it will
ever actually see in production (kohya always trains against real FLUX.1,
whose width IS 3072 — the hardcoding is invisible outside a synthetic tiny
model).

Net: no converter needed. Diffusers 0.39 handles every kohya FLUX module
name this repo's own vocabulary recognises (ground-facts.md), for the only
widths a real checkpoint will ever have.
"""

from __future__ import annotations

import torch

from conftest import kohya_module
from lorafactory.constants import FLUX_NUM_DOUBLE_BLOCKS, FLUX_NUM_SINGLE_BLOCKS

NUM_LAYERS = FLUX_NUM_DOUBLE_BLOCKS  # 19 — full count: key-grammar realism over width
NUM_SINGLE_LAYERS = FLUX_NUM_SINGLE_BLOCKS  # 38
HEAD_DIM = 8
HEADS = 4
HIDDEN = HEAD_DIM * HEADS  # 32 — tiny, deliberately not 3072 (see module docstring)
RANK = 2
ALPHA = 2.0

# axes_dims_rope entries must be even and sum to attention_head_dim (real FLUX.1
# default is (16, 56, 56), summing to its head_dim of 128); scaled down to match
# our tiny HEAD_DIM=8 so FluxPosEmbed's rotary table lines up with the model.
AXES_DIMS_ROPE = (2, 2, 4)

_DOUBLE_LEAVES = [
    # (kohya leaf, out_dim, in_dim) — fused qkv is 3x, mlp is 4x, mod_lin is 6x
    # (AdaLayerNormZero's shift/scale/gate x2), all ordinary nn.Linear widths.
    ("img_attn_qkv", 3 * HIDDEN, HIDDEN),
    ("img_attn_proj", HIDDEN, HIDDEN),
    ("txt_attn_qkv", 3 * HIDDEN, HIDDEN),
    ("txt_attn_proj", HIDDEN, HIDDEN),
    ("img_mlp_0", 4 * HIDDEN, HIDDEN),
    ("img_mlp_2", HIDDEN, 4 * HIDDEN),
    ("txt_mlp_0", 4 * HIDDEN, HIDDEN),
    ("txt_mlp_2", HIDDEN, 4 * HIDDEN),
    ("img_mod_lin", 6 * HIDDEN, HIDDEN),
    ("txt_mod_lin", 6 * HIDDEN, HIDDEN),
]


def tiny_flux_transformer():
    from diffusers import FluxTransformer2DModel

    torch.manual_seed(0)
    return FluxTransformer2DModel(
        patch_size=1,
        in_channels=8,
        out_channels=8,
        num_layers=NUM_LAYERS,
        num_single_layers=NUM_SINGLE_LAYERS,
        attention_head_dim=HEAD_DIM,
        num_attention_heads=HEADS,
        joint_attention_dim=HIDDEN,
        pooled_projection_dim=HIDDEN,
        guidance_embeds=False,
        axes_dims_rope=AXES_DIMS_ROPE,
    )


def kohya_sd_for_tiny_model() -> dict[str, torch.Tensor]:
    """Full kohya FLUX vocabulary except single-stream `linear1` (see module
    docstring: diffusers 0.39 hardcodes real-FLUX.1 dims for that one split,
    so it cannot be exercised at tiny width — covered separately below)."""
    sd: dict[str, torch.Tensor] = {}
    seed = 0
    for i in range(NUM_LAYERS):
        p = f"lora_unet_double_blocks_{i}"
        for leaf, out_dim, in_dim in _DOUBLE_LEAVES:
            sd.update(kohya_module(f"{p}_{leaf}", RANK, out_dim, in_dim, ALPHA, seed=seed))
            seed += 1

    for i in range(NUM_SINGLE_LAYERS):
        p = f"lora_unet_single_blocks_{i}"
        # linear2 -> proj_out: fused attn-out + mlp-out, in = dim + 4*dim = 5*dim
        sd.update(kohya_module(f"{p}_linear2", RANK, HIDDEN, 5 * HIDDEN, ALPHA, seed=seed))
        seed += 1
        # modulation_lin -> norm.linear: AdaLayerNormZeroSingle's shift/scale/gate
        sd.update(kohya_module(f"{p}_modulation_lin", RANK, 3 * HIDDEN, HIDDEN, ALPHA, seed=seed))
        seed += 1
    return sd


def test_convert_attach_forward():
    """The decision test: kohya-layout FLUX LoRA (12 of 13 module leaves,
    full 19+38 block count) through diffusers' own conversion + PEFT attach
    + a real forward pass on a tiny FluxTransformer2DModel."""
    from diffusers.loaders.lora_pipeline import FluxLoraLoaderMixin

    model = tiny_flux_transformer()
    kohya_sd = kohya_sd_for_tiny_model()

    converted = FluxLoraLoaderMixin.lora_state_dict(dict(kohya_sd))
    model.load_lora_adapter(converted, prefix="transformer")

    # PEFT registers both "...lora_A" (a ModuleDict) and "...lora_A.default_0"
    # (the actual Linear inside it) in named_modules(); count only the dict.
    peft_layers = [n for n, _ in model.named_modules() if n.endswith("lora_A")]
    assert peft_layers, "no LoRA layers attached"
    # Per double block: fused qkv splits into 3 attached modules each (x and
    # context), so 10 kohya leaves -> 14 attached PEFT modules; per single
    # block, linear2/modulation_lin stay 1:1 -> 2 attached PEFT modules.
    assert len(peft_layers) == NUM_LAYERS * 14 + NUM_SINGLE_LAYERS * 2

    img_seq, txt_seq = 16, 8
    hidden = torch.randn(1, img_seq, 8)
    encoder = torch.randn(1, txt_seq, HIDDEN)
    pooled = torch.randn(1, HIDDEN)
    timestep = torch.tensor([1.0])
    img_ids = torch.zeros(img_seq, 3)
    txt_ids = torch.zeros(txt_seq, 3)
    with torch.no_grad():
        out = model(
            hidden_states=hidden,
            encoder_hidden_states=encoder,
            pooled_projections=pooled,
            timestep=timestep,
            img_ids=img_ids,
            txt_ids=txt_ids,
        ).sample
    assert out.shape == (1, img_seq, 8)
    assert torch.isfinite(out).all()


def test_single_block_linear1_conversion_at_real_flux_dims():
    """The 13th leaf: single-stream `linear1` fuses attn q/k/v + mlp-in into
    one kohya module. diffusers' `_convert_to_ai_toolkit_cat` splits it with
    a literal `dims=[3072, 3072, 3072, 12288]` (real FLUX.1's inner_dim=3072,
    mlp_hidden_dim=4*3072=12288) instead of deriving the split from the
    tensor it is given — verified by probing at tiny width, which raises
    `AssertionError` at `lora_conversion_utils.py:436`. A real kohya-trained
    FLUX LoRA is always shaped for real FLUX.1 (that width is fixed by the
    pretrained architecture, not user-configurable), so this only matters at
    real dims — checked here directly against the real conversion function,
    with LoRA-sized (not model-sized) tensors, so it stays CPU-cheap.
    """
    from diffusers.loaders.lora_pipeline import FluxLoraLoaderMixin

    real_dim = 3072
    real_mlp = 4 * real_dim
    rank = 4
    sd = {
        "lora_unet_single_blocks_0_linear1.lora_down.weight": torch.randn(rank, real_dim),
        "lora_unet_single_blocks_0_linear1.lora_up.weight": torch.randn(
            3 * real_dim + real_mlp, rank
        ),
        "lora_unet_single_blocks_0_linear1.alpha": torch.tensor(float(rank)),
    }

    converted = FluxLoraLoaderMixin.lora_state_dict(dict(sd))

    prefix = "transformer.single_transformer_blocks.0"
    expected_out = {
        f"{prefix}.attn.to_q": real_dim,
        f"{prefix}.attn.to_k": real_dim,
        f"{prefix}.attn.to_v": real_dim,
        f"{prefix}.proj_mlp": real_mlp,
    }
    assert set(converted) == {
        f"{name}.{part}.weight" for name in expected_out for part in ("lora_A", "lora_B")
    }
    for name, out_dim in expected_out.items():
        assert converted[f"{name}.lora_A.weight"].shape == (rank, real_dim)
        assert converted[f"{name}.lora_B.weight"].shape == (out_dim, rank)
