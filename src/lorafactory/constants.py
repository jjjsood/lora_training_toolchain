"""Pinned module vocabulary (README §4). Pinned, never discovered.

Upstream renames must surface as "0 modules matched", not a quietly
narrower adapter — so these names are hardcoded constants, not introspected
from a live model.
"""

SD3_NUM_BLOCKS = 24
SD3_CONTEXT_PRE_ONLY_BLOCK = 23
#: Highest transformer block index of SD3-Medium (24 blocks, 0-23).
SD3_MAX_BLOCK_INDEX = SD3_NUM_BLOCKS - 1

SD3_ATTN_LEAVES = (
    "attn.to_q", "attn.to_k", "attn.to_v", "attn.to_out.0",
    "attn.add_q_proj", "attn.add_k_proj", "attn.add_v_proj", "attn.to_add_out",
)

SD3_MLP_LEAVES = (
    "ff.net.0.proj", "ff.net.2",
    "ff_context.net.0.proj", "ff_context.net.2",
)

# Leaves absent on the context-pre-only last block (23).
SD3_BLOCK23_ABSENT = (
    "attn.to_add_out", "ff_context.net.0.proj", "ff_context.net.2",
)

ADAPTER_IDS = (
    "L-F", "L-A", "L-M", "L-E", "L-L", "L-R4", "L-R64", "L-T",
    "O-F", "O-A", "O-M", "F-F",
)

CLIP_L_NUM_LAYERS = 12
CLIP_G_NUM_LAYERS = 32

CLIP_ATTN_LEAVES = (
    "self_attn.q_proj", "self_attn.k_proj", "self_attn.v_proj", "self_attn.out_proj",
)
CLIP_MLP_LEAVES = ("mlp.fc1", "mlp.fc2")

FLUX_NUM_DOUBLE_BLOCKS = 19
FLUX_NUM_SINGLE_BLOCKS = 38
#: Highest double-stream block index of FLUX (19 blocks, 0-18).
FLUX_MAX_DOUBLE_BLOCK_INDEX = FLUX_NUM_DOUBLE_BLOCKS - 1
#: Highest single-stream block index of FLUX (38 blocks, 0-37).
FLUX_MAX_SINGLE_BLOCK_INDEX = FLUX_NUM_SINGLE_BLOCKS - 1

FLUX_DOUBLE_ATTN_LEAVES = SD3_ATTN_LEAVES
FLUX_DOUBLE_MLP_LEAVES = SD3_MLP_LEAVES
FLUX_SINGLE_ATTN_LEAVES = ("attn.to_q", "attn.to_k", "attn.to_v")
FLUX_SINGLE_MLP_LEAVES = ("proj_mlp", "proj_out")

#: All double-stream leaves, uniform across every block (no SD3-style
#: context-pre-only quirk — see tests/test_flux_vocabulary.py).
FLUX_DOUBLE_LEAVES = FLUX_DOUBLE_ATTN_LEAVES + FLUX_DOUBLE_MLP_LEAVES
#: All single-stream leaves. Single blocks fuse attn-out/mlp into
#: proj_mlp/proj_out — no attn.to_out.0, no separate ff.* family.
FLUX_SINGLE_LEAVES = FLUX_SINGLE_ATTN_LEAVES + FLUX_SINGLE_MLP_LEAVES
