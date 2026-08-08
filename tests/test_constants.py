"""Pinned module vocabulary (README §4 table). Pinned, never discovered."""

from lorafactory import constants as c


def test_sd3_block_count():
    assert c.SD3_NUM_BLOCKS == 24
    assert c.SD3_CONTEXT_PRE_ONLY_BLOCK == 23


def test_sd3_attn_leaves():
    assert set(c.SD3_ATTN_LEAVES) == {
        "attn.to_q", "attn.to_k", "attn.to_v", "attn.to_out.0",
        "attn.add_q_proj", "attn.add_k_proj", "attn.add_v_proj",
        "attn.to_add_out",
    }


def test_sd3_mlp_leaves():
    assert set(c.SD3_MLP_LEAVES) == {
        "ff.net.0.proj", "ff.net.2",
        "ff_context.net.0.proj", "ff_context.net.2",
    }


def test_block23_absent_leaves():
    assert set(c.SD3_BLOCK23_ABSENT) == {
        "attn.to_add_out", "ff_context.net.0.proj", "ff_context.net.2",
    }


def test_adapter_ids():
    assert set(c.ADAPTER_IDS) == {
        "L-F", "L-A", "L-M", "L-E", "L-L", "L-R4", "L-R64", "L-T",
        "O-F", "O-A", "O-M", "F-F",
    }


def test_text_encoder_vocabulary():
    assert c.CLIP_L_NUM_LAYERS == 12
    assert c.CLIP_G_NUM_LAYERS == 32
    assert set(c.CLIP_ATTN_LEAVES) == {
        "self_attn.q_proj", "self_attn.k_proj",
        "self_attn.v_proj", "self_attn.out_proj",
    }
    assert set(c.CLIP_MLP_LEAVES) == {"mlp.fc1", "mlp.fc2"}


def test_flux_block_counts():
    assert c.FLUX_NUM_DOUBLE_BLOCKS == 19
    assert c.FLUX_NUM_SINGLE_BLOCKS == 38
