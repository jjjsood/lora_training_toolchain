"""Kohya TOML emitter: budget keys fixed, augmentation impossible, keys valid.

API pinned:
    from lorafactory.kohya.toml_emitter import emit
    result = emit(config_dict, out_dir)
    result.train_toml     # Path to kohya train config TOML
    result.dataset_toml   # Path to kohya dataset config TOML
Emit validates every train-TOML key against the arg registry (kohya silently
ignores unknown keys — a typo'd budget key must be OUR error, not silence).
"""

import tomllib

from conftest import MATRIX
from lorafactory.config.loader import resolve
from lorafactory.kohya.toml_emitter import emit

FORBIDDEN_KEYS = {
    "shuffle_caption", "color_aug", "flip_aug", "random_crop",
    "caption_dropout_rate", "caption_tag_dropout_rate", "caption_dropout_every_n_epochs",
}


def flat(toml_dict):
    """Kohya flattens sections; mirror that for assertions."""
    out = {}
    for k, v in toml_dict.items():
        if isinstance(v, dict):
            out.update(v)
        else:
            out[k] = v
    return out


def emit_for(aid, tmp_path):
    rc = resolve(MATRIX / f"{aid}.yaml")
    res = emit(rc.data, tmp_path / aid)
    with open(res.train_toml, "rb") as f:
        train = flat(tomllib.load(f))
    with open(res.dataset_toml, "rb") as f:
        dataset = tomllib.load(f)
    return train, dataset


def test_determinism_and_cache_keys(tmp_path):
    train, _ = emit_for("L-E", tmp_path)
    assert train["max_data_loader_n_workers"] == 0
    assert train["cache_latents"] is True
    assert train["cache_latents_to_disk"] is False
    assert train["cache_text_encoder_outputs"] is True
    assert train["cache_text_encoder_outputs_to_disk"] is False
    assert isinstance(train["seed"], int)
    assert train["gradient_checkpointing"] is True
    assert train["mixed_precision"] == "bf16"
    assert train["save_model_as"] == "safetensors"


def test_no_augmentation_keys(tmp_path):
    train, dataset = emit_for("L-E", tmp_path)
    assert not (FORBIDDEN_KEYS & set(train)), "augmentation keys forbidden"
    subsets = [s for d in dataset["datasets"] for s in d["subsets"]]
    for s in subsets:
        assert not (FORBIDDEN_KEYS & set(s))


def test_network_section(tmp_path):
    train, _ = emit_for("L-A", tmp_path)
    assert train["network_module"] == "networks.lora_sd3"
    assert train["network_dim"] == 16
    assert train["network_alpha"] == 16
    assert set(train["network_args"]) >= {
        "context_mlp_dim=0", "x_mlp_dim=0", "context_mod_dim=0", "x_mod_dim=0"}


def test_te_only_flag(tmp_path):
    train, _ = emit_for("L-T", tmp_path)
    assert train["network_train_text_encoder_only"] is True


def test_dataset_toml_shape(tmp_path):
    _, dataset = emit_for("L-F", tmp_path)
    ds = dataset["datasets"][0]
    subset = ds["subsets"][0]
    assert subset["num_repeats"] == 1
    assert subset["caption_extension"] == ".txt"
    assert "image_dir" in subset


def test_rank_variants(tmp_path):
    t4, _ = emit_for("L-R4", tmp_path)
    t64, _ = emit_for("L-R64", tmp_path)
    assert (t4["network_dim"], t4["network_alpha"]) == (4, 4)
    assert (t64["network_dim"], t64["network_alpha"]) == (64, 64)
