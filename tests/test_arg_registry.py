"""Registry of kohya argparse dests @ v0.11.1 — defence against silent typos.

API pinned:
    from lorafactory.kohya.arg_registry import (
        load_registry, validate_keys, UnknownKohyaKeyError)
    dests = load_registry("sd3")      # frozenset[str]; also "flux"
    validate_keys(["seed", ...], "sd3")   # raises UnknownKohyaKeyError

The committed JSON snapshots (src/lorafactory/kohya/kohya_args_{sd3,flux}.json)
must contain the real argparse dests of sd3_train_network.py /
flux_train_network.py at sd-scripts v0.11.1. tools/dump_kohya_args.py
regenerates them from a checkout.
"""

import pytest

from lorafactory.kohya.arg_registry import (
    UnknownKohyaKeyError,
    load_registry,
    validate_keys,
)

KNOWN_SD3_DESTS = {
    "seed", "max_train_steps", "learning_rate", "train_batch_size",
    "optimizer_type", "mixed_precision", "save_precision", "save_model_as",
    "gradient_checkpointing", "cache_latents", "cache_latents_to_disk",
    "cache_text_encoder_outputs", "cache_text_encoder_outputs_to_disk",
    "max_data_loader_n_workers", "network_module", "network_dim",
    "network_alpha", "network_args", "network_train_text_encoder_only",
    "pretrained_model_name_or_path", "clip_l", "clip_g", "t5xxl",
    "output_dir", "output_name", "logging_dir", "blocks_to_swap",
}


def test_sd3_registry_contains_known_dests():
    dests = load_registry("sd3")
    missing = KNOWN_SD3_DESTS - set(dests)
    assert not missing, f"registry missing known dests: {missing}"


def test_flux_registry_contains_fp8_base():
    dests = load_registry("flux")
    assert "fp8_base" in dests
    assert "network_args" in dests


def test_unknown_key_rejected():
    with pytest.raises(UnknownKohyaKeyError, match="shuffle_captoin"):
        validate_keys(["seed", "shuffle_captoin"], "sd3")


def test_valid_keys_pass():
    validate_keys(sorted(KNOWN_SD3_DESTS), "sd3")
