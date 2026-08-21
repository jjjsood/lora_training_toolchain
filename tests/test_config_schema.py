"""config/schema.py — pydantic validation of a resolved config.

The loader itself accepts any YAML shape, so a missing key like
`model.train_file` would otherwise slip through: the emitter reads with
`.get()` and writes whatever it finds.

Validation runs on the RESOLVED config (post-overlay), because that is the
object every consumer downstream — emitter, runner, models, provenance —
actually receives.
"""

import pytest

from conftest import ALL_MATRIX_IDS, CONFIGS, FALLBACK_IDS, MATRIX
from lorafactory.config.loader import resolve
from lorafactory.config.schema import ARCH_DEFAULTS, ConfigSchemaError, ModelSection, validate


def test_every_matrix_config_validates():
    for aid in ALL_MATRIX_IDS:
        validate(resolve(MATRIX / f"{aid}.yaml").data)


def test_every_fallback_config_validates():
    for aid in FALLBACK_IDS:
        validate(resolve(CONFIGS / "fallback" / f"{aid}.yaml").data)


def test_missing_section_is_rejected():
    data = dict(resolve(MATRIX / "L-F.yaml").data)
    del data["train"]
    with pytest.raises(ConfigSchemaError):
        validate(data)


def test_unknown_top_level_section_is_rejected():
    """kohya silently ignores unknown keys; a typo'd section must be OUR error
    rather than a config that quietly does nothing."""
    data = dict(resolve(MATRIX / "L-F.yaml").data)
    data["trian"] = {"seed": 1}
    with pytest.raises(ConfigSchemaError):
        validate(data)


def test_wrong_type_is_rejected():
    data = dict(resolve(MATRIX / "L-F.yaml").data)
    data["train"] = {**data["train"], "seed": "not-an-int"}
    with pytest.raises(ConfigSchemaError):
        validate(data)


def test_model_without_arch_default_must_name_a_checkpoint_file():
    """An arch with no ARCH_DEFAULTS entry still requires every model field —
    only known archs (sd3, flux) get defaults."""
    data = dict(resolve(MATRIX / "L-F.yaml").data)
    data["model"] = {
        k: v for k, v in data["model"].items() if k != "train_file"
    }
    data["model"]["arch"] = "unknown-arch"
    with pytest.raises(ConfigSchemaError):
        validate(data)


def test_short_revision_still_validates_but_warns():
    """Relaxed pattern accepts branch/tag-shaped strings; a non-pinned
    revision is a warning, never a hard failure — see
    docs/superpowers/specs/2026-08-22-config-dx-relaxation-design.md."""
    data = dict(resolve(MATRIX / "L-F.yaml").data)
    data["model"] = {**data["model"], "train_revision": "main"}
    with pytest.warns(UserWarning, match="train_revision"):
        validate(data)


def test_sha_revision_not_matching_the_verified_pin_warns():
    data = dict(resolve(MATRIX / "L-F.yaml").data)
    data["model"] = {
        **data["model"],
        "train_revision": "0" * 40,
    }
    with pytest.warns(UserWarning, match="does not match the verified pin"):
        validate(data)


def test_sd3_arch_default_fills_every_model_field():
    filled = ModelSection.model_validate({"arch": "sd3"})
    assert filled.train_repo == ARCH_DEFAULTS["sd3"]["train_repo"]
    assert filled.train_revision == ARCH_DEFAULTS["sd3"]["train_revision"]
    assert filled.train_file == ARCH_DEFAULTS["sd3"]["train_file"]
    assert filled.eval_repo == ARCH_DEFAULTS["sd3"]["eval_repo"]
    assert filled.eval_revision == ARCH_DEFAULTS["sd3"]["eval_revision"]
    assert filled.text_encoders == ARCH_DEFAULTS["sd3"]["text_encoders"]


def test_flux_arch_default_fills_every_model_field():
    filled = ModelSection.model_validate({"arch": "flux"})
    assert filled.train_repo == ARCH_DEFAULTS["flux"]["train_repo"]
    assert filled.ae == ARCH_DEFAULTS["flux"]["ae"]
    assert filled.text_encoders == ARCH_DEFAULTS["flux"]["text_encoders"]


def test_explicit_model_fields_override_arch_defaults():
    with pytest.warns(UserWarning, match="does not match the verified pin"):
        filled = ModelSection.model_validate({
            "arch": "sd3",
            "train_revision": "a" * 40,
        })
    assert filled.train_revision == "a" * 40
    assert filled.train_repo == ARCH_DEFAULTS["sd3"]["train_repo"]


def test_minimal_sd3_config_resolves_with_real_verified_pins():
    """The DX case the whole task exists for: `model: {arch: sd3}` alone."""
    data = dict(resolve(MATRIX / "L-F.yaml").data)
    data["model"] = {"arch": "sd3"}
    resolved = validate(data)
    assert resolved.model.train_revision == "19b7f516efea082d257947e057e6f419e26fd497"
    assert resolved.model.eval_revision == "ea42f8cef0f178587cf766dc8129abd379c90671"


def test_target_block_range_is_bounds_checked():
    """kohya does not bounds-check train_block_indices — we must."""
    data = dict(resolve(MATRIX / "L-F.yaml").data)
    data["target"] = {**data["target"], "blocks": [0, 99]}
    with pytest.raises(ConfigSchemaError):
        validate(data)


def test_negative_steps_are_rejected():
    data = dict(resolve(MATRIX / "L-F.yaml").data)
    data["train"] = {**data["train"], "max_train_steps": -1}
    with pytest.raises(ConfigSchemaError):
        validate(data)


def test_flux_f_f_validates():
    validate(resolve(MATRIX / "F-F.yaml").data)


def test_flux_rejects_blocks():
    """`blocks` is SD3-only; flux uses blocks_double/blocks_single."""
    data = dict(resolve(MATRIX / "F-F.yaml").data)
    data["target"] = {**data["target"], "blocks": [0, 23]}
    with pytest.raises(ConfigSchemaError):
        validate(data)


def test_sd3_rejects_blocks_double():
    data = dict(resolve(MATRIX / "L-F.yaml").data)
    data["target"] = {**data["target"], "blocks_double": [0, 18]}
    with pytest.raises(ConfigSchemaError):
        validate(data)


def test_sd3_rejects_blocks_single():
    data = dict(resolve(MATRIX / "L-F.yaml").data)
    data["target"] = {**data["target"], "blocks_single": [0, 37]}
    with pytest.raises(ConfigSchemaError):
        validate(data)


def test_flux_transformer_scope_requires_a_block_field():
    data = dict(resolve(MATRIX / "F-F.yaml").data)
    data["target"] = {
        k: v for k, v in data["target"].items()
        if k not in ("blocks_double", "blocks_single")
    }
    with pytest.raises(ConfigSchemaError):
        validate(data)


def test_flux_blocks_double_upper_bound_is_checked():
    data = dict(resolve(MATRIX / "F-F.yaml").data)
    data["target"] = {**data["target"], "blocks_double": [0, 19]}
    with pytest.raises(ConfigSchemaError):
        validate(data)


def test_flux_blocks_double_upper_edge_passes():
    data = dict(resolve(MATRIX / "F-F.yaml").data)
    data["target"] = {**data["target"], "blocks_double": [0, 18]}
    validate(data)


def test_flux_blocks_single_upper_bound_is_checked():
    data = dict(resolve(MATRIX / "F-F.yaml").data)
    data["target"] = {**data["target"], "blocks_single": [0, 38]}
    with pytest.raises(ConfigSchemaError):
        validate(data)


def test_flux_blocks_single_upper_edge_passes():
    data = dict(resolve(MATRIX / "F-F.yaml").data)
    data["target"] = {**data["target"], "blocks_single": [0, 37]}
    validate(data)


def test_flux_blocks_double_descending_is_rejected():
    data = dict(resolve(MATRIX / "F-F.yaml").data)
    data["target"] = {**data["target"], "blocks_double": [18, 0]}
    with pytest.raises(ConfigSchemaError):
        validate(data)


def test_flux_blocks_single_descending_is_rejected():
    data = dict(resolve(MATRIX / "F-F.yaml").data)
    data["target"] = {**data["target"], "blocks_single": [37, 0]}
    with pytest.raises(ConfigSchemaError):
        validate(data)


def test_sd3_block_upper_edge_still_passes():
    """23 is the top SD3 edge; the schema must not have tightened it."""
    data = dict(resolve(MATRIX / "L-F.yaml").data)
    data["target"] = {**data["target"], "blocks": [0, 23]}
    validate(data)


def test_sd3_block_edge_24_still_fails():
    """Existing gap kept green: 24 is one past the SD3 top edge."""
    data = dict(resolve(MATRIX / "L-F.yaml").data)
    data["target"] = {**data["target"], "blocks": [0, 24]}
    with pytest.raises(ConfigSchemaError):
        validate(data)
