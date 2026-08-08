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
from lorafactory.config.schema import ConfigSchemaError, validate


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


def test_model_must_name_a_checkpoint_file():
    data = dict(resolve(MATRIX / "L-F.yaml").data)
    data["model"] = {k: v for k, v in data["model"].items() if k != "train_file"}
    with pytest.raises(ConfigSchemaError):
        validate(data)


def test_revision_must_be_a_full_sha():
    """A short or truncated revision resolves differently over time and
    destroys the reproducibility claim the pins exist for."""
    data = dict(resolve(MATRIX / "L-F.yaml").data)
    data["model"] = {**data["model"], "train_revision": "19b7f51"}
    with pytest.raises(ConfigSchemaError):
        validate(data)


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
