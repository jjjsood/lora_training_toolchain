"""config/schema.py — pydantic validation of a resolved config.

The loader itself accepts any YAML shape, so a missing key like
`model.train_file` would otherwise slip through: the emitter reads with
`.get()` and writes whatever it finds.

Validation runs on the RESOLVED config (post-overlay), because that is the
object every consumer downstream — emitter, runner, models, provenance —
actually receives.
"""

import warnings

import pytest
from pydantic import ValidationError

from conftest import ALL_MATRIX_IDS, CONFIGS, FALLBACK_IDS, MATRIX
from lorafactory.config.loader import resolve
from lorafactory.config.schema import (
    ARCH_DEFAULTS,
    ConfigSchemaError,
    DatasetSection,
    DeterminismSection,
    LocalDatasetSource,
    ModelSection,
    RemoteDatasetSource,
    TrainSection,
    validate,
)


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


def test_non_string_arch_fails_cleanly_rather_than_raising_typeerror():
    """`arch: [sd3]` (a YAML list, unhashable) must raise pydantic's own clean
    `string_type` ConfigSchemaError, not a raw TypeError from
    `ARCH_DEFAULTS.get()` — a regression from before arch defaults existed."""
    data = dict(resolve(MATRIX / "L-F.yaml").data)
    data["model"] = {**data["model"], "arch": ["sd3"]}
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


def test_arch_defaults_match_the_verified_pins():
    """`ARCH_DEFAULTS` (schema.py) is a fourth hand-kept copy of the same pins
    `tests/test_model_pins.py` verifies against the HF API — nothing tied it
    to that source of truth, so a pin could move there (the documented
    procedure) and leave ARCH_DEFAULTS silently stale."""
    from test_model_pins import VERIFIED_REVISIONS

    for arch in ("sd3", "flux"):
        defaults = ARCH_DEFAULTS[arch]
        assert defaults["train_revision"] == VERIFIED_REVISIONS[defaults["train_repo"]]
        assert defaults["eval_revision"] == VERIFIED_REVISIONS[defaults["eval_repo"]]


def test_foreign_repo_with_a_borrowed_verified_sha_warns():
    """A SHA that IS a real verified pin, paired with a repo that is NOT the
    one it was verified for (e.g. a careless copy-paste into a fork's
    config), must warn — worse than an unrecognized repo/SHA pair with no
    ground truth to contradict, which is deliberately let through silently."""
    data = dict(resolve(MATRIX / "L-F.yaml").data)
    data["model"] = {
        **data["model"],
        "train_repo": "me/my-fork",
        "train_revision": ARCH_DEFAULTS["flux"]["train_revision"],
    }
    with pytest.warns(UserWarning, match="verified pin for a different repo"):
        validate(data)


def test_unrecognized_repo_and_unrecognized_sha_does_not_warn():
    """The pre-existing, spec-sanctioned silent case: neither the repo nor
    the SHA is anything we have ground truth for."""
    data = dict(resolve(MATRIX / "L-F.yaml").data)
    data["model"] = {
        **data["model"],
        "train_repo": "me/my-fork",
        "train_revision": "b" * 40,
    }
    with warnings.catch_warnings():
        warnings.simplefilter("error")
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


def test_determinism_section_all_fields_default():
    section = DeterminismSection.model_validate({})
    assert section.cublas_workspace_config == ":4096:8"
    assert section.pythonhashseed == "0"
    assert section.deterministic_algorithms is True


def test_pythonhashseed_int_zero_coerces_to_string():
    section = DeterminismSection.model_validate({"pythonhashseed": 0})
    assert section.pythonhashseed == "0"


def test_determinism_section_explicit_values_are_respected():
    section = DeterminismSection.model_validate({
        "cublas_workspace_config": ":16:8",
        "pythonhashseed": 7,
        "deterministic_algorithms": False,
    })
    assert section.cublas_workspace_config == ":16:8"
    assert section.pythonhashseed == "7"
    assert section.deterministic_algorithms is False


def test_train_section_optional_fields_default():
    section = TrainSection.model_validate({
        "seed": 1, "max_train_steps": 10,
        "learning_rate": 0.0001, "train_batch_size": 1,
        "mixed_precision": "bf16", "save_precision": "bf16",
    })
    assert section.optimizer_type == "AdamW8bit"
    assert section.save_model_as == "safetensors"
    assert section.gradient_checkpointing is True
    assert section.logging_dir == "logs"


def test_train_section_still_requires_run_specific_fields():
    with pytest.raises(ValidationError):
        TrainSection.model_validate({"seed": 1})


def test_local_dataset_source_needs_only_path():
    source = LocalDatasetSource.model_validate({
        "path": "STYLE", "caption": "a photo of sks_style",
    })
    assert source.type == "local"
    assert source.files == ["STYLE/*.png"]
    assert source.author is None


def test_local_dataset_source_files_override_the_default_glob():
    source = LocalDatasetSource.model_validate({
        "path": "STYLE", "caption": "x", "files": ["STYLE/*.jpg"],
    })
    assert source.files == ["STYLE/*.jpg"]


def test_remote_dataset_source_unchanged_behavior():
    with pytest.raises(ValidationError):
        RemoteDatasetSource.model_validate({"path": "STYLE"})  # no repo/revision/etc


def test_dataset_source_without_type_and_with_repo_infers_remote():
    section = DatasetSection.model_validate({
        "name": "STYLE", "path": "STYLE", "manifest": "STYLE/manifest.csv",
        "source": {
            "repo": "huggan/few-shot-aurora",
            "revision": "ccf645535bc3b5f755d03567374780ae9473d66b",
            "parquet": "data/train-00000-of-00001.parquet",
            "caption": "x", "author": "unknown", "licence": "unknown",
            "licence_url": "https://example.com", "acquisition_date": "2026-08-08",
        },
    })
    assert isinstance(section.source, RemoteDatasetSource)


def test_dataset_source_with_typoed_repo_key_still_infers_remote():
    """A typo'd `repo` key (e.g. `repos:`) must not be diagnosed as a local
    source: the presence of any other remote-only key (revision/parquet/
    licence/...) is enough to infer `remote`, so the resulting error is
    about the missing `repo`, not about 'extra' fields that don't belong on
    a local source."""
    with pytest.raises(ValidationError) as excinfo:
        DatasetSection.model_validate({
            "name": "STYLE", "path": "STYLE", "manifest": "STYLE/manifest.csv",
            "source": {
                "repos": "huggan/few-shot-aurora",  # typo: should be 'repo'
                "revision": "ccf645535bc3b5f755d03567374780ae9473d66b",
                "parquet": "data/train-00000-of-00001.parquet",
                "caption": "x", "author": "unknown", "licence": "unknown",
                "licence_url": "https://example.com", "acquisition_date": "2026-08-08",
            },
        })
    message = str(excinfo.value)
    assert "repo" in message
    # It must not be misdiagnosed as local and complain about the fields a
    # remote source legitimately carries.
    assert "revision" not in message
    assert "parquet" not in message


def test_dataset_source_without_type_and_without_repo_infers_local():
    section = DatasetSection.model_validate({
        "name": "STYLE", "path": "STYLE", "manifest": "STYLE/manifest.csv",
        "source": {"path": "STYLE", "caption": "x"},
    })
    assert isinstance(section.source, LocalDatasetSource)


def test_local_source_with_optional_licence_fields_is_not_misdiagnosed_remote():
    """Regression: I3's `_REMOTE_ONLY_KEYS` over-corrected by including
    `licence`/`licence_url`/`acquisition_date` — fields that are also valid
    OPTIONAL fields on `LocalDatasetSource` itself. A genuinely local,
    `type`-less source that happens to fill them in must still infer
    `local`, not get misclassified as `remote` (and then fail validation for
    a missing `repo`/`revision` and a now-'extra' `path`)."""
    section = DatasetSection.model_validate({
        "name": "STYLE", "path": "STYLE",
        "source": {
            "path": "STYLE", "caption": "x", "licence": "CC0",
            "licence_url": "https://example.com/cc0", "acquisition_date": "2026-08-08",
        },
    })
    assert isinstance(section.source, LocalDatasetSource)


def test_dataset_section_resolution_defaults_to_1024():
    section = DatasetSection.model_validate({
        "name": "STYLE", "path": "STYLE", "manifest": "STYLE/manifest.csv",
    })
    assert section.resolution == 1024


def test_dataset_section_manifest_defaults_from_path():
    section = DatasetSection.model_validate({"name": "STYLE", "path": "STYLE"})
    assert section.manifest == "STYLE/manifest.csv"


def test_dataset_section_explicit_manifest_is_respected():
    section = DatasetSection.model_validate({
        "name": "STYLE", "path": "STYLE", "manifest": "elsewhere/m.csv",
    })
    assert section.manifest == "elsewhere/m.csv"


def test_config_schema_error_is_one_line_per_field():
    data = dict(resolve(MATRIX / "L-F.yaml").data)
    del data["train"]["max_train_steps"]
    del data["train"]["seed"]
    with pytest.raises(ConfigSchemaError) as excinfo:
        validate(data)
    message = str(excinfo.value)
    assert "train.max_train_steps" in message
    assert "train.seed" in message
    assert "For further information visit" not in message  # pydantic's own footer
