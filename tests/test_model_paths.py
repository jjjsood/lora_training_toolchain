"""`lorafactory.models` — where the weight files actually live.

kohya-ss/sd-scripts takes a FILE for --pretrained_model_name_or_path and a file
per text encoder. Configs name those files relative to a model root so the repo
stays machine-independent; this module is the single place that turns
(config, root) into absolute paths, and the same place `fetch-models` reads to
decide what to download.

Before this existed the emitter wrote the HF repo id ("stabilityai/stable-
diffusion-3-medium") straight into the TOML — syntactically fine, silently
accepted by the emitter's key validation, and unopenable by kohya. That class
of error is only catchable here, on CPU, before any GPU time is spent.
"""

from pathlib import Path

import pytest

from conftest import MATRIX
from lorafactory.config.loader import resolve
from lorafactory.models import (
    ModelPaths,
    dataset_root,
    download_plan,
    model_root,
    resolve_dataset_paths,
    resolve_model_paths,
)

ENV_VAR = "LORAFACTORY_MODELS_DIR"
DATASET_ENV_VAR = "LORAFACTORY_DATASETS_DIR"


def test_model_root_comes_from_the_environment(tmp_path, monkeypatch):
    monkeypatch.setenv(ENV_VAR, str(tmp_path))
    assert model_root() == tmp_path


def test_model_root_has_a_default_when_unset(monkeypatch):
    monkeypatch.delenv(ENV_VAR, raising=False)
    # A default keeps the container working without extra env wiring; what
    # matters is that it is absolute, so nothing depends on the cwd.
    assert model_root().is_absolute()


def test_resolved_paths_are_absolute_and_under_the_root(tmp_path, monkeypatch):
    monkeypatch.setenv(ENV_VAR, str(tmp_path))
    config = resolve(MATRIX / "L-F.yaml").data
    paths = resolve_model_paths(config)

    assert isinstance(paths, ModelPaths)
    assert paths.checkpoint == tmp_path / "sd3_medium.safetensors"
    assert paths.checkpoint.is_absolute()
    for name in ("clip_l", "clip_g", "t5xxl"):
        p = paths.text_encoders[name]
        assert p.is_absolute()
        assert str(p).startswith(str(tmp_path))
        assert p.suffix == ".safetensors"


def test_flux_config_resolves_its_autoencoder(tmp_path, monkeypatch):
    monkeypatch.setenv(ENV_VAR, str(tmp_path))
    paths = resolve_model_paths(resolve(MATRIX / "F-F.yaml").data)
    assert paths.checkpoint == tmp_path / "flux1-dev.safetensors"
    assert paths.ae == tmp_path / "ae.safetensors"


def test_sd3_config_has_no_autoencoder_entry(tmp_path, monkeypatch):
    monkeypatch.setenv(ENV_VAR, str(tmp_path))
    assert resolve_model_paths(resolve(MATRIX / "L-F.yaml").data).ae is None


def test_resolving_does_not_require_the_files_to_exist(tmp_path, monkeypatch):
    """Path resolution is pure: `emit-kohya` must work before any download."""
    monkeypatch.setenv(ENV_VAR, str(tmp_path / "nothing-here"))
    resolve_model_paths(resolve(MATRIX / "L-F.yaml").data)


def test_download_plan_names_repo_revision_and_filename(tmp_path, monkeypatch):
    monkeypatch.setenv(ENV_VAR, str(tmp_path))
    plan = download_plan(resolve(MATRIX / "L-F.yaml").data)

    assert plan, "nothing to download?"
    for entry in plan:
        assert entry["repo_id"]
        assert len(entry["revision"]) == 40, entry
        assert entry["filename"]
        assert entry["local_path"].is_absolute()

    names = {e["filename"] for e in plan}
    assert "sd3_medium.safetensors" in names
    assert "text_encoders/t5xxl_fp16.safetensors" in names


def test_download_plan_pins_the_configs_revision(tmp_path, monkeypatch):
    """Every download must be pinned — an unpinned pull silently changes the
    weights an adapter was trained on and breaks the provenance record."""
    monkeypatch.setenv(ENV_VAR, str(tmp_path))
    config = resolve(MATRIX / "L-F.yaml").data
    plan = download_plan(config)
    expected = config["model"]["train_revision"]
    assert {e["revision"] for e in plan} == {expected}


def test_missing_train_file_is_a_clear_error(tmp_path, monkeypatch):
    monkeypatch.setenv(ENV_VAR, str(tmp_path))
    broken = dict(resolve(MATRIX / "L-F.yaml").data)
    broken["model"] = {k: v for k, v in broken["model"].items() if k != "train_file"}
    with pytest.raises(Exception, match="train_file"):
        resolve_model_paths(broken)


# --------------------------------------------------------------------------
# Datasets follow the same root-relative rule, for the same reason: the
# matched-budget block must be byte-identical across machines, so an absolute
# host path cannot live in the config.
# --------------------------------------------------------------------------

def test_dataset_root_comes_from_the_environment(tmp_path, monkeypatch):
    monkeypatch.setenv(DATASET_ENV_VAR, str(tmp_path))
    assert dataset_root() == tmp_path


def test_dataset_root_defaults_to_the_container_mount(monkeypatch):
    monkeypatch.delenv(DATASET_ENV_VAR, raising=False)
    assert dataset_root().is_absolute()


def test_style_dataset_resolves_under_the_root(tmp_path, monkeypatch):
    monkeypatch.setenv(DATASET_ENV_VAR, str(tmp_path))
    paths = resolve_dataset_paths(resolve(MATRIX / "L-F.yaml").data)
    assert paths.image_dir == tmp_path / "STYLE"
    assert paths.manifest == tmp_path / "STYLE" / "manifest.csv"


def test_obj_dataset_resolves_under_the_root(tmp_path, monkeypatch):
    monkeypatch.setenv(DATASET_ENV_VAR, str(tmp_path))
    paths = resolve_dataset_paths(resolve(MATRIX / "O-F.yaml").data)
    assert paths.image_dir == tmp_path / "OBJ"


def test_no_config_hardcodes_an_absolute_path():
    """No config may contain an absolute path, anywhere.

    Every absolute path in a config is a machine the toolchain now only runs
    on. This was first caught as `dataset.path: /workspace/datasets/STYLE`,
    then again as `output.base_dir: /workspace/runs` — which the emitter
    copied into the kohya TOML, pointing training output at a directory that
    exists only inside the container. The rule is general, so the check is
    too: roots come from the environment, configs name things relative to
    them.
    """
    import yaml

    from conftest import CONFIGS

    def walk(node, trail):
        if isinstance(node, dict):
            for k, v in node.items():
                yield from walk(v, f"{trail}.{k}" if trail else str(k))
        elif isinstance(node, list):
            for i, v in enumerate(node):
                yield from walk(v, f"{trail}[{i}]")
        elif isinstance(node, str) and node.startswith("/"):
            yield trail, node

    offenders = []
    for path in sorted(CONFIGS.rglob("*.yaml")):
        with open(path) as f:
            data = yaml.safe_load(f) or {}
        for trail, value in walk(data, ""):
            offenders.append(f"{path.name}:{trail}={value}")
    assert not offenders, offenders


def test_emitted_output_dir_is_the_run_directory(tmp_path, monkeypatch):
    """kohya writes the trained adapter to output_dir; it must be the run
    directory the CLI actually created, not a path baked into a config."""
    import tomllib

    from lorafactory.kohya.toml_emitter import emit

    monkeypatch.setenv(ENV_VAR, str(tmp_path / "models"))
    monkeypatch.setenv(DATASET_ENV_VAR, str(tmp_path / "datasets"))

    run_dir = tmp_path / "runs" / "L-E-a01"
    result = emit(resolve(MATRIX / "L-E.yaml").data, run_dir)

    with open(result.train_toml, "rb") as f:
        train = tomllib.load(f)
    flat = {}
    for k, v in train.items():
        flat.update(v) if isinstance(v, dict) else flat.__setitem__(k, v)

    out_dir = Path(flat["output_dir"])
    assert out_dir.is_absolute(), flat["output_dir"]
    assert out_dir == run_dir, f"output_dir {out_dir} is not the run dir {run_dir}"
    assert flat["output_name"] == "L-E"
