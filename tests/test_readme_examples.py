"""Every fenced ```yaml block in README.md under a 'lorafactory-example'
marker must actually resolve+validate — a doc example that silently rots
is worse than no example.

Beyond schema validation, the two flagship examples (`minimal-base.yaml`,
`local-dataset.yaml`) are driven through the real pipeline a user would
actually invoke — `cli._resolve_config()` then the kohya emitter / dataset
fetch planner — because a config can pass Pydantic validation yet still
carry a raw, un-defaulted dict past it if `_resolve_config` ever regresses
to returning the loader's pre-default `rc.data` (see C1 in the final
review's fix wave: arch/dataset/determinism/train defaults must actually
reach `emit-kohya`/`fetch-dataset`, not just `validate-config`)."""

import re
import tomllib

import pytest

from conftest import ROOT
from lorafactory.cli import _resolve_config
from lorafactory.config.loader import resolve
from lorafactory.config.schema import validate
from lorafactory.data.fetch import fetch_plan
from lorafactory.kohya.toml_emitter import emit

_BLOCK_RE = re.compile(
    r"<!-- lorafactory-example: (?P<name>[\w.-]+) -->\n```yaml\n(?P<body>.*?)\n```",
    re.DOTALL,
)


def _readme_examples() -> list[tuple[str, str]]:
    text = (ROOT / "README.md").read_text()
    return [(m.group("name"), m.group("body")) for m in _BLOCK_RE.finditer(text)]


@pytest.mark.parametrize(
    "name,body", _readme_examples(), ids=[name for name, _ in _readme_examples()]
)
def test_readme_example_resolves_and_validates(name, body, tmp_path):
    path = tmp_path / name
    path.write_text(body)
    validate(resolve(path).data)


def test_readme_has_at_least_the_minimal_and_local_dataset_examples():
    names = {name for name, _ in _readme_examples()}
    assert "minimal-base.yaml" in names
    assert "local-dataset.yaml" in names


def _readme_example_body(name: str) -> str:
    body = dict(_readme_examples()).get(name)
    assert body is not None, f"README.md has no lorafactory-example: {name}"
    return body


def test_minimal_base_example_emits_defaulted_kohya_toml(tmp_path, monkeypatch):
    """The README's own `minimal-base.yaml` example — `model.arch: sd3` plus
    only the budget-relevant fields — must actually emit a usable kohya TOML
    through the real CLI resolution path, with every arch/train default
    (`optimizer_type`, `save_model_as`, `gradient_checkpointing`,
    `logging_dir`) and dataset default (`resolution`) landing in the
    artifact, not just in the validated Pydantic object."""
    monkeypatch.setenv("LORAFACTORY_MODELS_DIR", str(tmp_path / "models"))
    monkeypatch.setenv("LORAFACTORY_DATASETS_DIR", str(tmp_path / "datasets"))

    config_path = tmp_path / "minimal-base.yaml"
    config_path.write_text(_readme_example_body("minimal-base.yaml"))

    rc = _resolve_config(config_path)
    result = emit(rc.data, tmp_path / "out")
    with open(result.train_toml, "rb") as f:
        train = tomllib.load(f)
    with open(result.dataset_toml, "rb") as f:
        dataset = tomllib.load(f)

    assert train["optimizer_type"] == "AdamW8bit"
    assert train["save_model_as"] == "safetensors"
    assert train["gradient_checkpointing"] is True
    assert train["logging_dir"] == "logs"
    # model.arch defaults must have resolved to a real single-file checkpoint
    # path, not silently stayed None (that would raise ModelPathError below
    # `_resolve_config`, which is exactly the C1 bug this test guards).
    assert train["pretrained_model_name_or_path"].endswith("sd3_medium.safetensors")

    ds = dataset["datasets"][0]
    assert ds["resolution"] == 1024


def test_local_dataset_example_fetch_plan_has_defaulted_resolution(tmp_path, monkeypatch):
    """The README's `local-dataset.yaml` example must carry a real
    `dataset.resolution` (default 1024) through `_resolve_config` into
    `fetch_plan` — `fetch_dataset` raises `DatasetSourceError` on a missing
    resolution, which is what the un-defaulted `rc.data` bug (C1) produced."""
    monkeypatch.setenv("LORAFACTORY_MODELS_DIR", str(tmp_path / "models"))
    monkeypatch.setenv("LORAFACTORY_DATASETS_DIR", str(tmp_path / "datasets"))

    config_path = tmp_path / "local-dataset.yaml"
    config_path.write_text(_readme_example_body("local-dataset.yaml"))

    rc = _resolve_config(config_path)
    plan = fetch_plan(rc.data)
    assert plan["resolution"] == 1024
