"""Config-declared dataset acquisition (`dataset.source` -> fetch-dataset).

API pinned:
    from lorafactory.data.fetch import fetch_plan, fetch_dataset, DatasetSourceError

`fetch_plan` is the pure half — it must stay callable with no network and no
weights present, the same guarantee `models.download_plan` gives, because that
is what makes `--dry-run` meaningful on a machine that has neither.

The test configs under configs/test/ are checked here rather than in
test_config_loader.py: they are deliberately outside the placement matrix, so
the matrix's ID lists must not grow to cover them.
"""

import pytest
import yaml

from conftest import CONFIGS
from lorafactory.config.loader import resolve
from lorafactory.config.schema import validate
from lorafactory.data.fetch import DatasetSourceError, fetch_dataset, fetch_plan
from lorafactory.data.manifest import COLUMNS

TEST_CONFIGS = CONFIGS / "test"
SMOKE = TEST_CONFIGS / "smoke.yaml"
SMOKE_OBJ = TEST_CONFIGS / "smoke-obj.yaml"


def test_smoke_configs_resolve_validate_and_are_exempt():
    for path in (SMOKE, SMOKE_OBJ):
        rc = resolve(path)
        validate(rc.data)
        assert rc.data["budget_exempt"] is True, f"{path.name} is not budget_exempt"
        # A smoke config that quietly ran the full budget would look like a
        # matrix result in runs/.
        assert rc.data["train"]["max_train_steps"] < 2000


def test_smoke_config_inherits_the_pinned_base_model():
    """The test path must exercise the real weights, not a convenience model."""
    with open(CONFIGS / "base.yaml") as f:
        base = yaml.safe_load(f)["model"]
    assert resolve(SMOKE).data["model"] == base


def test_plan_reports_the_pinned_source_without_touching_the_network():
    plan = fetch_plan(resolve(SMOKE).data)
    assert plan["repo"] == "huggan/few-shot-aurora"
    assert plan["form"] == "parquet"
    assert plan["limit"] == 50
    assert plan["image_dir"].name == "STYLE"
    assert plan["manifest"].name == "manifest.csv"
    assert len(plan["revision"]) == 40


def test_file_source_plan_reports_patterns_not_filenames():
    """Expanding a glob means listing the repo, which is network — not in a plan."""
    plan = fetch_plan(resolve(SMOKE_OBJ).data)
    assert plan["form"] == "files"
    assert plan["source"] == ["dataset/candle/*.jpg"]


def test_every_provenance_column_is_filled_by_the_plan():
    """manifest.py rejects empty provenance, so the config must supply all of it."""
    plan = fetch_plan(resolve(SMOKE).data)
    provenance = plan["provenance"]
    for column in COLUMNS:
        if column in ("file", "sha256", "caption", "source_url"):
            continue  # derived per image, not config-supplied
        assert provenance.get(column, "").strip(), f"empty provenance column {column}"
    assert plan["caption"].strip()


def test_source_url_pins_the_revision():
    from lorafactory.data.fetch import _blob_url

    plan = fetch_plan(resolve(SMOKE).data)
    url = _blob_url(plan["repo"], plan["revision"], plan["source"], row=0)
    assert plan["revision"] in url, "provenance URL points at a branch, not a commit"


def test_matrix_config_has_no_source_and_says_so_clearly():
    """The matrix data was acquired by hand; fetch-dataset must not guess."""
    matrix = resolve(CONFIGS / "matrix" / "L-A.yaml").data
    with pytest.raises(DatasetSourceError, match="no 'source'"):
        fetch_plan(matrix)


def test_source_must_name_exactly_one_form():
    config = resolve(SMOKE).data
    both = dict(config)
    both["dataset"] = dict(config["dataset"])
    both["dataset"]["source"] = dict(config["dataset"]["source"])
    both["dataset"]["source"]["files"] = ["dataset/x/*.jpg"]
    with pytest.raises(DatasetSourceError, match="exactly one"):
        fetch_plan(both)


def test_local_source_plan_globs_the_configured_directory(tmp_path):
    (tmp_path / "img_0001.png").write_bytes(b"\x89PNG\r\n")
    config = {
        "dataset": {
            "name": "STYLE", "path": "STYLE", "manifest": "STYLE/manifest.csv",
            "resolution": 64,
            "source": {
                "type": "local", "path": str(tmp_path), "caption": "a photo",
            },
        },
    }
    plan = fetch_plan(config)
    assert plan["form"] == "files"
    assert plan["source"] == [f"{tmp_path}/*.png"]
    assert plan["repo"] is None
    assert plan["caption"] == "a photo"


def test_local_source_fetch_copies_files_and_fills_sentinel_provenance(tmp_path, monkeypatch):
    from PIL import Image

    src_dir = tmp_path / "src"
    src_dir.mkdir()
    Image.new("RGB", (64, 64), "red").save(src_dir / "a.png")
    Image.new("RGB", (64, 64), "blue").save(src_dir / "b.png")

    datasets_dir = tmp_path / "datasets"
    monkeypatch.setenv("LORAFACTORY_DATASETS_DIR", str(datasets_dir))

    config = {
        "dataset": {
            "name": "STYLE", "path": "STYLE", "manifest": "STYLE/manifest.csv",
            "resolution": 64,
            "source": {
                "type": "local", "path": str(src_dir), "caption": "a photo of sks_style",
            },
        },
    }
    report = fetch_dataset(config)
    assert report.action == "fetched"
    assert report.count == 2
    assert report.ok, report.errors

    from lorafactory.data.manifest import check_dataset
    result = check_dataset(report.image_dir)
    assert result.ok, result.errors


def test_local_source_explicit_provenance_is_respected(tmp_path, monkeypatch):
    from PIL import Image

    src_dir = tmp_path / "src"
    src_dir.mkdir()
    Image.new("RGB", (64, 64), "red").save(src_dir / "a.png")

    datasets_dir = tmp_path / "datasets"
    monkeypatch.setenv("LORAFACTORY_DATASETS_DIR", str(datasets_dir))

    config = {
        "dataset": {
            "name": "STYLE", "path": "STYLE", "manifest": "STYLE/manifest.csv",
            "resolution": 64,
            "source": {
                "type": "local", "path": str(src_dir), "caption": "x",
                "author": "Jane Doe", "licence": "CC0",
            },
        },
    }
    report = fetch_dataset(config)
    import csv
    with open(report.image_dir / "manifest.csv", newline="") as f:
        row = next(csv.DictReader(f))
    assert row["author"] == "Jane Doe"
    assert row["licence"] == "CC0"
    assert row["licence_url"] == "local"
    assert row["acquisition_date"] == "unknown"
