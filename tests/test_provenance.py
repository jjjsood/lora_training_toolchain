"""Provenance sidecar + training log.

API pinned:
    from lorafactory.provenance import (
        write_provenance, append_training_log, ProvenanceError,
        REQUIRED_FIELDS, TRAINING_LOG_COLUMNS)
"""

import csv
import json

import pytest

from lorafactory.provenance import (
    REQUIRED_FIELDS,
    TRAINING_LOG_COLUMNS,
    ProvenanceError,
    append_training_log,
    write_provenance,
)

EXPECTED_REQUIRED = {
    "run_id", "adapter_id", "config_hash", "seed", "dataset_name",
    "dataset_manifest_hash", "base_model_repo", "base_model_revision",
    "sd_scripts_ref", "versions", "determinism_env", "wall_clock_s",
    "final_loss", "adapter_sha256",
}

EXPECTED_LOG_COLUMNS = [
    "run_id", "adapter_id", "config_hash", "dataset_name",
    "dataset_manifest_hash", "seed", "steps", "learning_rate", "batch_size",
    "wall_clock_s", "final_loss", "adapter_sha256",
]


def full_record():
    return {
        "run_id": "L-E-a01", "adapter_id": "L-E", "config_hash": "c" * 64,
        "seed": 42, "dataset_name": "STYLE", "dataset_manifest_hash": "d" * 64,
        "base_model_repo": "stabilityai/stable-diffusion-3-medium",
        "base_model_revision": "abc123", "sd_scripts_ref": "v0.11.1",
        "versions": {"torch": "x"}, "determinism_env": {"PYTHONHASHSEED": "0"},
        "wall_clock_s": 12.5, "final_loss": 0.123, "adapter_sha256": "e" * 64,
    }


def test_required_fields_pinned():
    assert EXPECTED_REQUIRED <= set(REQUIRED_FIELDS)
    assert TRAINING_LOG_COLUMNS == EXPECTED_LOG_COLUMNS


def test_write_provenance(tmp_path):
    out = write_provenance(tmp_path, full_record())
    data = json.loads(out.read_text())
    assert data["run_id"] == "L-E-a01"
    assert out.name == "provenance.json"


def test_missing_field_raises(tmp_path):
    rec = full_record()
    del rec["adapter_sha256"]
    with pytest.raises(ProvenanceError, match="adapter_sha256"):
        write_provenance(tmp_path, rec)


def test_training_log_append(tmp_path):
    log = tmp_path / "training_log.csv"
    row = {c: "v" for c in EXPECTED_LOG_COLUMNS}
    append_training_log(log, row)
    append_training_log(log, row)
    with open(log) as f:
        rows = list(csv.reader(f))
    assert rows[0] == EXPECTED_LOG_COLUMNS
    assert len(rows) == 3
