"""Provenance sidecar + training log.

    from lorafactory.provenance import (
        write_provenance, append_training_log, ProvenanceError,
        REQUIRED_FIELDS, TRAINING_LOG_COLUMNS)
"""

from __future__ import annotations

import csv
import json
from pathlib import Path

REQUIRED_FIELDS = [
    "run_id", "adapter_id", "config_hash", "seed", "dataset_name",
    "dataset_manifest_hash", "base_model_repo", "base_model_revision",
    "sd_scripts_ref", "versions", "determinism_env", "wall_clock_s",
    "final_loss", "adapter_sha256",
]

TRAINING_LOG_COLUMNS = [
    "run_id", "adapter_id", "config_hash", "dataset_name",
    "dataset_manifest_hash", "seed", "steps", "learning_rate", "batch_size",
    "wall_clock_s", "final_loss", "adapter_sha256",
]

PROVENANCE_FILENAME = "provenance.json"


class ProvenanceError(Exception):
    """A provenance record is missing a required field."""


def write_provenance(run_dir: Path, record: dict) -> Path:
    missing = [f for f in REQUIRED_FIELDS if f not in record]
    if missing:
        raise ProvenanceError(f"missing required field(s): {', '.join(missing)}")

    out = Path(run_dir) / PROVENANCE_FILENAME
    out.write_text(json.dumps(record, indent=2, sort_keys=True))
    return out


def append_training_log(log_path: Path, row: dict) -> None:
    log_path = Path(log_path)
    write_header = not log_path.exists()
    with open(log_path, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=TRAINING_LOG_COLUMNS)
        if write_header:
            w.writeheader()
        w.writerow({c: row.get(c, "") for c in TRAINING_LOG_COLUMNS})
