"""Run directories: idempotent, resumable, never silently retrain.

API pinned:
    from lorafactory.runs import prepare_run, next_attempt, RunConflictError
    h = prepare_run(runs_root, adapter_id, config_hash)
    h.action    # "create" | "skip" | "resume"
    h.run_dir   # Path, named <adapter_id>-a<NN> (attempt, 2 digits, from 01)

Semantics:
- No run dir yet -> action "create"; dir + config.hash file are created.
- Dir exists, hash matches, adapter/<ID>.safetensors present -> "skip" (no-op).
- Dir exists, hash matches, no adapter file -> "resume".
- Dir exists, hash differs -> RunConflictError (matched budget: a requeued
  job must never silently retrain).
- next_attempt(runs_root, adapter_id) -> new dir with incremented attempt
  (forced retraining; predecessor stays untouched).
"""

import pytest

from lorafactory.runs import RunConflictError, next_attempt, prepare_run


def complete(run_dir, adapter_id):
    (run_dir / "adapter").mkdir(parents=True, exist_ok=True)
    (run_dir / "adapter" / f"{adapter_id}.safetensors").write_bytes(b"stub")


def test_create_then_skip(tmp_path):
    h = prepare_run(tmp_path, "L-E", "hash-1")
    assert h.action == "create"
    assert h.run_dir.name == "L-E-a01"
    assert h.run_dir.is_dir()
    assert (h.run_dir / "config.hash").read_text().strip() == "hash-1"

    complete(h.run_dir, "L-E")
    h2 = prepare_run(tmp_path, "L-E", "hash-1")
    assert h2.action == "skip"
    assert h2.run_dir == h.run_dir


def test_resume_when_incomplete(tmp_path):
    h = prepare_run(tmp_path, "L-A", "hash-1")
    assert h.action == "create"
    h2 = prepare_run(tmp_path, "L-A", "hash-1")
    assert h2.action == "resume"


def test_hash_mismatch_refuses(tmp_path):
    h = prepare_run(tmp_path, "L-M", "hash-1")
    complete(h.run_dir, "L-M")
    with pytest.raises(RunConflictError):
        prepare_run(tmp_path, "L-M", "hash-2")


def test_next_attempt_keeps_predecessor(tmp_path):
    h = prepare_run(tmp_path, "L-F", "hash-1")
    complete(h.run_dir, "L-F")
    d2 = next_attempt(tmp_path, "L-F")
    assert d2.name == "L-F-a02"
    assert d2.is_dir()
    assert (tmp_path / "L-F-a01" / "adapter" / "L-F.safetensors").exists()
