"""Run directories: idempotent, resumable, never silently retrain.

    from lorafactory.runs import prepare_run, next_attempt, RunConflictError
    h = prepare_run(runs_root, adapter_id, config_hash)
    h.action    # "create" | "skip" | "resume"
    h.run_dir   # Path, named <adapter_id>-a<NN> (attempt, 2 digits, from 01)

Semantics:
- No run dir yet -> action "create"; dir + config.hash file are created.
- Dir exists, hash matches, adapter/<ID>.safetensors present -> "skip".
- Dir exists, hash matches, no adapter file -> "resume".
- Dir exists, hash differs -> RunConflictError.
- next_attempt(runs_root, adapter_id) -> new dir with incremented attempt
  (predecessor stays untouched).
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

CONFIG_HASH_FILENAME = "config.hash"
_ATTEMPT_RE = re.compile(r"^(?P<adapter_id>.+)-a(?P<attempt>\d{2})$")


class RunConflictError(Exception):
    """A run dir exists with a different config hash than requested."""


@dataclass
class RunHandle:
    action: str
    run_dir: Path
    attempt: int


def _run_dir_name(adapter_id: str, attempt: int) -> str:
    return f"{adapter_id}-a{attempt:02d}"


def _existing_attempts(runs_root: Path, adapter_id: str) -> list[int]:
    if not runs_root.is_dir():
        return []
    attempts = []
    for p in runs_root.iterdir():
        if not p.is_dir():
            continue
        m = _ATTEMPT_RE.match(p.name)
        if m and m.group("adapter_id") == adapter_id:
            attempts.append(int(m.group("attempt")))
    return sorted(attempts)


def _is_complete(run_dir: Path, adapter_id: str) -> bool:
    return (run_dir / "adapter" / f"{adapter_id}.safetensors").exists()


def prepare_run(runs_root: Path, adapter_id: str, config_hash: str) -> RunHandle:
    runs_root = Path(runs_root)
    attempts = _existing_attempts(runs_root, adapter_id)

    if not attempts:
        attempt = 1
        run_dir = runs_root / _run_dir_name(adapter_id, attempt)
        run_dir.mkdir(parents=True, exist_ok=True)
        (run_dir / CONFIG_HASH_FILENAME).write_text(config_hash)
        return RunHandle(action="create", run_dir=run_dir, attempt=attempt)

    attempt = attempts[-1]
    run_dir = runs_root / _run_dir_name(adapter_id, attempt)
    existing_hash = (run_dir / CONFIG_HASH_FILENAME).read_text().strip()

    if existing_hash != config_hash:
        raise RunConflictError(
            f"run {run_dir} has config hash {existing_hash!r}, "
            f"requested {config_hash!r}"
        )

    if _is_complete(run_dir, adapter_id):
        return RunHandle(action="skip", run_dir=run_dir, attempt=attempt)
    return RunHandle(action="resume", run_dir=run_dir, attempt=attempt)


def next_attempt(runs_root: Path, adapter_id: str) -> Path:
    runs_root = Path(runs_root)
    attempts = _existing_attempts(runs_root, adapter_id)
    attempt = (attempts[-1] + 1) if attempts else 1
    run_dir = runs_root / _run_dir_name(adapter_id, attempt)
    run_dir.mkdir(parents=True, exist_ok=True)
    return run_dir
