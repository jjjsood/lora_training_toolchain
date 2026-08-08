"""Thin, importable builder for the kohya `accelerate launch` subprocess command.

Not executed by the check suite (no GPU there) — this module must stay
import-safe (no side effects at import time) and only ever spawns a process
when `run_train()` is explicitly called.

Paths into the pinned kohya venv / sd-scripts v0.11.1 checkout come from env:
    LORAFACTORY_KOHYA_PYTHON    python interpreter inside the kohya venv
    LORAFACTORY_SDSCRIPTS_DIR   sd-scripts v0.11.1 checkout (cwd for the launch)

    from lorafactory.kohya.runner import build_train_command, run_train
    cmd = build_train_command("sd3", train_toml_path)
    cmd.argv, cmd.env, cmd.cwd     # what a caller subprocess.run()s
    run_train("sd3", train_toml_path, run_dir / "train.log")
"""

from __future__ import annotations

import os
import subprocess
from dataclasses import dataclass
from pathlib import Path

from ..determinism import build_env

LORAFACTORY_KOHYA_PYTHON = "LORAFACTORY_KOHYA_PYTHON"
LORAFACTORY_SDSCRIPTS_DIR = "LORAFACTORY_SDSCRIPTS_DIR"

_ENGINE_SCRIPTS = {
    "sd3": "sd3_train_network.py",
    "flux": "flux_train_network.py",
}


class RunnerConfigError(Exception):
    """The kohya venv / sd-scripts checkout env vars are missing or invalid."""


@dataclass(frozen=True)
class TrainCommand:
    argv: list[str]
    env: dict
    cwd: Path


def _require_env(var_name: str, base_env: dict | None) -> str:
    env = base_env if base_env is not None else os.environ
    value = env.get(var_name)
    if not value:
        raise RunnerConfigError(f"{var_name} is not set")
    return value


def build_train_command(engine: str, train_toml: Path, *,
                         accelerate_args: list[str] | None = None,
                         base_env: dict | None = None) -> TrainCommand:
    """Build (never run) the `accelerate launch` command for a kohya train TOML.

    `engine` ("sd3" | "flux") selects sd3_train_network.py / flux_train_network.py.
    The returned env has CUBLAS_WORKSPACE_CONFIG / PYTHONHASHSEED injected via
    lorafactory.determinism.build_env; a caller only needs subprocess.run(
    command.argv, cwd=command.cwd, env=command.env).
    """
    try:
        script_name = _ENGINE_SCRIPTS[engine]
    except KeyError:
        raise RunnerConfigError(
            f"unknown kohya engine {engine!r}; expected one of {sorted(_ENGINE_SCRIPTS)}"
        ) from None

    kohya_python = _require_env(LORAFACTORY_KOHYA_PYTHON, base_env)
    sdscripts_dir = Path(_require_env(LORAFACTORY_SDSCRIPTS_DIR, base_env))
    script_path = sdscripts_dir / script_name

    argv = [
        kohya_python, "-m", "accelerate", "launch",
        *(accelerate_args or []),
        str(script_path),
        "--config_file", str(train_toml),
    ]

    return TrainCommand(argv=argv, env=build_env(base_env), cwd=sdscripts_dir)


def run_train(engine: str, train_toml: Path, log_path: Path, *,
              accelerate_args: list[str] | None = None,
              base_env: dict | None = None) -> int:
    """Run the kohya training subprocess, streaming combined stdout+stderr to `log_path`.

    Never called by the check suite; kept here so callers (e.g. `lorafactory
    train`) don't need to know the accelerate/env wiring.
    """
    command = build_train_command(
        engine, train_toml, accelerate_args=accelerate_args, base_env=base_env)
    log_path = Path(log_path)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with open(log_path, "w") as log_file:
        result = subprocess.run(
            command.argv, cwd=command.cwd, env=command.env,
            stdout=log_file, stderr=subprocess.STDOUT,
            check=False,
        )
    return result.returncode
