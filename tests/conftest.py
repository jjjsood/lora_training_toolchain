"""Shared fixtures and factories for the test suite.

These tests pin the public API of the `lorafactory` package.
"""

from pathlib import Path

import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
CONFIGS = ROOT / "configs"
MATRIX = CONFIGS / "matrix"

SD3_MATRIX_IDS = [
    "L-F", "L-A", "L-M", "L-E", "L-L", "L-R4", "L-R64", "L-T",
    "O-F", "O-A", "O-M",
]
ALL_MATRIX_IDS = SD3_MATRIX_IDS + ["F-F"]
FALLBACK_IDS = ["X-R128", "X-SHORT", "X-OVERFIT"]


@pytest.fixture(autouse=True)
def _isolated_runs_dir(monkeypatch, tmp_path) -> None:
    """Redirect every test's runs root into tmp_path.

    `_runs_root` in `lorafactory.cli` honors `LORAFACTORY_RUNS_DIR` before
    falling back to cwd/runs, so any test that runs a CLI command without an
    explicit `--runs-dir` still lands its output outside the repo instead of
    poisoning the tracked `runs/` directory with pytest tmp paths.
    """
    monkeypatch.setenv("LORAFACTORY_RUNS_DIR", str(tmp_path / "runs"))


@pytest.fixture(autouse=True)
def _isolated_introspect_cache_dir(monkeypatch, tmp_path) -> None:
    """Redirect every test's default introspect base-subspace cache root.

    Mirrors `_isolated_runs_dir`: `introspect --cache-dir` defaults to the
    repo-level `.cache/introspect` when unset, so any CLI test that doesn't
    pass `--cache-dir` explicitly would otherwise write real cache files into
    the tracked repo instead of into an ephemeral test directory.
    """
    monkeypatch.setenv("LORAFACTORY_INTROSPECT_CACHE_DIR", str(tmp_path / "introspect-cache"))


@pytest.fixture(scope="session", autouse=True)
def _guard_runs_dir_stays_clean():
    """Fail the suite if any test leaks output into the repo's `runs/` dir.

    `runs/` is gitignored and holds only the checked-in T-SMOKE fixtures; a
    test that forgets to redirect its runs root would otherwise silently
    reintroduce pytest tmp-path artefacts there.
    """
    runs_dir = ROOT / "runs"
    before = set(runs_dir.iterdir()) if runs_dir.exists() else set()
    yield
    after = set(runs_dir.iterdir()) if runs_dir.exists() else set()
    leaked = after - before
    assert not leaked, (
        f"test run leaked into repo runs/ dir: {sorted(p.name for p in leaked)}"
    )


@pytest.fixture(scope="session", autouse=True)
def _guard_introspect_cache_stays_clean():
    """Fail the suite if any test leaks a subspace cache into the repo's
    `.cache/introspect/` dir — the default `introspect --cache-dir` resolves
    to, which `_isolated_introspect_cache_dir` exists specifically to avoid."""
    cache_dir = ROOT / ".cache" / "introspect"
    before = set(cache_dir.iterdir()) if cache_dir.exists() else set()
    yield
    after = set(cache_dir.iterdir()) if cache_dir.exists() else set()
    leaked = after - before
    assert not leaked, (
        f"test run leaked into repo .cache/introspect dir: {sorted(p.name for p in leaked)}"
    )


@pytest.fixture(scope="session")
def repo_root() -> Path:
    return ROOT


@pytest.fixture(scope="session")
def configs_dir() -> Path:
    return CONFIGS


@pytest.fixture(scope="session")
def matrix_dir() -> Path:
    return MATRIX


def kohya_module(prefix: str, rank: int, out_dim: int, in_dim: int, alpha: float,
                 seed: int = 0) -> dict:
    """One kohya-layout LoRA module: lora_down/lora_up/alpha."""
    g = torch.Generator().manual_seed(seed)
    return {
        f"{prefix}.lora_down.weight": torch.randn(rank, in_dim, generator=g),
        f"{prefix}.lora_up.weight": torch.randn(out_dim, rank, generator=g),
        f"{prefix}.alpha": torch.tensor(float(alpha)),
    }


def peft_module(name: str, rank: int, out_dim: int, in_dim: int,
                seed: int = 0) -> dict:
    """One diffusers/PEFT-layout LoRA module: lora_A/lora_B."""
    g = torch.Generator().manual_seed(seed)
    return {
        f"{name}.lora_A.weight": torch.randn(rank, in_dim, generator=g),
        f"{name}.lora_B.weight": torch.randn(out_dim, rank, generator=g),
    }


def delta_w_kohya(sd: dict, prefix: str) -> torch.Tensor:
    """Effective ΔW of a kohya module: (alpha/rank) * up @ down."""
    down = sd[f"{prefix}.lora_down.weight"].float()
    up = sd[f"{prefix}.lora_up.weight"].float()
    alpha = float(sd[f"{prefix}.alpha"])
    rank = down.shape[0]
    return (alpha / rank) * (up @ down)


def delta_w_peft(sd: dict, name: str) -> torch.Tensor:
    """Effective ΔW of a converted PEFT module (alpha folded => scale 1)."""
    a = sd[f"{name}.lora_A.weight"].float()
    b = sd[f"{name}.lora_B.weight"].float()
    return b @ a
