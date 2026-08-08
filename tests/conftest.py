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
