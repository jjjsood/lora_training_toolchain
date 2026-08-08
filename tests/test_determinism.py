"""Determinism plumbing (kohya itself has none — we own it).

API pinned:
    from lorafactory.determinism import build_env, adapters_identical
    env = build_env()                    # or build_env(dict(os.environ))
    adapters_identical(path_a, path_b)   # tensor-exact safetensors compare
"""

import torch
from safetensors.torch import save_file

from lorafactory.determinism import adapters_identical, build_env


def test_env_contains_required_flags():
    env = build_env()
    assert env["CUBLAS_WORKSPACE_CONFIG"] == ":4096:8"
    assert env["PYTHONHASHSEED"] == "0"


def test_env_preserves_base():
    env = build_env({"HF_TOKEN": "x", "PATH": "/usr/bin"})
    assert env["HF_TOKEN"] == "x"
    assert env["CUBLAS_WORKSPACE_CONFIG"] == ":4096:8"


def test_adapters_identical(tmp_path):
    a = {"m.lora_A.weight": torch.randn(4, 8), "m.lora_B.weight": torch.randn(8, 4)}
    p1, p2, p3 = (tmp_path / n for n in ("a.safetensors", "b.safetensors",
                                         "c.safetensors"))
    save_file(a, p1)
    save_file(dict(a), p2)
    b = {k: v.clone() for k, v in a.items()}
    b["m.lora_A.weight"][0, 0] += 1e-3
    save_file(b, p3)
    assert adapters_identical(p1, p2)
    assert not adapters_identical(p1, p3)
