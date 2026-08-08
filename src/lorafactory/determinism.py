"""Determinism plumbing (kohya itself has none — we own it).

    from lorafactory.determinism import build_env, adapters_identical
    env = build_env()                    # or build_env(dict(os.environ))
    adapters_identical(path_a, path_b)   # tensor-exact safetensors compare
"""

from __future__ import annotations

import os
from pathlib import Path

from safetensors import safe_open

REQUIRED_ENV = {
    "CUBLAS_WORKSPACE_CONFIG": ":4096:8",
    "PYTHONHASHSEED": "0",
}


def build_env(base: dict | None = None) -> dict:
    env = dict(base) if base is not None else dict(os.environ)
    env.update(REQUIRED_ENV)
    return env


def adapters_identical(path_a: Path, path_b: Path) -> bool:
    with safe_open(str(path_a), framework="pt") as fa, \
            safe_open(str(path_b), framework="pt") as fb:
        keys_a = set(fa.keys())
        keys_b = set(fb.keys())
        if keys_a != keys_b:
            return False
        for key in keys_a:
            ta = fa.get_tensor(key)
            tb = fb.get_tensor(key)
            if ta.shape != tb.shape or ta.dtype != tb.dtype:
                return False
            if not (ta == tb).all():
                return False
    return True
