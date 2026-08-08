"""Cliff's delta — pure numpy, exhaustive pairs.

API pinned:
    from lorafactory.gate.stats import cliffs_delta
    d = cliffs_delta(x, y)   # >0 when x tends larger than y; in [-1, 1]
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np

ArrayLike = Sequence[float] | np.ndarray


def cliffs_delta(x: ArrayLike, y: ArrayLike) -> float:
    """Exhaustive-pairs Cliff's delta between samples x and y.

    delta = (#{x_i > y_j} - #{x_i < y_j}) / (len(x) * len(y))

    Positive delta means x tends to be larger than y. Result is bounded to
    [-1, 1] and 0 for identical distributions.
    """
    xa = np.asarray(x, dtype=float).reshape(-1)
    ya = np.asarray(y, dtype=float).reshape(-1)

    # Exhaustive pairwise comparison via broadcasting.
    diff = xa[:, None] - ya[None, :]
    greater = np.sum(diff > 0)
    less = np.sum(diff < 0)
    total = xa.size * ya.size

    return float((greater - less) / total)
