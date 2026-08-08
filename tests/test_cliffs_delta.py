"""Cliff's delta — pure numpy, exhaustive pairs.

API pinned:
    from lorafactory.gate.stats import cliffs_delta
    d = cliffs_delta(x, y)   # >0 when x tends larger than y; in [-1, 1]
"""

import numpy as np
import pytest

from lorafactory.gate.stats import cliffs_delta


def test_complete_separation():
    assert cliffs_delta([1.0, 2.0, 3.0], [0.0, 0.1, 0.2]) == 1.0
    assert cliffs_delta([0.0, 0.1], [5.0, 6.0]) == -1.0


def test_identical_distributions():
    assert cliffs_delta([1.0, 1.0], [1.0, 1.0]) == 0.0


def test_hand_computed_mixed_case():
    # x=[1,2], y=[1.5]: pairs (1,1.5)->-1, (2,1.5)->+1 => delta 0
    assert cliffs_delta([1.0, 2.0], [1.5]) == 0.0
    # x=[1,2,3], y=[2]: (1,2)->-1 (2,2)->0 (3,2)->+1 => 0
    assert cliffs_delta([1.0, 2.0, 3.0], [2.0]) == 0.0
    # x=[2,3], y=[1,2]: pairs: (2,1)+ (2,2)0 (3,1)+ (3,2)+ => 3/4
    assert cliffs_delta([2.0, 3.0], [1.0, 2.0]) == pytest.approx(0.75)


def test_antisymmetry():
    rng = np.random.default_rng(0)
    x = rng.normal(0.5, 1.0, 25)
    y = rng.normal(0.0, 1.0, 30)
    assert cliffs_delta(x, y) == pytest.approx(-cliffs_delta(y, x))


def test_bounds():
    rng = np.random.default_rng(1)
    for _ in range(5):
        x = rng.normal(size=13)
        y = rng.normal(size=17)
        assert -1.0 <= cliffs_delta(x, y) <= 1.0


def test_accepts_lists_and_arrays():
    assert cliffs_delta(np.array([2.0]), [1.0]) == 1.0
