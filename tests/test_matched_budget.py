"""Matched-budget check across the real config matrix (README §3).

API pinned:
    from lorafactory.config.budget import check_matrix
    report = check_matrix(configs_matrix_dir)
    report.ok            # bool
    report.violations    # list[str], each naming the offending adapter id(s)
    report.write(path)   # writes JSON report, returns Path

Semantics: within the SD3 STYLE block {L-F,L-A,L-M,L-E,L-L,L-R4,L-R64,L-T}
every section except `target` must be identical. O-* additionally may differ
in `dataset` (and only there) vs their L-partner. F-F and budget_exempt
configs are reported but exempt. A diff outside the allowance is a violation
even if the overlay declared it in overrides_allowed.
"""

import json
import shutil

from conftest import MATRIX
from lorafactory.config.budget import check_matrix


def test_real_matrix_passes(tmp_path):
    report = check_matrix(MATRIX)
    assert report.ok, f"violations: {report.violations}"
    out = report.write(tmp_path / "budget.json")
    data = json.loads(out.read_text())
    assert isinstance(data, dict) and data


def test_tampered_seed_is_flagged(tmp_path):
    work = tmp_path / "matrix"
    shutil.copytree(MATRIX, work)
    base_src = (MATRIX.parent / "base.yaml").read_text()
    (tmp_path / "base.yaml").write_text(base_src)

    le = work / "L-E.yaml"
    text = le.read_text()
    text = text.replace("overrides_allowed: [target]",
                        "overrides_allowed: [target, train]")
    text += "\ntrain:\n  seed: 4242\n"
    le.write_text(text)

    report = check_matrix(work)
    assert not report.ok
    assert any("L-E" in v for v in report.violations)
