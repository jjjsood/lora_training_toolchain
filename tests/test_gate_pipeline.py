"""gate/metrics.py, gate/run.py, gate/generate.py — the three gate modules.

Hard constraint on all three: **importing them must not load a model or touch
the network.** LPIPS pulls AlexNet weights and CLIP pulls a transformer; if
that happened at import time the test suite would try to download hundreds of
megabytes on a machine that may have no network at all. Weight loading belongs
behind an explicit call, and only the GPU path makes that call.
"""

import json
import subprocess
import sys

import pytest

from conftest import CONFIGS
from lorafactory.gate import generate, metrics, run

GATE_CONFIG = CONFIGS / "gate" / "e_img.yaml"


# --------------------------------------------------------------------------
# metrics
# --------------------------------------------------------------------------

def test_metric_names_match_the_gate_contract():
    # gate/report.py scores exactly these two per prompt; a third metric name
    # here would silently never be evaluated.
    assert tuple(metrics.METRIC_NAMES) == ("lpips", "clip_distance")


def test_embedding_distance_is_zero_for_identical_vectors():
    import torch
    v = torch.tensor([0.3, -0.4, 0.5, 1.0])
    assert metrics.embedding_distance(v, v) == pytest.approx(0.0, abs=1e-6)


def test_embedding_distance_is_symmetric_and_non_negative():
    import torch
    g = torch.Generator().manual_seed(0)
    a = torch.randn(64, generator=g)
    b = torch.randn(64, generator=g)
    d_ab = metrics.embedding_distance(a, b)
    d_ba = metrics.embedding_distance(b, a)
    assert d_ab == pytest.approx(d_ba, abs=1e-6)
    assert d_ab >= 0.0


def test_embedding_distance_grows_with_dissimilarity():
    import torch
    ref = torch.tensor([1.0, 0.0, 0.0, 0.0])
    near = torch.tensor([1.0, 0.1, 0.0, 0.0])
    far = torch.tensor([-1.0, 0.0, 0.0, 0.0])
    assert metrics.embedding_distance(ref, near) < metrics.embedding_distance(ref, far)


def test_importing_metrics_loads_no_weights():
    """A fresh interpreter importing the module must stay offline and fast."""
    code = (
        "import os; os.environ['HF_HUB_OFFLINE']='1'; "
        "os.environ['TRANSFORMERS_OFFLINE']='1'; "
        "import lorafactory.gate.metrics as m; "
        "print(tuple(m.METRIC_NAMES))"
    )
    proc = subprocess.run([sys.executable, "-c", code], capture_output=True,
                          text=True, timeout=120, check=False)
    assert proc.returncode == 0, proc.stderr
    assert "lpips" in proc.stdout


# --------------------------------------------------------------------------
# run — CSV in, verdict out (this is what the `gate` CLI command calls)
# --------------------------------------------------------------------------

def test_load_gate_config_reads_the_real_file():
    cfg = run.load_gate_config(GATE_CONFIG)
    assert cfg["delta_threshold"] == 0.8
    assert cfg["metric_rule"] == "both"
    assert len(cfg["prompts"]) >= cfg["n_prompts_min"]
    assert len(cfg["seeds"]) >= cfg["n_seeds_min"]


def test_evaluate_csv_pairs_arms_by_prompt_and_seed(tmp_path):
    import csv as csv_mod
    path = tmp_path / "results.csv"
    null_lpips = [0.08, 0.12, 0.09, 0.11, 0.10, 0.10, 0.11, 0.09, 0.12, 0.08]
    with open(path, "w", newline="") as f:
        w = csv_mod.DictWriter(f, fieldnames=[
            "checkpoint", "prompt", "seed", "arm", "lpips", "clip_distance"])
        w.writeheader()
        for prompt in ("p1", "p2", "p3"):
            for i in range(10):
                w.writerow({"checkpoint": "L-F", "prompt": prompt, "seed": i,
                            "arm": "lora", "lpips": 0.40 + i * 0.01,
                            "clip_distance": 0.30 + i * 0.01})
                w.writerow({"checkpoint": "L-F", "prompt": prompt, "seed": i,
                            "arm": "null", "lpips": null_lpips[i],
                            "clip_distance": 0.05})
    result = run.evaluate_csv(path, run.load_gate_config(GATE_CONFIG))
    assert result.verdict == "pass"
    assert result.null_sanity_ok is True


def test_evaluate_csv_rejects_an_unknown_arm(tmp_path):
    import csv as csv_mod
    path = tmp_path / "bad.csv"
    with open(path, "w", newline="") as f:
        w = csv_mod.DictWriter(f, fieldnames=[
            "checkpoint", "prompt", "seed", "arm", "lpips", "clip_distance"])
        w.writeheader()
        w.writerow({"checkpoint": "X", "prompt": "p", "seed": 0,
                    "arm": "typo", "lpips": 0.1, "clip_distance": 0.1})
    with pytest.raises(ValueError):
        run.evaluate_csv(path, run.load_gate_config(GATE_CONFIG))


# --------------------------------------------------------------------------
# generate — the image grid is planned on CPU, only rendering needs a GPU
# --------------------------------------------------------------------------

def test_plan_images_covers_every_prompt_seed_arm_cell(tmp_path):
    cfg = run.load_gate_config(GATE_CONFIG)
    plan = generate.plan_images(cfg, checkpoint="L-F", out_dir=tmp_path)

    assert len(plan) == len(cfg["prompts"]) * len(cfg["seeds"]) * 2
    assert {img["arm"] for img in plan} == {"lora", "null"}
    # Seed pairing is the gate's core invariant: each (prompt, seed) must exist
    # in both arms, or gate/manifest.py's validation fails downstream.
    lora = {(i["prompt"], i["seed"]) for i in plan if i["arm"] == "lora"}
    null = {(i["prompt"], i["seed"]) for i in plan if i["arm"] == "null"}
    assert lora == null


def test_plan_images_gives_every_cell_a_distinct_path(tmp_path):
    cfg = run.load_gate_config(GATE_CONFIG)
    plan = generate.plan_images(cfg, checkpoint="L-F", out_dir=tmp_path)
    paths = [str(img["path"]) for img in plan]
    assert len(set(paths)) == len(paths), "two cells would overwrite each other"


def test_plan_images_writes_nothing(tmp_path):
    cfg = run.load_gate_config(GATE_CONFIG)
    generate.plan_images(cfg, checkpoint="L-F", out_dir=tmp_path)
    assert list(tmp_path.iterdir()) == [], "planning must not touch the filesystem"


def test_plan_is_json_serialisable(tmp_path):
    """`gen-gate-images --dry-run` prints this plan, so it has to survive
    json.dumps without custom encoders."""
    cfg = run.load_gate_config(GATE_CONFIG)
    plan = generate.plan_images(cfg, checkpoint="L-F", out_dir=tmp_path)
    json.dumps([{**img, "path": str(img["path"])} for img in plan])
