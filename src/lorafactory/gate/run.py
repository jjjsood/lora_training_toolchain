"""Gate driver — config in, CSV in, verdict out.

API pinned:
    from lorafactory.gate.run import load_gate_config, evaluate_csv
    cfg = load_gate_config("configs/gate/e_img.yaml")
    result = evaluate_csv("results.csv", cfg)
    result.verdict          # "pass" | "fail"
    result.null_sanity_ok   # bool

CSV columns: checkpoint,prompt,seed,arm,lpips,clip_distance with arm in
{"lora", "null"}. The statistics live in gate.report/gate.stats; this module
only reads, coerces and forwards.
"""

from __future__ import annotations

import csv
from pathlib import Path

import yaml

from lorafactory.gate.generate import ARMS
from lorafactory.gate.report import GateResult, evaluate


def load_gate_config(path: str | Path) -> dict:
    """Read the gate config YAML (configs/gate/e_img.yaml) into a dict."""
    with open(Path(path)) as f:
        cfg = yaml.safe_load(f)
    if not isinstance(cfg, dict):
        raise ValueError(f"gate config {path} did not parse to a mapping")
    return cfg


def _rows_to_pairs(csv_path: Path) -> list[dict]:
    pairs: list[dict] = []
    with open(csv_path, newline="") as f:
        for row in csv.DictReader(f):
            arm = (row["arm"] or "").strip()
            if arm not in ARMS:
                raise ValueError(f"unknown arm {arm!r} in {csv_path}")
            pairs.append(
                {
                    "prompt": row["prompt"],
                    "kind": arm,  # report.evaluate calls the arm key "kind"
                    "seed": row["seed"],
                    "lpips": float(row["lpips"]),
                    "clip_distance": float(row["clip_distance"]),
                }
            )
    return pairs


def evaluate_csv(csv_path: str | Path, cfg: dict) -> GateResult:
    """Evaluate a gate results CSV against the thresholds in ``cfg``."""
    pairs = _rows_to_pairs(Path(csv_path))
    return evaluate(
        pairs,
        delta_threshold=float(cfg.get("delta_threshold", 0.8)),
        min_prompts=int(cfg.get("n_prompts_min", 3)),
        min_seeds=int(cfg.get("n_seeds_min", 10)),
        null_max_abs_delta=float(cfg.get("null_sanity_delta_max", 0.2)),
    )
