"""Gate verdict logic.

API pinned:
    from lorafactory.gate.report import evaluate, write_report_csv
    result = evaluate(pairs, delta_threshold=0.8, min_prompts=3,
                      min_seeds=10, null_max_abs_delta=0.2)
    result.verdict          # "pass" | "fail"
    result.null_sanity_ok   # bool (all prompts)
    result.prompts          # dict[prompt] -> {"lpips_delta": float,
                            #   "clip_delta": float, "passed": bool}

Pair shape: {"prompt": str, "kind": "lora"|"null",
             "lpips": float, "clip_distance": float}
Per prompt: passed iff both metric deltas (lora vs null) > threshold and
lora pair count >= min_seeds. Verdict pass iff >= min_prompts prompts pass
AND null sanity holds.
Null sanity: per prompt, split the null pairs into first and
second half in given order; |cliffs_delta(half1, half2)| < null_max_abs_delta
for both metrics.

    write_report_csv(rows, path, checkpoint_id)
    # header: checkpoint,prompt,seed,arm,lpips,clip_distance
"""

from __future__ import annotations

import csv
from collections import defaultdict
from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field
from pathlib import Path

from lorafactory.gate.stats import cliffs_delta


@dataclass
class GateResult:
    verdict: str
    null_sanity_ok: bool
    prompts: dict = field(default_factory=dict)


def _split_half(values: list[float]) -> tuple[list[float], list[float]]:
    mid = len(values) // 2
    return values[:mid], values[mid:]


def evaluate(
    pairs: Iterable[Mapping],
    delta_threshold: float = 0.8,
    min_prompts: int = 3,
    min_seeds: int = 10,
    null_max_abs_delta: float = 0.2,
) -> GateResult:
    by_prompt: dict[str, dict[str, dict[str, list]]] = defaultdict(
        lambda: {
            "lora": {"lpips": [], "clip_distance": []},
            "null": {"lpips": [], "clip_distance": []},
        }
    )

    for pair in pairs:
        prompt = pair["prompt"]
        kind = pair["kind"]
        if kind not in ("lora", "null"):
            raise ValueError(f"unknown pair kind {kind!r}")
        by_prompt[prompt][kind]["lpips"].append(pair["lpips"])
        by_prompt[prompt][kind]["clip_distance"].append(pair["clip_distance"])

    prompts_out: dict[str, dict] = {}
    null_sanity_ok = True
    passed_count = 0

    for prompt, arms in by_prompt.items():
        lora = arms["lora"]
        null = arms["null"]

        lpips_delta = cliffs_delta(lora["lpips"], null["lpips"])
        clip_delta = cliffs_delta(lora["clip_distance"], null["clip_distance"])

        lora_pair_count = len(lora["lpips"])
        passed = (
            lpips_delta > delta_threshold
            and clip_delta > delta_threshold
            and lora_pair_count >= min_seeds
        )
        if passed:
            passed_count += 1

        prompts_out[prompt] = {
            "lpips_delta": lpips_delta,
            "clip_delta": clip_delta,
            "passed": passed,
        }

        lpips_h1, lpips_h2 = _split_half(null["lpips"])
        clip_h1, clip_h2 = _split_half(null["clip_distance"])
        null_lpips_delta = cliffs_delta(lpips_h1, lpips_h2)
        null_clip_delta = cliffs_delta(clip_h1, clip_h2)
        if (
            abs(null_lpips_delta) >= null_max_abs_delta
            or abs(null_clip_delta) >= null_max_abs_delta
        ):
            null_sanity_ok = False

    verdict = "pass" if (passed_count >= min_prompts and null_sanity_ok) else "fail"

    return GateResult(
        verdict=verdict,
        null_sanity_ok=null_sanity_ok,
        prompts=prompts_out,
    )


def write_report_csv(rows: Iterable[Mapping], path, checkpoint_id: str) -> None:
    """Write the gate report CSV.

    Header: checkpoint,prompt,seed,arm,lpips,clip_distance
    """
    path = Path(path)
    with open(path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(
            ["checkpoint", "prompt", "seed", "arm", "lpips", "clip_distance"]
        )
        for row in rows:
            writer.writerow(
                [
                    checkpoint_id,
                    row["prompt"],
                    row["seed"],
                    row["arm"],
                    row["lpips"],
                    row["clip_distance"],
                ]
            )
