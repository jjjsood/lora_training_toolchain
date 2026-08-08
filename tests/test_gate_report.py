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

import csv

from lorafactory.gate.report import evaluate, write_report_csv


def make_pairs(prompt, lora_lpips, null_lpips, lora_clip=None, null_clip=None):
    lora_clip = lora_clip or lora_lpips
    null_clip = null_clip or null_lpips
    pairs = []
    for v, c in zip(lora_lpips, lora_clip, strict=True):
        pairs.append({"prompt": prompt, "kind": "lora",
                      "lpips": v, "clip_distance": c})
    for v, c in zip(null_lpips, null_clip, strict=True):
        pairs.append({"prompt": prompt, "kind": "null",
                      "lpips": v, "clip_distance": c})
    return pairs


def separated(prompt):
    """LoRA distances clearly above the null distribution."""
    lora = [0.55 + 0.01 * i for i in range(10)]
    null = [0.08 + 0.001 * (i % 5) for i in range(20)]
    return make_pairs(prompt, lora, null)


def overlapping(prompt):
    lora = [0.10 + 0.01 * (i % 4) for i in range(10)]
    null = [0.09 + 0.012 * (i % 5) for i in range(20)]
    return make_pairs(prompt, lora, null)


def test_pass_verdict():
    pairs = separated("p0") + separated("p1") + separated("p2")
    r = evaluate(pairs)
    assert r.verdict == "pass"
    assert r.null_sanity_ok
    for p in ("p0", "p1", "p2"):
        assert r.prompts[p]["passed"]
        assert r.prompts[p]["lpips_delta"] > 0.8


def test_fail_when_distributions_overlap():
    pairs = overlapping("p0") + overlapping("p1") + overlapping("p2")
    r = evaluate(pairs)
    assert r.verdict == "fail"


def test_fail_when_too_few_prompts_pass():
    pairs = separated("p0") + separated("p1") + overlapping("p2")
    assert evaluate(pairs).verdict == "fail"


def test_fail_when_too_few_seeds():
    lora = [0.6] * 9  # only 9 lora pairs
    null = [0.1] * 20
    pairs = make_pairs("p0", lora, null) + separated("p1") + separated("p2")
    r = evaluate(pairs)
    assert not r.prompts["p0"]["passed"]
    assert r.verdict == "fail"


def test_null_sanity_blocks_verdict():
    """Trending null distances (first half << second half) => gate unusable."""
    lora = [5.0 + 0.1 * i for i in range(10)]
    null = [0.01 * i for i in range(20)]  # strictly increasing
    pairs = (make_pairs("p0", lora, null)
             + make_pairs("p1", lora, null)
             + make_pairs("p2", lora, null))
    r = evaluate(pairs)
    assert not r.null_sanity_ok
    assert r.verdict == "fail"


def test_csv_report_shape(tmp_path):
    rows = [
        {"prompt": "p0", "seed": "1000", "arm": "lora",
         "lpips": 0.5, "clip_distance": 0.3},
        {"prompt": "p0", "seed": "1000-1001", "arm": "null",
         "lpips": 0.1, "clip_distance": 0.05},
    ]
    out = tmp_path / "report.csv"
    write_report_csv(rows, out, checkpoint_id="L-E")
    with open(out) as f:
        reader = list(csv.reader(f))
    assert reader[0] == ["checkpoint", "prompt", "seed", "arm",
                         "lpips", "clip_distance"]
    assert len(reader) == 3
    assert reader[1][0] == "L-E"
