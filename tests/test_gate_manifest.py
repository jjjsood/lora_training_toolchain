"""Gate image manifest: seed-pairing invariant + minimum design size.

API pinned:
    from lorafactory.gate.manifest import validate_manifest, ManifestError
    validate_manifest(rows, min_prompts=3, min_seeds=10)  # None or raises

Row shape: {"prompt": str, "seed": int, "arm": "base"|"lora", "file": str}
Invariant (§5): a lora row at seed s requires a base row at seed s of the
same prompt. Null pairs are built from distinct-seed base rows only.
"""

import pytest

from lorafactory.gate.manifest import ManifestError, validate_manifest


def rows_for(prompts=3, seeds=10):
    rows = []
    for p in range(prompts):
        for s in range(seeds):
            rows.append({"prompt": f"p{p}", "seed": 1000 + s, "arm": "base",
                         "file": f"base_p{p}_{s}.png"})
            rows.append({"prompt": f"p{p}", "seed": 1000 + s, "arm": "lora",
                         "file": f"lora_p{p}_{s}.png"})
    return rows


def test_valid_manifest_passes():
    validate_manifest(rows_for())


def test_lora_seed_without_base_partner_rejected():
    rows = rows_for()
    rows.append({"prompt": "p0", "seed": 9999, "arm": "lora", "file": "x.png"})
    with pytest.raises(ManifestError):
        validate_manifest(rows)


def test_too_few_prompts_rejected():
    with pytest.raises(ManifestError):
        validate_manifest(rows_for(prompts=2))


def test_too_few_seeds_rejected():
    with pytest.raises(ManifestError):
        validate_manifest(rows_for(seeds=9))


def test_thresholds_are_parameters():
    validate_manifest(rows_for(prompts=2, seeds=5), min_prompts=2, min_seeds=5)
