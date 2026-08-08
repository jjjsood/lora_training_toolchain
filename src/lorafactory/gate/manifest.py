"""Gate image manifest: seed-pairing invariant + minimum design size.

API pinned:
    from lorafactory.gate.manifest import validate_manifest, ManifestError
    validate_manifest(rows, min_prompts=3, min_seeds=10)  # None or raises

Row shape: {"prompt": str, "seed": int, "arm": "base"|"lora", "file": str}
Invariant (§5): a lora row at seed s requires a base row at seed s of the
same prompt. Null pairs are built from distinct-seed base rows only.

Two different things are called "arm" in this package, and they do not mix:
here it is the *image* arm, and the no-LoRA render is called "base"
(`gate/generate.plan_images` names the same render "null"). In `gate/run` and
`gate/report` "arm"/"kind" is instead the *pair* kind — a lora pair is
(base_s, lora_s), a null pair is (base_s, base_s') — so a plan_images plan is
not a manifest and must not be handed to `validate_manifest`.
"""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Iterable, Mapping


class ManifestError(ValueError):
    """Raised when a gate manifest violates the seed-pairing invariant or
    fails to meet the minimum design size."""


def validate_manifest(
    rows: Iterable[Mapping],
    min_prompts: int = 3,
    min_seeds: int = 10,
) -> None:
    """Validate a gate image manifest.

    Checks:
      * every "lora" row's (prompt, seed) has a matching "base" row.
      * at least ``min_prompts`` distinct prompts are present.
      * each prompt has at least ``min_seeds`` distinct (paired) seeds.

    Raises ManifestError on any violation; returns None otherwise.
    """
    by_prompt: dict[str, dict[str, set]] = defaultdict(
        lambda: {"base": set(), "lora": set()}
    )

    for row in rows:
        prompt = row["prompt"]
        seed = row["seed"]
        arm = row["arm"]
        if arm not in ("base", "lora"):
            raise ManifestError(f"unknown arm {arm!r} for prompt {prompt!r}")
        by_prompt[prompt][arm].add(seed)

    if len(by_prompt) < min_prompts:
        raise ManifestError(
            f"too few prompts: got {len(by_prompt)}, need >= {min_prompts}"
        )

    for prompt, arms in by_prompt.items():
        base_seeds = arms["base"]
        lora_seeds = arms["lora"]

        missing = lora_seeds - base_seeds
        if missing:
            raise ManifestError(
                f"prompt {prompt!r}: lora seeds without a base partner: "
                f"{sorted(missing)}"
            )

        paired_seeds = base_seeds & lora_seeds
        if len(paired_seeds) < min_seeds:
            raise ManifestError(
                f"prompt {prompt!r}: only {len(paired_seeds)} paired seeds, "
                f"need >= {min_seeds}"
            )
