"""Gate image-grid planning.

The (prompt x seed x arm) grid is planned on CPU and is pure data: planning
writes nothing to disk and loads no model. Only rendering needs a GPU, and
rendering is never reached from :func:`plan_images`.

API pinned:
    from lorafactory.gate.generate import plan_images
    plan = plan_images(cfg, checkpoint="L-F", out_dir=Path("out"))
    # one dict per cell: prompt, prompt_index, seed, arm, checkpoint, path
"""

from __future__ import annotations

from pathlib import Path

#: The two arms every gate cell is rendered in; gate/run.py reads the same
#: vocabulary back out of the results CSV. "null" is the no-LoRA render —
#: gate/manifest.py calls that same image "base" (see its module docstring).
ARMS = ("lora", "null")


def plan_images(cfg: dict, checkpoint: str, out_dir: str | Path) -> list[dict]:
    """Plan every (prompt, seed, arm) cell of the gate grid.

    Returns ``len(prompts) * len(seeds) * 2`` dicts with identical
    (prompt, seed) coverage per arm and a distinct ``path`` per cell. Nothing
    is written: the returned paths do not exist yet.
    """
    prompts = list(cfg["prompts"])
    seeds = list(cfg["seeds"])
    root = Path(out_dir)

    plan: list[dict] = []
    for arm in ARMS:
        for p_idx, prompt in enumerate(prompts):
            for seed in seeds:
                seed_int = int(seed)
                name = f"{checkpoint}_p{p_idx:02d}_s{seed_int:04d}_{arm}.png"
                plan.append(
                    {
                        "checkpoint": str(checkpoint),
                        "prompt": prompt,
                        "prompt_index": p_idx,
                        "seed": seed_int,
                        "arm": arm,
                        "path": root / arm / name,
                    }
                )
    return plan


def render_plan(plan: list[dict], *, pipeline) -> None:  # pragma: no cover - GPU path
    """Render a plan produced by :func:`plan_images`.

    GPU-only entry point; never called by planning. ``pipeline`` must be an
    already-constructed diffusion pipeline — this module loads no weights.
    """
    for cell in plan:
        path = Path(cell["path"])
        path.parent.mkdir(parents=True, exist_ok=True)
        image = pipeline(cell["prompt"], seed=cell["seed"]).images[0]
        image.save(path)
