"""Gate image-grid planning.

The (prompt x seed x arm) grid is planned on CPU and is pure data: planning
writes nothing to disk and loads no model. Only rendering needs a GPU, and
rendering is never reached from :func:`plan_images`.

API pinned:
    from lorafactory.gate.generate import plan_images
    plan = plan_images(cfg, checkpoint="L-F", out_dir=Path("out"))
    # one dict per cell: prompt, prompt_index, seed, arm, checkpoint, path

Render families: "sd3" (current/default) and "flux_schnell". Each family has
its own vocabulary of render kwargs; :func:`build_render_kwargs` validates a
gate config's `render` section against that vocabulary and raises
`GateRenderError` on a key or value from the wrong family rather than
forwarding it silently into a pipeline call. Building kwargs is pure CPU
data work, same as planning — it still loads no model.
"""

from __future__ import annotations

from pathlib import Path

#: The two arms every gate cell is rendered in; gate/run.py reads the same
#: vocabulary back out of the results CSV. "null" is the no-LoRA render —
#: gate/manifest.py calls that same image "base" (see its module docstring).
ARMS = ("lora", "null")

#: Render families the gate knows how to build kwargs for.
FAMILIES = ("sd3", "flux_schnell")

#: FLUX.1-schnell is guidance-distilled: it was trained to run with no
#: classifier-free guidance and a fixed short schedule. These are the only
#: values `build_render_kwargs` accepts for the flux_schnell family.
FLUX_SCHNELL_GUIDANCE_SCALE = 0.0
FLUX_SCHNELL_NUM_INFERENCE_STEPS = 4
FLUX_SCHNELL_SCHEDULER = "euler"

#: `negative_prompt` drives classifier-free guidance, which flux_schnell
#: does not use (guidance_scale is pinned to 0.0); diffusers would silently
#: ignore it, so the gate rejects it instead of forwarding it.
SD3_ONLY_KEYS = frozenset({"negative_prompt"})

#: `max_sequence_length` bounds the FLUX T5 prompt-embedding sequence; SD3's
#: pipeline has no such parameter.
FLUX_ONLY_KEYS = frozenset({"max_sequence_length"})


class GateRenderError(ValueError):
    """A render config key or value is incompatible with the render family."""


def build_render_kwargs(family: str, render_cfg: dict | None = None) -> dict:
    """Validate a gate config's `render` section and return pipeline kwargs.

    ``family`` selects which render vocabulary is valid ("sd3" or
    "flux_schnell"). A key or value that belongs to the other family raises
    :class:`GateRenderError` — this function never forwards it silently.
    """
    cfg = dict(render_cfg or {})

    if family == "flux_schnell":
        forbidden = SD3_ONLY_KEYS & cfg.keys()
        if forbidden:
            raise GateRenderError(
                f"flux_schnell render config has SD3-only keys: {sorted(forbidden)}"
            )

        guidance_scale = cfg.get("guidance_scale", FLUX_SCHNELL_GUIDANCE_SCALE)
        if float(guidance_scale) != FLUX_SCHNELL_GUIDANCE_SCALE:
            raise GateRenderError(
                f"flux_schnell guidance_scale must be {FLUX_SCHNELL_GUIDANCE_SCALE}, "
                f"got {guidance_scale!r}"
            )

        num_inference_steps = cfg.get(
            "num_inference_steps", FLUX_SCHNELL_NUM_INFERENCE_STEPS
        )
        if int(num_inference_steps) != FLUX_SCHNELL_NUM_INFERENCE_STEPS:
            raise GateRenderError(
                "flux_schnell num_inference_steps must be "
                f"{FLUX_SCHNELL_NUM_INFERENCE_STEPS}, got {num_inference_steps!r}"
            )

        scheduler = cfg.get("scheduler", FLUX_SCHNELL_SCHEDULER)
        if scheduler != FLUX_SCHNELL_SCHEDULER:
            raise GateRenderError(
                f"flux_schnell scheduler must be {FLUX_SCHNELL_SCHEDULER!r}, "
                f"got {scheduler!r}"
            )

        return {
            "guidance_scale": FLUX_SCHNELL_GUIDANCE_SCALE,
            "num_inference_steps": FLUX_SCHNELL_NUM_INFERENCE_STEPS,
            "scheduler": FLUX_SCHNELL_SCHEDULER,
        }

    if family == "sd3":
        forbidden = FLUX_ONLY_KEYS & cfg.keys()
        if forbidden:
            raise GateRenderError(
                f"sd3 render config has FLUX-only keys: {sorted(forbidden)}"
            )
        return cfg

    raise GateRenderError(f"unknown render family {family!r}; expected one of {FAMILIES}")


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


def render_plan(
    plan: list[dict],
    *,
    pipeline,
    family: str = "sd3",
    render_cfg: dict | None = None,
) -> None:  # pragma: no cover - GPU path
    """Render a plan produced by :func:`plan_images`.

    GPU-only entry point; never called by planning. ``pipeline`` must be an
    already-constructed diffusion pipeline — this module loads no weights.
    ``family`` and ``render_cfg`` are validated through
    :func:`build_render_kwargs` before any cell is rendered, so an
    incompatible render config fails before the first image is written.
    """
    render_kwargs = build_render_kwargs(family, render_cfg)
    for cell in plan:
        path = Path(cell["path"])
        path.parent.mkdir(parents=True, exist_ok=True)
        image = pipeline(cell["prompt"], seed=cell["seed"], **render_kwargs).images[0]
        image.save(path)
