# WS4 (T4): FLUX load path — converter or waiver

## Setup correction (before any WS4 work)

The worktree's branch (`worktree-agent-ae1601a4fcde0729b`) was checked out at `93b02f4`,
*not* on top of `flux-t1-t7` as the task briefing assumed — it was missing WS1/2/3/5/7
entirely (FLUX block targeting, verify-keys FLUX support, gate FLUX family, weight
introspection; `tests/test_flux_vocabulary.py`, `FLUX_DOUBLE_LEAVES`/`FLUX_SINGLE_LEAVES`
in `constants.py`, `parse_flux_diffusers_lora_key` in `keymap.py` all absent). Verified
the branch's own history was a strict ancestor of `flux-t1-t7` (`git log flux-t1-t7..HEAD`
was empty — no unique commits would be lost), then fast-forwarded:
`git merge --ff-only flux-t1-t7`. Post-merge: 327 tests passing, confirming the intended
base state.

## The experiment

Built a tiny `FluxTransformer2DModel` (full block count: `num_layers=19`,
`num_single_layers=38`, per `FLUX_NUM_DOUBLE_BLOCKS`/`FLUX_NUM_SINGLE_BLOCKS`; tiny width:
`attention_head_dim=8`, `num_attention_heads=4`), synthesized a kohya-layout FLUX LoRA
covering all 10 double-stream module leaves + `linear2`/`modulation_lin` on single-stream
(798 kohya tensors), and ran it through the real
`diffusers.loaders.lora_pipeline.FluxLoraLoaderMixin.lora_state_dict` →
`model.load_lora_adapter` → forward pass. This is `tests/test_flux_load_smoke.py::test_convert_attach_forward`.

**Result: green.** 342 PEFT `lora_A` modules attached, forward pass runs, output finite.

## A real finding along the way

While probing before committing to the tiny-model design, direct experimentation (not
source-reading) turned up that diffusers 0.39's `_convert_to_ai_toolkit_cat` — the helper
`_convert_kohya_flux_lora_to_diffusers` uses to split the single-stream fused `linear1`
module (kohya `lora_unet_single_blocks_{N}_linear1`, which fuses attn q/k/v + mlp-in) —
calls it with a **literal `dims=[3072, 3072, 3072, 12288]`** (real FLUX.1's
`inner_dim`/`mlp_hidden_dim`), not derived from the tensors it's given
(`lora_conversion_utils.py:436`: `assert sum(dims) == up_weight.shape[0]`). Feeding it a
tiny (non-3072) width raises `AssertionError` before conversion completes — reproduced
directly, traceback confirmed at that exact line.

Every other one of the 13 kohya FLUX module leaves is dimension-agnostic (either a direct
1:1 conversion or a `dims=None` dynamic split) and converts fine at any width — only
`linear1` has this hardcoded literal. Since real production usage always trains against
real FLUX.1 (a fixed architecture, not user-configurable — kohya's own `lora_up` for
`linear1` will always be exactly 20496 rows wide), this quirk is invisible outside a
synthetic tiny model; it does not block the waiver. But it does mean the tiny-model test
can't exercise `linear1` through a full model attach without reconstructing real FLUX.1's
actual width (~12B params — an OOM-risk-tier CPU allocation I did not do without asking,
and unnecessary here). Instead, `linear1` is verified separately, in
`test_single_block_linear1_conversion_at_real_flux_dims`, feeding the real conversion
function real FLUX.1 dims (3072/12288) — those are LoRA-sized tensors (rank × 3072), not
model-sized, so it stays CPU-cheap (confirmed: correct output keys/shapes for
`to_q`/`to_k`/`to_v`/`proj_mlp`).

## Decision: WAIVER

Both CPU tests pass, together exercising all 13 kohya FLUX module leaf types through
diffusers' real (not mocked) conversion code, at widths that are either arbitrary (12
leaves) or the only width real usage will ever produce (`linear1`). No converter written;
`lorafactory convert` remains SD3-only, unchanged.

## Deliverables

- `pyproject.toml`: registered `gpu` marker + `addopts = "-m 'not gpu'"`.
- `tests/test_flux_load_smoke.py` (new, 2 tests, CPU): the decision experiment.
- `tests/test_flux_kohya_load.py` (new, 1 test, `@pytest.mark.gpu`): real `FluxPipeline` +
  `load_lora_weights` + one denoise step against a real kohya-layout FLUX adapter path
  (via `LORAFACTORY_TEST_FLUX_ADAPTER` env var, skips if unset). **Written, not run**, per
  the hard user OOM/no-GPU constraint. Collection verified clean:
  `uv run pytest --collect-only -m gpu tests/test_flux_kohya_load.py` → 1 test collected,
  no import errors.
- `README.md`: waiver paragraph added directly after the existing SD3 `convert` sentence
  (~line 242, "The adapter comes out in kohya key layout...").

## Verification

- `uv run pytest -q`: 329 passed, 1 deselected (327 pre-existing + 2 new CPU tests; the
  GPU test collects but is excluded by default — confirms `addopts` doesn't change the
  non-gpu run's pass count beyond the intentional +2).
- `uv run ruff check .`: all checks passed.
- `uv run ruff format --check tests/test_flux_load_smoke.py tests/test_flux_kohya_load.py`:
  both already formatted (repo-wide `ruff format --check .` flags 49 pre-existing files
  unrelated to this change — not touched, out of WS4 scope).

## Self-review (task step 6)

- Waiver claim is backed by tests that run diffusers' actual `_convert_kohya_flux_lora_to_diffusers`
  code path — not a stub — across the full pinned kohya FLUX vocabulary, with a real
  `FluxTransformer2DModel` PEFT attach + forward pass for 12 of 13 leaves and a real-dims
  conversion check for the 13th. Not a trivial no-op.
- GPU test file collects cleanly under both the default (`not gpu`, deselected) and
  `-m gpu` (1 collected, 0 errors) invocations. Not executed, per policy.
