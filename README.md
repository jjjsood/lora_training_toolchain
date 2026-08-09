# lorafactory

**Config-driven LoRA training toolchain for SD3-Medium (and FLUX): trains a matched-budget adapter matrix, verifies where each adapter actually landed, gates it on whether it changed anything, and synthesises norm-matched random controls.**

[![Python](https://img.shields.io/badge/python-3.11-3776AB?logo=python&logoColor=white)](https://www.python.org/)
[![Tests](https://img.shields.io/badge/tests-211%20passing-success)](#development)
[![Trainer](https://img.shields.io/badge/trainer-kohya--ss%2Fsd--scripts%20v0.11.1-orange)](https://github.com/kohya-ss/sd-scripts)
[![Ruff](https://img.shields.io/badge/lint-ruff-D7FF64?logo=ruff&logoColor=black)](https://docs.astral.sh/ruff/)
[![License: MIT](https://img.shields.io/badge/license-MIT-green)](LICENSE)
[![Status](https://img.shields.io/badge/status-CPU%20pipeline%20complete%20%C2%B7%20no%20GPU%20run%20yet-yellow)](#status)

Training runs are declared in YAML, not in code. Twelve adapters differ **only** in where the adapter sits — a difference the toolchain enforces before training and re-checks on the finished checkpoint.

It produces adapters. It does not measure them.

---

## Contents

- [Quickstart](#quickstart)
- [Pipeline](#pipeline)
- [CLI](#cli)
- [The adapter matrix](#the-adapter-matrix)
- [Targeting and verification](#targeting-and-verification)
- [Configs](#configs)
- [What a run writes](#what-a-run-writes)
- [The E_img gate](#the-e_img-gate)
- [Docker](#docker)
- [Requirements](#requirements)
- [Definition of done](#definition-of-done)
- [Repository layout](#repository-layout)
- [Development](#development)
- [Status](#status)
- [License](#license)

---

## Quickstart

The whole CPU half — config resolution, budget checking, kohya TOML emission, key verification, gate statistics, synthesis — runs with **no GPU, no weights and no network**.

```bash
git clone <repo-url> lora_training_toolchain
cd lora_training_toolchain

uv sync
uv run pytest -q                     # 279 tests, ~4 s, offline

uv run lorafactory resolve-config configs/matrix/L-E.yaml
uv run lorafactory check-budget --configs configs/matrix
uv run lorafactory train --config configs/matrix/L-E.yaml --dry-run
```

`--dry-run` is on every command that would otherwise need a GPU or the network (`train`, `train-matrix`, `gen-gate-images`, `fetch-models`, `fetch-dataset`, `test-run`). It does all local work and prints the resulting plan as JSON instead of executing it.

Real training needs the container, the pinned weights and a dataset — see [Docker](#docker).

---

## Pipeline

```
config.yaml ──resolve──► validated config ──check-budget──► matched matrix
     │
     └──emit-kohya──► kohya_config.toml + dataset_config.toml
                             │
                             └──train──► sd-scripts v0.11.1 (subprocess)
                                              │
                                              └──► adapter (kohya key layout)
                                                       │
                                    convert ──► diffusers/PEFT key layout
                                                       │
                        verify-keys ──► module inventory diff vs config
                                                       │
              gen-gate-images + gate ──► E_img verdict (Cliff's δ)
                                                       │
                                  provenance ──► provenance.json
```

Two side branches: `synth` builds norm-matched random controls from a trained adapter, and `screen` runs a downloaded community checkpoint through introspection plus the same gate.

Training itself is **not** reimplemented here. sd-scripts does it, pinned to `v0.11.1` and asserted against its exact commit at image build time. This repo owns everything around it: what gets trained, whether the matrix stays comparable, and whether the result is what the config asked for.

---

## CLI

Installed as `lorafactory`. Every command exits non-zero on a failed report.

| Command | Purpose |
|---|---|
| `resolve-config` | Resolve an overlay (`extends` + `overrides_allowed`), validate it, print merged config + hash. |
| `check-budget` | Diff all matrix configs; fail if two differ in a section neither declared overridable. |
| `check-dataset` | Validate a dataset directory against its `manifest.csv`. |
| `emit-kohya` | Write the `kohya_config.toml` + `dataset_config.toml` pair. |
| `train` | Run one sd-scripts training into a run directory. |
| `train-matrix` | Same, over every config in a directory, in sequence. |
| `convert` | kohya key layout → diffusers/PEFT key layout. |
| `verify-keys` | Diff a converted checkpoint's module inventory against the `target` it claims. |
| `gen-gate-images` | Render the gate's LoRA-arm / null image grid. |
| `gate` | Score the gate CSV: Cliff's δ on LPIPS + CLIP distance, pass/fail. |
| `synth` | Synthesise a norm-matched random-control adapter from a reference checkpoint. |
| `introspect` | Per-module spectra of one adapter: Frobenius norm, effective rank, top singular values, intruder dimensions. |
| `screen` | Screen a community checkpoint (introspection + gate). |
| `fetch-models` | Pull the pinned base and text-encoder weights. |
| `fetch-dataset` | Materialise a config's declared `dataset.source`. |
| `test-run` | One-shot smoke path: weights + dataset + 50 training steps. |
| `determinism-check` | Tensor-exact compare of two checkpoints. |
| `provenance` | Write `provenance.json` for a finished run. |

---

## The adapter matrix

Eleven SD3 adapters share data, steps, LR, batch size and seed. The **only** difference inside the placement block is where the adapter sits.

| ID | Dataset | Modules | Blocks | Rank | Role |
|---|---|---|---|---|---|
| `L-F` | STYLE | attn + mlp | 0–23 | 16 | reference adapter |
| `L-A` | STYLE | attn | 0–23 | 16 | module-type ground truth (attention) |
| `L-M` | STYLE | mlp | 0–23 | 16 | module-type ground truth (MLP) |
| `L-E` | STYLE | attn + mlp | **0–7** | 16 | **site-of-action ground truth, early** |
| `L-L` | STYLE | attn + mlp | **16–23** | 16 | **site-of-action ground truth, late** |
| `L-R4` | STYLE | attn + mlp | 0–23 | **4** | rank axis |
| `L-R64` | STYLE | attn + mlp | 0–23 | **64** | rank axis |
| `L-T` | STYLE | text encoders | — | 16 | **negative control**: no transformer weight touched |
| `O-F` | OBJ | attn + mlp | 0–23 | 16 | objective axis, paired with `L-F` |
| `O-A` | OBJ | attn | 0–23 | 16 | objective × module type |
| `O-M` | OBJ | mlp | 0–23 | 16 | objective × module type |
| `F-F` | STYLE | attn + mlp | see note | 16 | FLUX replication of `L-F` |

`L-E` and `L-L` are the ones that matter most: their site of action is known in advance, so they are the check on whether downstream analysis can recover an answer it was already told. `L-T` is the matching negative control — it adapts CLIP-L and CLIP-G only, no transformer block and no T5 (diffusers 0.39 has no `text_encoder_3` LoRA path, so a T5-adapted checkpoint would load silently incomplete).

`F-F` trains on FLUX.1-dev, overrides `model`/`train` (fp8 base, `blocks_to_swap: 16`) and is exempt from the SD3 matched budget. **FLUX support is partial**: the network-args path emits `networks.lora_flux`, but block-range bounds checking and `verify-keys` are SD3-only. See [Status](#status).

Three fallback configs (`configs/fallback/X-OVERFIT.yaml`, `X-R128.yaml`, `X-SHORT.yaml`) are documented retrain recipes for gate-rejected adapters. They carry `budget_exempt: true` and are excluded from the matched-budget check by design.

---

## Targeting and verification

**The hazard.** Substring matching on block names is the standard way to scope a LoRA, and `transformer_blocks.1` also matches `.11`, `.12` … `.19`. A mis-targeted adapter trains without error, writes a valid checkpoint and produces plausible numbers. Nothing downstream catches it.

The toolchain closes that two ways, neither of which trusts a pattern:

**1. Targeting is emitted as explicit kohya arguments, not patterns.** A `target` section becomes a `network_args` list for `networks.lora_sd3` / `networks.lora_flux`:

| Config | Emitted |
|---|---|
| `blocks: [0, 7]` | `train_block_indices=0-7` — an index range, no string matching |
| `module_classes: [attn]` | `context_mlp_dim=0`, `x_mlp_dim=0` — the excluded class is zeroed |
| `scope: text_encoders` | `network_train_text_encoder_only=true`, no transformer args |
| always | `context_mod_dim=0` / `x_mod_dim=0` (SD3), `img_/txt_/single_mod_dim=0` (FLUX) — adaLN is outside the spec vocabulary |

kohya does **not** bounds-check `train_block_indices`, so `network_args.py` enforces 0–23 itself and refuses an empty `module_classes` or a descending range.

**2. The finished checkpoint is diffed against the config it claims.** `verify-keys` builds the full expected module-name set from pinned constants — never introspected from a live model — and compares it to the converted state dict, reporting `unexpected`, `missing` and `rank_mismatches`. An upstream rename surfaces as a loud diff rather than a quietly narrower adapter.

Pinned module vocabulary (`src/lorafactory/constants.py`):

| Scope | Namespace | Attention leaves | MLP leaves |
|---|---|---|---|
| SD3-Medium | `transformer_blocks.{0..23}` | `attn.to_{q,k,v}`, `to_out.0`, `attn.add_{q,k,v}_proj`, `to_add_out` | `ff.net.0.proj`, `ff.net.2`, `ff_context.net.0.proj`, `ff_context.net.2` |
| CLIP-L / CLIP-G | `text_model.encoder.layers.{0..11}` / `{0..31}` | `self_attn.{q,k,v,out}_proj` | `mlp.fc1`, `mlp.fc2` |
| FLUX double | `transformer_blocks.{0..18}` | as SD3 | as SD3 |
| FLUX single | `single_transformer_blocks.{0..37}` | `attn.to_{q,k,v}` | `proj_mlp`, `proj_out` |

Two architecture facts the verifier encodes: SD3's **block 23 is context-pre-only** — no `ff_context`, no `attn.to_add_out`, so expecting them there is a false "missing". And FLUX's 19 double + 38 single blocks live in **two separate index namespaces**; a FLUX block range is not an SD3 block range.

---

## Configs

Two levels, nothing more:

```
configs/base.yaml            shared training budget — every matched key lives here
configs/matrix/<ID>.yaml     overlay; declares which sections it may change
configs/fallback/X-*.yaml    retrain recipes, budget_exempt
configs/gate/e_img.yaml      gate prompts, seeds, metrics, thresholds
configs/synth/R-*.yaml       norm-matched random control recipes
configs/survey/*.yaml        community checkpoint survey and screening
configs/test/smoke*.yaml     self-contained smoke run (declares its own dataset source)
```

An overlay names its parent and the sections it is allowed to touch:

```yaml
adapter_id: L-E
extends: ../base.yaml
overrides_allowed: [target]
target:
  scope: transformer
  blocks: [0, 7]
  module_classes: [attn, mlp]
  rank: 16
  alpha: 16
```

`L-E` may override `target`. It may not override `train`. `check-budget` diffs all resolved configs and fails on any section that differs between two members without both declaring it overridable — so "identical data, steps, LR, batch size and seed" is a check, not a claim. Every resolved config is hashed (SHA-256 over canonical JSON), and the hash gates run reuse.

Paths resolve against environment roots, so the same config works in the container and on a host checkout:

| Variable | Default | Meaning |
|---|---|---|
| `HF_TOKEN` | — | required for the gated SD3 / FLUX repos |
| `CIVITAI_TOKEN` | — | optional, community survey |
| `LORAFACTORY_DATASETS_DIR` | `/workspace/datasets` | dataset root |
| `LORAFACTORY_MODELS_DIR` | `/workspace/models` | weight root |
| `LORAFACTORY_RUNS_DIR` | `./runs` | run output root |
| `LORAFACTORY_KOHYA_PYTHON` | `/opt/venv-kohya/bin/python` | interpreter sd-scripts runs under |
| `LORAFACTORY_SDSCRIPTS_DIR` | `/opt/sd-scripts` | sd-scripts checkout |
| `TOOLCHAIN_MEM_LIMIT` | `40g` | container RAM cap (host OOM guard) |

---

## What a run writes

```
runs/<ID>-a01/
  config.hash              SHA-256 of the resolved config
  kohya_config.toml        emitted trainer config
  dataset_config.toml      emitted dataset config
  train.log                sd-scripts stdout/stderr
  adapter/<ID>.safetensors the trained adapter, in kohya key layout
  provenance.json          written by `lorafactory provenance`
```

The `-aNN` suffix is the attempt number. `prepare_run` decides what happens when the directory already exists:

| State | Action |
|---|---|
| no directory | **create** |
| hash matches, adapter present | **skip** — a requeued job never silently retrains |
| hash matches, no adapter | **resume** |
| hash differs | **conflict** — hard error |

A deliberate retrain gets a fresh attempt directory; the predecessor is left untouched.

`provenance.json` carries `run_id`, `adapter_id`, `config_hash`, `seed`, dataset name and manifest hash, base-model repo and revision, the sd-scripts ref, library versions, determinism env, wall clock, final loss and the adapter's sha256 — enough to trace a checkpoint back to its config a year later. Missing any required field is an error, not a warning.

The adapter comes out in **kohya key layout**; `convert` turns it into the diffusers/PEFT layout, which is what `verify-keys` and any consumer read.

---

## The E_img gate

An adapter that does not visibly change the image cannot be expected to change anything downstream, and including it dilutes every aggregate it appears in — invisibly, because nothing reports "this arm contributed nothing".

`gen-gate-images` renders a `prompt × seed × arm` grid (default 3 prompts × 10 seeds, arms `lora` and `null`, LoRA scale 1.0). `gate` reads the resulting CSV (`checkpoint,prompt,seed,arm,lpips,clip_distance`) and applies `configs/gate/e_img.yaml`:

- **pass** requires Cliff's δ > `0.8` on **both** LPIPS and CLIP distance (`metric_rule: both`),
- **and** null sanity: the base-vs-base δ must stay ≤ `0.2`.

Same gate screens downloaded community checkpoints via `screen`. A rejection is documented and retrained from `configs/fallback/`, not quietly dropped.

`synth` is the matching control on the other side: Gaussian adapters rescaled **per module** so `‖B·A‖_F` matches a reference adapter module-for-module, with a recipe recording target norm, achieved norm and rank per module. It answers *"would any perturbation of that size do this?"*.

---

## Docker

One image. Everything mutable is a mount.

| Mount | Purpose |
|---|---|
| `/workspace/configs` | configs (read-only) |
| `/workspace/datasets` | STYLE and OBJ (read-only) |
| `/workspace/models` | weight cache — shared, so a second job does not re-download 15 GB |
| `/workspace/runs` | adapters, logs, provenance (read-write) |

```bash
docker build -t lorafactory .
```

Or pull the prebuilt image instead of building locally — `.github/workflows/docker-publish.yml` builds and pushes it to GitHub Container Registry on every push to `main`/`master` and on `v*` tags:

```bash
docker pull ghcr.io/jjjsood/lora_training_toolchain:latest
```

(GHCR packages default to private on first publish — set the package visibility to public once under github.com/jjjsood → Packages settings, otherwise pulling from a GPU cluster needs `docker login ghcr.io` first.)

The image provisions two Python environments — `/opt/venv-tool` for this repo, `/opt/venv-kohya` for the training deps — plus an sd-scripts checkout pinned to `v0.11.1` and asserted against its exact commit, so a moved tag fails the build loudly instead of silently training against different code. Base image is CUDA 12.8, matching the cu128 torch wheels (Blackwell / sm_120).

The entrypoint execs the `lorafactory` CLI, so container args are CLI args:

```bash
docker run --gpus all \
  -v "$PWD/configs:/workspace/configs:ro" \
  -v "$PWD/datasets:/workspace/datasets:ro" \
  -v /mnt/models:/workspace/models \
  -v "$PWD/runs:/workspace/runs" \
  -e HF_TOKEN \
  lorafactory:latest \
  train --config /workspace/configs/matrix/L-E.yaml
```

Whole matrix:

```bash
docker compose run --rm lorafactory train-matrix --configs /workspace/configs/matrix
```

**Smoke run.** `configs/test/smoke.yaml` declares where its images come from, so the container fetches the pinned weights, rebuilds the dataset and trains 50 steps by itself:

```bash
docker compose run --rm lorafactory-test
```

That service differs in exactly two mounts: `datasets` is read-write (the only service allowed to create image data) and `models` is a host bind rather than the named volume. It proves the pipeline runs; it proves nothing about the matrix.

A job needs no state beyond its config path, so the twelve configs map onto twelve independent jobs with no ordering constraint — the image is the portable unit and no scheduler-specific code lives here. Warm the shared model cache once before submitting a batch, or every job pulls the base weights at the same time.

---

## Requirements

### Hardware

| | Minimum | Notes |
|---|---|---|
| GPU | 16 GB VRAM | SD3-Medium LoRA at 1024 px fits with bf16 + gradient checkpointing + 8-bit optimiser + cached text-encoder outputs. It does **not** fit with T5-XXL resident — caching text embeddings and unloading the encoders is a requirement, not an optimisation. |
| GPU (FLUX) | 16 GB VRAM, tight | `F-F` needs `fp8_base` and `blocks_to_swap: 16`. FLUX-side only; does not touch the SD3 budget. |
| RAM | 32 GB | 64 GB comfortable. Text-embedding precompute is the peak. The container is capped at `TOOLCHAIN_MEM_LIMIT` (40 GB default) as host OOM protection. |
| Disk | ~120 GB | base weights (SD3 ≈ 15 GB, FLUX.1-dev ≈ 24 GB), datasets, adapters, gate images. |
| Driver | CUDA 12.8 capable | matches the cu128 torch wheels in `docker/kohya-requirements.lock.txt`. |

### Software

Python 3.11 (`>=3.11,<3.12`) via [uv](https://docs.astral.sh/uv/); `torch`, `diffusers==0.39.0`, `transformers`, `peft`, `safetensors`, `huggingface_hub`, `lpips`, `pydantic`, `click`, `pyarrow`. Inside the image, additionally `accelerate`, `bitsandbytes`, `sentencepiece` and `protobuf` (the last two are T5 requirements, easy to forget until the tokenizer fails). Versions are pinned in `uv.lock` and `docker/kohya-requirements.lock.txt` — the exact `peft` version decides adapter key naming.

Docker with the NVIDIA Container Toolkit (`--gpus all` must work) for anything that trains.

### Model access

**SD3-Medium** and **FLUX.1-dev** are gated on Hugging Face: accept the licence, set `HF_TOKEN`. Revisions are pinned to commit SHAs and held by `tests/test_model_pins.py`, so they cannot drift back into plausible-looking but invented hashes. sd-scripts reads the SAI single-file layout, not the `-diffusers` repos — both are pinned, since conversion and eval need the diffusers one.

### Data

The STYLE and OBJ image sets **do not exist yet** and are the blocking item. The contract they must satisfy is in [`datasets/README.md`](datasets/README.md). Each image needs a `manifest.csv` row with `file, sha256, caption, source_url, author, licence, licence_url, acquisition_date` — every field non-empty, checked against disk before training. A file on disk with no manifest row is an error.

- **STYLE** — 40–60 images, one consistent visual style, **no recurring subject** (a recurring subject confounds the style axis).
- **OBJ** — 25–40 images of one rare object with a rare trigger token (`sks_lantern`), varied backgrounds (a fixed background trains the background).

### Determinism

`CUBLAS_WORKSPACE_CONFIG=:4096:8`, `PYTHONHASHSEED=0`, `max_data_loader_n_workers: 0` (workers reintroduce ordering nondeterminism), and one seed governing init, data order and noise. `determinism-check` compares two checkpoints tensor-exactly.

---

## Definition of done

1. `L-A` contains only attention keys; `L-M` only `ff` / `ff_context` keys.
2. `L-E` contains only blocks 0–7; `L-L` only 16–23.
3. `L-T` contains zero transformer keys.
4. All 11 SD3 adapters pass the `E_img` gate, or their rejection and retrain is documented and reported alongside every comparison that uses them.
5. `check-budget` passes across the placement block, report archived.
6. The three norm-matched controls reproduce their reference `‖B·A‖_F` per module within tolerance, recipe written alongside.
7. Two runs of the same config produce the same adapter.

1–3 are checked by `verify-keys` against pinned constants, so the substring defect cannot pass.

---

## Repository layout

```
src/lorafactory/
  cli.py           17 click commands
  constants.py     pinned module vocabulary, block counts, adapter IDs
  models.py        model paths, pinned revisions, download plans
  determinism.py   determinism env + tensor-exact checkpoint compare
  provenance.py    provenance record + training log
  runs.py          run directories: create / skip / resume / conflict
  config/          loader (extends + overrides_allowed), schema, hashing, budget diff
  kohya/           sd-scripts arg registry, network_args, TOML emitter, subprocess runner
  convert/         kohya -> diffusers/PEFT key mapping
  verify/          checkpoint key-inventory diff
  gate/            image plan, generation, LPIPS/CLIP metrics, Cliff's δ, report
  synth/           norm-matched random adapter synthesiser
  introspect/      per-module spectra: norms, effective rank, intruder dimensions
  survey/          community checkpoint introspection + screening
  data/            dataset manifest validation and fetch
configs/           base + matrix + fallback + gate + synth + survey + test
docker/            kohya lockfile, entrypoint
tests/             28 modules, CPU-only, offline
tools/             dump_kohya_args.py — one-shot sd-scripts argparse dump
```

## Development

```bash
uv sync
uv run pytest -q        # 279 tests, CPU-only, no network, no weights
uv run ruff check .
uv run ruff format --check .
```

The suite never builds the image and never downloads weights. Ruff runs `E, F, I, UP, B, PL` at line length 100 — the default `E4/E7/E9/F` set silently skips line length, import order and the pyupgrade/pylint findings.

## Status

| Area | State |
|---|---|
| CPU pipeline — configs, budget check, kohya emission, conversion, key verification, gate stats, synth, introspection, provenance | implemented, 279 tests passing |
| Intruder-dimension statistic | SD3 only — the base-key mapping is pinned for SD3's single-file layout; FLUX's fused `double_blocks.N.img_attn.qkv` needs its own slice convention. Norms, effective rank and top singular values work on any checkpoint. |
| FLUX path | partial — `networks.lora_flux` args emitted; block bounds and `verify-keys` are SD3-only |
| Docker image | defined, built by hand, not exercised by CI |
| STYLE / OBJ datasets | **do not exist — blocking item** |
| Training runs, adapters, gate results | none |

Everything downstream waits on images with resolved licensing.

## License

[MIT](LICENSE) — this code only. The models and artefacts carry their own terms:

- **SD3-Medium** and **FLUX.1-dev** are gated, non-commercial licences requiring acceptance on Hugging Face.
- Trained adapters inherit the base model's licence; an MIT trainer does not make a non-commercial checkpoint commercial.
- Training images keep their per-file licences, recorded in the dataset manifest.
