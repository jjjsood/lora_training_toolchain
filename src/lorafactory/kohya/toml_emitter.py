"""Resolved adapter config -> kohya TOML pair (train config + dataset config).

    from lorafactory.kohya.toml_emitter import emit
    result = emit(config_dict, out_dir)
    result.train_toml     # Path to kohya train config TOML
    result.dataset_toml   # Path to kohya dataset config TOML

kohya's `--config_file` flattens TOML sections (cosmetic nesting only) and
**silently ignores unknown keys** — so `emit()` writes a flat train TOML and
validates every emitted key against `arg_registry` before writing, turning a
typo'd budget/network key into OUR error instead of a silently-dropped one.
The dataset config is the separate `[[datasets]]` / `[[datasets.subsets]]`
TOML kohya's dataset loader expects: one subset per adapter, `num_repeats=1`,
`caption_extension=".txt"`, no augmentation keys anywhere (this toolchain
never augments — augmentation would break the matched-budget/gate design).

Every path kohya has to open — the checkpoint, each text encoder, the FLUX
autoencoder, the dataset `image_dir` — is written absolute, resolved through
`lorafactory.models`. kohya runs with its own cwd and takes a *file* for
`--pretrained_model_name_or_path`; a repo id or a relative path here is
syntactically valid TOML that fails only once a GPU is already burning.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import tomli_w

from ..models import resolve_dataset_paths, resolve_model_paths
from .arg_registry import validate_keys
from .network_args import NetworkSpec, build_network_args

TRAIN_TOML_NAME = "kohya_config.toml"
DATASET_TOML_NAME = "dataset_config.toml"

_TEXT_ENCODER_DEST_KEYS = ("clip_l", "clip_g", "t5xxl")


@dataclass(frozen=True)
class EmitResult:
    train_toml: Path
    dataset_toml: Path


def engine_for(config: dict) -> str:
    """Which kohya train script a config runs on: "flux" or (default) "sd3"."""
    return "flux" if (config.get("model") or {}).get("arch") == "flux" else "sd3"


def _build_train_dict(config: dict, spec: NetworkSpec, engine: str,
                      dataset_toml_path: Path, train_toml_path: Path) -> dict:
    train = dict(config.get("train", {}))

    train["network_module"] = spec.network_module
    train["network_dim"] = spec.network_dim
    train["network_alpha"] = spec.network_alpha
    train["network_args"] = list(spec.network_args)
    train.update(spec.flags)

    # kohya opens these as files from its own cwd, so every one of them is an
    # absolute path resolved under the model root — never the HF repo id.
    model_paths = resolve_model_paths(config)
    train["pretrained_model_name_or_path"] = str(model_paths.checkpoint)
    for te_key, te_path in model_paths.text_encoders.items():
        if te_key in _TEXT_ENCODER_DEST_KEYS:
            train[te_key] = str(te_path)
    # Only flux_train_network.py has an `ae` dest; the sd3 registry has none.
    if engine == "flux" and model_paths.ae is not None:
        train["ae"] = str(model_paths.ae)

    adapter_id = config.get("adapter_id", "adapter")
    # kohya writes the trained adapter next to the TOML it was launched from:
    # the run directory the caller chose, never a path baked into the config.
    train["output_dir"] = str(train_toml_path.parent)
    train["output_name"] = adapter_id
    train["dataset_config"] = str(dataset_toml_path)

    return train


def _build_dataset_dict(config: dict) -> dict:
    dataset_cfg = config.get("dataset", {})
    # kohya resolves image_dir against the sd-scripts cwd, not ours.
    dataset_table = {
        "subsets": [{
            "image_dir": str(resolve_dataset_paths(config).image_dir),
            "num_repeats": 1,
            "caption_extension": ".txt",
        }],
    }
    if "resolution" in dataset_cfg:
        dataset_table["resolution"] = dataset_cfg["resolution"]
    return {"datasets": [dataset_table]}


def emit(config: dict, out_dir) -> EmitResult:
    """Write kohya_config.toml + dataset_config.toml for `config` into `out_dir`."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    train_toml_path = out_dir / TRAIN_TOML_NAME
    dataset_toml_path = out_dir / DATASET_TOML_NAME

    spec = build_network_args(config)
    engine = engine_for(config)

    train = _build_train_dict(config, spec, engine, dataset_toml_path,
                              train_toml_path)
    validate_keys(train.keys(), engine)

    dataset = _build_dataset_dict(config)

    with open(train_toml_path, "wb") as f:
        tomli_w.dump(train, f)
    with open(dataset_toml_path, "wb") as f:
        tomli_w.dump(dataset, f)

    return EmitResult(train_toml=train_toml_path, dataset_toml=dataset_toml_path)
