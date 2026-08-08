"""Where the weight files and datasets actually live on this machine.

kohya-ss/sd-scripts takes a **file** for ``--pretrained_model_name_or_path``
and one file per text encoder, and it resolves ``image_dir`` against its own
cwd. Configs must stay machine-independent (the matched-budget block is
compared byte-for-byte across machines), so they name files *relative to a
root*:

    LORAFACTORY_MODELS_DIR    default /workspace/models
    LORAFACTORY_DATASETS_DIR  default /workspace/datasets

This module is the only place that joins the two halves. Resolution is pure:
it never stats, creates or downloads anything, so ``emit-kohya`` works on a
laptop with no weights present. ``download_plan()`` is the same join expressed
as "what would have to be fetched", which is what ``fetch-models`` reads.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

MODELS_DIR_ENV = "LORAFACTORY_MODELS_DIR"
DATASETS_DIR_ENV = "LORAFACTORY_DATASETS_DIR"

DEFAULT_MODEL_ROOT = "/workspace/models"
DEFAULT_DATASET_ROOT = "/workspace/datasets"


class ModelPathError(Exception):
    """A config does not name the model/dataset files this toolchain needs."""


@dataclass(frozen=True)
class ModelPaths:
    """Absolute locations of every weight file one adapter config refers to."""

    checkpoint: Path
    text_encoders: dict[str, Path] = field(default_factory=dict)
    ae: Path | None = None


@dataclass(frozen=True)
class DatasetPaths:
    """Absolute locations of one adapter's image directory and manifest."""

    image_dir: Path
    manifest: Path | None = None


def _root(env_var: str, default: str) -> Path:
    """Read `env_var` at call time (tests monkeypatch it) as an absolute Path."""
    value = os.environ.get(env_var) or default
    return Path(value).expanduser().absolute()


def model_root() -> Path:
    """Root directory holding the checkpoints, text encoders and autoencoder."""
    return _root(MODELS_DIR_ENV, DEFAULT_MODEL_ROOT)


def dataset_root() -> Path:
    """Root directory holding the training datasets."""
    return _root(DATASETS_DIR_ENV, DEFAULT_DATASET_ROOT)


def _under(root: Path, relative) -> Path:
    """Join a config-relative name under `root` (an absolute name wins as-is)."""
    path = Path(relative).expanduser()
    return path if path.is_absolute() else root / path


def resolve_model_paths(config: dict) -> ModelPaths:
    """Absolute weight paths for `config`. Pure — the files need not exist."""
    model = config.get("model") or {}

    train_file = model.get("train_file")
    if not train_file:
        raise ModelPathError(
            "config model section has no 'train_file': kohya needs a single-file "
            "checkpoint path, not a repo id"
        )

    root = model_root()
    text_encoders = {
        name: _under(root, relative)
        for name, relative in (model.get("text_encoders") or {}).items()
    }
    ae = model.get("ae")

    return ModelPaths(
        checkpoint=_under(root, train_file),
        text_encoders=text_encoders,
        ae=_under(root, ae) if ae else None,
    )


def resolve_dataset_paths(config: dict) -> DatasetPaths:
    """Absolute image_dir / manifest paths for `config`. Pure, like the above."""
    dataset = config.get("dataset") or {}

    path = dataset.get("path")
    if not path:
        raise ModelPathError(
            "config dataset section has no 'path': the training images must be "
            "named relative to the dataset root"
        )

    root = dataset_root()
    manifest = dataset.get("manifest")

    return DatasetPaths(
        image_dir=_under(root, path),
        manifest=_under(root, manifest) if manifest else None,
    )


def download_plan(config: dict) -> list[dict]:
    """What `fetch-models` would download for `config`, without downloading it.

    One entry per file — checkpoint, every text encoder, and the FLUX
    autoencoder when the config has one — each pinned to the config's
    `model.train_revision`. An unpinned pull would silently change the weights
    an adapter was trained on and break the provenance record.
    """
    model = config.get("model") or {}
    paths = resolve_model_paths(config)

    repo_id = model.get("train_repo")
    if not repo_id:
        raise ModelPathError("config model section has no 'train_repo' to fetch from")
    revision = model.get("train_revision")
    if not revision:
        raise ModelPathError(
            f"model.train_revision missing for {repo_id}: every download must be "
            "pinned to a commit sha"
        )

    entries = [(model["train_file"], paths.checkpoint)]
    entries += [
        (model["text_encoders"][name], path)
        for name, path in paths.text_encoders.items()
    ]
    if paths.ae is not None:
        entries.append((model["ae"], paths.ae))

    return [
        {
            "repo_id": repo_id,
            "revision": revision,
            "filename": str(filename),
            "local_path": local_path,
        }
        for filename, local_path in entries
    ]
