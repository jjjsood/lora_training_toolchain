"""Pydantic validation of a RESOLVED config (post-overlay).

`loader.resolve()` merges an overlay onto its base and returns the merged dict;
nothing until now checked its shape, so a missing `model.train_file` or a
truncated `train_revision` would travel all the way into an emitted kohya TOML.

`validate()` runs on that merged dict — the object every downstream consumer
(emitter, runner, provenance) actually receives — and raises
`ConfigSchemaError` on any violation. `cli._resolve_config()` calls it on the
way out of `resolve()`, so no command can reach a config it has not checked.

Deliberate asymmetry: the top level is `extra="forbid"` (a typo'd section such
as `trian` is silently a no-op everywhere else, so it must be our error), while
`train` allows extra keys because it carries kohya passthrough arguments that
legitimately differ per adapter (e.g. `fp8_base` only on the FLUX config).
"""

from __future__ import annotations

import re
import warnings
from typing import Annotated, Literal

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    StrictBool,
    StrictInt,
    StrictStr,
    ValidationError,
    field_validator,
    model_validator,
)

from lorafactory.constants import (
    FLUX_MAX_DOUBLE_BLOCK_INDEX,
    FLUX_MAX_SINGLE_BLOCK_INDEX,
    SD3_MAX_BLOCK_INDEX,
)

__all__ = ["ARCH_DEFAULTS", "ConfigSchemaError", "validate"]

_SHA1_RE = r"^[0-9a-f]{40}$"
_SHA1_ONLY = re.compile(_SHA1_RE)
#: Relaxed: a full SHA, or anything branch/tag-shaped. Non-SHA values still
#: work — see `_warn_unpinned_revisions` below, which flags them without
#: blocking, preserving the reproducibility guarantee as a warning rather
#: than turning it into friction for local/dev configs.
_REVISION_RE = r"^[0-9a-f]{40}$|^[A-Za-z0-9_.\-/]+$"

#: Highest transformer block index of SD3-Medium (24 blocks, 0-23). Kept as
#: its own name — referenced from outside this module — but derived from the
#: single source of truth in `constants.py`.
MAX_BLOCK_INDEX = SD3_MAX_BLOCK_INDEX

#: `blocks` is an inclusive [first, last] pair, never an enumeration.
_BLOCKS_PAIR_LEN = 2

PositiveInt = Annotated[StrictInt, Field(gt=0)]
Revision = Annotated[StrictStr, Field(pattern=_REVISION_RE)]

#: Known-good defaults per `model.arch`, copied from the real verified pins in
#: configs/base.yaml, configs/matrix/F-F.yaml and tests/test_model_pins.py —
#: this repo already keeps that trio in sync by hand; this is a fourth copy of
#: the same convention, not a new one. Production code cannot import
#: tests/test_model_pins.py, so this cannot literally share that dict.
ARCH_DEFAULTS: dict[str, dict] = {
    "sd3": {
        "train_repo": "stabilityai/stable-diffusion-3-medium",
        "train_revision": "19b7f516efea082d257947e057e6f419e26fd497",
        "train_file": "sd3_medium.safetensors",
        "eval_repo": "stabilityai/stable-diffusion-3-medium-diffusers",
        "eval_revision": "ea42f8cef0f178587cf766dc8129abd379c90671",
        "text_encoders": {
            "clip_l": "text_encoders/clip_l.safetensors",
            "clip_g": "text_encoders/clip_g.safetensors",
            "t5xxl": "text_encoders/t5xxl_fp16.safetensors",
        },
    },
    "flux": {
        "train_repo": "black-forest-labs/FLUX.1-dev",
        "train_revision": "3de623fc3c33e44ffbe2bad470d0f45bccf2eb21",
        "train_file": "flux1-dev.safetensors",
        "eval_repo": "black-forest-labs/FLUX.1-dev",
        "eval_revision": "3de623fc3c33e44ffbe2bad470d0f45bccf2eb21",
        "ae": "ae.safetensors",
        "text_encoders": {
            "clip_l": "text_encoders/clip_l.safetensors",
            "t5xxl": "text_encoders/t5xxl_fp16.safetensors",
        },
    },
}

#: repo -> verified HEAD sha, derived from ARCH_DEFAULTS above. Used only to
#: warn on drift, never to block — see `_warn_unpinned_revisions`.
_VERIFIED_REVISIONS: dict[str, str] = {}
for _arch_defaults in ARCH_DEFAULTS.values():
    _VERIFIED_REVISIONS[_arch_defaults["train_repo"]] = _arch_defaults["train_revision"]
    _VERIFIED_REVISIONS[_arch_defaults["eval_repo"]] = _arch_defaults["eval_revision"]
del _arch_defaults

#: The reverse of `_VERIFIED_REVISIONS`: every SHA we have actually verified,
#: regardless of which repo it belongs to. Lets `_warn_unpinned_revisions`
#: catch a SHA that IS a real verified pin, just copy-pasted next to the
#: wrong (unrecognized) repo — worse than an unrecognized repo/SHA pair,
#: which has no ground truth to contradict and is deliberately let through.
_VERIFIED_SHAS: set[str] = set(_VERIFIED_REVISIONS.values())


class ConfigSchemaError(Exception):
    """A resolved config violates the schema."""


class _Section(BaseModel):
    model_config = ConfigDict(extra="forbid")


class ModelSection(_Section):
    # `protected_namespaces` cleared: this section is literally named "model".
    model_config = ConfigDict(extra="forbid", protected_namespaces=())

    arch: StrictStr
    train_repo: StrictStr | None = None
    train_revision: Revision | None = None
    #: kohya takes a FILE for --pretrained_model_name_or_path, never a repo id.
    train_file: StrictStr | None = None
    eval_repo: StrictStr | None = None
    eval_revision: Revision | None = None
    text_encoders: dict[StrictStr, StrictStr] | None = None
    ae: StrictStr | None = None

    @model_validator(mode="before")
    @classmethod
    def _fill_arch_defaults(cls, data):
        if not isinstance(data, dict):
            return data
        arch = data.get("arch")
        # `arch` must be hashable to key ARCH_DEFAULTS.get(); a non-string
        # value (e.g. a YAML list) would otherwise raise a raw TypeError here,
        # bypassing pydantic's own clean `string_type` error entirely.
        defaults = ARCH_DEFAULTS.get(arch) if isinstance(arch, str) else None
        if not defaults:
            return data
        filled = dict(data)
        for key, value in defaults.items():
            filled.setdefault(key, value)
        return filled

    @model_validator(mode="after")
    def _required_after_defaults(self) -> ModelSection:
        """Fields with no ARCH_DEFAULTS entry for this arch are still required."""
        for name in (
            "train_repo", "train_revision", "train_file",
            "eval_repo", "eval_revision", "text_encoders",
        ):
            if getattr(self, name) is None:
                raise ValueError(
                    f"model.{name} is required (no default for arch {self.arch!r})"
                )
        return self

    @model_validator(mode="after")
    def _warn_unpinned_revisions(self) -> ModelSection:
        """Never blocks — see docs/superpowers/specs/2026-08-22-config-dx-relaxation-design.md."""
        for repo_field, rev_field in (
            ("train_repo", "train_revision"), ("eval_repo", "eval_revision"),
        ):
            repo, rev = getattr(self, repo_field), getattr(self, rev_field)
            if repo is None or rev is None:
                continue
            if not _SHA1_ONLY.fullmatch(rev):
                warnings.warn(
                    f"model.{rev_field} {rev!r} is not a pinned commit SHA — "
                    "reproducibility is not guaranteed for this run",
                    stacklevel=2,
                )
                continue
            expected = _VERIFIED_REVISIONS.get(repo)
            if expected is not None and rev != expected:
                warnings.warn(
                    f"model.{rev_field} {rev!r} does not match the verified pin "
                    f"for {repo} ({expected!r}) — reproducibility is not guaranteed",
                    stacklevel=2,
                )
            elif expected is None and rev in _VERIFIED_SHAS:
                warnings.warn(
                    f"model.{rev_field} {rev!r} is a verified pin for a different "
                    f"repo, not {repo!r} — reproducibility is not guaranteed",
                    stacklevel=2,
                )
        return self


class RemoteDatasetSource(_Section):
    """A dataset pinned hard enough to be re-fetched from Hugging Face.

    `fetch-dataset` can rebuild the directory byte-for-byte from `repo` @
    `revision`, and the licence fields below are what lands in the
    manifest's provenance columns — which is why none of them may be empty.
    """

    type: Literal["remote", "hf"] = "remote"
    repo: StrictStr
    #: A dataset repo commit sha, same 40-hex rule as the model pins: an
    #: unpinned pull would silently change the images an adapter was trained on.
    revision: Revision
    #: Exactly one of these two: a parquet file inside the repo, or a list of
    #: image paths (fnmatch patterns allowed) to pull as individual files.
    parquet: StrictStr | None = None
    files: list[StrictStr] | None = None
    #: How many images to take, in source order (never a random sample).
    limit: PositiveInt | None = None
    #: Caption written to every img_NNNN.txt — carries the trigger token.
    caption: StrictStr
    author: StrictStr
    licence: StrictStr
    licence_url: StrictStr
    acquisition_date: StrictStr

    @field_validator("files")
    @classmethod
    def _files_not_empty(cls, v: list[str] | None) -> list[str] | None:
        if v is not None and not v:
            raise ValueError("files must not be an empty list")
        return v

    @model_validator(mode="after")
    def _exactly_one_source_form(self) -> RemoteDatasetSource:
        if bool(self.parquet) == bool(self.files):
            raise ValueError("give exactly one of 'parquet' or 'files'")
        return self


class LocalDatasetSource(_Section):
    """A dataset that already lives on disk — no HF metadata required.

    `fetch-dataset` stages it the same way it stages an HF repo: it copies
    the files matching `files` (default every `.png` directly under `path`)
    into the dataset's image directory.
    """

    type: Literal["local"] = "local"
    path: StrictStr
    files: list[StrictStr] | None = None
    limit: PositiveInt | None = None
    caption: StrictStr
    #: Unlike RemoteDatasetSource, all optional: local data has no HF-style
    #: provenance, so `data/fetch.py` fills non-empty sentinels when absent
    #: rather than forcing the user to invent metadata that doesn't exist.
    author: StrictStr | None = None
    licence: StrictStr | None = None
    licence_url: StrictStr | None = None
    acquisition_date: StrictStr | None = None

    @model_validator(mode="before")
    @classmethod
    def _default_files_glob(cls, data):
        if not isinstance(data, dict):
            return data
        if not data.get("files") and data.get("path"):
            data = dict(data)
            data["files"] = [f"{data['path']}/*.png"]
        return data

    @field_validator("files")
    @classmethod
    def _files_not_empty(cls, v: list[str] | None) -> list[str] | None:
        if v is not None and not v:
            raise ValueError("files must not be an empty list")
        return v


#: Discriminated on `type`: a `local` source needs only a path; a `remote`
#: (or legacy-spelled `hf`) source keeps the full HF metadata contract.
DatasetSourceSection = Annotated[
    LocalDatasetSource | RemoteDatasetSource, Field(discriminator="type")
]


#: Keys that only ever appear on a remote (HF) source — i.e. genuinely absent
#: from `LocalDatasetSource` entirely. Used to infer `type` when it's
#: omitted — checking all of them, not just `repo`, means a typo'd `repo` key
#: (e.g. `repos:`) still gets diagnosed as a remote source with a missing
#: `repo`, rather than as a local source with a pile of confusing "extra"
#: fields (revision/parquet/...) to delete.
#:
#: `licence`/`licence_url`/`acquisition_date` are deliberately NOT here even
#: though `RemoteDatasetSource` requires them: `LocalDatasetSource` also
#: models them as optional fields (filled with sentinel defaults by
#: `data/fetch.py` when a local config omits them), so including them here
#: misclassified a genuinely local source that happens to supply e.g.
#: `licence: "CC0"` as remote, and it then failed validation (missing
#: `repo`/`revision`, `path` reported as extra).
_REMOTE_ONLY_KEYS = ("repo", "revision", "parquet")


class DatasetSection(_Section):
    name: StrictStr
    path: StrictStr
    manifest: StrictStr
    resolution: PositiveInt = 1024
    #: Absent on the matrix configs; see LocalDatasetSource/RemoteDatasetSource.
    source: DatasetSourceSection | None = None

    @model_validator(mode="before")
    @classmethod
    def _default_manifest(cls, data):
        if not isinstance(data, dict):
            return data
        if data.get("manifest") or not data.get("path"):
            return data
        data = dict(data)
        data["manifest"] = f"{data['path']}/manifest.csv"
        return data

    @field_validator("source", mode="before")
    @classmethod
    def _infer_source_type(cls, v):
        """A `source` with no `type` but any remote-only key (`repo` — the
        pre-existing backward-compat case — or `revision`/`parquet`/licence
        metadata) is the remote shape; otherwise it's a local source."""
        if not isinstance(v, dict) or "type" in v:
            return v
        v = dict(v)
        v["type"] = "remote" if any(k in v for k in _REMOTE_ONLY_KEYS) else "local"
        return v


class TrainSection(BaseModel):
    """Known keys are type-checked; unknown ones pass through to kohya."""

    model_config = ConfigDict(extra="allow")

    seed: StrictInt
    max_train_steps: PositiveInt
    learning_rate: float
    train_batch_size: PositiveInt
    optimizer_type: StrictStr = "AdamW8bit"
    mixed_precision: StrictStr
    save_precision: StrictStr
    save_model_as: StrictStr = "safetensors"
    gradient_checkpointing: StrictBool = True
    logging_dir: StrictStr = "logs"

    @field_validator("learning_rate")
    @classmethod
    def _lr_positive(cls, v: float) -> float:
        if not v > 0:
            raise ValueError("learning_rate must be > 0")
        return v


class TargetSection(_Section):
    scope: StrictStr
    rank: PositiveInt
    alpha: PositiveInt
    #: SD3 only. Absent when the scope adapts everything or no transformer at all.
    blocks: list[StrictInt] | None = None
    #: FLUX only. Double-stream (`train_double_block_indices`) block range.
    blocks_double: list[StrictInt] | None = None
    #: FLUX only. Single-stream (`train_single_block_indices`) block range.
    blocks_single: list[StrictInt] | None = None
    module_classes: list[StrictStr] | None = None
    te_encoders: list[StrictStr] | None = None

    @field_validator("blocks", "blocks_double", "blocks_single")
    @classmethod
    def _block_range_shape(cls, v: list[int] | None) -> list[int] | None:
        """Shape-only: `[first, last]`, non-negative, ascending.

        The upper bound is arch-dependent (SD3 vs FLUX-double vs FLUX-single)
        and arch lives in `ModelSection`, invisible from here — that check
        happens in `ResolvedConfigModel`'s model_validator instead.
        """
        if v is None:
            return v
        if len(v) != _BLOCKS_PAIR_LEN:
            raise ValueError("block range must be [first, last]")
        first, last = v
        for i in v:
            if i < 0:
                raise ValueError(f"block index {i} must be non-negative")
        if first > last:
            raise ValueError("block range must be ordered [first, last]")
        return v


class DeterminismSection(_Section):
    cublas_workspace_config: StrictStr = ":4096:8"
    pythonhashseed: StrictStr = "0"
    deterministic_algorithms: StrictBool = True

    @field_validator("pythonhashseed", mode="before")
    @classmethod
    def _coerce_pythonhashseed(cls, v):
        if isinstance(v, int):
            return str(v)
        return v


class ResolvedConfigModel(BaseModel):
    model_config = ConfigDict(extra="forbid", protected_namespaces=())

    model: ModelSection
    dataset: DatasetSection
    train: TrainSection
    target: TargetSection
    determinism: DeterminismSection
    #: There is no `output` section, and `extra="forbid"` above rejects one: where a
    #: run is written is a runtime decision (CLI --runs-dir / LORAFACTORY_RUNS_DIR),
    #: not part of the matched budget. It used to hold `base_dir: /workspace/runs`,
    #: which the emitter copied into the kohya TOML's output_dir — a container-only
    #: path that produced nothing on a host checkout.

    # Meta keys the loader stamps onto the resolved dict.
    adapter_id: StrictStr | None = None
    budget_exempt: StrictBool = False

    @model_validator(mode="after")
    def _arch_specific_block_targeting(self) -> ResolvedConfigModel:
        """`target.blocks*` fields are arch-exclusive and arch-bounded.

        `TargetSection` cannot see `model.arch`, so the cross-section check
        — which block field is legal for this arch, and what its upper bound
        is — happens here instead.
        """
        arch = self.model.arch
        target = self.target

        if arch == "flux":
            if target.blocks is not None:
                raise ValueError(
                    "target.blocks is SD3-only; flux uses "
                    "blocks_double/blocks_single"
                )
            if (
                target.scope == "transformer"
                and target.blocks_double is None
                and target.blocks_single is None
            ):
                raise ValueError(
                    "flux transformer scope requires at least one of "
                    "target.blocks_double / target.blocks_single"
                )
            for name, bound in (
                ("blocks_double", FLUX_MAX_DOUBLE_BLOCK_INDEX),
                ("blocks_single", FLUX_MAX_SINGLE_BLOCK_INDEX),
            ):
                pair = getattr(target, name)
                if pair is None:
                    continue
                for i in pair:
                    if not 0 <= i <= bound:
                        raise ValueError(f"{name} index {i} outside 0..{bound}")
        else:
            if target.blocks_double is not None or target.blocks_single is not None:
                raise ValueError(
                    "target.blocks_double/blocks_single are flux-only; "
                    "sd3 uses target.blocks"
                )
            if target.blocks is not None:
                for i in target.blocks:
                    if not 0 <= i <= SD3_MAX_BLOCK_INDEX:
                        raise ValueError(
                            f"block index {i} outside 0..{SD3_MAX_BLOCK_INDEX}"
                        )

        return self


_ERROR_HINTS = {
    "missing": "provide a value",
    "string_pattern_mismatch": "check the expected format",
    "string_type": "expected a string",
    "int_type": "expected an integer",
    "int_parsing": "expected an integer",
    "float_parsing": "expected a number",
    "bool_type": "expected true or false",
    "extra_forbidden": "remove this key, or check for a typo",
    "literal_error": "check the allowed values",
}


def _format_validation_error(exc: ValidationError) -> str:
    """One line per Pydantic error: `<field.path>: <message> (<hint>)`."""
    lines = []
    for err in exc.errors():
        loc = ".".join(str(part) for part in err["loc"]) or "<root>"
        hint = _ERROR_HINTS.get(err["type"])
        line = f"{loc}: {err['msg']}"
        if hint:
            line += f" ({hint})"
        lines.append(line)
    return "\n".join(lines)


def validate(resolved: dict) -> ResolvedConfigModel:
    """Validate a resolved config dict, raising ConfigSchemaError on any error."""
    try:
        return ResolvedConfigModel.model_validate(resolved)
    except ValidationError as exc:
        raise ConfigSchemaError(_format_validation_error(exc)) from exc
