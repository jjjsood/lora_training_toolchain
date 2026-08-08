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

from typing import Annotated

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

__all__ = ["ConfigSchemaError", "validate"]

_SHA1_RE = r"^[0-9a-f]{40}$"

#: Highest transformer block index of SD3-Medium (24 blocks, 0-23).
MAX_BLOCK_INDEX = 23

#: `blocks` is an inclusive [first, last] pair, never an enumeration.
_BLOCKS_PAIR_LEN = 2

PositiveInt = Annotated[StrictInt, Field(gt=0)]
Revision = Annotated[StrictStr, Field(pattern=_SHA1_RE)]


class ConfigSchemaError(Exception):
    """A resolved config violates the schema."""


class _Section(BaseModel):
    model_config = ConfigDict(extra="forbid")


class ModelSection(_Section):
    # `protected_namespaces` cleared: this section is literally named "model".
    model_config = ConfigDict(extra="forbid", protected_namespaces=())

    arch: StrictStr
    train_repo: StrictStr
    train_revision: Revision
    #: kohya takes a FILE for --pretrained_model_name_or_path, never a repo id.
    train_file: StrictStr
    eval_repo: StrictStr
    eval_revision: Revision
    text_encoders: dict[StrictStr, StrictStr]
    ae: StrictStr | None = None


class DatasetSourceSection(_Section):
    """Where the images come from, pinned hard enough to be re-fetchable.

    Optional and only present on the test configs: the matrix configs describe
    data that was acquired by hand and is documented in datasets/README.md.
    When it *is* present, `fetch-dataset` can rebuild the directory byte-for-byte
    from `repo` @ `revision`, and the licence fields below are what lands in the
    manifest's provenance columns — which is why none of them may be empty.
    """

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
    def _exactly_one_source_form(self) -> DatasetSourceSection:
        if bool(self.parquet) == bool(self.files):
            raise ValueError("give exactly one of 'parquet' or 'files'")
        return self


class DatasetSection(_Section):
    name: StrictStr
    path: StrictStr
    manifest: StrictStr
    resolution: PositiveInt
    #: Absent on the matrix configs; see DatasetSourceSection.
    source: DatasetSourceSection | None = None


class TrainSection(BaseModel):
    """Known keys are type-checked; unknown ones pass through to kohya."""

    model_config = ConfigDict(extra="allow")

    seed: StrictInt
    max_train_steps: PositiveInt
    learning_rate: float
    train_batch_size: PositiveInt
    optimizer_type: StrictStr
    mixed_precision: StrictStr
    save_precision: StrictStr
    save_model_as: StrictStr
    gradient_checkpointing: StrictBool
    logging_dir: StrictStr

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
    #: Absent when the scope adapts everything (FLUX) or no transformer at all.
    blocks: list[StrictInt] | None = None
    module_classes: list[StrictStr] | None = None
    te_encoders: list[StrictStr] | None = None

    @field_validator("blocks")
    @classmethod
    def _blocks_are_an_inclusive_pair(
        cls, v: list[int] | None
    ) -> list[int] | None:
        if v is None:
            return v
        if len(v) != _BLOCKS_PAIR_LEN:
            raise ValueError("blocks must be [first, last]")
        first, last = v
        for i in v:
            if not 0 <= i <= MAX_BLOCK_INDEX:
                raise ValueError(f"block index {i} outside 0..{MAX_BLOCK_INDEX}")
        if first > last:
            raise ValueError("blocks must be ordered [first, last]")
        return v


class DeterminismSection(_Section):
    cublas_workspace_config: StrictStr
    pythonhashseed: StrictStr
    deterministic_algorithms: StrictBool


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


def validate(resolved: dict) -> ResolvedConfigModel:
    """Validate a resolved config dict, raising ConfigSchemaError on any error."""
    try:
        return ResolvedConfigModel.model_validate(resolved)
    except ValidationError as exc:
        raise ConfigSchemaError(str(exc)) from exc
