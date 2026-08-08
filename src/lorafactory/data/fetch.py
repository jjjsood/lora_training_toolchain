"""Rebuild a dataset directory from the `dataset.source` block of a config.

The matrix configs describe images acquired by hand; `datasets/README.md` is
their contract and nothing here touches them. A config that *does* carry a
`dataset.source` block declares its images as a pinned Hugging Face dataset
(repo + 40-hex revision), and this module turns that declaration into a
directory `check-dataset` accepts: `img_0001.png` / `img_0001.txt` pairs plus a
`manifest.csv` whose provenance columns come straight out of the config.

Deliberately split like ``lorafactory.models``:

``fetch_plan()``   pure — no network, no ``stat``. What *would* be fetched.
``fetch_dataset()``  the same join, executed.

Two properties the rest of the toolchain depends on:

* **Deterministic selection.** ``limit`` takes the first N images in source
  order (parquet row order, or sorted repo path order) — never a sample. Two
  machines running this against the same revision get the same pixels, which is
  what makes ``dataset_manifest_hash`` comparable across them.
* **Idempotence.** A directory that already validates against its manifest with
  the expected image count is left alone, so the container's one-shot
  ``test-run`` can be re-invoked without re-downloading anything.
"""

from __future__ import annotations

import csv
import fnmatch
import io
import re
from dataclasses import dataclass, field
from pathlib import Path

import pyarrow.parquet as pq
from huggingface_hub import hf_hub_download, list_repo_files
from PIL import Image

from ..models import resolve_dataset_paths
from .manifest import COLUMNS, MANIFEST_FILENAME, check_dataset, sha256_file

#: Filename pattern from datasets/README.md; also what a rebuild is allowed to
#: delete, so a stale run of 50 images cannot leave orphans behind a run of 20.
IMAGE_SUFFIX = ".png"
CAPTION_SUFFIX = ".txt"
_MANAGED_NAME = re.compile(r"^img_\d{4}(\.png|\.txt)$")


class DatasetSourceError(Exception):
    """A config cannot be turned into a dataset fetch."""


@dataclass
class FetchReport:
    action: str  # "fetched" | "skipped"
    image_dir: Path
    count: int
    manifest_hash: str = ""
    errors: list[str] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return not self.errors


def _dataset(config: dict) -> dict:
    return config.get("dataset") or {}


def _source(config: dict) -> dict:
    source = _dataset(config).get("source")
    if not source:
        raise DatasetSourceError(
            "config dataset section has no 'source': only configs that declare "
            "where their images come from can be fetched (see configs/test/)"
        )
    return source


def fetch_plan(config: dict) -> dict:
    """What `fetch_dataset` would download for `config`. Pure — nothing is read.

    File sources are reported as the *patterns* the config names, not as
    resolved filenames: expanding them means listing the repo, which is network.
    """
    source = _source(config)
    paths = resolve_dataset_paths(config)

    for required in ("repo", "revision", "caption"):
        if not source.get(required):
            raise DatasetSourceError(f"dataset.source has no '{required}'")

    parquet, files = source.get("parquet"), source.get("files")
    if bool(parquet) == bool(files):
        raise DatasetSourceError(
            "dataset.source must name exactly one of 'parquet' or 'files'"
        )

    return {
        "repo": source["repo"],
        "revision": source["revision"],
        "form": "parquet" if parquet else "files",
        "source": parquet if parquet else list(files),
        "limit": source.get("limit"),
        "resolution": _dataset(config).get("resolution"),
        "image_dir": paths.image_dir,
        "manifest": paths.image_dir / MANIFEST_FILENAME,
        "caption": source["caption"],
        "provenance": {
            "author": source.get("author", ""),
            "licence": source.get("licence", ""),
            "licence_url": source.get("licence_url", ""),
            "acquisition_date": source.get("acquisition_date", ""),
        },
    }


def _blob_url(repo: str, revision: str, path: str, row: int | None = None) -> str:
    """Pin the provenance URL to the revision, not to the branch head."""
    url = f"https://huggingface.co/datasets/{repo}/blob/{revision}/{path}"
    return f"{url}#row={row}" if row is not None else url


def _square(image, resolution: int):
    """Centre-crop to 1:1, then resize to `resolution` — no padding, no letterbox."""
    image = image.convert("RGB")
    width, height = image.size
    side = min(width, height)
    left = (width - side) // 2
    top = (height - side) // 2
    image = image.crop((left, top, left + side, top + side))
    if side != resolution:
        image = image.resize((resolution, resolution), Image.LANCZOS)
    return image


def _parquet_images(repo: str, revision: str, parquet: str, limit: int | None):
    """(row_index, image_bytes) pairs in parquet row order."""
    local = hf_hub_download(
        repo_id=repo, revision=revision, filename=parquet, repo_type="dataset"
    )
    table = pq.read_table(local, columns=["image"])
    column = table.column("image")

    total = len(column) if limit is None else min(limit, len(column))
    for row in range(total):
        cell = column[row].as_py()
        # HF's Image feature stores {"bytes": ..., "path": ...} in parquet.
        data = cell.get("bytes") if isinstance(cell, dict) else cell
        if not data:
            raise DatasetSourceError(
                f"{repo}@{revision[:8]} row {row}: image column holds no bytes "
                "(the dataset may store external paths rather than the images)"
            )
        yield row, data


def _file_images(repo: str, revision: str, patterns: list[str], limit: int | None):
    """(repo_path, image_bytes) pairs in sorted repo-path order."""
    names = sorted(
        name for name in list_repo_files(repo, revision=revision, repo_type="dataset")
        if any(fnmatch.fnmatch(name, pattern) for pattern in patterns)
    )
    if not names:
        raise DatasetSourceError(
            f"{repo}@{revision[:8]}: no repo file matches {patterns}"
        )
    if limit is not None:
        names = names[:limit]

    for name in names:
        local = hf_hub_download(
            repo_id=repo, revision=revision, filename=name, repo_type="dataset"
        )
        yield name, Path(local).read_bytes()


def _image_count(image_dir: Path) -> int:
    return sum(1 for p in image_dir.iterdir() if p.suffix == IMAGE_SUFFIX)


def _clear_managed(image_dir: Path) -> None:
    """Drop a previous rebuild's img_NNNN.* so a shorter run leaves no orphans."""
    for path in image_dir.iterdir():
        if path.is_file() and _MANAGED_NAME.match(path.name):
            path.unlink()


def _already_valid(image_dir: Path, expected: int | None) -> tuple[bool, str]:
    if not (image_dir / MANIFEST_FILENAME).exists():
        return False, ""
    report = check_dataset(image_dir)
    if not report.ok:
        return False, ""
    if expected is not None and _image_count(image_dir) != expected:
        return False, ""
    return True, report.manifest_hash


def fetch_dataset(config: dict, force: bool = False) -> FetchReport:
    """Materialise `config`'s declared dataset, then validate what was written."""
    plan = fetch_plan(config)
    image_dir: Path = plan["image_dir"]
    resolution = plan["resolution"]
    if not resolution:
        raise DatasetSourceError("dataset section has no 'resolution'")

    if not force:
        valid, manifest_hash = _already_valid(image_dir, plan["limit"])
        if valid:
            return FetchReport("skipped", image_dir, _image_count(image_dir),
                               manifest_hash)

    image_dir.mkdir(parents=True, exist_ok=True)
    _clear_managed(image_dir)

    is_parquet = plan["form"] == "parquet"
    reader = _parquet_images if is_parquet else _file_images
    items = reader(plan["repo"], plan["revision"], plan["source"], plan["limit"])

    def source_url(key) -> str:
        """`key` is a parquet row index or, for file sources, a repo path."""
        if is_parquet:
            return _blob_url(plan["repo"], plan["revision"], plan["source"], row=key)
        return _blob_url(plan["repo"], plan["revision"], key)

    rows = []
    for index, (key, data) in enumerate(items, start=1):
        stem = f"img_{index:04d}"
        image_path = image_dir / f"{stem}{IMAGE_SUFFIX}"
        _square(Image.open(io.BytesIO(data)), resolution).save(image_path, "PNG")
        (image_dir / f"{stem}{CAPTION_SUFFIX}").write_text(plan["caption"])

        rows.append({
            "file": image_path.name,
            # Hashed from the file we just wrote, because that is the byte
            # sequence check-dataset re-hashes — not the upstream original.
            "sha256": sha256_file(image_path),
            "caption": plan["caption"],
            "source_url": source_url(key),
            **plan["provenance"],
        })

    if not rows:
        raise DatasetSourceError(
            f"{plan['repo']}@{plan['revision'][:8]}: source yielded no images"
        )

    manifest_path = image_dir / MANIFEST_FILENAME
    with open(manifest_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=COLUMNS)
        writer.writeheader()
        writer.writerows(rows)

    report = check_dataset(image_dir)
    return FetchReport(
        action="fetched",
        image_dir=image_dir,
        count=len(rows),
        manifest_hash=report.manifest_hash,
        errors=list(report.errors),
    )
