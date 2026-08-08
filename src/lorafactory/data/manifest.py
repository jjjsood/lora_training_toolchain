"""Dataset provenance manifest — CAP-24 / §5.4.

manifest.csv columns (exact):
    file,sha256,caption,source_url,author,licence,licence_url,acquisition_date

Errors: file on disk without row; row without file; sha256 mismatch;
empty provenance field. Caption files (*.txt) and manifest.csv itself are
not dataset images.
"""

from __future__ import annotations

import csv
import hashlib
from dataclasses import dataclass, field
from pathlib import Path

COLUMNS = [
    "file", "sha256", "caption", "source_url", "author",
    "licence", "licence_url", "acquisition_date",
]

NON_IMAGE_SUFFIXES = {".txt"}
MANIFEST_FILENAME = "manifest.csv"


@dataclass
class ManifestReport:
    ok: bool
    errors: list[str] = field(default_factory=list)
    manifest_hash: str = ""


def sha256_file(path: Path) -> str:
    """sha256 of a file's bytes, read in chunks — adapters run to gigabytes."""
    digest = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def check_dataset(dataset_dir: Path) -> ManifestReport:
    dataset_dir = Path(dataset_dir)
    manifest_path = dataset_dir / MANIFEST_FILENAME
    errors: list[str] = []

    manifest_bytes = manifest_path.read_bytes()
    manifest_hash = hashlib.sha256(manifest_bytes).hexdigest()

    with open(manifest_path, newline="") as f:
        reader = csv.DictReader(f)
        rows = list(reader)

    row_by_file = {}
    for row in rows:
        name = row.get("file", "")
        row_by_file[name] = row
        for col in COLUMNS:
            if not (row.get(col) or "").strip():
                errors.append(f"row for {name!r}: empty field {col!r}")

    disk_images = {
        p.name for p in dataset_dir.iterdir()
        if p.is_file()
        and p.name != MANIFEST_FILENAME
        and p.suffix.lower() not in NON_IMAGE_SUFFIXES
    }

    for name in sorted(disk_images - row_by_file.keys()):
        errors.append(f"file on disk without manifest row: {name}")

    for name in sorted(row_by_file.keys() - disk_images):
        errors.append(f"manifest row without file on disk: {name}")

    for name in sorted(disk_images & row_by_file.keys()):
        row = row_by_file[name]
        actual = sha256_file(dataset_dir / name)
        expected = row.get("sha256", "")
        if actual != expected:
            errors.append(
                f"sha256 mismatch for {name}: expected {expected}, got {actual}"
            )

    return ManifestReport(ok=not errors, errors=errors, manifest_hash=manifest_hash)
