"""Dataset provenance manifest — CAP-24 / §5.4.

API pinned:
    from lorafactory.data.manifest import check_dataset
    report = check_dataset(dataset_dir)      # manifest.csv inside dataset_dir
    report.ok             # bool
    report.errors         # list[str]
    report.manifest_hash  # sha256 hex of the manifest file bytes (ok or not)

manifest.csv columns (exact):
    file,sha256,caption,source_url,author,licence,licence_url,acquisition_date
Errors: file on disk without row; row without file; sha256 mismatch;
empty provenance field. Caption files (*.txt) are not dataset images.
"""

import csv
import hashlib

from lorafactory.data.manifest import check_dataset

COLUMNS = ["file", "sha256", "caption", "source_url", "author",
           "licence", "licence_url", "acquisition_date"]


def write_image(d, name, content):
    p = d / name
    p.write_bytes(content)
    return hashlib.sha256(content).hexdigest()


def write_manifest(d, rows):
    with open(d / "manifest.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=COLUMNS)
        w.writeheader()
        w.writerows(rows)


def row(name, digest, **overrides):
    r = {"file": name, "sha256": digest, "caption": "a picture",
         "source_url": "https://example.org/img", "author": "someone",
         "licence": "CC-BY-4.0", "licence_url": "https://example.org/licence",
         "acquisition_date": "2026-08-08"}
    r.update(overrides)
    return r


def test_consistent_dataset_passes(tmp_path):
    h1 = write_image(tmp_path, "a.png", b"img-a")
    h2 = write_image(tmp_path, "b.png", b"img-b")
    (tmp_path / "a.txt").write_text("caption a")  # caption sidecar, not an image
    write_manifest(tmp_path, [row("a.png", h1), row("b.png", h2)])
    r = check_dataset(tmp_path)
    assert r.ok, r.errors
    assert len(r.manifest_hash) == 64


def test_file_without_row_is_error(tmp_path):
    h1 = write_image(tmp_path, "a.png", b"img-a")
    write_image(tmp_path, "orphan.png", b"img-x")
    write_manifest(tmp_path, [row("a.png", h1)])
    r = check_dataset(tmp_path)
    assert not r.ok
    assert any("orphan.png" in e for e in r.errors)


def test_row_without_file_is_error(tmp_path):
    h1 = write_image(tmp_path, "a.png", b"img-a")
    write_manifest(tmp_path, [row("a.png", h1), row("ghost.png", "0" * 64)])
    r = check_dataset(tmp_path)
    assert not r.ok
    assert any("ghost.png" in e for e in r.errors)


def test_hash_mismatch_is_error(tmp_path):
    write_image(tmp_path, "a.png", b"img-a")
    write_manifest(tmp_path, [row("a.png", "0" * 64)])
    assert not check_dataset(tmp_path).ok


def test_empty_provenance_field_is_error(tmp_path):
    h1 = write_image(tmp_path, "a.png", b"img-a")
    write_manifest(tmp_path, [row("a.png", h1, licence="")])
    r = check_dataset(tmp_path)
    assert not r.ok
    assert any("licence" in e for e in r.errors)


def test_manifest_hash_is_stable(tmp_path):
    h1 = write_image(tmp_path, "a.png", b"img-a")
    write_manifest(tmp_path, [row("a.png", h1)])
    expected = hashlib.sha256((tmp_path / "manifest.csv").read_bytes()).hexdigest()
    assert check_dataset(tmp_path).manifest_hash == expected
