# Datasets

The images themselves are **not** in this repo — they are licensed material and
too large to version. This file is the contract they have to satisfy so that
`lorafactory check-dataset` accepts them and the provenance record can pin them.

## Where they go

Everything resolves under a dataset root:

```
LORAFACTORY_DATASETS_DIR   default: /workspace/datasets  (the container mount)
```

Configs name datasets relative to that root (`path: STYLE`), so the same
matched-budget config block works inside the container and on a host checkout.
On a host checkout, point it at wherever the images actually live:

```bash
export LORAFACTORY_DATASETS_DIR=~/datasets
```

## Layout

```
$LORAFACTORY_DATASETS_DIR/
  STYLE/                 # the style axis (base.yaml)
    manifest.csv
    img_0001.png
    img_0001.txt         # caption, same stem as the image
    ...
  OBJ/                   # the objective axis (O-* overlays)
    manifest.csv
    ...
```

One caption `.txt` per image, same stem. `caption_extension = ".txt"` and
`num_repeats = 1` are emitted into the kohya dataset TOML; no augmentation keys
are ever emitted, because augmentation would break the matched-budget design.

## manifest.csv

Header, exactly these eight columns in this order:

```csv
file,sha256,caption,source_url,author,licence,licence_url,acquisition_date
```

| column | meaning |
|---|---|
| `file` | filename relative to the dataset directory |
| `sha256` | sha256 of the image bytes — re-checked by `check-dataset` |
| `caption` | must match the `.txt` file's content |
| `source_url` | where the image came from |
| `author` | attribution |
| `licence` / `licence_url` | licence identifier and its text |
| `acquisition_date` | ISO date the file was obtained |

`check-dataset` re-hashes every image and fails if a `sha256` is stale, so
editing an image after writing the manifest is caught rather than silently
training on different pixels. The sha256 of the manifest file itself becomes
`dataset_manifest_hash` in the provenance record — that is what makes "these
adapters saw identical data" a checkable claim rather than an assertion.

## Size

The matched-budget block trains every adapter on the same data for the same
2000 steps at batch size 1, so the dataset only has to be large enough that the
run is a genuine style/objective fit rather than memorisation of a handful of
images. 30–100 captioned images per axis is the usual working range; the
toolchain does not enforce a count, but `check-dataset` will report what it
found.

## Checking

```bash
lorafactory check-dataset --dataset "$LORAFACTORY_DATASETS_DIR/STYLE"
```

Exit 0 and a printed manifest hash means the dataset is usable.

## Datasets a config can rebuild by itself

The matrix configs describe images acquired by hand — this file is the only
contract they have. A config may instead *declare* its images, by adding a
`source` block to its `dataset` section:

```yaml
dataset:
  name: STYLE
  path: STYLE
  manifest: STYLE/manifest.csv
  resolution: 1024
  source:
    repo: huggan/few-shot-aurora
    revision: ccf645535bc3b5f755d03567374780ae9473d66b   # 40-hex, like the model pins
    parquet: data/train-00000-of-00001.parquet           # or: files: ["dataset/candle/*.jpg"]
    limit: 50
    caption: a photograph in the sks_aurora style
    author: unknown
    licence: unknown
    licence_url: https://huggingface.co/datasets/huggan/few-shot-aurora
    acquisition_date: "2026-08-08"
```

```bash
lorafactory fetch-dataset --config configs/test/smoke.yaml [--dry-run] [--force]
```

writes the `img_NNNN.png` / `img_NNNN.txt` pairs and a `manifest.csv` whose four
provenance columns come from the block above — which is why none of them may be
empty. Selection is `limit` images in **source order**, never a sample, so two
machines fetching the same revision get the same pixels and the same manifest
hash. A directory that already validates is left alone; `--force` rebuilds it.

This is how `configs/test/` works, and it is what lets the container fetch its
own data (`docker compose run --rm lorafactory-test`). It does not replace
hand-curated data for the matrix.
