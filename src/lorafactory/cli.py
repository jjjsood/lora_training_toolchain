"""CLI surface: click group `lorafactory` (pyproject `project.scripts`).

Subcommands: resolve-config, check-budget, check-dataset,
emit-kohya, train, train-matrix, convert, verify-keys, gen-gate-images, gate, synth,
introspect, screen, fetch-models, determinism-check, provenance. Plus two that exist only to
make a first real run reproducible from a checkout: `fetch-dataset` (materialise a
config's declared `dataset.source`) and `test-run` (fetch weights + dataset, then
train — what the container's test service invokes).

Every command is wired to the module that owns its logic; this file only parses
options, joins the pieces and turns a failed report into a non-zero exit. The
four commands that need a GPU or the network (`train`, `train-matrix`,
`gen-gate-images`, `fetch-models`) additionally take `--dry-run`, which performs
every CPU step and prints the resulting plan as JSON on stdout instead of
executing it — no subprocess is spawned, no weights are downloaded, no image is
rendered. That is what makes the whole CPU half of a training run checkable on a
machine with no GPU.

`train --dry-run` writes its kohya TOML pair into a durable run directory
(`--out`, default `$LORAFACTORY_RUNS_DIR` or `./runs`) rather than a temporary
one, because the printed plan names that TOML and a caller must be able to open
it after the command exits.
"""

from __future__ import annotations

import json
import os
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path

import click
from huggingface_hub import hf_hub_download
from safetensors.torch import load_file, save_file

from lorafactory.config.budget import check_matrix
from lorafactory.config.loader import ConfigViolation, ResolvedConfig, resolve
from lorafactory.config.schema import ConfigSchemaError, validate
from lorafactory.convert.keymap import UnconvertibleKeyError
from lorafactory.convert.sd3_kohya_to_diffusers import convert as convert_sd3_to_diffusers
from lorafactory.data.fetch import DatasetSourceError, fetch_dataset, fetch_plan
from lorafactory.data.manifest import MANIFEST_FILENAME, sha256_file
from lorafactory.data.manifest import check_dataset as check_dataset_manifest
from lorafactory.determinism import REQUIRED_ENV, adapters_identical, build_env
from lorafactory.gate.generate import plan_images
from lorafactory.gate.report import GateResult
from lorafactory.gate.run import evaluate_csv, load_gate_config
from lorafactory.introspect.base_cache import BaseSubspaceCache
from lorafactory.introspect.report import introspect_checkpoint, write_config_json, write_csv
from lorafactory.kohya.runner import RunnerConfigError, build_train_command, run_train
from lorafactory.kohya.toml_emitter import emit, engine_for
from lorafactory.models import ModelPathError, download_plan, resolve_dataset_paths
from lorafactory.provenance import REQUIRED_FIELDS, ProvenanceError, write_provenance
from lorafactory.runs import prepare_run
from lorafactory.survey.screen import screen_plan
from lorafactory.synth.synthesiser import synthesise
from lorafactory.verify.key_inventory import verify as verify_key_inventory

#: sd-scripts tag this toolchain is pinned to (see kohya/arg_registry.py).
SD_SCRIPTS_REF = "v0.11.1"

RUNS_DIR_ENV = "LORAFACTORY_RUNS_DIR"
DEFAULT_RUNS_DIRNAME = "runs"

#: `introspect --cache-dir` default override, same pattern as RUNS_DIR_ENV: an
#: explicit env var beats the repo-relative default, and tests redirect it so
#: the suite never writes real cache files into a tracked repo directory.
INTROSPECT_CACHE_DIR_ENV = "LORAFACTORY_INTROSPECT_CACHE_DIR"

_REPO_ROOT = Path(__file__).resolve().parents[2]
_GATE_CONFIG_RELPATH = Path("configs") / "gate" / "e_img.yaml"


def _default_gate_config() -> Path:
    """The repo's E_img gate config, whether run from a checkout or installed."""
    packaged = _REPO_ROOT / _GATE_CONFIG_RELPATH
    return packaged if packaged.exists() else Path(_GATE_CONFIG_RELPATH)


def _default_introspect_cache_root() -> Path:
    """Repo-level root for `introspect`'s shared base-subspace cache.

    `BaseSubspaceCache` writes under `<root>/.cache/introspect/<revision>/`
    (base_cache.py), so this is the *root* passed in, not that full path.
    Defaults to the repo checkout so an introspect matrix over many adapters
    shares one set of base-weight SVDs regardless of each adapter's `--out`;
    ``$LORAFACTORY_INTROSPECT_CACHE_DIR`` overrides it (tests redirect it here
    so the suite never writes cache files into the tracked repo).
    """
    override = os.environ.get(INTROSPECT_CACHE_DIR_ENV)
    return Path(override) if override else _REPO_ROOT


def _resolve_config(config_path: Path) -> ResolvedConfig:
    """Resolve an adapter config and hold it to the schema before anyone uses it.

    Every command reaches its config through here, so this is where a typo'd
    section or a truncated revision has to fail: further down it would travel
    into an emitted kohya TOML, and kohya silently ignores what it cannot use.
    """
    try:
        rc = resolve(config_path)
    except ConfigViolation as exc:
        raise click.ClickException(str(exc)) from exc
    try:
        validate(rc.data)
    except ConfigSchemaError as exc:
        raise click.ClickException(f"{config_path}: {exc}") from exc
    return rc


def _adapter_id(config: dict, config_path: Path) -> str:
    return str(config.get("adapter_id") or config_path.stem)


def _runs_root(explicit: Path | None) -> Path:
    if explicit is not None:
        return Path(explicit)
    return Path(os.environ.get(RUNS_DIR_ENV) or Path.cwd() / DEFAULT_RUNS_DIRNAME)


def _require_usable_dataset(report) -> None:
    """Turn a failed manifest/fetch report into a non-zero exit, errors first."""
    if report.ok:
        return
    for error in report.errors:
        click.echo(error, err=True)
    raise click.ClickException(f"{len(report.errors)} dataset error(s)")


def _launch_training(plan: dict, runs_root: Path | None) -> None:
    """Run one plan from `_plan_train` through the run directory into kohya."""
    handle = prepare_run(_runs_root(runs_root), plan["adapter_id"], plan["config_hash"])
    if handle.action == "skip":
        click.echo(f"{plan['adapter_id']}: already trained in {handle.run_dir}, skipping")
        return

    log_path = handle.run_dir / "train.log"
    returncode = run_train(plan["engine"], Path(plan["train_toml"]), log_path)
    if returncode != 0:
        raise click.ClickException(
            f"{plan['adapter_id']}: kohya exited {returncode} (see {log_path})"
        )
    click.echo(f"{plan['adapter_id']}: training finished in {handle.run_dir}")


def _jsonable(value):
    """JSON-safe view of a plan: Paths become strings, containers recurse."""
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {k: _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    return value


def _echo_json(payload) -> None:
    click.echo(json.dumps(_jsonable(payload), indent=2, sort_keys=True))


@click.group()
@click.version_option(package_name="lorafactory")
def cli():
    """lorafactory: matched-budget SD3/FLUX LoRA training toolchain."""


@cli.command("resolve-config")
@click.argument("config_path", type=click.Path(exists=True, path_type=Path))
def resolve_config(config_path: Path):
    """Resolve an overlay config (extends + overrides_allowed) and print its canonical hash."""
    rc = _resolve_config(config_path)
    click.echo(f"sha256: {rc.sha256}")
    click.echo(json.dumps(rc.data, indent=2, sort_keys=True))


@cli.command("check-budget")
@click.option("--configs", "configs_dir", required=True,
              type=click.Path(exists=True, path_type=Path))
@click.option("--out", "out_path", type=click.Path(path_type=Path), default=None,
              help="Optional path to write the JSON budget report.")
def check_budget(configs_dir: Path, out_path: Path | None):
    """Check the matched-budget matrix: STYLE/O-* sections must agree."""
    report = check_matrix(configs_dir)
    if out_path is not None:
        report.write(out_path)
    if report.ok:
        click.echo(f"budget OK ({len(report.details)} configs checked)")
        return
    for violation in report.violations:
        click.echo(violation, err=True)
    raise click.ClickException(f"{len(report.violations)} budget violation(s)")


@cli.command("check-dataset")
@click.option("--dataset", required=True, type=click.Path(exists=True, path_type=Path))
def check_dataset(dataset: Path):
    """Validate a dataset directory against data/manifest.py's manifest.csv contract."""
    manifest_path = dataset / MANIFEST_FILENAME
    if not manifest_path.exists():
        raise click.ClickException(f"no {MANIFEST_FILENAME} in {dataset}")

    report = check_dataset_manifest(dataset)
    click.echo(f"manifest_hash: {report.manifest_hash}")
    _require_usable_dataset(report)
    click.echo("dataset OK")


@cli.command("emit-kohya")
@click.option("--config", "config_path", required=True,
              type=click.Path(exists=True, path_type=Path))
@click.option("--out", "out_dir", required=True, type=click.Path(path_type=Path))
def emit_kohya(config_path: Path, out_dir: Path):
    """Resolve a config and emit kohya_config.toml + dataset_config.toml into --out."""
    rc = _resolve_config(config_path)
    result = emit(rc.data, out_dir)
    click.echo(str(result.train_toml))
    click.echo(str(result.dataset_toml))


def _plan_train(config_path: Path, out_dir: Path | None, runs_root: Path | None) -> dict:
    """Everything a training launch needs, computed on CPU: TOML pair + argv + env.

    The TOML pair is written to a durable directory (never a temp dir) because
    the plan's ``--config_file`` argument must still resolve after we return.
    """
    rc = _resolve_config(config_path)
    adapter_id = _adapter_id(rc.data, config_path)
    engine = engine_for(rc.data)

    target_dir = Path(out_dir) if out_dir is not None else _runs_root(runs_root) / adapter_id
    result = emit(rc.data, target_dir)

    try:
        command = build_train_command(engine, result.train_toml)
    except RunnerConfigError as exc:
        # Missing LORAFACTORY_KOHYA_PYTHON / _SDSCRIPTS_DIR is a setup mistake,
        # not a crash: --dry-run of all things must say so in one line.
        raise click.ClickException(str(exc)) from exc
    return {
        "adapter_id": adapter_id,
        "config": str(config_path),
        "config_hash": rc.sha256,
        "engine": engine,
        "argv": list(command.argv),
        "cwd": str(command.cwd),
        # The determinism overrides layered on top of the inherited environment.
        "env": {key: command.env[key] for key in REQUIRED_ENV},
        "train_toml": str(result.train_toml),
        "dataset_toml": str(result.dataset_toml),
    }


@cli.command("train")
@click.option("--config", "config_path", required=True,
              type=click.Path(exists=True, path_type=Path))
@click.option("--out", "out_dir", type=click.Path(path_type=Path), default=None,
              help="Directory for the emitted kohya TOML pair (default: <runs>/<adapter_id>).")
@click.option("--runs-dir", "runs_root", type=click.Path(path_type=Path), default=None,
              help=f"Runs root (default: ${RUNS_DIR_ENV} or ./{DEFAULT_RUNS_DIRNAME}).")
@click.option("--dry-run", is_flag=True,
              help="Do every CPU step and print the launch plan as JSON; spawn nothing.")
def train(config_path: Path, out_dir: Path | None, runs_root: Path | None, dry_run: bool):
    """Run a single kohya-ss/sd-scripts training as a subprocess (GPU; --dry-run is CPU-only)."""
    plan = _plan_train(config_path, out_dir, runs_root)
    if dry_run:
        _echo_json(plan)
        return
    _launch_training(plan, runs_root)


@cli.command("train-matrix")
@click.option("--configs", "configs_dir", required=True,
              type=click.Path(exists=True, path_type=Path))
@click.option("--runs-dir", "runs_root", type=click.Path(path_type=Path), default=None,
              help=f"Runs root (default: ${RUNS_DIR_ENV} or ./{DEFAULT_RUNS_DIRNAME}).")
@click.option("--dry-run", is_flag=True,
              help="Print the launch plan of every config in --configs as a JSON list.")
def train_matrix(configs_dir: Path, runs_root: Path | None, dry_run: bool):
    """Run every config in the placement matrix in sequence (GPU; --dry-run is CPU-only)."""
    config_paths = sorted(Path(configs_dir).glob("*.yaml"))
    if not config_paths:
        raise click.ClickException(f"no *.yaml configs in {configs_dir}")

    plans = [_plan_train(path, None, runs_root) for path in config_paths]
    if dry_run:
        _echo_json(plans)
        return

    for plan in plans:
        _launch_training(plan, runs_root)


@cli.command("convert")
@click.option("--in", "src_path", required=True, type=click.Path(exists=True, path_type=Path))
@click.option("--out", "out_path", required=True, type=click.Path(path_type=Path))
def convert(src_path: Path, out_path: Path):
    """Convert a kohya-layout SD3 LoRA checkpoint to diffusers/PEFT layout."""
    kohya_sd = load_file(str(src_path))
    try:
        converted = convert_sd3_to_diffusers(kohya_sd)
    except UnconvertibleKeyError as exc:
        raise click.ClickException(f"unconvertible key: {exc}") from exc

    out_path.parent.mkdir(parents=True, exist_ok=True)
    save_file(converted, str(out_path))
    click.echo(f"{len(converted)} tensors -> {out_path}")


@cli.command("verify-keys")
@click.option("--checkpoint", required=True, type=click.Path(exists=True, path_type=Path))
@click.option("--config", "config_path", required=True,
              type=click.Path(exists=True, path_type=Path),
              help="Adapter config whose `target` section defines the expected inventory.")
def verify_keys(checkpoint: Path, config_path: Path):
    """Verify a converted checkpoint's module-name inventory against target selection."""
    rc = _resolve_config(config_path)
    target = rc.data.get("target")
    if not target:
        raise click.ClickException(f"{config_path} has no `target` section")

    sd = load_file(str(checkpoint))
    arch = rc.data["model"]["arch"]
    report = verify_key_inventory(sd, target, arch=arch)

    click.echo(f"missing: {len(report.missing)}")
    click.echo(f"unexpected: {len(report.unexpected)}")
    click.echo(f"rank mismatches: {len(report.rank_mismatches)}")
    for name in sorted(report.missing):
        click.echo(f"missing module: {name}", err=True)
    for name in sorted(report.unexpected):
        click.echo(f"unexpected module: {name}", err=True)
    for name, rank in sorted(report.rank_mismatches.items()):
        click.echo(f"rank mismatch: {name} has rank {rank}", err=True)

    if not report.ok:
        raise click.ClickException("key inventory does not match the config target")
    click.echo("key inventory OK")


@cli.command("gen-gate-images")
@click.option("--checkpoint", required=True, type=click.Path(exists=True, path_type=Path))
@click.option("--out", "out_dir", required=True, type=click.Path(path_type=Path))
@click.option("--gate-config", "gate_config_path", type=click.Path(path_type=Path),
              default=None, help="Gate config (default: configs/gate/e_img.yaml).")
@click.option("--dry-run", is_flag=True,
              help="Print the (prompt x seed x arm) grid as JSON; render nothing.")
def gen_gate_images(checkpoint: Path, out_dir: Path, gate_config_path: Path | None,
                    dry_run: bool):
    """Generate the E_img gate's LoRA-arm/null image pairs (GPU; --dry-run is CPU-only)."""
    cfg = load_gate_config(gate_config_path or _default_gate_config())
    plan = plan_images(cfg, checkpoint=Path(checkpoint).stem, out_dir=out_dir)
    if dry_run:
        _echo_json({"checkpoint": str(checkpoint), "out": str(out_dir), "images": plan})
        return
    raise click.ClickException(
        "gen-gate-images needs a GPU pipeline; run with --dry-run to inspect the "
        f"{len(plan)}-cell grid it would render"
    )


@cli.command("gate")
@click.option("--manifest", required=True, type=click.Path(exists=True, path_type=Path))
@click.option("--out", "out_path", required=True, type=click.Path(path_type=Path),
              help="Where to write the JSON gate report.")
@click.option("--gate-config", "gate_config_path", type=click.Path(path_type=Path),
              default=None, help="Gate config (default: configs/gate/e_img.yaml).")
def gate(manifest: Path, out_path: Path, gate_config_path: Path | None):
    """Compute the E_img success gate (LPIPS + CLIP distance, Cliff's delta) from a manifest."""
    gate_config = Path(gate_config_path or _default_gate_config())
    cfg = load_gate_config(gate_config)
    result: GateResult = evaluate_csv(manifest, cfg)

    payload = {
        "manifest": str(manifest),
        "gate_config": str(gate_config),
        "verdict": result.verdict,
        "null_sanity_ok": result.null_sanity_ok,
        "prompts": result.prompts,
    }
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, indent=2, sort_keys=True))

    # A "fail" verdict is a legitimate answer, not a CLI error: the caller reads
    # the report. Only a broken input raises.
    click.echo(f"verdict: {result.verdict} (null_sanity_ok={result.null_sanity_ok})")
    click.echo(str(out_path))


@cli.command("synth")
@click.option("--reference", required=True, type=click.Path(exists=True, path_type=Path))
@click.option("--out", "out_path", required=True, type=click.Path(path_type=Path))
@click.option("--seed", type=int, default=0, show_default=True,
              help="Seed for the Gaussian draw; a fixed seed is reproducible.")
@click.option("--tolerance", type=float, default=1e-4, show_default=True)
def synth(reference: Path, out_path: Path, seed: int, tolerance: float):
    """Synthesise a norm-matched random-control adapter from a reference checkpoint."""
    reference_sd = load_file(str(reference))
    out_sd, recipe = synthesise(reference_sd, seed=seed, tolerance=tolerance)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    save_file(out_sd, str(out_path))
    recipe_path = out_path.with_suffix(".recipe.json")
    recipe_path.write_text(json.dumps(recipe, indent=2, sort_keys=True))

    click.echo(f"{len(recipe['modules'])} modules -> {out_path}")
    click.echo(str(recipe_path))


#: Filenames `introspect` writes into its --out directory.
INTROSPECT_CSV = "introspect.csv"
INTROSPECT_CONFIG = "introspect.config.json"


@cli.command("introspect")
@click.argument("checkpoint", type=click.Path(exists=True, path_type=Path))
@click.option("--out", "out_dir", required=True, type=click.Path(path_type=Path),
              help="Directory for introspect.csv + introspect.config.json.")
@click.option("--base", "base_checkpoint", default=None,
              type=click.Path(exists=True, path_type=Path),
              help="Base checkpoint for the intruder statistic (SD3; omit to leave it empty).")
@click.option("--base-revision", default="", help="Revision the --base weights are pinned to; "
              "keys the subspace cache, so a different base never reuses entries.")
@click.option("--cache-dir", "cache_dir", default=None,
              type=click.Path(path_type=Path),
              help="Root for the shared base-subspace cache, independent of --out so an "
              "introspect matrix reuses one adapter's SVDs across every other adapter's "
              f"--out directory (default: <repo>/.cache/introspect; override via "
              f"${INTROSPECT_CACHE_DIR_ENV}).")
@click.option("--k", type=int, default=10, show_default=True,
              help="How many singular values/base directions to report per module.")
@click.option("--tau", type=float, default=0.5, show_default=True,
              help="Overlap below which an adapter direction counts as an intruder.")
@click.option("--adapter-id", default=None,
              help="Value of the CSV's `adapter` column (default: the checkpoint's stem).")
def introspect(checkpoint: Path, out_dir: Path, base_checkpoint: Path | None,  # noqa: PLR0913, PLR0917 - a click command's parameter list *is* its CLI surface
               base_revision: str, cache_dir: Path | None, k: int, tau: float,
               adapter_id: str | None):
    """Per-module spectra of one adapter: norms, effective rank, top sigmas, intruders.

    CPU-only and offline — the checkpoint is streamed tensor by tensor and
    ΔW is never materialised. One explicit checkpoint path, no enumeration.
    """
    if k < 1:
        raise click.ClickException("--k must be at least 1")
    if base_checkpoint is not None and not base_revision:
        # The subspace cache is keyed by revision. A placeholder revision would
        # let a second run against different weights silently reuse the first
        # run's subspaces, i.e. report an intruder count for the wrong model.
        raise click.ClickException(
            "--base requires --base-revision: the subspace cache is keyed by it, "
            "and an unnamed base would let two different checkpoints share a cache"
        )

    base_cache = None
    if base_checkpoint is not None or base_revision:
        cache_root = cache_dir if cache_dir is not None else _default_introspect_cache_root()
        base_cache = BaseSubspaceCache(
            cache_root, base_revision, base_checkpoint=base_checkpoint
        )

    try:
        report = introspect_checkpoint(checkpoint, base_cache=base_cache, k=k, tau=tau)
    except ValueError as exc:
        raise click.ClickException(f"{checkpoint}: {exc}") from exc

    out_dir = Path(out_dir)
    csv_path = out_dir / INTROSPECT_CSV
    config_path = out_dir / INTROSPECT_CONFIG
    write_csv(report, csv_path, adapter_id or Path(checkpoint).stem)
    write_config_json(report, config_path)

    click.echo(f"{len(report.rows)} modules ({report.layout} layout) -> {csv_path}")
    if base_cache is not None:
        matched = report.intruder_matched
        unmatched = report.intruder_unmatched
        click.echo(f"intruder subspaces: {matched} matched, {unmatched} unmatched")
        if unmatched:
            examples = ", ".join(report.unmatched_modules(limit=3))
            click.echo(
                f"{unmatched} module(s) have no base subspace (intruder columns "
                f"empty), first: {examples}",
                err=True,
            )
    click.echo(str(config_path))


@cli.command("screen")
@click.option("--config", "config_path", required=True,
              type=click.Path(exists=True, path_type=Path))
def screen(config_path: Path):
    """Screen a downloaded community checkpoint through the E_img gate."""
    plan = screen_plan(config_path)
    _echo_json(plan)


@cli.command("fetch-models")
@click.option("--config", "config_path", required=True,
              type=click.Path(exists=True, path_type=Path))
@click.option("--dry-run", is_flag=True,
              help="Print exactly what would be downloaded (repo/revision/file) as JSON.")
def fetch_models(config_path: Path, dry_run: bool):
    """Pull pinned base/text-encoder weights via huggingface_hub (network; --dry-run is offline)."""
    rc = _resolve_config(config_path)
    try:
        files = download_plan(rc.data)
    except ModelPathError as exc:
        raise click.ClickException(str(exc)) from exc

    if dry_run:
        _echo_json({"adapter_id": _adapter_id(rc.data, config_path), "files": files})
        return

    _download_weights(files)


def _local_dir_for(local_path: Path, filename: str) -> Path:
    """The `local_dir` under which hf_hub_download lands `filename` on `local_path`.

    hf_hub_download recreates the repo-relative path below `local_dir`, so the
    root is `local_path` with as many components stripped as `filename` has —
    not `parent.parent`, which only happens to be right at one level of nesting.
    """
    root = local_path
    for _ in Path(filename).parts:
        root = root.parent
    return root


def _download_weights(files: list[dict]) -> None:
    """Pull every entry of a `download_plan` to its resolved local path."""
    for entry in files:
        local_path = Path(entry["local_path"])
        local_path.parent.mkdir(parents=True, exist_ok=True)
        hf_hub_download(
            repo_id=entry["repo_id"],
            revision=entry["revision"],
            filename=entry["filename"],
            local_dir=str(_local_dir_for(local_path, str(entry["filename"]))),
        )
        click.echo(f"fetched {entry['filename']} @ {entry['revision'][:8]}")


@cli.command("fetch-dataset")
@click.option("--config", "config_path", required=True,
              type=click.Path(exists=True, path_type=Path))
@click.option("--force", is_flag=True,
              help="Rebuild even if the directory already validates.")
@click.option("--dry-run", is_flag=True,
              help="Print the declared source (repo/revision/target) as JSON; fetch nothing.")
def fetch_dataset_cmd(config_path: Path, force: bool, dry_run: bool):
    """Materialise a config's `dataset.source` into a manifest-checked image directory."""
    rc = _resolve_config(config_path)
    try:
        if dry_run:
            _echo_json({"adapter_id": _adapter_id(rc.data, config_path),
                        "dataset": fetch_plan(rc.data)})
            return
        report = fetch_dataset(rc.data, force=force)
    except DatasetSourceError as exc:
        raise click.ClickException(str(exc)) from exc

    click.echo(f"{report.action}: {report.count} images in {report.image_dir}")
    click.echo(f"manifest_hash: {report.manifest_hash}")
    _require_usable_dataset(report)


@cli.command("test-run")
@click.option("--config", "config_path", required=True,
              type=click.Path(exists=True, path_type=Path))
@click.option("--runs-dir", "runs_root", type=click.Path(path_type=Path), default=None,
              help=f"Runs root (default: ${RUNS_DIR_ENV} or ./{DEFAULT_RUNS_DIRNAME}).")
@click.option("--dry-run", is_flag=True,
              help="Print the whole plan (weights + dataset + launch) as JSON; do nothing.")
def test_run(config_path: Path, runs_root: Path | None, dry_run: bool):
    """One-shot: fetch the pinned weights and the declared dataset, then train.

    This is the entry point the container's test service runs. It only composes
    commands that already exist, so a green `test-run` says nothing more than
    `fetch-models` + `fetch-dataset` + `train` would have said separately.
    """
    rc = _resolve_config(config_path)
    try:
        weights = download_plan(rc.data)
        dataset = fetch_plan(rc.data)
    except (ModelPathError, DatasetSourceError) as exc:
        raise click.ClickException(str(exc)) from exc

    if dry_run:
        _echo_json({
            "weights": weights,
            "dataset": dataset,
            "train": _plan_train(config_path, None, runs_root),
        })
        return

    _download_weights(weights)

    try:
        report = fetch_dataset(rc.data)
    except DatasetSourceError as exc:
        raise click.ClickException(str(exc)) from exc
    _require_usable_dataset(report)
    click.echo(f"dataset {report.action}: {report.count} images, "
               f"manifest_hash {report.manifest_hash}")

    _launch_training(_plan_train(config_path, None, runs_root), runs_root)


@cli.command("determinism-check")
@click.option("--a", "checkpoint_a", required=True, type=click.Path(exists=True, path_type=Path))
@click.option("--b", "checkpoint_b", required=True, type=click.Path(exists=True, path_type=Path))
def determinism_check(checkpoint_a: Path, checkpoint_b: Path):
    """Tensor-exact compare two safetensors checkpoints (determinism.adapters_identical)."""
    if adapters_identical(checkpoint_a, checkpoint_b):
        click.echo("identical")
        return
    raise click.ClickException(f"{checkpoint_a} and {checkpoint_b} differ")


def _package_versions() -> dict:
    versions = {}
    for package in ("lorafactory", "torch", "safetensors", "diffusers",
                    "transformers", "accelerate"):
        try:
            versions[package] = version(package)
        except PackageNotFoundError:
            continue
    return versions


def _dataset_manifest_hash(config: dict) -> str:
    """The dataset's manifest hash when the dataset is present on this machine."""
    try:
        paths = resolve_dataset_paths(config)
    except ModelPathError:
        return ""
    if not (paths.image_dir / MANIFEST_FILENAME).exists():
        return ""
    return check_dataset_manifest(paths.image_dir).manifest_hash


def _find_adapter(run_dir: Path, adapter_id: str) -> Path:
    candidates = sorted(run_dir.rglob("*.safetensors"))
    if not candidates:
        raise click.ClickException(f"no .safetensors adapter under {run_dir}")
    for candidate in candidates:
        if candidate.stem == adapter_id:
            return candidate
    return candidates[0]


@cli.command("provenance")
@click.option("--run-dir", required=True, type=click.Path(exists=True, path_type=Path))
@click.option("--config", "config_path", required=True,
              type=click.Path(exists=True, path_type=Path),
              help="Adapter config the run was trained from (supplies the config hash).")
def provenance(run_dir: Path, config_path: Path):
    """Emit the provenance record for a completed run directory."""
    rc = _resolve_config(config_path)
    config = rc.data
    adapter_id = _adapter_id(config, config_path)
    adapter_path = _find_adapter(run_dir, adapter_id)

    model = config.get("model") or {}
    dataset = config.get("dataset") or {}
    metrics_path = run_dir / "metrics.json"
    metrics = json.loads(metrics_path.read_text()) if metrics_path.exists() else {}

    record = {
        "run_id": run_dir.name,
        "adapter_id": adapter_id,
        "config_hash": rc.sha256,
        "seed": (config.get("train") or {}).get("seed"),
        "dataset_name": dataset.get("name", ""),
        "dataset_manifest_hash": _dataset_manifest_hash(config),
        "base_model_repo": model.get("train_repo", ""),
        "base_model_revision": model.get("train_revision", ""),
        "sd_scripts_ref": SD_SCRIPTS_REF,
        "versions": _package_versions(),
        "determinism_env": build_env({}),
        "wall_clock_s": metrics.get("wall_clock_s"),
        "final_loss": metrics.get("final_loss"),
        "adapter_sha256": sha256_file(adapter_path),
    }
    missing = [field for field in REQUIRED_FIELDS if field not in record]
    if missing:
        raise click.ClickException(f"provenance record missing {missing}")

    try:
        out = write_provenance(run_dir, record)
    except ProvenanceError as exc:
        raise click.ClickException(str(exc)) from exc
    click.echo(str(out))


if __name__ == "__main__":
    cli()
