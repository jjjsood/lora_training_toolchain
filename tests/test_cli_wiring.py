"""Every CLI subcommand must do its job, not merely appear in `--help`.

test_cli.py only asserts that the command names are listed, which a body of
`raise click.ClickException("<cmd>: not implemented yet")` would satisfy.

These tests run the commands for real, on CPU, against fixtures built here:
seven do actual work, four are GPU/network commands that must produce a
complete `--dry-run` plan (everything except the execution step itself).

Contracts pinned here beyond the bare command list:
  * `verify-keys` and `provenance` take `--config` in addition to their input,
    because `verify(sd, target)` / `write_provenance(run_dir, record)` cannot
    be called without the resolved config.
  * GPU/network commands take `--dry-run` and print JSON to stdout.
  * Model files are located under `LORAFACTORY_MODELS_DIR`; configs name files
    relative to it, and the emitter/runner resolve them to absolute paths.
"""

import csv
import json

import torch
from click.testing import CliRunner
from safetensors.torch import save_file

from conftest import CONFIGS, MATRIX, peft_module
from lorafactory.cli import cli

GATE_CSV_COLUMNS = ["checkpoint", "prompt", "seed", "arm", "lpips", "clip_distance"]

PROMPTS = ["prompt a", "prompt b", "prompt c"]
SEEDS = list(range(10))

# Symmetric around the split point, so the null arm's first/second half are the
# same multiset and null-sanity Cliff's delta is exactly 0 — the fixture tests
# CLI wiring, not the gate statistics (test_gate_report.py owns those).
NULL_LPIPS = [0.08, 0.12, 0.09, 0.11, 0.10, 0.10, 0.11, 0.09, 0.12, 0.08]
NULL_CLIP = [0.04, 0.06, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.06, 0.04]


def run(*args):
    return CliRunner().invoke(cli, list(args))


def write_peft_checkpoint(path, names, rank=4, dim=8, seed=0):
    sd = {}
    for i, name in enumerate(names):
        sd.update(peft_module(name, rank, dim, dim, seed=seed + i))
    save_file(sd, str(path))
    return sd


def make_dataset(root):
    """Minimal but valid dataset: images + captions + a correct manifest.csv."""
    root.mkdir(parents=True, exist_ok=True)
    rows = []
    for i in range(3):
        img = root / f"img_{i}.png"
        img.write_bytes(f"not-a-real-png-{i}".encode())
        (root / f"img_{i}.txt").write_text(f"caption {i}")
        import hashlib
        rows.append({
            "file": img.name,
            "sha256": hashlib.sha256(img.read_bytes()).hexdigest(),
            "caption": f"caption {i}",
            "source_url": f"https://example.invalid/{i}",
            "author": "test",
            "licence": "CC0",
            "licence_url": "https://creativecommons.org/publicdomain/zero/1.0/",
            "acquisition_date": "2026-01-01",
        })
    with open(root / "manifest.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=[
            "file", "sha256", "caption", "source_url", "author",
            "licence", "licence_url", "acquisition_date"])
        w.writeheader()
        w.writerows(rows)
    return root


def make_gate_csv(path, checkpoint="L-F"):
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=GATE_CSV_COLUMNS)
        w.writeheader()
        for prompt in PROMPTS:
            for i, seed in enumerate(SEEDS):
                w.writerow({
                    "checkpoint": checkpoint, "prompt": prompt, "seed": seed,
                    "arm": "lora", "lpips": 0.40 + i * 0.01,
                    "clip_distance": 0.30 + i * 0.01,
                })
                w.writerow({
                    "checkpoint": checkpoint, "prompt": prompt, "seed": seed,
                    "arm": "null", "lpips": NULL_LPIPS[i],
                    "clip_distance": NULL_CLIP[i],
                })
    return path


def fake_model_root(tmp_path, monkeypatch):
    """A model root with empty placeholder files — no 15 GB download needed."""
    root = tmp_path / "models"
    (root / "text_encoders").mkdir(parents=True, exist_ok=True)
    for name in ("sd3_medium.safetensors", "text_encoders/clip_l.safetensors",
                 "text_encoders/clip_g.safetensors",
                 "text_encoders/t5xxl_fp16.safetensors"):
        (root / name).touch()
    monkeypatch.setenv("LORAFACTORY_MODELS_DIR", str(root))
    return root


def fake_kohya_env(tmp_path, monkeypatch):
    py = tmp_path / "venv-kohya" / "bin" / "python"
    sd_scripts = tmp_path / "sd-scripts"
    py.parent.mkdir(parents=True, exist_ok=True)
    sd_scripts.mkdir(parents=True, exist_ok=True)
    py.touch()
    (sd_scripts / "sd3_train_network.py").touch()
    (sd_scripts / "flux_train_network.py").touch()
    monkeypatch.setenv("LORAFACTORY_KOHYA_PYTHON", str(py))
    monkeypatch.setenv("LORAFACTORY_SDSCRIPTS_DIR", str(sd_scripts))


# --------------------------------------------------------------------------
# The seven commands that do real CPU work.
# --------------------------------------------------------------------------

def test_check_dataset_accepts_a_valid_dataset(tmp_path):
    ds = make_dataset(tmp_path / "STYLE")
    result = run("check-dataset", "--dataset", str(ds))
    assert result.exit_code == 0, result.output
    # The manifest hash is the provenance anchor — it must be reported.
    assert any(len(t) == 64 for t in result.output.split()), result.output


def test_check_dataset_rejects_a_tampered_manifest(tmp_path):
    ds = make_dataset(tmp_path / "STYLE")
    (ds / "img_0.png").write_bytes(b"changed after hashing")
    result = run("check-dataset", "--dataset", str(ds))
    assert result.exit_code != 0, "a stale sha256 must fail the check"


def test_convert_writes_a_diffusers_checkpoint(tmp_path):
    from conftest import kohya_module
    sd = {}
    for i in range(2):
        p = f"lora_unet_joint_blocks_{i}_x_block_attn_proj"
        sd.update(kohya_module(p, 4, 8, 8, 4.0, seed=i))
    src = tmp_path / "kohya.safetensors"
    save_file(sd, str(src))

    out = tmp_path / "diffusers.safetensors"
    result = run("convert", "--in", str(src), "--out", str(out))
    assert result.exit_code == 0, result.output
    assert out.exists()

    from safetensors.torch import load_file
    converted = load_file(str(out))
    assert any(k.endswith(".lora_A.weight") for k in converted)
    assert not any(k.endswith(".alpha") for k in converted), "alpha must be folded"


def test_convert_rejects_an_unconvertible_key(tmp_path):
    src = tmp_path / "bad.safetensors"
    save_file({"lora_te3_something.lora_down.weight": torch.zeros(2, 2),
               "lora_te3_something.lora_up.weight": torch.zeros(2, 2)}, str(src))
    result = run("convert", "--in", str(src), "--out", str(tmp_path / "o.safetensors"))
    assert result.exit_code != 0, "lora_te3 has no diffusers path — must fail loudly"


def test_verify_keys_reports_against_the_config_target(tmp_path):
    from lorafactory.config.loader import resolve
    from lorafactory.verify.key_inventory import expected_module_names

    target = resolve(MATRIX / "L-E.yaml").data["target"]
    names = sorted(expected_module_names(target))[:4]
    ckpt = tmp_path / "adapter.safetensors"
    write_peft_checkpoint(ckpt, names)

    result = run("verify-keys", "--checkpoint", str(ckpt),
                 "--config", str(MATRIX / "L-E.yaml"))
    # A partial checkpoint is a real mismatch, so a non-zero exit is correct —
    # what matters is that it reports the inventory rather than crashing.
    assert "missing" in result.output.lower(), result.output


def test_verify_keys_passes_on_a_complete_checkpoint(tmp_path):
    from lorafactory.config.loader import resolve
    from lorafactory.verify.key_inventory import expected_module_names

    target = resolve(MATRIX / "L-E.yaml").data["target"]
    ckpt = tmp_path / "full.safetensors"
    write_peft_checkpoint(ckpt, sorted(expected_module_names(target)),
                          rank=target["rank"])

    result = run("verify-keys", "--checkpoint", str(ckpt),
                 "--config", str(MATRIX / "L-E.yaml"))
    assert result.exit_code == 0, result.output


def test_verify_keys_passes_flux_arch_from_config(tmp_path):
    """`verify-keys` must dispatch on `model.arch` from the config, not
    default to SD3, or a complete F-F (FLUX) checkpoint would misreport."""
    from lorafactory.config.loader import resolve
    from lorafactory.verify.key_inventory import expected_module_names

    target = resolve(MATRIX / "F-F.yaml").data["target"]
    names = sorted(expected_module_names(target, arch="flux"))
    ckpt = tmp_path / "full.safetensors"
    write_peft_checkpoint(ckpt, names, rank=target["rank"])

    result = run("verify-keys", "--checkpoint", str(ckpt),
                 "--config", str(MATRIX / "F-F.yaml"))
    assert result.exit_code == 0, result.output
    assert "key inventory OK" in result.output


def test_verify_keys_flux_partial_checkpoint_reports_missing(tmp_path):
    from lorafactory.config.loader import resolve
    from lorafactory.verify.key_inventory import expected_module_names

    target = resolve(MATRIX / "F-F.yaml").data["target"]
    names = sorted(expected_module_names(target, arch="flux"))[:4]
    ckpt = tmp_path / "adapter.safetensors"
    write_peft_checkpoint(ckpt, names)

    result = run("verify-keys", "--checkpoint", str(ckpt),
                 "--config", str(MATRIX / "F-F.yaml"))
    assert result.exit_code != 0
    assert "missing" in result.output.lower()


def test_verify_keys_passes_a_genuine_kohya_layout_flux_checkpoint(tmp_path):
    """The WS3/WS4 seam. `train`/`train-matrix` write kohya-layout FLUX
    checkpoints and nothing else (T4: WAIVER — `convert` stays SD3-only), so
    this is the only FLUX artifact `verify-keys` is ever actually pointed at
    outside a test. Every FLUX fixture above builds a diffusers-layout dict
    directly (no producer); this one builds the real kohya key grammar an
    F-F run would write, at the full pinned block count, and checks the
    bridge in `key_inventory.verify` converts it and passes clean."""
    from conftest import kohya_module
    from lorafactory.config.loader import resolve
    from lorafactory.constants import FLUX_NUM_DOUBLE_BLOCKS, FLUX_NUM_SINGLE_BLOCKS

    target = resolve(MATRIX / "F-F.yaml").data["target"]
    rank = target["rank"]

    hidden = 32  # dimension-agnostic leaves: any width converts correctly
    real_dim = 3072  # single-stream linear1: diffusers hardcodes this split width
    real_mlp = 4 * real_dim

    double_leaves = [
        ("img_attn_qkv", 3 * hidden, hidden),
        ("img_attn_proj", hidden, hidden),
        ("txt_attn_qkv", 3 * hidden, hidden),
        ("txt_attn_proj", hidden, hidden),
        ("img_mlp_0", 4 * hidden, hidden),
        ("img_mlp_2", hidden, 4 * hidden),
        ("txt_mlp_0", 4 * hidden, hidden),
        ("txt_mlp_2", hidden, 4 * hidden),
    ]

    sd = {}
    seed = 0
    for i in range(FLUX_NUM_DOUBLE_BLOCKS):
        prefix = f"lora_unet_double_blocks_{i}"
        for leaf, out_dim, in_dim in double_leaves:
            sd.update(kohya_module(
                f"{prefix}_{leaf}", rank, out_dim, in_dim, float(rank), seed=seed
            ))
            seed += 1
    for i in range(FLUX_NUM_SINGLE_BLOCKS):
        prefix = f"lora_unet_single_blocks_{i}"
        sd.update(kohya_module(
            f"{prefix}_linear1", rank, 3 * real_dim + real_mlp, real_dim, float(rank), seed=seed
        ))
        seed += 1
        sd.update(kohya_module(
            f"{prefix}_linear2", rank, hidden, 5 * hidden, float(rank), seed=seed
        ))
        seed += 1

    ckpt = tmp_path / "kohya_full.safetensors"
    save_file(sd, str(ckpt))

    result = run("verify-keys", "--checkpoint", str(ckpt), "--config", str(MATRIX / "F-F.yaml"))
    assert result.exit_code == 0, result.output
    assert "key inventory OK" in result.output


def test_gate_computes_a_verdict_from_a_results_csv(tmp_path):
    csv_path = make_gate_csv(tmp_path / "results.csv")
    out = tmp_path / "gate.json"
    result = run("gate", "--manifest", str(csv_path), "--out", str(out))
    assert result.exit_code == 0, result.output

    report = json.loads(out.read_text())
    assert report["verdict"] == "pass", report
    assert report["null_sanity_ok"] is True, report


def test_gate_fails_when_the_lora_arm_is_indistinguishable(tmp_path):
    csv_path = tmp_path / "flat.csv"
    with open(csv_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=GATE_CSV_COLUMNS)
        w.writeheader()
        for prompt in PROMPTS:
            for i, seed in enumerate(SEEDS):
                for arm in ("lora", "null"):
                    w.writerow({
                        "checkpoint": "X", "prompt": prompt, "seed": seed,
                        "arm": arm, "lpips": NULL_LPIPS[i],
                        "clip_distance": NULL_CLIP[i],
                    })
    out = tmp_path / "gate.json"
    result = run("gate", "--manifest", str(csv_path), "--out", str(out))
    # A "fail" verdict is an answer, not a CLI error: the report is the output.
    assert result.exit_code == 0, result.output
    report = json.loads(out.read_text())
    assert report["verdict"] == "fail", report


def test_synth_matches_the_reference_norm(tmp_path):
    from safetensors.torch import load_file

    from lorafactory.synth.norms import lora_frobenius_norm

    ref = tmp_path / "ref.safetensors"
    ref_sd = write_peft_checkpoint(ref, ["transformer.transformer_blocks.0.attn.to_q"])
    out = tmp_path / "random.safetensors"

    result = run("synth", "--reference", str(ref), "--out", str(out))
    assert result.exit_code == 0, result.output

    got = load_file(str(out))
    name = "transformer.transformer_blocks.0.attn.to_q"
    ref_norm = lora_frobenius_norm(ref_sd[f"{name}.lora_A.weight"],
                                   ref_sd[f"{name}.lora_B.weight"])
    got_norm = lora_frobenius_norm(got[f"{name}.lora_A.weight"],
                                   got[f"{name}.lora_B.weight"])
    assert abs(got_norm - ref_norm) / ref_norm < 1e-3


def test_synth_is_reproducible_for_a_fixed_seed(tmp_path):
    from safetensors.torch import load_file
    ref = tmp_path / "ref.safetensors"
    write_peft_checkpoint(ref, ["transformer.transformer_blocks.0.attn.to_q"])

    outs = []
    for i in (1, 2):
        out = tmp_path / f"r{i}.safetensors"
        assert run("synth", "--reference", str(ref), "--out", str(out),
                   "--seed", "1234").exit_code == 0
        outs.append(load_file(str(out)))
    for key in outs[0]:
        assert torch.equal(outs[0][key], outs[1][key]), key


def test_determinism_check_distinguishes_identical_from_different(tmp_path):
    a = tmp_path / "a.safetensors"
    b = tmp_path / "b.safetensors"
    c = tmp_path / "c.safetensors"
    names = ["transformer.transformer_blocks.0.attn.to_q"]
    write_peft_checkpoint(a, names, seed=1)
    write_peft_checkpoint(b, names, seed=1)
    write_peft_checkpoint(c, names, seed=99)

    assert run("determinism-check", "--a", str(a), "--b", str(b)).exit_code == 0
    assert run("determinism-check", "--a", str(a), "--b", str(c)).exit_code != 0


def test_provenance_writes_a_complete_record(tmp_path, monkeypatch):
    from lorafactory.provenance import PROVENANCE_FILENAME, REQUIRED_FIELDS

    fake_model_root(tmp_path, monkeypatch)
    run_dir = tmp_path / "runs" / "L-E-a01"
    run_dir.mkdir(parents=True)
    (run_dir / "L-E.safetensors").write_bytes(b"adapter bytes")

    result = run("provenance", "--run-dir", str(run_dir),
                 "--config", str(MATRIX / "L-E.yaml"))
    assert result.exit_code == 0, result.output

    record = json.loads((run_dir / PROVENANCE_FILENAME).read_text())
    missing = [f for f in REQUIRED_FIELDS if f not in record]
    assert not missing, f"provenance record missing {missing}"
    assert record["adapter_id"] == "L-E"
    assert len(record["config_hash"]) == 64


# --------------------------------------------------------------------------
# The four GPU/network commands: --dry-run must produce a complete plan.
# --------------------------------------------------------------------------

def test_train_dry_run_emits_a_launch_plan(tmp_path, monkeypatch):
    fake_model_root(tmp_path, monkeypatch)
    fake_kohya_env(tmp_path, monkeypatch)

    result = run("train", "--config", str(MATRIX / "L-E.yaml"),
                 "--runs-dir", str(tmp_path / "runs"), "--dry-run")
    assert result.exit_code == 0, result.output

    plan = json.loads(result.output)
    assert any(a.endswith("sd3_train_network.py") for a in plan["argv"]), plan
    assert "--config_file" in plan["argv"]
    assert plan["env"]["CUBLAS_WORKSPACE_CONFIG"] == ":4096:8"
    assert plan["env"]["PYTHONHASHSEED"] == "0"


def test_train_dry_run_spawns_nothing(tmp_path, monkeypatch):
    """A dry run must not reach subprocess at all — that is what makes it safe
    to call from the check suite on a machine with no GPU."""
    import subprocess

    fake_model_root(tmp_path, monkeypatch)
    fake_kohya_env(tmp_path, monkeypatch)

    def explode(*args, **kwargs):
        raise AssertionError("--dry-run must not spawn a process")

    monkeypatch.setattr(subprocess, "run", explode)
    monkeypatch.setattr(subprocess, "Popen", explode)
    assert run("train", "--config", str(MATRIX / "L-E.yaml"),
               "--runs-dir", str(tmp_path / "runs"),
               "--dry-run").exit_code == 0


def test_train_matrix_dry_run_covers_every_adapter(tmp_path, monkeypatch):
    from conftest import ALL_MATRIX_IDS

    fake_model_root(tmp_path, monkeypatch)
    fake_kohya_env(tmp_path, monkeypatch)

    result = run("train-matrix", "--configs", str(MATRIX),
                 "--runs-dir", str(tmp_path / "runs"), "--dry-run")
    assert result.exit_code == 0, result.output
    plans = json.loads(result.output)
    assert {p["adapter_id"] for p in plans} == set(ALL_MATRIX_IDS)


def test_fetch_models_dry_run_names_files_and_revisions(tmp_path, monkeypatch):
    fake_model_root(tmp_path, monkeypatch)
    result = run("fetch-models", "--config", str(MATRIX / "L-F.yaml"), "--dry-run")
    assert result.exit_code == 0, result.output

    plan = json.loads(result.output)
    names = {f["filename"] for f in plan["files"]}
    assert "sd3_medium.safetensors" in names, names
    assert {"text_encoders/clip_l.safetensors",
            "text_encoders/clip_g.safetensors",
            "text_encoders/t5xxl_fp16.safetensors"} <= names, names
    for entry in plan["files"]:
        assert len(entry["revision"]) == 40, entry
        assert entry["repo_id"], entry


def test_fetch_models_gated_repo_gives_a_clean_diagnosis_not_a_traceback(tmp_path, monkeypatch):
    import httpx
    from huggingface_hub.errors import GatedRepoError

    fake_model_root(tmp_path, monkeypatch)
    response = httpx.Response(403, request=httpx.Request("GET", "https://huggingface.co"))

    def _raise_gated(**kwargs):
        raise GatedRepoError("403 Client Error", response=response)

    monkeypatch.setattr("lorafactory.cli.hf_hub_download", _raise_gated)
    result = run("fetch-models", "--config", str(MATRIX / "L-F.yaml"))

    assert result.exit_code != 0
    assert "Traceback" not in result.output
    assert "gated" in result.output.lower()
    assert "HF_TOKEN" in result.output


def test_fetch_models_missing_file_gives_a_clean_diagnosis(tmp_path, monkeypatch):
    from huggingface_hub.errors import EntryNotFoundError

    fake_model_root(tmp_path, monkeypatch)

    def _raise_missing(**kwargs):
        raise EntryNotFoundError("404 Client Error")

    monkeypatch.setattr("lorafactory.cli.hf_hub_download", _raise_missing)
    result = run("fetch-models", "--config", str(MATRIX / "L-F.yaml"))

    assert result.exit_code != 0
    assert "Traceback" not in result.output
    assert "filename" in result.output.lower() or "commit" in result.output.lower()


def test_gen_gate_images_dry_run_plans_the_full_grid(tmp_path, monkeypatch):
    fake_model_root(tmp_path, monkeypatch)
    ckpt = tmp_path / "adapter.safetensors"
    write_peft_checkpoint(ckpt, ["transformer.transformer_blocks.0.attn.to_q"])

    result = run("gen-gate-images", "--checkpoint", str(ckpt),
                 "--out", str(tmp_path / "imgs"), "--dry-run")
    assert result.exit_code == 0, result.output

    plan = json.loads(result.output)
    # Both arms for every prompt/seed cell — the gate needs the paired null.
    assert len(plan["images"]) == len(PROMPTS) * len(SEEDS) * 2, len(plan["images"])
    assert {img["arm"] for img in plan["images"]} == {"lora", "null"}
    assert not any((tmp_path / "imgs").glob("*.png")), "dry run wrote images"
    # SD3's default family, with no render kwargs restricted.
    assert plan["family"] == "sd3"
    assert plan["render_kwargs"] == {}


def test_gen_gate_images_dry_run_fails_on_an_invalid_render_config(tmp_path, monkeypatch):
    """Finding 4: WS5's render-config validation must actually fire from the
    CLI, not just from tests that call `build_render_kwargs`/`render_settings`
    directly. A `negative_prompt` under a `flux_schnell` family config is
    exactly the kind of SD3-only key that would silently reach a real
    pipeline call if `--dry-run` never validated the render section."""
    fake_model_root(tmp_path, monkeypatch)
    ckpt = tmp_path / "adapter.safetensors"
    write_peft_checkpoint(ckpt, ["transformer.transformer_blocks.0.attn.to_q"])

    bad_config = tmp_path / "bad_flux_gate.yaml"
    bad_config.write_text(
        "family: flux_schnell\n"
        "prompts: [a]\n"
        "seeds: [0]\n"
        "render:\n"
        "  negative_prompt: bad\n"
    )

    result = run("gen-gate-images", "--checkpoint", str(ckpt),
                 "--out", str(tmp_path / "imgs"), "--gate-config", str(bad_config),
                 "--dry-run")
    assert result.exit_code != 0
    assert "negative_prompt" in result.output


def test_gen_gate_images_flux_dry_run_reports_the_resolved_render_kwargs(tmp_path, monkeypatch):
    """A valid FLUX dry-run's JSON must visibly carry the family and the
    resolved render kwargs (guidance_scale 0.0, 4 steps, euler), not just
    the image grid — that is what makes WS5's validation checkable from a
    `--dry-run` invocation instead of only from a unit test."""
    fake_model_root(tmp_path, monkeypatch)
    ckpt = tmp_path / "adapter.safetensors"
    write_peft_checkpoint(ckpt, ["transformer.transformer_blocks.0.attn.to_q"])

    result = run("gen-gate-images", "--checkpoint", str(ckpt),
                 "--out", str(tmp_path / "imgs"), "--gate-config",
                 str(CONFIGS / "gate" / "e_img_flux.yaml"), "--dry-run")
    assert result.exit_code == 0, result.output

    plan = json.loads(result.output)
    assert plan["family"] == "flux_schnell"
    assert plan["render_kwargs"] == {
        "guidance_scale": 0.0,
        "num_inference_steps": 4,
        "scheduler": "euler",
    }
