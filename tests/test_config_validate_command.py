"""`lorafactory validate-config` — human-readable schema errors."""

from click.testing import CliRunner

from conftest import MATRIX
from lorafactory.cli import cli


def test_validate_config_ok_on_a_real_matrix_config():
    runner = CliRunner()
    result = runner.invoke(cli, ["validate-config", str(MATRIX / "L-F.yaml")])
    assert result.exit_code == 0
    assert "OK" in result.output


def test_validate_config_warns_but_never_blocks_on_drifted_revision(tmp_path):
    """The branch's headline guarantee: a drifted/non-SHA `train_revision`
    (e.g. a branch name like 'main') is always a warning at the CLI level
    too, never a raised exception or a non-zero exit code.

    `loader.resolve()` replaces an overridden section wholesale (never a
    deep merge), so the override below restates the whole `model` section
    from `configs/base.yaml` rather than naming `train_revision` alone."""
    drifted = tmp_path / "drifted.yaml"
    drifted.write_text(f"""
extends: {MATRIX.parent / "base.yaml"}
overrides_allowed: [model]
model:
  arch: sd3
  train_repo: stabilityai/stable-diffusion-3-medium
  train_revision: main
  train_file: sd3_medium.safetensors
  eval_repo: stabilityai/stable-diffusion-3-medium-diffusers
  eval_revision: ea42f8cef0f178587cf766dc8129abd379c90671
  text_encoders:
    clip_l: text_encoders/clip_l.safetensors
    clip_g: text_encoders/clip_g.safetensors
    t5xxl: text_encoders/t5xxl_fp16.safetensors
""")
    runner = CliRunner()
    result = runner.invoke(cli, ["validate-config", str(drifted)])
    assert result.exit_code == 0, result.output
    assert "OK" in result.output


def test_validate_config_reports_missing_field_in_one_line(tmp_path):
    bad = tmp_path / "bad.yaml"
    bad.write_text(f"""
extends: {MATRIX / "L-F.yaml"}
overrides_allowed: [train]
train:
  seed: 1
""")
    runner = CliRunner()
    result = runner.invoke(cli, ["validate-config", str(bad)])
    assert result.exit_code != 0
    assert "Traceback" not in result.output
    assert "train.max_train_steps" in result.output
