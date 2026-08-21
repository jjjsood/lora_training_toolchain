"""`lorafactory validate-config` — human-readable schema errors."""

from click.testing import CliRunner

from conftest import MATRIX
from lorafactory.cli import cli


def test_validate_config_ok_on_a_real_matrix_config():
    runner = CliRunner()
    result = runner.invoke(cli, ["validate-config", str(MATRIX / "L-F.yaml")])
    assert result.exit_code == 0
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
