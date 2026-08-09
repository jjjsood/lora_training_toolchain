"""CLI surface — click group `lorafactory`."""

from click.testing import CliRunner

from conftest import MATRIX
from lorafactory.cli import cli

EXPECTED_COMMANDS = {
    "resolve-config", "check-budget", "check-dataset", "emit-kohya",
    "train", "train-matrix", "convert", "verify-keys",
    "gen-gate-images", "gate", "synth", "introspect", "screen",
    "fetch-models", "determinism-check", "provenance",
}


def test_help_lists_all_subcommands():
    result = CliRunner().invoke(cli, ["--help"])
    assert result.exit_code == 0
    missing = {c for c in EXPECTED_COMMANDS if c not in result.output}
    assert not missing, f"missing subcommands: {missing}"


def test_resolve_config_prints_hash():
    result = CliRunner().invoke(cli, ["resolve-config", str(MATRIX / "L-E.yaml")])
    assert result.exit_code == 0, result.output
    assert "sha256" in result.output.lower() or len(
        [t for t in result.output.split() if len(t) == 64]) > 0


def test_check_budget_passes_on_real_matrix():
    result = CliRunner().invoke(cli, ["check-budget", "--configs", str(MATRIX)])
    assert result.exit_code == 0, result.output
