"""The launch half of `train`, `train-matrix` and `test-run`.

test_cli_wiring.py covers every `--dry-run` plan; the branch that actually
calls `prepare_run` + `run_train` — create/skip, a non-zero kohya exit, and
the fetch-then-train composition of `test-run` — had no coverage at all, so
the three copies of that block could drift apart unnoticed.

Nothing here spawns a process: `lorafactory.cli.run_train` is replaced by a
recorder, which is exactly the seam the real command uses.
"""

import pytest

from conftest import CONFIGS, MATRIX
from lorafactory.data.fetch import FetchReport
from lorafactory.runs import CONFIG_HASH_FILENAME
from test_cli_wiring import fake_kohya_env, fake_model_root, run

CONFIG = MATRIX / "L-E.yaml"
SMOKE = CONFIGS / "test" / "smoke.yaml"


class Launches(list):
    """Every kohya launch the command asked for, plus the exit code to fake."""

    returncode = 0

    def set_returncode(self, returncode: int) -> None:
        self.returncode = returncode


@pytest.fixture
def calls(monkeypatch):
    """Record every kohya launch instead of spawning one; default exit 0."""
    recorded = Launches()

    def fake_run_train(engine, train_toml, log_path, **kwargs):
        recorded.append({"engine": engine, "toml": train_toml, "log": log_path})
        return recorded.returncode

    monkeypatch.setattr("lorafactory.cli.run_train", fake_run_train)
    return recorded


@pytest.fixture
def env(tmp_path, monkeypatch):
    fake_model_root(tmp_path, monkeypatch)
    fake_kohya_env(tmp_path, monkeypatch)
    return tmp_path / "runs"


def test_train_launches_kohya_and_reports_the_run_dir(env, calls):
    result = run("train", "--config", str(CONFIG), "--runs-dir", str(env))

    assert result.exit_code == 0, result.output
    assert len(calls) == 1, calls
    assert calls[0]["engine"] == "sd3"
    assert calls[0]["log"] == env / "L-E-a01" / "train.log"
    assert "L-E" in result.output and "finished" in result.output
    assert (env / "L-E-a01" / CONFIG_HASH_FILENAME).exists()


def test_train_fails_loudly_on_a_non_zero_kohya_exit(env, calls):
    calls.set_returncode(3)

    result = run("train", "--config", str(CONFIG), "--runs-dir", str(env))

    assert result.exit_code != 0
    assert "3" in result.output
    assert "train.log" in result.output, result.output


def test_train_skips_a_run_that_already_produced_an_adapter(env, calls):
    first = run("train", "--config", str(CONFIG), "--runs-dir", str(env))
    assert first.exit_code == 0, first.output
    adapter = env / "L-E-a01" / "adapter" / "L-E.safetensors"
    adapter.parent.mkdir(parents=True, exist_ok=True)
    adapter.touch()

    second = run("train", "--config", str(CONFIG), "--runs-dir", str(env))

    assert second.exit_code == 0, second.output
    assert "skipping" in second.output
    assert len(calls) == 1, "a completed run must not be trained again"


def test_train_matrix_launches_every_config_in_the_directory(env, calls):
    result = run("train-matrix", "--configs", str(MATRIX), "--runs-dir", str(env))

    assert result.exit_code == 0, result.output
    assert len(calls) == len(list(MATRIX.glob("*.yaml")))
    assert {c["engine"] for c in calls} == {"sd3", "flux"}


def test_train_matrix_stops_at_the_first_failing_adapter(env, calls):
    calls.set_returncode(1)

    result = run("train-matrix", "--configs", str(MATRIX), "--runs-dir", str(env))

    assert result.exit_code != 0
    assert len(calls) == 1, "a failed adapter must not be followed by the next"


class Fetches(list):
    """Weight entries the command asked for, plus the dataset errors to fake."""

    dataset_errors: list = []

    def set_dataset_errors(self, errors: list) -> None:
        self.dataset_errors = errors


@pytest.fixture
def fetches(monkeypatch, tmp_path):
    """Stub the two network halves of `test-run`: weights and dataset."""
    downloaded = Fetches()
    monkeypatch.setattr("lorafactory.cli._download_weights", downloaded.extend)

    image_dir = tmp_path / "datasets" / "STYLE"

    def fake_fetch_dataset(config, force=False):
        return FetchReport("fetched", image_dir, 50, "abc123",
                           list(downloaded.dataset_errors))

    monkeypatch.setattr("lorafactory.cli.fetch_dataset", fake_fetch_dataset)
    return downloaded


def test_test_run_fetches_weights_and_dataset_then_trains(env, calls, fetches):
    result = run("test-run", "--config", str(SMOKE), "--runs-dir", str(env))

    assert result.exit_code == 0, result.output
    assert fetches, "weights were never fetched"
    assert "50 images" in result.output
    assert len(calls) == 1
    assert calls[0]["log"] == env / "T-SMOKE-a01" / "train.log"


def test_test_run_refuses_to_train_on_an_unusable_dataset(env, calls, fetches):
    fetches.set_dataset_errors(["sha256 mismatch for img_0001.png"])

    result = run("test-run", "--config", str(SMOKE), "--runs-dir", str(env))

    assert result.exit_code != 0
    assert "sha256 mismatch" in result.output
    assert not calls, "training started on a dataset that does not validate"


# --------------------------------------------------------------------------
# A config only ever reaches a command through cli._resolve_config, which is
# where the schema has to stop it — downstream, kohya silently ignores keys it
# does not know and the run burns a GPU before saying so.
# --------------------------------------------------------------------------

BROKEN = """\
adapter_id: BROKEN
extends: {base}
overrides_allowed: [train]
train:
  seed: 1
  max_train_steps: -5
  learning_rate: 0.0001
  train_batch_size: 1
  optimizer_type: AdamW8bit
  mixed_precision: bf16
  save_precision: bf16
  save_model_as: safetensors
  gradient_checkpointing: true
  logging_dir: logs
"""


@pytest.fixture
def broken_config(tmp_path):
    path = tmp_path / "broken.yaml"
    path.write_text(BROKEN.format(base=CONFIGS / "base.yaml"))
    return path


def test_train_rejects_a_config_the_schema_refuses(env, calls, broken_config):
    result = run("train", "--config", str(broken_config), "--runs-dir", str(env))

    assert result.exit_code != 0
    assert "max_train_steps" in result.output, result.output
    assert not calls, "kohya was launched on a config that does not validate"


def test_emit_kohya_rejects_a_config_the_schema_refuses(tmp_path, broken_config):
    result = run("emit-kohya", "--config", str(broken_config),
                 "--out", str(tmp_path / "emit"))

    assert result.exit_code != 0
    assert not (tmp_path / "emit" / "kohya_config.toml").exists()


def test_train_dry_run_names_the_missing_kohya_env_instead_of_crashing(
        tmp_path, monkeypatch):
    fake_model_root(tmp_path, monkeypatch)
    monkeypatch.delenv("LORAFACTORY_KOHYA_PYTHON", raising=False)
    monkeypatch.delenv("LORAFACTORY_SDSCRIPTS_DIR", raising=False)

    result = run("train", "--config", str(CONFIG), "--dry-run",
                 "--runs-dir", str(tmp_path / "runs"))

    assert result.exit_code != 0
    assert "LORAFACTORY_KOHYA_PYTHON" in result.output
    assert "Traceback" not in result.output


def test_a_nested_weight_filename_lands_where_the_plan_says(tmp_path):
    """hf_hub_download recreates the repo path under local_dir, so the root it
    is given must be stripped of exactly as many components as the filename
    has — one level of nesting must not be the only case that works."""
    from lorafactory.cli import _local_dir_for

    root = tmp_path / "models"
    for filename in ("sd3_medium.safetensors",
                     "text_encoders/clip_l.safetensors",
                     "a/b/c/deep.safetensors"):
        local_path = root / filename
        assert _local_dir_for(local_path, filename) == root
        assert _local_dir_for(local_path, filename) / filename == local_path


def _weight_entry(tmp_path):
    return {"repo_id": "org/repo", "revision": "deadbeefcafe",
            "filename": "sd3_medium.safetensors",
            "local_path": str(tmp_path / "models" / "sd3_medium.safetensors")}


def test_download_retries_transient_failures(tmp_path, monkeypatch):
    """A DNS flake or CDN reset mid-download must not kill the run: the
    download is resumable, so transient errors are retried with backoff."""
    from lorafactory import cli as cli_mod

    attempts = []

    def flaky(**kwargs):
        attempts.append(kwargs)
        if len(attempts) < 3:
            raise RuntimeError("error sending request")

    slept = []
    monkeypatch.setattr(cli_mod, "hf_hub_download", flaky)
    monkeypatch.setattr(cli_mod.time, "sleep", slept.append)

    cli_mod._download_weights([_weight_entry(tmp_path)])

    assert len(attempts) == 3
    assert slept == [5, 25]


def test_download_gives_up_after_max_attempts(tmp_path, monkeypatch):
    from lorafactory import cli as cli_mod

    attempts = []

    def always_down(**kwargs):
        attempts.append(kwargs)
        raise RuntimeError("error sending request")

    monkeypatch.setattr(cli_mod, "hf_hub_download", always_down)
    monkeypatch.setattr(cli_mod.time, "sleep", lambda s: None)

    with pytest.raises(RuntimeError):
        cli_mod._download_weights([_weight_entry(tmp_path)])

    assert len(attempts) == cli_mod.DOWNLOAD_ATTEMPTS


def test_download_never_retries_permanent_errors(tmp_path, monkeypatch):
    """404/403-class answers mean the plan or token is wrong; retrying them
    would only delay the real error by the whole backoff schedule."""
    from types import SimpleNamespace

    from huggingface_hub.errors import RepositoryNotFoundError

    from lorafactory import cli as cli_mod

    attempts = []
    fake_response = SimpleNamespace(headers={}, status_code=404,
                                    request=SimpleNamespace(url="u"))

    def gone(**kwargs):
        attempts.append(kwargs)
        raise RepositoryNotFoundError("no such repo", response=fake_response)

    monkeypatch.setattr(cli_mod, "hf_hub_download", gone)
    monkeypatch.setattr(cli_mod.time, "sleep",
                        lambda s: pytest.fail("must not sleep"))

    with pytest.raises(cli_mod.ModelDownloadError) as excinfo:
        cli_mod._download_weights([_weight_entry(tmp_path)])

    assert isinstance(excinfo.value.__cause__, RepositoryNotFoundError)
    assert "HF_TOKEN" in str(excinfo.value)
    assert len(attempts) == 1
