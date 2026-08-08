"""Config overlay loader: extends + overrides_allowed + canonical hash.

API pinned:
    from lorafactory.config.loader import resolve, ConfigViolation
    rc = resolve(path)         # pure: no filesystem checks beyond reading YAML
    rc.data                    # resolved canonical dict (no 'extends' key)
    rc.sha256                  # SHA-256 hex over canonical JSON of rc.data
    from lorafactory.config.hashing import config_hash
"""

import textwrap

import pytest

from conftest import ALL_MATRIX_IDS, FALLBACK_IDS, MATRIX
from lorafactory.config.hashing import config_hash
from lorafactory.config.loader import ConfigViolation, resolve


def test_all_matrix_configs_resolve():
    for aid in ALL_MATRIX_IDS:
        rc = resolve(MATRIX / f"{aid}.yaml")
        assert rc.data["adapter_id"] == aid
        assert len(rc.sha256) == 64


def test_fallback_configs_resolve_and_are_exempt(configs_dir):
    for aid in FALLBACK_IDS:
        rc = resolve(configs_dir / "fallback" / f"{aid}.yaml")
        assert rc.data.get("budget_exempt") is True


def test_matrix_target_selection_values():
    assert resolve(MATRIX / "L-E.yaml").data["target"]["blocks"] == [0, 7]
    assert resolve(MATRIX / "L-L.yaml").data["target"]["blocks"] == [16, 23]
    assert resolve(MATRIX / "L-A.yaml").data["target"]["module_classes"] == ["attn"]
    assert resolve(MATRIX / "L-M.yaml").data["target"]["module_classes"] == ["mlp"]
    assert resolve(MATRIX / "L-R4.yaml").data["target"]["rank"] == 4
    assert resolve(MATRIX / "L-R4.yaml").data["target"]["alpha"] == 4
    assert resolve(MATRIX / "L-R64.yaml").data["target"]["rank"] == 64
    lt = resolve(MATRIX / "L-T.yaml").data["target"]
    assert lt["scope"] == "text_encoders"
    assert lt["te_encoders"] == ["clip_l", "clip_g"]  # no T5: diffusers 0.39 limit
    for oid in ("O-F", "O-A", "O-M"):
        assert resolve(MATRIX / f"{oid}.yaml").data["dataset"]["name"] == "OBJ"
    assert resolve(MATRIX / "F-F.yaml").data["model"]["arch"] == "flux"


def test_base_pins_matched_budget_keys():
    rc = resolve(MATRIX / "L-F.yaml")
    train = rc.data["train"]
    assert isinstance(train["seed"], int)
    assert train["max_data_loader_n_workers"] == 0
    assert rc.data["dataset"]["name"] == "STYLE"
    assert rc.data["model"]["train_revision"], "base model revision must be pinned"
    assert rc.data["model"]["eval_revision"], "eval model revision must be pinned"


def test_undeclared_override_raises(tmp_path):
    base = MATRIX.parent / "base.yaml"
    overlay = tmp_path / "bad.yaml"
    overlay.write_text(textwrap.dedent(f"""\
        adapter_id: BAD
        extends: {base}
        overrides_allowed: [target]
        train:
          seed: 999
    """))
    with pytest.raises(ConfigViolation):
        resolve(overlay)


def test_hash_stable_and_order_independent():
    a = {"x": 1, "y": {"b": 2, "a": 3}}
    b = {"y": {"a": 3, "b": 2}, "x": 1}
    assert config_hash(a) == config_hash(b)
    assert len(config_hash(a)) == 64
    assert config_hash(a) != config_hash({"x": 2, "y": {"b": 2, "a": 3}})


def test_resolve_is_deterministic():
    r1 = resolve(MATRIX / "L-E.yaml")
    r2 = resolve(MATRIX / "L-E.yaml")
    assert r1.sha256 == r2.sha256
