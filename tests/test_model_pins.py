"""Model identity pins — the weights the instrument is actually trained on.

Every hash below was read from the Hugging Face API (`GET /api/models/<repo>`,
field `sha`) and is the real HEAD commit of that repo. They live here rather
than only in configs/ because a plausible-looking 40-hex string that points at
nothing fails only at download time, after the whole CPU pipeline has already
reported green.

A pinned revision is a scientific claim ("this adapter was trained on exactly
these weights"), so it belongs in the test suite, not in a comment.

If a pin legitimately has to move (repo re-tagged upstream), update it HERE
first, with a fresh API read — never the other way round.
"""

import re

import yaml

from conftest import CONFIGS, MATRIX

# repo id -> real HEAD sha, verified against the HF API.
VERIFIED_REVISIONS = {
    "stabilityai/stable-diffusion-3-medium": "19b7f516efea082d257947e057e6f419e26fd497",
    "stabilityai/stable-diffusion-3-medium-diffusers": (
        "ea42f8cef0f178587cf766dc8129abd379c90671"
    ),
    "black-forest-labs/FLUX.1-dev": "3de623fc3c33e44ffbe2bad470d0f45bccf2eb21",
}

# kohya-ss/sd-scripts loads a single-file checkpoint; these names exist in the
# SAI/BFL repos above (verified against the API's sibling list).
SD3_TRAIN_FILE = "sd3_medium.safetensors"
FLUX_TRAIN_FILE = "flux1-dev.safetensors"

# The SD3 SAI repo ships the encoders under text_encoders/; kohya wants each
# as its own file path, so the config must name files, not a repo.
SD3_TEXT_ENCODERS = {
    "clip_l": "text_encoders/clip_l.safetensors",
    "clip_g": "text_encoders/clip_g.safetensors",
    "t5xxl": "text_encoders/t5xxl_fp16.safetensors",
}

_SHA_RE = re.compile(r"^[0-9a-f]{40}$")


def load_yaml(path):
    with open(path) as f:
        return yaml.safe_load(f)


def test_base_model_revisions_are_the_verified_ones():
    model = load_yaml(CONFIGS / "base.yaml")["model"]
    assert model["train_repo"] == "stabilityai/stable-diffusion-3-medium"
    assert model["eval_repo"] == "stabilityai/stable-diffusion-3-medium-diffusers"
    assert model["train_revision"] == VERIFIED_REVISIONS[model["train_repo"]]
    assert model["eval_revision"] == VERIFIED_REVISIONS[model["eval_repo"]]


def test_flux_model_revisions_are_the_verified_ones():
    model = load_yaml(MATRIX / "F-F.yaml")["model"]
    assert model["train_repo"] == "black-forest-labs/FLUX.1-dev"
    assert model["train_revision"] == VERIFIED_REVISIONS[model["train_repo"]]
    assert model["eval_revision"] == VERIFIED_REVISIONS[model["eval_repo"]]


def test_every_revision_in_configs_is_a_known_verified_sha():
    """No config may pin a revision we have not actually checked against HF."""
    unknown = []
    for path in sorted(CONFIGS.rglob("*.yaml")):
        model = (load_yaml(path) or {}).get("model") or {}
        for field in ("train_revision", "eval_revision"):
            rev = model.get(field)
            if rev is None:
                continue
            assert _SHA_RE.match(str(rev)), f"{path.name}:{field} is not a 40-hex sha"
            if rev not in VERIFIED_REVISIONS.values():
                unknown.append(f"{path.name}:{field}={rev}")
    assert not unknown, (
        "unverified revision pins (add them to VERIFIED_REVISIONS only after "
        f"reading the real sha from the HF API): {unknown}"
    )


def test_single_file_checkpoints_are_named():
    """kohya takes a FILE for --pretrained_model_name_or_path, not a repo id."""
    base_model = load_yaml(CONFIGS / "base.yaml")["model"]
    assert base_model["train_file"] == SD3_TRAIN_FILE

    flux_model = load_yaml(MATRIX / "F-F.yaml")["model"]
    assert flux_model["train_file"] == FLUX_TRAIN_FILE
    # FLUX training additionally needs the autoencoder as its own file.
    assert flux_model["ae"] == "ae.safetensors"


def test_sd3_text_encoder_files():
    assert load_yaml(CONFIGS / "base.yaml")["model"]["text_encoders"] == SD3_TEXT_ENCODERS
