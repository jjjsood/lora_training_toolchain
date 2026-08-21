"""Every fenced ```yaml block in README.md under a 'lorafactory-example'
marker must actually resolve+validate — a doc example that silently rots
is worse than no example."""

import re

import pytest

from conftest import ROOT
from lorafactory.config.loader import resolve
from lorafactory.config.schema import validate

_BLOCK_RE = re.compile(
    r"<!-- lorafactory-example: (?P<name>[\w.-]+) -->\n```yaml\n(?P<body>.*?)\n```",
    re.DOTALL,
)


def _readme_examples() -> list[tuple[str, str]]:
    text = (ROOT / "README.md").read_text()
    return [(m.group("name"), m.group("body")) for m in _BLOCK_RE.finditer(text)]


@pytest.mark.parametrize(
    "name,body", _readme_examples(), ids=[name for name, _ in _readme_examples()]
)
def test_readme_example_resolves_and_validates(name, body, tmp_path):
    path = tmp_path / name
    path.write_text(body)
    validate(resolve(path).data)


def test_readme_has_at_least_the_minimal_and_local_dataset_examples():
    names = {name for name, _ in _readme_examples()}
    assert "minimal-base.yaml" in names
    assert "local-dataset.yaml" in names
