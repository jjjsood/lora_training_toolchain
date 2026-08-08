"""Screening plan for a community-checkpoint survey (CAP-28).

API pinned:
    from lorafactory.survey import screen
    screen.screen_plan("configs/survey/sd3_community.yaml")

One entry per survey slot, each carrying the gate config it will be screened
against, so no checkpoint can be accepted without passing E_img.
"""

from __future__ import annotations

from pathlib import Path

import yaml


class SurveyConfigError(ValueError):
    """The survey YAML is missing targets or a gate config."""


def load_survey(survey_config_path) -> dict:
    """Read a survey YAML into a plain dict."""
    path = Path(survey_config_path)
    with open(path) as f:
        return yaml.safe_load(f) or {}


def _resolve(path_value: str, base_dir: Path) -> Path:
    path = Path(path_value)
    if not path.is_absolute():
        path = (base_dir / path).resolve()
    return path


def screen_plan(survey_config_path) -> list[dict]:
    """Build the screening plan for ``survey_config_path``.

    Returns one dict per slot, in YAML order, with at least ``slot``,
    ``status`` and ``gate_config`` (a Path resolved relative to the survey
    file's own directory). Extra target keys are carried through.
    """
    path = Path(survey_config_path)
    config = load_survey(path)

    targets = config.get("targets") or []
    if not targets:
        raise SurveyConfigError(f"{path}: no targets declared")

    gate_config = config.get("gate_config")
    if not gate_config:
        raise SurveyConfigError(f"{path}: no gate_config declared")
    gate_path = _resolve(gate_config, path.parent)

    base_arch = config.get("base_arch")
    fallback = config.get("fallback") or {}

    plan: list[dict] = []
    for target in targets:
        slot = target.get("slot")
        if not slot:
            raise SurveyConfigError(f"{path}: a target has no slot name")
        entry = dict(target)
        entry.update(
            {
                "slot": slot,
                "status": target.get("status", "pending_acquisition"),
                "gate_config": gate_path,
                "base_arch": base_arch,
                "survey_config": path,
                "min_usable": fallback.get("min_usable"),
                "on_shortfall": fallback.get("on_shortfall"),
            }
        )
        plan.append(entry)

    return plan
