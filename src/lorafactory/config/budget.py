"""Matched-budget check across the config matrix (README §3)."""

import json
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path

from .loader import resolve

STYLE_IDS = ("L-F", "L-A", "L-M", "L-E", "L-L", "L-R4", "L-R64", "L-T")
O_PARTNER = {"O-F": "L-F", "O-A": "L-A", "O-M": "L-M"}
SECTIONS = ("model", "dataset", "train", "determinism")


def _canon(value) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"))


@dataclass
class BudgetReport:
    ok: bool
    violations: list = field(default_factory=list)
    details: dict = field(default_factory=dict)

    def write(self, path) -> Path:
        path = Path(path)
        path.write_text(json.dumps(
            {"ok": self.ok, "violations": self.violations, "details": self.details},
            indent=2, sort_keys=True,
        ))
        return path


def check_matrix(configs_dir) -> BudgetReport:
    configs_dir = Path(configs_dir)
    resolved = {}
    for yaml_path in sorted(configs_dir.glob("*.yaml")):
        rc = resolve(yaml_path)
        resolved[rc.data["adapter_id"]] = rc.data

    violations = []
    exempt = {aid for aid, d in resolved.items() if d.get("budget_exempt") is True}
    exempt.add("F-F")

    style_present = [aid for aid in STYLE_IDS if aid in resolved]
    for section in SECTIONS:
        values = {aid: _canon(resolved[aid].get(section)) for aid in style_present}
        if not values:
            continue
        common, _ = Counter(values.values()).most_common(1)[0]
        for aid, v in values.items():
            if v != common:
                violations.append(
                    f"adapter '{aid}': section '{section}' diverges from the "
                    f"matched STYLE budget"
                )

    for oid, partner in O_PARTNER.items():
        if oid not in resolved or partner not in resolved:
            continue
        for section in SECTIONS:
            if section == "dataset":
                continue
            a = _canon(resolved[oid].get(section))
            b = _canon(resolved[partner].get(section))
            if a != b:
                violations.append(
                    f"adapter '{oid}': section '{section}' diverges from matched "
                    f"partner '{partner}'"
                )

    ok = not violations
    details = {aid: ("exempt" if aid in exempt else "checked") for aid in resolved}
    return BudgetReport(ok=ok, violations=violations, details=details)
