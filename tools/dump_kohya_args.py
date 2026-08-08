#!/usr/bin/env python3
"""Static dump of kohya-ss/sd-scripts argparse dests, v0.11.1.

sd-scripts builds its CLI parser at runtime via a chain of ``add_*_arguments(parser)``
calls spread across `train_network.py`, `sd3_train_network.py` / `flux_train_network.py`
and several `library/*.py` modules. Actually importing those modules pulls in the full
kohya dependency set (accelerate, bitsandbytes, ...), which this repo does not need just
to know which argparse `dest` names exist.

Instead this tool parses the relevant files with `ast` (no execution, no sd-scripts
dependencies required) and statically walks the same call graph `setup_parser()` would
walk at runtime: starting from `sd3_train_network.setup_parser` / `flux_train_network.
setup_parser`, it follows every call to a locally-defined or imported `add_*(parser,
...)` function (resolving `import`/`from ... import` aliases against files on disk) and
collects every `<obj>.add_argument(...)` call it finds along the way. The `dest` of each
argument is taken from an explicit `dest=` kwarg if present, else derived from the first
long option string (`--foo-bar` -> `foo_bar`), matching argparse's own rule.

Conditionally-added arguments (functions that take extra bool flags gating some
`add_argument` calls) are collected unconditionally -- the registry is a defence against
typos (README/PLAN "arg registry"), so a superset of legal dests is safe; a subset is not.

Usage:
    git clone --depth 1 --branch v0.11.1 https://github.com/kohya-ss/sd-scripts .cache/sd-scripts
    uv run python tools/dump_kohya_args.py [--sdscripts-dir .cache/sd-scripts]

Writes sorted dest lists to:
    src/lorafactory/kohya/kohya_args_sd3.json
    src/lorafactory/kohya/kohya_args_flux.json
"""

from __future__ import annotations

import argparse
import ast
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SDSCRIPTS_DIR = REPO_ROOT / ".cache" / "sd-scripts"
OUT_DIR = REPO_ROOT / "src" / "lorafactory" / "kohya"

# entry point (module-relative-to-checkout-root file, function name) per engine
ENGINES = {
    "sd3": ("sd3_train_network.py", "setup_parser"),
    "flux": ("flux_train_network.py", "setup_parser"),
}


def _module_file(root: Path, dotted: str) -> Path | None:
    """Resolve a dotted module path to a .py file under `root`, if one exists."""
    parts = dotted.split(".")
    candidate = root.joinpath(*parts).with_suffix(".py")
    if candidate.is_file():
        return candidate
    candidate = root.joinpath(*parts, "__init__.py")
    if candidate.is_file():
        return candidate
    return None


def _import_map(tree: ast.Module) -> dict:
    """Map names bound at module scope to what they refer to.

    Value is either ("module", dotted_path) for `import x.y as name` / a name bound to
    a submodule via `from pkg import name`, or ("from", dotted_pkg, orig_name) for a
    plain `from pkg import name` whose target turns out (on resolution) not to be a
    module file, i.e. a function.
    """
    mapping: dict[str, tuple] = {}
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                bound = alias.asname or alias.name.split(".")[0]
                mapping[bound] = ("module", alias.name)
        elif isinstance(node, ast.ImportFrom):
            if node.module is None or node.level:  # skip relative imports; unused here
                continue
            for alias in node.names:
                bound = alias.asname or alias.name
                mapping[bound] = ("from", node.module, alias.name)
    return mapping


def _resolve(name: str, imports: dict, root: Path):
    """Resolve a bound name to ("module", file) or ("func", file, func_name), if possible."""
    target = imports.get(name)
    if target is None:
        return None
    if target[0] == "module":
        f = _module_file(root, target[1])
        return ("module", f) if f else None
    if target[0] == "from":
        pkg_dotted, orig_name = target[1], target[2]
        as_module = _module_file(root, f"{pkg_dotted}.{orig_name}")
        if as_module:
            return ("module", as_module)
        pkg_file = _module_file(root, pkg_dotted)
        if pkg_file:
            return ("func", pkg_file, orig_name)
    return None


def _is_followable(func_name: str) -> bool:
    """Names worth recursing into: the `add_*_arguments` registration helpers plus the
    base-parser constructors (`setup_parser`) the entry points delegate to."""
    return func_name.startswith("add_") or func_name == "setup_parser"


def _find_function(tree: ast.Module, name: str) -> ast.FunctionDef | None:
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and node.name == name:
            return node
    return None


def _dest_of(call: ast.Call) -> str | None:
    """argparse's own dest-derivation rule: explicit dest= wins, else the first long
    option string with leading dashes stripped and internal dashes turned into
    underscores; falls back to the first (short) option string if no long one exists."""
    for kw in call.keywords:
        if (kw.arg == "dest" and isinstance(kw.value, ast.Constant)
                and isinstance(kw.value.value, str)):
            return kw.value.value

    option_strings = [a.value for a in call.args
                      if isinstance(a, ast.Constant) and isinstance(a.value, str)]
    if not option_strings:
        return None
    long_opts = [o for o in option_strings if o.startswith("--")]
    chosen = long_opts[0] if long_opts else option_strings[0]
    return chosen.lstrip("-").replace("-", "_")


class Collector:
    def __init__(self, root: Path):
        self.root = root
        self._trees: dict[Path, tuple[ast.Module, dict]] = {}
        self._visited: set[tuple[Path, str]] = set()
        self.dests: set[str] = set()

    def _load(self, path: Path) -> tuple[ast.Module, dict]:
        if path not in self._trees:
            tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
            self._trees[path] = (tree, _import_map(tree))
        return self._trees[path]

    def collect(self, path: Path, func_name: str) -> None:
        key = (path, func_name)
        if key in self._visited:
            return
        self._visited.add(key)

        tree, imports = self._load(path)
        fn = _find_function(tree, func_name)
        if fn is None:
            print(f"warning: {func_name} not found in {path}", file=sys.stderr)
            return

        for node in ast.walk(fn):
            if not isinstance(node, ast.Call):
                continue

            if isinstance(node.func, ast.Attribute):
                attr = node.func.attr
                obj = node.func.value
                if attr == "add_argument":
                    dest = _dest_of(node)
                    if dest:
                        self.dests.add(dest)
                elif _is_followable(attr) and isinstance(obj, ast.Name):
                    resolved = _resolve(obj.id, imports, self.root)
                    if resolved and resolved[0] == "module":
                        self.collect(resolved[1], attr)

            elif isinstance(node.func, ast.Name):
                fname = node.func.id
                if not _is_followable(fname) or fname == func_name:
                    continue
                resolved = _resolve(fname, imports, self.root)
                if resolved and resolved[0] == "func":
                    self.collect(resolved[1], resolved[2])
                elif resolved is None and _find_function(tree, fname):
                    self.collect(path, fname)  # locally defined in the same file


def dump_engine(sdscripts_dir: Path, entry_file: str, entry_func: str) -> list[str]:
    collector = Collector(sdscripts_dir)
    entry_path = sdscripts_dir / entry_file
    if not entry_path.is_file():
        raise FileNotFoundError(
            f"{entry_path} not found -- clone sd-scripts first:\n"
            f"  git clone --depth 1 --branch v0.11.1 "
            f"https://github.com/kohya-ss/sd-scripts {sdscripts_dir}"
        )
    collector.collect(entry_path, entry_func)
    return sorted(collector.dests)


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--sdscripts-dir", type=Path, default=DEFAULT_SDSCRIPTS_DIR)
    parser.add_argument("--out-dir", type=Path, default=OUT_DIR)
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    for engine, (entry_file, entry_func) in ENGINES.items():
        dests = dump_engine(args.sdscripts_dir, entry_file, entry_func)
        out_path = args.out_dir / f"kohya_args_{engine}.json"
        out_path.write_text(json.dumps(dests, indent=2) + "\n")
        print(f"{engine}: {len(dests)} dests -> {out_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
