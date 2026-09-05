"""Materialize or inspect recipe-declared train/eval topology banks."""

from __future__ import annotations

import argparse
from collections.abc import Sequence

from jaxborg.recipe import REPO_ROOT, load
from jaxborg.topology_banks import expand_topology_bank, materialize_topology_bank


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description="Materialize recipe topology seed ranges into snapshot banks",
    )
    parser.add_argument("--recipe", required=True, help="Recipe name or YAML path")
    parser.add_argument(
        "--scope",
        choices=("train", "eval", "all"),
        default="all",
        help="Which declared bank to materialize (default: all)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate and print deterministic paths without creating snapshots",
    )
    args = parser.parse_args(argv)

    recipe = load(args.recipe)
    scopes = ("train", "eval") if args.scope == "all" else (args.scope,)
    for scope in scopes:
        section = recipe.get(scope) or {}
        if args.dry_run:
            paths = expand_topology_bank(section, scope=scope, repo_root=REPO_ROOT)
        else:
            paths = materialize_topology_bank(section, scope=scope, repo_root=REPO_ROOT)
        print(f"{scope}: {len(paths)} topology snapshot(s)")
        for path in paths:
            print(f"  {path}")


if __name__ == "__main__":
    main()
