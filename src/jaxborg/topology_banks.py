"""Recipe-driven topology snapshot materialization and split validation."""

from __future__ import annotations

import os
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

MAX_TOPOLOGY_SEED = 2**32 - 1
MAX_TOPOLOGY_BANK_SIZE = 10_000
_GENERATOR_SOURCES = {"jax": "generated", "cyborg": "cyborg"}
_GENERATION_KEYS = frozenset({"generator", "seed_start", "seed_end", "count", "cache_dir", "output_dir"})


@dataclass(frozen=True)
class TopologyGenerationSpec:
    generator: str
    seeds: tuple[int, ...]
    cache_dir: Path

    @property
    def source(self) -> str:
        return _GENERATOR_SOURCES[self.generator]

    @property
    def paths(self) -> tuple[Path, ...]:
        return tuple(self.cache_dir / f"{self.generator}_seed_{seed:010d}.snapshot.npz" for seed in self.seeds)


def _absolute_path(value: str | Path, *, repo_root: Path) -> Path:
    path = Path(value).expanduser()
    if not path.is_absolute():
        path = repo_root / path
    return path.resolve()


def _integer(value: Any, *, field: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"{field} must be an integer")
    return value


def explicit_topology_bank(section: Mapping[str, Any], *, repo_root: Path) -> tuple[Path, ...]:
    """Resolve an existing ``topology_bank`` declaration without materializing."""
    if "topology_bank" not in section:
        return ()
    bank = section["topology_bank"]
    if bank is None:
        raise ValueError("topology_bank must be a path or a sequence of paths, not null")
    if isinstance(bank, (str, Path)):
        if not str(bank).strip():
            raise ValueError("topology_bank must not be empty")
        bank = (bank,)
    if not isinstance(bank, Sequence):
        raise ValueError("topology_bank must be a path or a sequence of paths")
    if not bank:
        raise ValueError("topology_bank must contain at least one snapshot path")
    for index, entry in enumerate(bank):
        if not isinstance(entry, (str, Path)) or not str(entry).strip():
            raise ValueError(f"topology_bank[{index}] must be a non-empty path")
    paths = tuple(_absolute_path(entry, repo_root=repo_root) for entry in bank)
    seen: set[Path] = set()
    duplicates: set[Path] = set()
    for path in paths:
        if path in seen:
            duplicates.add(path)
        seen.add(path)
    if duplicates:
        labels = ", ".join(str(path) for path in sorted(duplicates))
        raise ValueError(f"topology_bank contains duplicate paths: {labels}")
    return paths


def expand_topology_bank(
    section: Mapping[str, Any],
    *,
    scope: str,
    repo_root: Path,
) -> tuple[Path, ...]:
    """Return the paths a declaration denotes without creating any files."""
    generated = parse_topology_generation(section, scope=scope, repo_root=repo_root)
    if generated is not None:
        return generated.paths
    return explicit_topology_bank(section, repo_root=repo_root)


def parse_topology_generation(
    section: Mapping[str, Any],
    *,
    scope: str,
    repo_root: Path,
) -> TopologyGenerationSpec | None:
    """Parse and validate one train/eval ``topology_generation`` mapping."""
    raw = section.get("topology_generation")
    if raw is None:
        return None
    if not isinstance(raw, Mapping):
        raise ValueError(f"{scope}.topology_generation must be a mapping")
    unknown = set(raw) - _GENERATION_KEYS
    if unknown:
        raise ValueError(f"{scope}.topology_generation has unknown keys: {sorted(unknown)}")
    if "topology_bank" in section:
        raise ValueError(f"{scope}.topology_bank and {scope}.topology_generation are mutually exclusive")

    generator = raw.get("generator")
    if generator not in _GENERATOR_SOURCES:
        raise ValueError(f"{scope}.topology_generation.generator must be one of {tuple(_GENERATOR_SOURCES)}")

    if "seed_start" not in raw:
        raise ValueError(f"{scope}.topology_generation.seed_start is required")
    seed_start = _integer(raw["seed_start"], field=f"{scope}.topology_generation.seed_start")
    has_end = "seed_end" in raw
    has_count = "count" in raw
    if has_end == has_count:
        raise ValueError(f"{scope}.topology_generation requires exactly one of seed_end or count")
    if has_count:
        count = _integer(raw["count"], field=f"{scope}.topology_generation.count")
        if count < 1:
            raise ValueError(f"{scope}.topology_generation.count must be positive")
        seed_end = seed_start + count - 1
    else:
        seed_end = _integer(raw["seed_end"], field=f"{scope}.topology_generation.seed_end")
        if seed_end < seed_start:
            raise ValueError(f"{scope}.topology_generation.seed_end must be >= seed_start")
    if seed_start < 0 or seed_end > MAX_TOPOLOGY_SEED:
        raise ValueError(f"{scope}.topology_generation seeds must be in [0, {MAX_TOPOLOGY_SEED}]")
    pool_size = seed_end - seed_start + 1
    if pool_size > MAX_TOPOLOGY_BANK_SIZE:
        raise ValueError(
            f"{scope}.topology_generation contains {pool_size} seeds; "
            f"the maximum supported bank size is {MAX_TOPOLOGY_BANK_SIZE}"
        )

    cache_dir = raw.get("cache_dir")
    output_dir = raw.get("output_dir")
    if cache_dir is not None and output_dir is not None:
        raise ValueError(f"{scope}.topology_generation cannot set both cache_dir and output_dir")
    directory = cache_dir if cache_dir is not None else output_dir
    if directory is None:
        raise ValueError(f"{scope}.topology_generation.cache_dir is required")
    if not isinstance(directory, (str, Path)) or not str(directory).strip():
        raise ValueError(f"{scope}.topology_generation.cache_dir must be a non-empty path")
    return TopologyGenerationSpec(
        generator=str(generator),
        seeds=tuple(range(seed_start, seed_end + 1)),
        cache_dir=_absolute_path(directory, repo_root=repo_root),
    )


def _path_provenance(path: Path) -> tuple[str, int] | None:
    if not path.is_file():
        return None
    from jaxborg.scenarios.cc4.topology import load_topology_metadata

    metadata = load_topology_metadata(path)
    source = metadata.get("source")
    seed = metadata.get("source_seed")
    if not isinstance(source, str) or isinstance(seed, bool) or not isinstance(seed, int):
        return None
    return source, seed


def _section_sources(
    section: Mapping[str, Any],
    *,
    scope: str,
    repo_root: Path,
) -> tuple[set[Path], set[tuple[str, int]], set[Path]]:
    explicit = explicit_topology_bank(section, repo_root=repo_root)
    generated = parse_topology_generation(section, scope=scope, repo_root=repo_root)
    paths = set(explicit)
    provenance = {path: _path_provenance(path) for path in explicit}
    sources = {source for source in provenance.values() if source is not None}
    unverified = {path for path, source in provenance.items() if source is None}
    if generated is not None:
        paths.update(generated.paths)
        sources.update((generated.source, seed) for seed in generated.seeds)
    return paths, sources, unverified


def validate_topology_split(recipe: Mapping[str, Any], *, repo_root: Path) -> None:
    """Validate topology declarations and reject train/eval leakage."""
    train = recipe.get("train") or {}
    evaluation = recipe.get("eval") or {}
    train_paths = set(expand_topology_bank(train, scope="train", repo_root=repo_root))
    eval_paths = set(expand_topology_bank(evaluation, scope="eval", repo_root=repo_root))

    path_overlap = train_paths & eval_paths
    if path_overlap:
        paths = ", ".join(str(path) for path in sorted(path_overlap))
        raise ValueError(f"train/eval topology path overlap: {paths}")
    if not train_paths or not eval_paths:
        return

    _, train_sources, train_unverified = _section_sources(train, scope="train", repo_root=repo_root)
    _, eval_sources, eval_unverified = _section_sources(evaluation, scope="eval", repo_root=repo_root)
    # Distinct paths can still contain snapshots generated from the same source
    # seed. If both sides declare pools, provenance is required to prove the
    # split disjoint before any generated snapshots are materialized.
    if train_paths and eval_paths and (train_unverified or eval_unverified):
        paths = ", ".join(str(path) for path in sorted(train_unverified | eval_unverified))
        raise ValueError(f"cannot validate train/eval topology source-seed split; missing provenance: {paths}")
    source_overlap = train_sources & eval_sources
    if source_overlap:
        labels = ", ".join(f"{source}:{seed}" for source, seed in sorted(source_overlap))
        raise ValueError(f"train/eval topology source-seed overlap: {labels}")


def validate_eval_topology_override(
    recipe: Mapping[str, Any],
    topology_paths: Sequence[str | Path],
    *,
    repo_root: Path,
) -> None:
    """Reject a CLI evaluation bank that leaks into the recipe's train pool."""
    candidate = {
        "train": recipe.get("train") or {},
        "eval": {"topology_bank": tuple(topology_paths)},
    }
    validate_topology_split(candidate, repo_root=repo_root)


def _validate_cached_snapshot(path: Path, *, source: str, seed: int) -> None:
    from jaxborg.scenarios.cc4.topology import (
        TOPOLOGY_SNAPSHOT_FORMAT,
        TOPOLOGY_SNAPSHOT_VERSION,
        load_topology,
        load_topology_metadata,
    )

    metadata = load_topology_metadata(path)
    actual_source = metadata.get("source")
    actual_seed = metadata.get("source_seed")
    if (
        not isinstance(actual_source, str)
        or isinstance(actual_seed, bool)
        or not isinstance(actual_seed, int)
        or (actual_source, actual_seed) != (source, seed)
    ):
        actual = (actual_source, actual_seed)
        raise ValueError(f"cached topology {path} has provenance {actual!r}; expected {(source, seed)!r}")
    actual_format_name = metadata.get("format")
    actual_format_version = metadata.get("format_version")
    actual_format = (actual_format_name, actual_format_version)
    expected_format = (TOPOLOGY_SNAPSHOT_FORMAT, TOPOLOGY_SNAPSHOT_VERSION)
    if (
        not isinstance(actual_format_name, str)
        or isinstance(actual_format_version, bool)
        or not isinstance(actual_format_version, int)
        or actual_format != expected_format
    ):
        raise ValueError(f"cached topology {path} has snapshot format {actual_format!r}; expected {expected_format!r}")
    # Validate required arrays and the scenario-configuration digest before a
    # generated file is published or reused. Generator revision metadata is
    # audit-only: the immutable snapshot defines the cached environment.
    load_topology(path)


def _export_snapshot(generator: str, seed: int, path: Path) -> None:
    """Import only the selected generator's implementation."""
    if generator == "jax":
        from jaxborg.scenarios.cc4.topology_cli import export_generated

        export_generated(seed, path)
        return
    from jaxborg.scenarios.cc4.topology_cli import export_cyborg

    export_cyborg(seed, path)


def materialize_topology_bank(
    section: Mapping[str, Any],
    *,
    scope: str,
    repo_root: Path,
) -> tuple[Path, ...]:
    """Resolve an explicit bank or deterministically materialize a generated one."""
    generated = parse_topology_generation(section, scope=scope, repo_root=repo_root)
    if generated is None:
        return explicit_topology_bank(section, repo_root=repo_root)

    generated.cache_dir.mkdir(parents=True, exist_ok=True)
    source = generated.source
    for seed, path in zip(generated.seeds, generated.paths, strict=True):
        if path.exists():
            _validate_cached_snapshot(path, source=source, seed=seed)
            continue
        handle = tempfile.NamedTemporaryFile(
            dir=generated.cache_dir,
            prefix=f".{path.stem}.",
            suffix=".tmp.npz",
            delete=False,
        )
        temporary = Path(handle.name)
        handle.close()
        try:
            _export_snapshot(generated.generator, seed, temporary)
            _validate_cached_snapshot(
                temporary,
                source=source,
                seed=seed,
            )
            temporary.chmod(0o644)
            # Hard-link publication is atomic and never replaces a winner
            # from another concurrent projector. Both paths are in the same
            # cache directory, so this does not cross filesystem boundaries.
            try:
                os.link(temporary, path)
            except FileExistsError:
                _validate_cached_snapshot(
                    path,
                    source=source,
                    seed=seed,
                )
        finally:
            temporary.unlink(missing_ok=True)
    return generated.paths


__all__ = [
    "MAX_TOPOLOGY_BANK_SIZE",
    "TopologyGenerationSpec",
    "expand_topology_bank",
    "explicit_topology_bank",
    "materialize_topology_bank",
    "parse_topology_generation",
    "validate_eval_topology_override",
    "validate_topology_split",
]
