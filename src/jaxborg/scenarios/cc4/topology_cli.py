from __future__ import annotations

import argparse
import functools
from pathlib import Path

import jax

from jaxborg.scenarios.cc4.topology import build_const_from_cyborg, build_topology, save_topology


def _positive_seed(value: str) -> int:
    seed = int(value)
    if seed < 0:
        raise argparse.ArgumentTypeError("seed must be non-negative")
    return seed


@functools.lru_cache(maxsize=None)
def _compiled_builder(op_zone_servers: int | None):
    """Return an XLA-compiled ``build_topology`` for one topology shape.

    Eager ``build_topology`` dispatches several hundred unfused ops per call
    (~2s each), so materializing a 100-snapshot bank spends minutes in
    dispatch overhead. Compiling once per ``op_zone_servers`` value and
    reusing it across every seed in the bank drops that to ~1ms per seed.
    """
    return jax.jit(functools.partial(build_topology, op_zone_min_servers=op_zone_servers))


def export_generated(
    seed: int,
    out: str | Path,
    *,
    op_zone_servers: int | None = None,
) -> None:
    const = _compiled_builder(op_zone_servers)(jax.random.PRNGKey(seed))
    # ``jit`` traces the ``max_steps`` field into an int32 array; restore the
    # Python int so snapshots stay byte-identical to eagerly built ones.
    const = const.replace(max_steps=int(const.max_steps))
    save_topology(
        const,
        out,
        metadata={
            "source": "generated",
            "source_seed": seed,
            "op_zone_servers": op_zone_servers,
        },
    )


def export_cyborg(seed: int, out: str | Path) -> None:
    from CybORG import CybORG
    from CybORG.Agents import EnterpriseGreenAgent, FiniteStateRedAgent, SleepAgent
    from CybORG.Simulator.Scenarios import EnterpriseScenarioGenerator

    scenario = EnterpriseScenarioGenerator(
        blue_agent_class=SleepAgent,
        green_agent_class=EnterpriseGreenAgent,
        red_agent_class=FiniteStateRedAgent,
        steps=500,
    )
    cyborg = CybORG(scenario_generator=scenario, seed=seed)
    cyborg.reset()
    const = build_const_from_cyborg(cyborg)
    save_topology(
        const,
        out,
        metadata={
            "source": "cyborg",
            "source_seed": seed,
        },
    )


def _parse_export_args(description: str) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument("--seed", type=_positive_seed, required=True)
    parser.add_argument("--out", type=Path, required=True)
    return parser.parse_args()


def export_generated_main() -> None:
    """Console-script entry point for ``export-generated-topology``."""
    args = _parse_export_args("Export a generated JAXborg CC4 topology snapshot")
    export_generated(args.seed, args.out)


def export_cyborg_main() -> None:
    """Console-script entry point for ``export-cyborg-topology``."""
    args = _parse_export_args("Export a live CybORG CC4 topology snapshot")
    export_cyborg(args.seed, args.out)
