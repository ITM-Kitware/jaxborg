#!/usr/bin/env python3
"""Parallel topology bank generator.

Generates topology, green-random, and red-policy banks using multiprocessing.
Saves to the same cache location that training expects (.bank_cache/).

Usage:
    uv run python scripts/train/generate_bank.py --bank-size 1024 --workers 56
    uv run python scripts/train/generate_bank.py --bank-size 1024 --only topo  # just topologies
"""

import argparse
import hashlib
import multiprocessing
import pickle
import signal
import time
from functools import partial
from pathlib import Path

import numpy as np

# ---------------------------------------------------------------------------
# Cache-key helpers (must match topology.py exactly)
# ---------------------------------------------------------------------------

_TOPOLOGY_PY = Path(__file__).resolve().parents[1] / "src" / "jaxborg" / "topology.py"
_BANK_CACHE_DIR = Path(__file__).resolve().parents[1] / ".bank_cache"


def _hash_paths(*abs_paths: Path) -> str:
    digest = hashlib.md5()
    for p in abs_paths:
        digest.update(p.read_bytes())
    return digest.hexdigest()[:12]


def _topo_cache_path(num_steps: int, bank_size: int) -> Path:
    h = _hash_paths(_TOPOLOGY_PY)
    return _BANK_CACHE_DIR / f"topo_steps{num_steps}_bank{bank_size}_{h}.pkl"


def _green_cache_path(num_steps: int, bank_size: int) -> Path:
    green_py = _TOPOLOGY_PY.parent / "cyborg_green_recorder.py"
    h = _hash_paths(_TOPOLOGY_PY, green_py)
    return _BANK_CACHE_DIR / f"green_steps{num_steps}_bank{bank_size}_{h}.pkl"


def _red_policy_cache_path(num_steps: int, bank_size: int) -> Path:
    red_py = _TOPOLOGY_PY.parent / "cyborg_red_policy_recorder.py"
    fsm_py = _TOPOLOGY_PY.parent / "agents" / "fsm_red.py"
    h = _hash_paths(_TOPOLOGY_PY, red_py, fsm_py)
    return _BANK_CACHE_DIR / f"red_policy_steps{num_steps}_bank{bank_size}_{h}.pkl"


# ---------------------------------------------------------------------------
# Per-seed worker functions (run in subprocesses)
# ---------------------------------------------------------------------------


def _init_worker():
    """Ignore SIGINT in workers so Ctrl-C is handled by the main process."""
    signal.signal(signal.SIGINT, signal.SIG_IGN)


def _build_one_topology(seed: int, num_steps: int) -> dict:
    """Build a single SimulatorConst from one CybORG seed; return as numpy dict."""
    import jax

    # Prevent workers from grabbing GPU
    jax.config.update("jax_platforms", "cpu")

    from CybORG import CybORG
    from CybORG.Agents import EnterpriseGreenAgent, FiniteStateRedAgent, SleepAgent
    from CybORG.Simulator.Scenarios import EnterpriseScenarioGenerator

    from jaxborg.scenarios.cc4.topology import build_const_from_cyborg

    scenario = EnterpriseScenarioGenerator(
        blue_agent_class=SleepAgent,
        green_agent_class=EnterpriseGreenAgent,
        red_agent_class=FiniteStateRedAgent,
        steps=num_steps,
    )
    cyborg = CybORG(scenario_generator=scenario, seed=seed)
    cyborg.reset()
    const = build_const_from_cyborg(cyborg)
    # Convert to numpy dict for pickling across processes
    return jax.tree.map(np.asarray, const)


def _build_one_green(seed: int, num_steps: int) -> np.ndarray:
    """Record green random tape for one CybORG seed; return numpy array."""
    import jax

    jax.config.update("jax_platforms", "cpu")

    from CybORG import CybORG
    from CybORG.Agents import EnterpriseGreenAgent, FiniteStateRedAgent, SleepAgent
    from CybORG.Agents.Wrappers import BlueFlatWrapper
    from CybORG.Simulator.Actions import Sleep
    from CybORG.Simulator.Scenarios import EnterpriseScenarioGenerator

    from jaxborg.parity.cyborg_green_recorder import GreenRecorder
    from jaxborg.parity.translate import build_mappings_from_cyborg

    scenario = EnterpriseScenarioGenerator(
        blue_agent_class=SleepAgent,
        green_agent_class=EnterpriseGreenAgent,
        red_agent_class=FiniteStateRedAgent,
        steps=num_steps,
    )
    cyborg = CybORG(scenario_generator=scenario, seed=seed)
    wrapper = BlueFlatWrapper(env=cyborg, pad_spaces=True)
    wrapper.reset()

    mappings = build_mappings_from_cyborg(cyborg)
    recorder = GreenRecorder()
    recorder.install(cyborg, mappings)

    sleep_actions = {agent: Sleep() for agent in wrapper.agents}
    for step_idx in range(num_steps):
        wrapper.step(actions=sleep_actions)
        recorder.extract_step(step_idx)

    return np.asarray(recorder.to_jax_array())


def _build_one_red_policy(seed: int, num_steps: int) -> np.ndarray:
    """Record red policy random tape for one CybORG seed; return numpy array."""
    import jax

    jax.config.update("jax_platforms", "cpu")

    from CybORG import CybORG
    from CybORG.Agents import EnterpriseGreenAgent, FiniteStateRedAgent, SleepAgent
    from CybORG.Agents.Wrappers import BlueFlatWrapper
    from CybORG.Simulator.Actions import Sleep
    from CybORG.Simulator.Scenarios import EnterpriseScenarioGenerator

    from jaxborg.parity.cyborg_red_policy_recorder import RedPolicyRecorder
    from jaxborg.parity.translate import build_mappings_from_cyborg

    scenario = EnterpriseScenarioGenerator(
        blue_agent_class=SleepAgent,
        green_agent_class=EnterpriseGreenAgent,
        red_agent_class=FiniteStateRedAgent,
        steps=num_steps,
    )
    cyborg = CybORG(scenario_generator=scenario, seed=seed)
    wrapper = BlueFlatWrapper(env=cyborg, pad_spaces=True)
    wrapper.reset()

    recorder = RedPolicyRecorder()
    recorder.install(cyborg, build_mappings_from_cyborg(cyborg))

    sleep_actions = {agent: Sleep() for agent in wrapper.agents}
    for _ in range(num_steps):
        wrapper.step(actions=sleep_actions)

    return np.asarray(recorder.to_jax_array())


# ---------------------------------------------------------------------------
# Main: parallel generation + caching
# ---------------------------------------------------------------------------


def generate_bank(bank_size: int, num_steps: int, workers: int, only: str | None = None):
    import jax

    _BANK_CACHE_DIR.mkdir(parents=True, exist_ok=True)

    components = ["topo", "green", "red_policy"] if only is None else [only]

    for component in components:
        if component == "topo":
            cache_path = _topo_cache_path(num_steps, bank_size)
        elif component == "green":
            cache_path = _green_cache_path(num_steps, bank_size)
        elif component == "red_policy":
            cache_path = _red_policy_cache_path(num_steps, bank_size)
        else:
            raise ValueError(f"Unknown component: {component}")

        if cache_path.exists():
            print(f"[{component}] Already cached at {cache_path} — skipping")
            continue

        print(f"[{component}] Generating bank_size={bank_size} with {workers} workers...")
        t0 = time.time()

        if component == "topo":
            worker_fn = partial(_build_one_topology, num_steps=num_steps)
        elif component == "green":
            worker_fn = partial(_build_one_green, num_steps=num_steps)
        elif component == "red_policy":
            worker_fn = partial(_build_one_red_policy, num_steps=num_steps)

        with multiprocessing.get_context("spawn").Pool(workers, initializer=_init_worker) as pool:
            results = []
            for i, result in enumerate(pool.imap(worker_fn, range(bank_size))):
                results.append(result)
                if (i + 1) % 50 == 0 or i + 1 == bank_size:
                    elapsed = time.time() - t0
                    rate = (i + 1) / elapsed
                    eta = (bank_size - i - 1) / rate if rate > 0 else 0
                    print(
                        f"  [{component}] {i + 1}/{bank_size} done ({elapsed:.0f}s elapsed, {eta:.0f}s remaining)",
                        flush=True,
                    )

        elapsed = time.time() - t0
        print(f"[{component}] All {bank_size} seeds done in {elapsed:.1f}s")

        # Stack and save
        if component == "topo":
            # results is a list of pytree dicts — stack with jax
            stacked = jax.tree.map(
                lambda *xs: np.stack(xs, axis=0),
                *results,
            )
            # Save as numpy for the cache
            with open(cache_path, "wb") as f:
                pickle.dump(stacked, f)
        else:
            # results is a list of numpy arrays — stack directly
            stacked = np.stack(results, axis=0)
            with open(cache_path, "wb") as f:
                pickle.dump(stacked, f)

        size_mb = cache_path.stat().st_size / 1e6
        print(f"[{component}] Cached to {cache_path} ({size_mb:.1f} MB)")

    print("\nDone! Bank is ready for training with +TOPOLOGY_BANK_SIZE={bank_size}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Parallel topology bank generator")
    parser.add_argument("--bank-size", type=int, default=1024)
    parser.add_argument("--num-steps", type=int, default=500, help="Episode length")
    parser.add_argument("--workers", type=int, default=56, help="Parallel workers")
    parser.add_argument(
        "--only",
        choices=["topo", "green", "red_policy"],
        default=None,
        help="Generate only one component",
    )
    args = parser.parse_args()
    generate_bank(args.bank_size, args.num_steps, args.workers, args.only)
