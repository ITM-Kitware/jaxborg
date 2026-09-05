"""JAX-native learned Blue-vs-learned Red matchup evaluation.

The simulator is always :class:`JointPolicyCC4Env`.  Policy inference may be
performed by two Flax bundles or two Torch bundles; mixing frameworks in one
matchup is rejected so reproducibility and deployment dependencies stay
explicit.  The legacy CybORG Blue-vs-scripted-Red evaluator remains separate.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal, Sequence

import jax
import jax.numpy as jnp
import numpy as np

from jaxborg.actions.encoding import (
    BLUE_ALLOW_TRAFFIC_END,
    BLUE_ALLOW_TRAFFIC_START,
    BLUE_ANALYSE_START,
    BLUE_BLOCK_TRAFFIC_START,
    BLUE_DECOY_START,
    BLUE_MONITOR,
    BLUE_REMOVE_START,
    BLUE_RESTORE_START,
    BLUE_SLEEP,
)
from jaxborg.checkpoint import (
    ModelBundle,
    PolicyBundleEntry,
    load_jax_bundle,
    load_torch_bundle,
    read_sidecar,
)
from jaxborg.constants import (
    BLUE_MAX_OBSERVED_SUBNETS,
    BLUE_OBS_SIZE,
    CYBORG_SUBNET_SUFFIX,
    NUM_SUBNETS,
    OBS_VECTOR_HOSTS_PER_SUBNET,
    SUBNET_NAMES,
)
from jaxborg.evaluation.jax_env_factory import make_joint_jax_env
from jaxborg.learned_red import RED_OBS_SIZE, RED_POLICY_ACTION_DIM
from jaxborg.policies import make_jax_policy, make_torch_policy
from jaxborg.recipe import team_recipe
from jaxborg.scenarios.cc4.game_variant import GameVariant

PolicyBackend = Literal["jax", "cyborg"]
TEAM_DIMS = {
    "blue": (BLUE_OBS_SIZE, BLUE_ALLOW_TRAFFIC_END),
    "red": (RED_OBS_SIZE, RED_POLICY_ACTION_DIM),
}


def cyborg_blue_flat_to_jax_lookup(const, agent_id: int) -> np.ndarray:
    """Map CybORG's padded 242-action Blue ordering to JAXborg indices.

    The two Blue APIs intentionally share a dimension, not an index ordering.
    CybORG groups only the subnets owned by a particular agent and pads at the
    end, whereas JAXborg reserves three host/subnet slots in every action
    block.  Host-slot and subnet assignment are fixed by the CC4 contract, so
    this lookup is derived from ``SimulatorConst`` without a shadow simulator.
    """

    observed = np.asarray(const.blue_obs_subnets[agent_id], dtype=np.int32)
    observed = observed[observed >= 0]
    host_count = len(observed) * OBS_VECTOR_HOSTS_PER_SUBNET
    lookup = np.full(BLUE_ALLOW_TRAFFIC_END, -1, dtype=np.int32)
    cursor = 0

    lookup[cursor : cursor + host_count] = BLUE_ANALYSE_START + np.arange(host_count)
    cursor += host_count
    lookup[cursor] = BLUE_MONITOR
    cursor += 1
    lookup[cursor : cursor + host_count] = BLUE_REMOVE_START + np.arange(host_count)
    cursor += host_count
    lookup[cursor : cursor + host_count] = BLUE_RESTORE_START + np.arange(host_count)
    cursor += host_count
    lookup[cursor] = BLUE_SLEEP
    cursor += 1

    # BlueFlatWrapper orders traffic outer-by-destination (the observed
    # subnets) and inner-by alphabetically sorted source subnet. JAXborg uses
    # outer-by compressed absolute source and inner-by one of three relative
    # destinations.
    cyborg_subnet_order = sorted(
        range(NUM_SUBNETS),
        key=lambda subnet_id: CYBORG_SUBNET_SUFFIX[SUBNET_NAMES[subnet_id]],
    )
    for canonical_start in (BLUE_ALLOW_TRAFFIC_START, BLUE_BLOCK_TRAFFIC_START):
        for relative_dst, dst in enumerate(observed):
            for src in (subnet for subnet in cyborg_subnet_order if subnet != int(dst)):
                src_offset = src if src < dst else src - 1
                lookup[cursor] = canonical_start + src_offset * BLUE_MAX_OBSERVED_SUBNETS + relative_dst
                cursor += 1

    lookup[cursor : cursor + host_count] = BLUE_DECOY_START + np.arange(host_count)
    cursor += host_count
    if cursor > BLUE_ALLOW_TRAFFIC_END:  # pragma: no cover - static contract guard
        raise ValueError(f"CybORG Blue action layout overflow for agent {agent_id}: {cursor}")
    return lookup


def jax_mask_to_cyborg_blue(mask: Any, lookup: np.ndarray) -> np.ndarray:
    """Reorder a canonical JAX Blue mask for a Torch/CybORG policy head."""

    canonical = np.asarray(mask, dtype=bool)
    out = np.zeros(BLUE_ALLOW_TRAFFIC_END, dtype=bool)
    valid = lookup >= 0
    out[valid] = canonical[lookup[valid]]
    return out


@dataclass(frozen=True)
class LoadedMatchupPolicy:
    team: str
    backend: PolicyBackend
    module: Any
    weights: Any
    source: dict[str, Any]


@dataclass(frozen=True)
class MatchupEvaluation:
    blue_returns: list[float]
    red_returns: list[float]
    episode_seeds: list[int]
    policies: dict[str, dict[str, Any]]
    topology_paths: list[str] = field(default_factory=list)
    episode_topology_paths: list[str | None] = field(default_factory=list)
    topology_sampling: str = "generative"


def _normalise_backend(backend: str) -> PolicyBackend:
    value = "cyborg" if backend == "torch" else backend
    if value not in ("jax", "cyborg"):
        raise ValueError(f"policy backend must be 'jax' or 'cyborg', got {backend!r}")
    return value  # type: ignore[return-value]


def _source_sidecar(path: Path) -> dict[str, Any] | None:
    try:
        return read_sidecar(path)
    except FileNotFoundError:
        return None


def _entry_arch(path: Path, entry: PolicyBundleEntry, team: str) -> tuple[dict, dict | None]:
    sidecar = _source_sidecar(path)
    if entry.arch.get("name"):
        return dict(entry.arch), sidecar
    if sidecar is None:
        raise ValueError(f"legacy {team} model {path} has no architecture metadata and no recipe sidecar")
    return dict(team_recipe(sidecar, team)["arch"]), sidecar


def _bundle_entry(bundle: ModelBundle, path: Path, team: str) -> PolicyBundleEntry:
    if team not in bundle.policies:
        raise ValueError(f"{path} has no {team} policy; available: {sorted(bundle.policies)}")
    entry = bundle.policies[team]
    obs_dim, action_dim = TEAM_DIMS[team]
    if entry.obs_dim not in (0, obs_dim):
        raise ValueError(f"{team} observation dimension mismatch: model={entry.obs_dim}, expected={obs_dim}")
    if entry.action_dim not in (0, action_dim):
        raise ValueError(f"{team} action dimension mismatch: model={entry.action_dim}, expected={action_dim}")
    return entry


def load_matchup_policy(path: str | Path, *, team: str, backend: str) -> LoadedMatchupPolicy:
    """Load one team policy with strict contract and backend validation."""
    if team not in TEAM_DIMS:
        raise ValueError(f"unknown team {team!r}")
    backend_name = _normalise_backend(backend)
    model_path = Path(path).expanduser().resolve()
    if not model_path.is_file():
        raise FileNotFoundError(f"Model not found: {model_path}")
    expected_suffix = ".safetensors" if backend_name == "jax" else ".pt"
    if model_path.suffix != expected_suffix:
        raise ValueError(
            f"{team} model {model_path} does not match {backend_name} backend (expected {expected_suffix})"
        )

    bundle = load_jax_bundle(model_path) if backend_name == "jax" else load_torch_bundle(model_path)
    if bundle.backend != backend_name:
        raise ValueError(f"{team} bundle backend is {bundle.backend!r}, expected {backend_name!r}")
    entry = _bundle_entry(bundle, model_path, team)
    arch, sidecar = _entry_arch(model_path, entry, team)
    obs_dim, action_dim = TEAM_DIMS[team]
    if backend_name == "jax":
        module = make_jax_policy(
            arch["name"],
            action_dim=action_dim,
            hidden_dim=int(arch.get("hidden_dim", 256)),
            hidden_layers=int(arch.get("hidden_layers", 2)),
            activation=arch.get("activation", "tanh"),
        )
    else:
        module = make_torch_policy(
            arch["name"],
            obs_dim=obs_dim,
            action_dim=action_dim,
            hidden_dim=int(arch.get("hidden_dim", 256)),
            hidden_layers=int(arch.get("hidden_layers", 2)),
        )
        module.load_state_dict(entry.weights)
        module.eval()

    run = (sidecar or {}).get("run", {})
    source = {
        "path": str(model_path),
        "backend": backend_name,
        "team": team,
        "bundle_schema": bundle.schema_version,
        "bundle_legacy": bundle.legacy,
        "bundle_teams": sorted(bundle.policies),
        "bundle_trainable": entry.trainable,
        "observation_dim": entry.obs_dim or obs_dim,
        "action_dim": entry.action_dim or action_dim,
        "bundle_source": entry.source,
        "bundle_provenance": bundle.provenance,
        "train_run_id": run.get("train_run_id"),
        "train_seed": run.get("seed"),
        "recipe_name": (sidecar or {}).get("meta", {}).get("name"),
        "arch": arch,
    }
    return LoadedMatchupPolicy(team, backend_name, module, entry.weights, source)


def _jax_actions(policy: LoadedMatchupPolicy, obs, mask, key, deterministic: bool):
    pi, _ = policy.module.apply(policy.weights, obs, mask)
    return jnp.argmax(pi.logits, axis=-1) if deterministic else pi.sample(seed=key)


def _torch_actions(policy: LoadedMatchupPolicy, obs, mask, seed: int, deterministic: bool):
    import torch

    # Categorical.sample has no generator argument. Isolate deterministic
    # episode/step randomness by reseeding immediately before inference.
    torch.manual_seed(int(seed) & 0x7FFF_FFFF_FFFF_FFFF)
    # JAX may expose read-only NumPy views; Torch warns when wrapping them
    # without a copy even though inference itself is read-only.
    obs_tensor = torch.tensor(np.asarray(obs), dtype=torch.float32)
    mask_tensor = torch.tensor(np.asarray(mask), dtype=torch.bool)
    with torch.no_grad():
        if deterministic:
            actions = policy.module.deterministic_action(obs_tensor, mask_tensor)
        else:
            actions = policy.module.get_action_and_value(obs_tensor, mask_tensor)[0]
    return np.asarray(actions.cpu(), dtype=np.int32)


def run_matchup_episode(
    policies: dict[str, LoadedMatchupPolicy],
    *,
    variant: GameVariant,
    seed: int,
    deterministic: bool = False,
    topology_path: str | Path | Sequence[str | Path] | None = None,
    env: Any | None = None,
    topology_index: int | jax.Array | None = None,
) -> float:
    """Run one episode and return the Blue game score."""
    if set(policies) != {"blue", "red"}:
        raise ValueError("a learned matchup requires both Blue and Red policies")
    backends = {policy.backend for policy in policies.values()}
    if len(backends) != 1:
        raise ValueError("mixed JAX/Torch matchup policies are not supported")
    backend = next(iter(backends))

    if env is None:
        env = make_joint_jax_env(
            variant,
            training_mode=False,
            topology_path=topology_path,
        )
    elif topology_path is not None:
        raise ValueError("topology_path cannot be supplied with a pre-built env")
    rng = jax.random.PRNGKey(seed)
    rng, reset_key = jax.random.split(rng)
    if topology_index is None:
        obs, state = env.reset(reset_key)
    else:
        obs, state = env.reset_at_topology(reset_key, topology_index)
    team_agents = {"blue": tuple(env.blue_agents), "red": tuple(env.red_agents)}
    total = 0.0

    for step_idx in range(variant.num_steps):
        masks = env.get_avail_actions(state)
        all_actions = {}
        # Both policies observe the same state before the sole environment step.
        for team in ("blue", "red"):
            names = team_agents[team]
            obs_batch = jnp.stack([obs[name] for name in names])
            mask_batch = jnp.stack([masks[name] for name in names])
            if backend == "jax":
                rng, policy_key = jax.random.split(rng)
                team_actions = _jax_actions(policies[team], obs_batch, mask_batch, policy_key, deterministic)
            else:
                torch_seed = seed * 1_000_003 + step_idx * 17 + (0 if team == "blue" else 1)
                policy_mask = mask_batch
                blue_lookups = None
                if team == "blue":
                    blue_lookups = [
                        cyborg_blue_flat_to_jax_lookup(state.const, agent_id) for agent_id in range(len(names))
                    ]
                    policy_mask = np.stack(
                        [
                            jax_mask_to_cyborg_blue(mask_batch[agent_id], lookup)
                            for agent_id, lookup in enumerate(blue_lookups)
                        ]
                    )
                team_actions = _torch_actions(
                    policies[team],
                    obs_batch,
                    policy_mask,
                    torch_seed,
                    deterministic,
                )
                if blue_lookups is not None:
                    team_actions = np.asarray(
                        [blue_lookups[agent_id][int(action)] for agent_id, action in enumerate(team_actions)],
                        dtype=np.int32,
                    )
                    if np.any(team_actions < 0):  # pragma: no cover - masked defensive guard
                        raise RuntimeError("Torch Blue policy selected a padded CybORG action")
            for idx, name in enumerate(names):
                all_actions[name] = jnp.asarray(team_actions[idx], dtype=jnp.int32)

        rng, step_key = jax.random.split(rng)
        obs, state, rewards, dones, _ = env.step(step_key, state, all_actions)
        total += float(rewards[team_agents["blue"][0]])
        if bool(dones["__all__"]):
            break
    return total


def evaluate_matchup(
    blue_model: str | Path,
    red_model: str | Path,
    *,
    backend: str,
    variant: GameVariant,
    seeds: list[int],
    episodes_per_seed: int = 1,
    deterministic: bool = False,
    progress: bool = True,
    topology_path: str | Path | Sequence[str | Path] | None = None,
    topology_sampling: str = "exhaustive",
) -> MatchupEvaluation:
    """Evaluate independently sourced learned policies in the JAX simulator.

    A configured bank defaults to exhaustive evaluation: every expanded
    episode seed is run once on every topology. ``random`` retains the
    training-style behavior where each reset samples the complete bank with
    replacement.
    """
    backend_name = _normalise_backend(backend)
    policies = {
        "blue": load_matchup_policy(blue_model, team="blue", backend=backend_name),
        "red": load_matchup_policy(red_model, team="red", backend=backend_name),
    }
    if topology_path is None:
        topology_paths: list[Path] = []
    elif isinstance(topology_path, (str, Path)):
        topology_paths = [Path(topology_path).expanduser().resolve()]
    else:
        topology_paths = [Path(path).expanduser().resolve() for path in topology_path]
        if not topology_paths:
            raise ValueError("topology_path must contain at least one snapshot path")
    if topology_sampling not in ("exhaustive", "random"):
        raise ValueError("topology_sampling must be 'exhaustive' or 'random'")

    # Load and stack the bank once. Reconstructing an environment per episode
    # becomes prohibitively expensive for exhaustive held-out evaluations.
    env = make_joint_jax_env(
        variant,
        training_mode=False,
        topology_path=topology_paths or None,
    )

    blue_returns = []
    episode_seeds = []
    episode_topology_paths: list[str | None] = []
    if topology_paths and topology_sampling == "exhaustive":
        topology_assignments: list[tuple[int | None, str | None]] = [
            (index, str(path)) for index, path in enumerate(topology_paths)
        ]
        sampling_label = "exhaustive"
    elif topology_paths:
        topology_assignments = [(None, None)]
        sampling_label = "random"
    else:
        topology_assignments = [(None, None)]
        sampling_label = "generative"
    total_episodes = len(topology_assignments) * len(seeds) * episodes_per_seed
    idx = 0
    for topology_index, topology_label in topology_assignments:
        for base_seed in seeds:
            for episode_idx in range(episodes_per_seed):
                episode_seed = base_seed + episode_idx
                score = run_matchup_episode(
                    policies,
                    variant=variant,
                    seed=episode_seed,
                    deterministic=deterministic,
                    env=env,
                    topology_index=topology_index,
                )
                blue_returns.append(score)
                episode_seeds.append(episode_seed)
                episode_topology_paths.append(topology_label)
                idx += 1
                if progress:
                    topology_text = f", topology={Path(topology_label).name}" if topology_label else ""
                    print(
                        f"  ep {idx}/{total_episodes} (seed={episode_seed}{topology_text}): Blue {score:.1f}",
                        flush=True,
                    )
    return MatchupEvaluation(
        blue_returns=blue_returns,
        red_returns=[-score for score in blue_returns],
        episode_seeds=episode_seeds,
        policies={team: policy.source for team, policy in policies.items()},
        topology_paths=[str(path) for path in topology_paths],
        episode_topology_paths=episode_topology_paths,
        topology_sampling=sampling_label,
    )


__all__ = [
    "LoadedMatchupPolicy",
    "MatchupEvaluation",
    "cyborg_blue_flat_to_jax_lookup",
    "evaluate_matchup",
    "jax_mask_to_cyborg_blue",
    "load_matchup_policy",
    "run_matchup_episode",
]
