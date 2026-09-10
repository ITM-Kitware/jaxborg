"""JAX-native learned Blue-vs-learned Red matchup evaluation.

The simulator is always :class:`JointPolicyCC4Env`.  Policy inference may be
performed by two Flax bundles or two Torch bundles; mixing frameworks in one
matchup is rejected so reproducibility and deployment dependencies stay
explicit.  The legacy CybORG Blue-vs-scripted-Red evaluator remains separate.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from functools import partial
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
from jaxborg.policies import initial_carry, is_recurrent, policy_from_arch, policy_step
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
    cia_metric: str | None = None
    cia_config: dict[str, Any] | None = None
    cia_summary: dict[str, Any] | None = None
    per_episode_cia: list[dict[str, float]] = field(default_factory=list)
    episode_role_map_ids: list[str] = field(default_factory=list)
    episode_topology_fingerprints: list[str] = field(default_factory=list)
    topology_role_maps: list[dict[str, Any]] = field(default_factory=list)


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
        module = policy_from_arch(arch, action_dim=action_dim)
    else:
        module = policy_from_arch(arch, action_dim=action_dim, backend=backend_name, obs_dim=obs_dim)
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


def _jax_actions(policy: LoadedMatchupPolicy, obs, mask, key, deterministic: bool, carry=None, reset=None):
    """Sample one joint action. Returns ``(actions, next_carry)``.

    ``carry`` is ``None`` for a feedforward policy and threads back unchanged,
    so the caller does not branch on the architecture.
    """
    pi, _, carry = policy_step(policy.module, policy.weights, obs, mask, carry=carry, reset=reset)
    actions = jnp.argmax(pi.logits, axis=-1) if deterministic else pi.sample(seed=key)
    return actions, carry


@partial(
    jax.jit,
    static_argnames=(
        "blue_module",
        "red_module",
        "env",
        "num_steps",
        "deterministic",
        "use_topology_index",
        "score_cia",
    ),
)
def _run_jax_matchup_episode_scan(
    blue_weights: Any,
    red_weights: Any,
    key: jax.Array,
    topology_index: jax.Array,
    host_resilience_role: jax.Array,
    *,
    blue_module: Any,
    red_module: Any,
    env: Any,
    num_steps: int,
    deterministic: bool,
    use_topology_index: bool,
    score_cia: bool,
) -> tuple[jax.Array, jax.Array]:
    """Compile policy inference and all simulator steps as one episode."""

    key, reset_key = jax.random.split(key)
    if use_topology_index:
        obs, state = env.reset_at_topology(reset_key, topology_index)
    else:
        obs, state = env.reset(reset_key)

    blue_agents = tuple(env.blue_agents)
    red_agents = tuple(env.red_agents)
    zero_cia = jnp.zeros(3, dtype=jnp.float32)
    # Only a recurrent Red has a sequence to restart, and only it needs the
    # simulator's activity mask read out on every step.
    red_recurrent = is_recurrent(red_module)

    def _active_step(rng, current_obs, current_state, carries, red_reset):
        masks = env.get_avail_actions(current_state)

        blue_obs = jnp.stack([current_obs[name] for name in blue_agents])
        blue_masks = jnp.stack([masks[name] for name in blue_agents])
        rng, blue_key = jax.random.split(rng)
        # Blue never goes dormant and the scan stops at termination, so within
        # one episode its sequence never restarts.
        blue_pi, _, carries["blue"] = policy_step(
            blue_module, blue_weights, blue_obs, blue_masks, carry=carries["blue"]
        )
        blue_actions = jnp.argmax(blue_pi.logits, axis=-1) if deterministic else blue_pi.sample(seed=blue_key)

        red_obs = jnp.stack([current_obs[name] for name in red_agents])
        red_masks = jnp.stack([masks[name] for name in red_agents])
        rng, red_key = jax.random.split(rng)
        # Red's sequence restarts whenever session reassignment revives a
        # dormant agent, matching the reset rule the joint trainer applies.
        red_pi, _, carries["red"] = policy_step(
            red_module, red_weights, red_obs, red_masks, carry=carries["red"], reset=red_reset
        )
        red_actions = jnp.argmax(red_pi.logits, axis=-1) if deterministic else red_pi.sample(seed=red_key)

        actions = {
            **{name: jnp.asarray(blue_actions[index], dtype=jnp.int32) for index, name in enumerate(blue_agents)},
            **{name: jnp.asarray(red_actions[index], dtype=jnp.int32) for index, name in enumerate(red_agents)},
        }
        rng, step_key = jax.random.split(rng)
        # ``JointPolicyCC4Env.step`` splits the caller key once before its
        # transition. Use that first child here so bypassing auto-reset does
        # not change the established reward-only rollout RNG stream.
        transition_key, _ = jax.random.split(step_key)
        next_obs, next_state, rewards, dones, _ = env.step_env(
            transition_key,
            current_state,
            actions,
        )
        if score_cia:
            from jaxborg.evaluation.cia.jax_resilience import score_resilience_state

            cia = score_resilience_state(next_state.state, host_resilience_role)
        else:
            cia = zero_cia
        reward = jnp.asarray(rewards[blue_agents[0]], dtype=jnp.float32)
        done = jnp.asarray(dones["__all__"], dtype=jnp.bool_)
        # Keyed on the pre-step activity, so the first live row after a gap is
        # the one that starts from a blank hidden state.
        next_red_reset = ~current_state.state.red_agent_active if red_recurrent else red_reset
        return rng, next_obs, next_state, reward, cia, done, carries, next_red_reset

    def _scan_step(carry, _):
        rng, current_obs, current_state, active, reward_sum, cia_sum, valid_steps, carries, red_reset = carry

        def run_active(_):
            next_rng, next_obs, next_state, reward, cia, done, next_carries, next_red_reset = _active_step(
                rng,
                current_obs,
                current_state,
                dict(carries),
                red_reset,
            )
            return (
                next_rng,
                next_obs,
                next_state,
                ~done,
                reward_sum + reward,
                cia_sum + cia,
                valid_steps + jnp.int32(1),
                next_carries,
                next_red_reset,
            )

        def keep_terminal(_):
            return (
                rng,
                current_obs,
                current_state,
                active,
                reward_sum,
                cia_sum,
                valid_steps,
                carries,
                red_reset,
            )

        return jax.lax.cond(active, run_active, keep_terminal, operand=None), None

    scan_carry = (
        key,
        obs,
        state,
        jnp.bool_(True),
        jnp.float32(0.0),
        zero_cia,
        jnp.int32(0),
        {
            "blue": initial_carry(blue_module, len(blue_agents)),
            "red": initial_carry(red_module, len(red_agents)),
        },
        jnp.ones((len(red_agents),), dtype=jnp.bool_),
    )
    final_carry, _ = jax.lax.scan(_scan_step, scan_carry, xs=None, length=num_steps)
    reward_sum = final_carry[4]
    cia_sum = final_carry[5]
    valid_steps = final_carry[6]
    cia_mean = jnp.where(valid_steps > 0, cia_sum / jnp.maximum(valid_steps, 1), zero_cia)
    return reward_sum, cia_mean


DEFAULT_EVAL_BATCH_SIZE = 64


def _eval_batch_size() -> int:
    """Episodes evaluated per vmapped call (``JAXBORG_EVAL_BATCH_SIZE``)."""
    import os

    raw = os.environ.get("JAXBORG_EVAL_BATCH_SIZE")
    if raw is None:
        return DEFAULT_EVAL_BATCH_SIZE
    value = int(raw)
    if value < 1:
        raise ValueError("JAXBORG_EVAL_BATCH_SIZE must be positive")
    return value


def _supports_batched_eval(env: Any) -> bool:
    """Whether ``env`` exposes the reset/step surface ``vmap`` batching needs.

    Test doubles substitute a stand-in env alongside a patched
    ``run_matchup_episode``; those fall back to the sequential seam instead of
    being vmapped.
    """
    return all(hasattr(env, name) for name in ("reset", "reset_at_topology", "step_env"))


def _run_jax_matchup_episodes_batched(
    policies: dict[str, LoadedMatchupPolicy],
    *,
    variant: GameVariant,
    env: Any,
    episode_seeds: Sequence[int],
    topology_indices: Sequence[int] | None,
    role_arrays: Sequence[Any] | None,
    deterministic: bool,
    batch_size: int | None = None,
    progress: bool = False,
    progress_label: str = "",
) -> tuple[list[float], list[list[float]]]:
    """Evaluate many episodes per compiled call via ``jax.vmap``.

    One episode is a 500-step scan over a single environment, which leaves an
    accelerator almost entirely idle — the work per kernel is tiny and the
    cost is dispatch latency. Mapping the identical per-episode scan over a
    batch gives the device the same shape of work training already runs at.
    Results are unchanged: each episode keeps its own key, topology and role
    map, so this is the sequential computation evaluated in parallel.
    """
    count = len(episode_seeds)
    if count == 0:
        return [], []
    score_cia = role_arrays is not None
    use_topology_index = topology_indices is not None
    chunk = batch_size or _eval_batch_size()

    scan = partial(
        _run_jax_matchup_episode_scan,
        blue_module=policies["blue"].module,
        red_module=policies["red"].module,
        env=env,
        num_steps=variant.num_steps,
        deterministic=deterministic,
        use_topology_index=use_topology_index,
        score_cia=score_cia,
    )
    # Weights are shared across the batch; keys, topologies and role maps vary.
    batched = jax.vmap(scan, in_axes=(None, None, 0, 0, 0))

    keys = jnp.stack([jax.random.PRNGKey(int(seed)) for seed in episode_seeds])
    if use_topology_index:
        indices = jnp.asarray(topology_indices, dtype=jnp.int32)
    else:
        indices = jnp.zeros(count, dtype=jnp.int32)
    if score_cia:
        roles = jnp.stack([jnp.asarray(role, dtype=jnp.int32) for role in role_arrays])
    else:
        roles = jnp.zeros((count, 1), dtype=jnp.int32)

    rewards: list[float] = []
    cia_scores: list[list[float]] = []
    for start in range(0, count, chunk):
        stop = min(start + chunk, count)
        reward_batch, cia_batch = batched(
            policies["blue"].weights,
            policies["red"].weights,
            keys[start:stop],
            indices[start:stop],
            roles[start:stop],
        )
        rewards.extend(float(value) for value in np.asarray(jax.device_get(reward_batch)))
        if score_cia:
            cia_scores.extend([float(value) for value in row] for row in np.asarray(jax.device_get(cia_batch)))
        else:
            cia_scores.extend([] for _ in range(stop - start))
        if progress:
            print(f"  {progress_label}episodes {stop}/{count}", flush=True)
    return rewards, cia_scores


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
    host_resilience_role: Any | None = None,
) -> float | tuple[float, list[float]]:
    """Run one episode and optionally return its temporal C/I/A means."""
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

    if backend == "jax":
        score_cia = host_resilience_role is not None
        role_array = jnp.asarray(host_resilience_role, dtype=jnp.int32) if score_cia else jnp.zeros(1, dtype=jnp.int32)
        reward, episode_cia = _run_jax_matchup_episode_scan(
            policies["blue"].weights,
            policies["red"].weights,
            jax.random.PRNGKey(seed),
            jnp.asarray(0 if topology_index is None else topology_index, dtype=jnp.int32),
            role_array,
            blue_module=policies["blue"].module,
            red_module=policies["red"].module,
            env=env,
            num_steps=variant.num_steps,
            deterministic=deterministic,
            use_topology_index=topology_index is not None,
            score_cia=score_cia,
        )
        reward_value = float(jax.device_get(reward))
        if not score_cia:
            return reward_value
        return reward_value, [float(value) for value in np.asarray(jax.device_get(episode_cia))]

    rng = jax.random.PRNGKey(seed)
    rng, reset_key = jax.random.split(rng)
    if topology_index is None:
        obs, state = env.reset(reset_key)
    else:
        obs, state = env.reset_at_topology(reset_key, topology_index)
    team_agents = {"blue": tuple(env.blue_agents), "red": tuple(env.red_agents)}
    # Recurrent policies only; `initial_carry` is None for the feedforward
    # archs and threads through the loop untouched.
    carries = {team: initial_carry(policies[team].module, len(names)) for team, names in team_agents.items()}
    red_recurrent = backend == "jax" and is_recurrent(policies["red"].module)
    red_reset = jnp.ones((len(team_agents["red"]),), dtype=jnp.bool_)
    total = 0.0
    cia_step_scores = []

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
                team_actions, carries[team] = _jax_actions(
                    policies[team],
                    obs_batch,
                    mask_batch,
                    policy_key,
                    deterministic,
                    carry=carries[team],
                    reset=red_reset if team == "red" else None,
                )
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

        if red_recurrent:
            # Keyed on the pre-step activity, so the first live row after a
            # dormancy gap is the one that starts from a blank hidden state.
            red_reset = ~state.state.red_agent_active
        rng, step_key = jax.random.split(rng)
        if host_resilience_role is None:
            obs, state, rewards, dones, _ = env.step(step_key, state, all_actions)
        else:
            # Keep the terminal state available for CIA scoring. ``step``
            # auto-resets and would otherwise score a healthy reset state on
            # the final timestep. Preserve ``step``'s transition RNG by using
            # the first child of its one key split.
            transition_key, _ = jax.random.split(step_key)
            obs, state, rewards, dones, _ = env.step_env(
                transition_key,
                state,
                all_actions,
            )
            from jaxborg.evaluation.cia.jax_resilience import score_resilience_state

            cia_step_scores.append(score_resilience_state(state.state, host_resilience_role))
        total += float(rewards[team_agents["blue"][0]])
        if bool(dones["__all__"]):
            break
    if host_resilience_role is None:
        return total

    from jaxborg.evaluation.cia.jax_resilience import mean_resilience_episode

    episode_cia = mean_resilience_episode(jnp.stack(cia_step_scores))
    return total, [float(value) for value in np.asarray(episode_cia)]


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
    cia: Mapping[str, Any] | None = None,
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

    from jaxborg.evaluation.cia.config import coerce_cia_settings, validate_cia_evaluation

    cia_settings = coerce_cia_settings(cia)
    validate_cia_evaluation(
        cia_settings,
        variant=variant,
        topology_sampling=topology_sampling,
        topology_paths=topology_paths,
        inspect_snapshots=cia_settings.enabled,
    )

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
    cia_episode_scores: list[list[float]] = []
    episode_role_map_ids: list[str] = []
    episode_topology_fingerprints: list[str] = []
    topology_role_maps: list[dict[str, Any]] = []
    if cia_settings.enabled:
        from jaxborg.evaluation.cia.fixed_topology import build_evaluation_cases

        cases = build_evaluation_cases(topology_paths, seeds, episodes_per_seed)
        seen_topologies: set[int] = set()
        for case in cases:
            if case.topology_index in seen_topologies:
                continue
            seen_topologies.add(case.topology_index)
            topology_role_maps.append(
                {
                    "topology_index": case.topology_index,
                    "topology_path": str(case.topology_path),
                    **case.audit_role_map(),
                }
            )

        total_episodes = len(cases)
        for case in cases:
            episode_seeds.append(case.episode_seed)
            episode_topology_paths.append(str(case.topology_path))
            episode_role_map_ids.append(case.role_map_id)
            episode_topology_fingerprints.append(case.topology_fingerprint)
        if backend_name == "jax" and _supports_batched_eval(env):
            blue_returns, cia_episode_scores = _run_jax_matchup_episodes_batched(
                policies,
                variant=variant,
                env=env,
                episode_seeds=[case.episode_seed for case in cases],
                topology_indices=[case.topology_index for case in cases],
                role_arrays=[case.role_array for case in cases],
                deterministic=deterministic,
                progress=progress,
            )
        else:
            for idx, case in enumerate(cases, start=1):
                score, episode_cia = run_matchup_episode(
                    policies,
                    variant=variant,
                    seed=case.episode_seed,
                    deterministic=deterministic,
                    env=env,
                    topology_index=case.topology_index,
                    host_resilience_role=case.role_array,
                )
                blue_returns.append(score)
                cia_episode_scores.append(episode_cia)
                if progress:
                    print(
                        f"  ep {idx}/{total_episodes} (seed={case.episode_seed}, "
                        f"topology={case.topology_path.name}): Blue {score:.1f}",
                        flush=True,
                    )
        sampling_label = "exhaustive"
    else:
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
        plan = [
            (base_seed + episode_idx, topology_index, topology_label)
            for topology_index, topology_label in topology_assignments
            for base_seed in seeds
            for episode_idx in range(episodes_per_seed)
        ]
        episode_seeds.extend(entry[0] for entry in plan)
        episode_topology_paths.extend(entry[2] for entry in plan)
        if backend_name == "jax" and _supports_batched_eval(env):
            # ``topology_index`` is None for random/generative sampling, where
            # each reset draws its own snapshot from its own key.
            indices = [entry[1] for entry in plan]
            blue_returns, _ = _run_jax_matchup_episodes_batched(
                policies,
                variant=variant,
                env=env,
                episode_seeds=[entry[0] for entry in plan],
                topology_indices=None if any(index is None for index in indices) else indices,
                role_arrays=None,
                deterministic=deterministic,
                progress=progress,
            )
        else:
            for idx, (episode_seed, topology_index, topology_label) in enumerate(plan, start=1):
                score = run_matchup_episode(
                    policies,
                    variant=variant,
                    seed=episode_seed,
                    deterministic=deterministic,
                    env=env,
                    topology_index=topology_index,
                )
                blue_returns.append(score)
                if progress:
                    topology_text = f", topology={Path(topology_label).name}" if topology_label else ""
                    print(
                        f"  ep {idx}/{total_episodes} (seed={episode_seed}{topology_text}): Blue {score:.1f}",
                        flush=True,
                    )

    cia_summary = None
    per_episode_cia: list[dict[str, float]] = []
    if cia_settings.enabled:
        from jaxborg.evaluation.cia.jax_resilience import (
            resilience_episode_records,
            summarize_resilience_episodes,
        )

        cia_summary = summarize_resilience_episodes(cia_episode_scores).to_dict()
        per_episode_cia = resilience_episode_records(cia_episode_scores)
    return MatchupEvaluation(
        blue_returns=blue_returns,
        red_returns=[-score for score in blue_returns],
        episode_seeds=episode_seeds,
        policies={team: policy.source for team, policy in policies.items()},
        topology_paths=[str(path) for path in topology_paths],
        episode_topology_paths=episode_topology_paths,
        topology_sampling=sampling_label,
        cia_metric=cia_settings.metric if cia_settings.enabled else None,
        cia_config=cia_settings.as_dict() if cia_settings.enabled else None,
        cia_summary=cia_summary,
        per_episode_cia=per_episode_cia,
        episode_role_map_ids=episode_role_map_ids,
        episode_topology_fingerprints=episode_topology_fingerprints,
        topology_role_maps=topology_role_maps,
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
